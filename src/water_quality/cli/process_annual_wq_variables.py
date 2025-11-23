import json
import os
import sys
import warnings
from itertools import chain

import click
import numpy as np
import xarray as xr
from datacube import Datacube
from deafrica_tools.dask import create_local_dask_cluster
from odc.geo.xr import write_cog
from odc.stats._cli_common import click_yaml_cfg
from odc.stats._text import split_and_check
from odc.stats.model import DateTimeRange

from water_quality.grid import check_resolution, get_waterbodies_grid
from water_quality.io import (
    check_directory_exists,
    check_file_exists,
    get_filesystem,
    get_parent_dir,
    get_wq_cog_url,
    get_wq_csv_url,
    get_wq_dataset_path,
    get_wq_stac_url,
    join_url,
)
from water_quality.logs import setup_logging
from water_quality.mapping.algorithms import WQ_vars
from water_quality.mapping.config import check_config
from water_quality.mapping.hue import hue_calculation
from water_quality.mapping.instruments import (
    check_instrument_dates,
    get_instruments_list,
)
from water_quality.mapping.load_data import (
    build_dc_queries,
    build_wq_agm_dataset,
)
from water_quality.mapping.optical_water_type import OWT_pixel
from water_quality.mapping.pixel_correction import R_correction
from water_quality.mapping.water_detection import water_analysis
from water_quality.metadata.prepare_metadata import prepare_dataset
from water_quality.tasks import parse_task_id


@click.command(
    name="process-annual-wq-variables",
    no_args_is_help=True,
)
@click.option(
    "--tasks",
    help="List of comma separated tasks in the format"
    "period/x{x:02d}/y{y:02d} to generate water quality variables for. "
    "For example `2015--P1Y/x200/y34,2015--P1Y/x178/y095, 2015--P1Y/x199y/100`",
)
@click.option(
    "--tasks-file",
    help="Optional path to a text file containing the tasks to generate"
    "water quality variables for. This file can be generated using the "
    "command `wq-generate-tiles`.",
)
@click.argument(
    "output-directory",
    type=str,
)
@click.argument(
    "max-parallel-steps",
    type=int,
)
@click.argument(
    "worker-idx",
    type=int,
)
@click_yaml_cfg(
    "--analysis-config",
    required=True,
    help="Config for the analysis parameters in yaml format, file or text",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help=(
        "If overwrite is True tasks that have already been processed "
        "will be rerun. "
    ),
)
def cli(
    tasks: str,
    tasks_file: str,
    output_directory: str,
    max_parallel_steps: int,
    worker_idx: int,
    analysis_config,
    overwrite: bool,
):
    """
    Get the annual Water Quality variables for the input tasks and write
    the resulting water quality variables to COG files.

    OUTPUT_DIRECTORY: The directory to write the water quality variables
    COG files to for each task.

    MAX_PARALLEL_STEPS: The total number of parallel workers or pods
    expected in the workflow. This value is used to divide the list of
    tasks to be processed among the available workers.

    WORKER_IDX: The sequential index (0-indexed) of the current worker.
    This index determines which subset of tasks the current worker will
    process.
    """
    log = setup_logging()

    # Enforce mutual exclusivity
    if tasks and tasks_file:
        raise click.UsageError("Use either --tasks or --tasks-file, not both.")

    if not tasks and not tasks_file:
        raise click.UsageError(
            "Please provide either --tasks or --tasks-file."
        )

    if tasks:
        all_task_ids = tasks.split(",")
        all_task_ids = [i.strip() for i in all_task_ids]

    if tasks_file:
        # Assumption here is the file is public-read.
        if not check_file_exists(tasks_file):
            raise FileNotFoundError(f"{tasks_file} does not exist!")
        else:
            fs = get_filesystem(tasks_file, anon=True)
            with fs.open(tasks_file, "r") as f:
                all_task_ids = f.readlines()
                all_task_ids = [i.strip() for i in all_task_ids]

    log.info(f"Total number of tasks found: {len(all_task_ids)}")

    # Split tasks equally among the workers
    task_chunks = np.array_split(np.array(all_task_ids), max_parallel_steps)
    task_chunks = [chunk.tolist() for chunk in task_chunks]
    task_chunks = list(filter(None, task_chunks))

    # In case of the index being bigger than the number of positions
    # in the array, the extra POD isn't necessary
    if len(task_chunks) <= worker_idx:
        log.warning(f"Worker {worker_idx} Skipped!")
        sys.exit(0)

    log.info(f"Executing worker {worker_idx}")
    task_ids = task_chunks[worker_idx]
    log.info(f"Worker {worker_idx} to process {len(task_ids)} tasks.")

    # ------------------------------------------------ #
    # Get water quality variables                      #
    # ------------------------------------------------ #
    analysis_config = check_config(analysis_config)

    resolution_m = check_resolution(int(analysis_config["resolution"]))
    WFTH = analysis_config["water_frequency_threshold_high"]
    WFTL = analysis_config["water_frequency_threshold_low"]
    PWT = analysis_config["permanent_water_threshold"]
    SC = analysis_config["sigma_coefficient"]
    product_info = analysis_config["product"]
    product_name = product_info["name"]
    product_version = product_info["version"]

    gridspec = get_waterbodies_grid(resolution_m)
    dc = Datacube(app="ProcessAnnualWQvariables")
    failed_tasks = []
    for idx, task_id in enumerate(task_ids):
        log.info(f"Processing task {task_id} {idx + 1} / {len(task_ids)}")

        try:
            temporal_id, tile_id = parse_task_id(task_id)

            # Enforce this command line tool only works for
            # annual tasks.
            _, freq = split_and_check(temporal_id, "--P", 2)
            if freq != "1Y":
                raise ValueError(
                    f"Expecting tasks with an annual frequency '1Y' not {freq}"
                )

            # Check if this task has been processed before by checking if
            # expected dataset metadata file exists.
            dataset_path = get_wq_dataset_path(
                output_directory=output_directory,
                tile_id=tile_id,
                temporal_id=temporal_id,
                product_name=product_name,
                product_version=product_version,
            )
            output_stac_url = get_wq_stac_url(dataset_path)
            exists = check_file_exists(output_stac_url)
            if not overwrite and exists:
                log.info(
                    f"{output_stac_url} exists! Skipping processing task {task_id}"
                )
                continue

            temporal_range = DateTimeRange(temporal_id)

            start_date = temporal_range.start.strftime("%Y-%m-%d")
            end_date = temporal_range.end.strftime("%Y-%m-%d")

            tile_geobox = gridspec.tile_geobox(tile_index=tile_id)

            # Reset instruments to use to instruments from the config
            # file.
            instruments_to_use = analysis_config["instruments_to_use"]
            # don't try to use instruments for which there are no data
            instruments_to_use = check_instrument_dates(
                instruments_to_use, start_date, end_date
            )
            instruments_list = get_instruments_list(instruments_to_use)

            log.info("Building the multivariate/multi-sensor dataset")
            # build the multivariate/multi-sensor dataset.
            dc_queries = build_dc_queries(
                instruments_to_use=instruments_to_use,
                start_date=start_date,
                end_date=end_date,
            )
            # Set up a dask client if on the sandbox.
            if bool(os.environ.get("JUPYTERHUB_USER", None)):
                client = create_local_dask_cluster(
                    display_client=False, return_client=True
                )
            else:
                client = None

            ds, source_datasets_uuids = build_wq_agm_dataset(
                dc_queries=dc_queries, tile_geobox=tile_geobox, dc=dc
            )

            if client is not None:
                client.close()

            log.info("Determining the pixels that are water")
            # Determine pixels that are water (sometimes, usually, permanent)
            ds = water_analysis(
                ds,
                water_frequency_threshold=WFTH,
                wofs_varname="wofs_ann_freq",
                permanent_water_threshold=PWT,
                sigma_coefficient=SC,
            )

            # Dark pixel correction
            ds = R_correction(ds, instruments_to_use, WFTL)

            if "msi_agm" in instruments_list.keys():
                log.info("Calculating the hue.")
                ds["hue"] = hue_calculation(ds, instrument="msi_agm")

            if "msi_agm" in instruments_list.keys():
                log.info(
                    "Determining the open water type for each pixel "
                    "using the instrument msi_agm"
                )
                ds["owt_msi"] = OWT_pixel(
                    ds,
                    instrument="msi_agm",
                    resample_rate=3,
                )

            if "oli_agm" in instruments_list.keys():
                log.info(
                    "Determining the open water type for each pixel "
                    "using the instrument oli_agm"
                )
                ds["owt_oli"] = OWT_pixel(
                    ds,
                    instrument="oli_agm",
                    resample_rate=3,
                )

            log.info("Applying the WQ algorithms to water areas.")

            ds, wq_vars_df = WQ_vars(
                ds.where(ds.wofs_ann_freq >= WFTL),
                instruments_list=instruments_list,
                stack_wq_vars=False,
            )
            # Get the list of all generated water quality variables
            # from the table
            wq_vars_list = list(
                chain.from_iterable(
                    [
                        wq_vars_df[col].dropna().to_list()
                        for col in wq_vars_df.columns
                    ]
                )
            )

            initial_keep_list = [
                # wofs_ann instrument
                "wofs_ann_freq",
                "wofs_ann_clearcount",
                "wofs_ann_wetcount",
                # water_analysis
                "wofs_ann_freq_sigma",
                "wofs_ann_confidence",
                "wofs_pw_threshold",
                "wofs_ann_pwater",
                "watermask",
                # optical water type
                "owt_msi",
                "owt_oli",
                # wq variables
                "tss",
                "chla",
            ]
            # The keeplist is not complete;
            # if the wq variables are retained as variables they will
            # appear in a listing of data_vars. Therefore, revert to the
            # instruments dictionary to list variables to drop
            droplist = []
            for instrument in list(instruments_list.keys()):
                for band in list(instruments_list[instrument].keys()):
                    variable = instruments_list[instrument][band]["varname"]
                    if variable not in initial_keep_list:
                        droplist = np.append(droplist, variable)
                        droplist = np.append(droplist, variable + "r")
            ds = ds.drop_vars(droplist, errors="ignore")

            # Save each band into a COG file.
            fs = get_filesystem(output_directory, anon=False)
            bands = list(ds.data_vars)
            for band in bands:
                output_cog_url = get_wq_cog_url(
                    output_directory=output_directory,
                    tile_id=tile_id,
                    temporal_id=temporal_id,
                    band_name=band,
                    product_name=product_name,
                    product_version=product_version,
                )

                # Enforce data type for all bands to float32
                da: xr.DataArray = ds[band].astype(np.float32)

                # No data and offset attributes for water quality
                # variables should be set in run_wq_algorithms
                if band not in wq_vars_list:
                    # Enforce no data for all bands to np.nan
                    da.attrs = dict(
                        nodata=np.nan,
                        # scale
                        scales=1,
                        # add_offset
                        offsets=0,
                        product_name=product_name,
                        product_version=product_version,
                    )
                else:
                    da.attrs.update(
                        dict(
                            product_name=product_name,
                            product_version=product_version,
                        )
                    )
                cog_bytes = write_cog(
                    geo_im=da,
                    fname=":mem:",
                    overwrite=True,
                    nodata=da.attrs["nodata"],
                    tags=da.attrs,
                )
                with fs.open(output_cog_url, "wb") as f:
                    f.write(cog_bytes)
                log.info(f"Band {band} saved to {output_cog_url}")

            # Save a table containing the water quality parameters
            output_csv_url = get_wq_csv_url(
                output_directory=output_directory,
                tile_id=tile_id,
                temporal_id=temporal_id,
                product_name=product_name,
                product_version=product_version,
            )
            with fs.open(output_csv_url, mode="w") as f:
                wq_vars_df.to_csv(f, index=False)

            # Generate the stac file for the task
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                log.info("Creating metadata STAC file ...")
                stac_file_url = prepare_dataset(  # noqa F841
                    dataset_path=get_parent_dir(output_csv_url),
                    source_datasets_uuids=source_datasets_uuids,
                )

        except Exception as error:
            log.exception(error)
            failed_tasks.append(task_id)

    if failed_tasks:
        failed_tasks_json_array = json.dumps(failed_tasks)

        tasks_directory = "/tmp/"
        failed_tasks_output_file = join_url(tasks_directory, "failed_tasks")

        fs = get_filesystem(path=tasks_directory, anon=False)
        if not check_directory_exists(path=tasks_directory):
            fs.mkdirs(path=tasks_directory, exist_ok=True)

        with fs.open(failed_tasks_output_file, "a") as file:
            file.write(failed_tasks_json_array + "\n")
        log.error(f"Failed tasks: {failed_tasks_json_array}")
        log.info(f"Failed tasks written to {failed_tasks_output_file}")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    cli()
