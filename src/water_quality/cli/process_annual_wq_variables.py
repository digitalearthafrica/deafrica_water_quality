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
import gc

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


def load_tasks(tasks=None, tasks_file=None):
    """Load task IDs from --tasks or --tasks-file."""
    if tasks and tasks_file:
        raise ValueError("Use either tasks or tasks_file, not both.")
    if not tasks and not tasks_file:
        raise ValueError("Must provide tasks or tasks_file.")

    if tasks:
        return [i.strip() for i in tasks.split(",")]

    if not check_file_exists(tasks_file):
        raise FileNotFoundError(f"{tasks_file} does not exist!")

    fs = get_filesystem(tasks_file, anon=True)
    with fs.open(tasks_file, "r") as f:
        return [i.strip() for i in f.readlines()]


def split_tasks(all_task_ids, max_parallel_steps, worker_idx):
    """Divide tasks across workers."""
    task_chunks = np.array_split(np.array(all_task_ids), max_parallel_steps)
    task_chunks = [chunk.tolist() for chunk in task_chunks if len(chunk) > 0]
    if len(task_chunks) <= worker_idx:
        return []
    return task_chunks[worker_idx]


def prepare_config(analysis_config):
    """Validate and extract config values."""
    cfg = check_config(analysis_config)
    resolution_m = check_resolution(int(cfg["resolution"]))
    product_info = cfg["product"]
    return dict(
        resolution=resolution_m,
        WFTH=cfg["water_frequency_threshold_high"],
        WFTL=cfg["water_frequency_threshold_low"],
        PWT=cfg["permanent_water_threshold"],
        SC=cfg["sigma_coefficient"],
        product_info=product_info,
        product_name=product_info["name"],
        product_version=product_info["version"],
        instruments_to_use=cfg["instruments_to_use"]
    )


def parse_task(task_id, gridspec):
    """Parse task ID into temporal + tile ID and build tile geobox."""
    temporal_id, tile_id = parse_task_id(task_id)
    tile_geobox = gridspec.tile_geobox(tile_index=tile_id)
    return temporal_id, tile_id, tile_geobox


def prepare_instruments(instruments_to_use, start_date, end_date):
    """Filter instruments by date and return valid instruments list."""
    instruments_to_use = check_instrument_dates(instruments_to_use, start_date, end_date)
    instruments_list = get_instruments_list(instruments_to_use)
    return instruments_to_use, instruments_list


def prepare_queries(instruments_to_use, start_date, end_date):
    """Build datacube queries for given instruments and dates."""
    return build_dc_queries(
        instruments_to_use=instruments_to_use,
        start_date=start_date,
        end_date=end_date,
    )


def setup_dask_if_needed():
    """Start local Dask cluster in Sandbox, else return None."""
    if bool(os.environ.get("JUPYTERHUB_USER", None)):
        return create_local_dask_cluster(display_client=False, return_client=True)
    return None


def load_data(dc_queries, tile_geobox, dc):
    """Load multi-sensor dataset from datacube."""
    return build_wq_agm_dataset(
        dc_queries=dc_queries, tile_geobox=tile_geobox, dc=dc
    )


@click.command(
    name="process-annual-wq-variables",
    no_args_is_help=True,
)
@click.option(
    "--tasks",
    help="List of comma separated tasks in the format "
    "period/x{x:02d}/y{y:02d} to generate water quality variables for. "
    "For example `2015--P1Y/x200/y34,2015--P1Y/x178/y095, 2015--P1Y/x199y/100`",
)
@click.option(
    "--tasks-file",
    help="Optional path to a text file containing the tasks to generate "
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

    # Load and validate configuration
    config = prepare_config(analysis_config)
    resolution_m = config['resolution']
    WFTH = config['WFTH']
    WFTL = config['WFTL']
    PWT = config['PWT']
    SC = config['SC']
    product_name = config['product_name']
    product_version = config['product_version']
    config_instruments_to_use = config['instruments_to_use']

    # Load all tasks and split for this worker
    all_task_ids = load_tasks(tasks, tasks_file)
    my_tasks = split_tasks(all_task_ids, max_parallel_steps, worker_idx)

    if not my_tasks:
        log.warning(f"Worker {worker_idx} has no tasks to process. Exiting.")
        sys.exit(0)

    log.info(f"Worker {worker_idx} processing {len(my_tasks)} tasks")

    # Initialize grid and datacube
    gridspec = get_waterbodies_grid(resolution_m)
    dc = Datacube(app="ProcessAnnualWQVariables")

    failed_tasks = []

    # Process each task
    for task_id in my_tasks:
        try:
            log.info(f"Processing task: {task_id}")

            # Parse task information
            temporal_id, tile_id, tile_geobox = parse_task(task_id, gridspec)
            temporal_range = DateTimeRange(temporal_id)
            start_date = temporal_range.start.strftime("%Y-%m-%d")
            end_date = temporal_range.end.strftime("%Y-%m-%d")

            # Check if task already processed
            if not overwrite:
                output_csv_url = get_wq_csv_url(
                    output_directory=output_directory,
                    tile_id=tile_id,
                    temporal_id=temporal_id,
                    product_name=product_name,
                    product_version=product_version,
                )
                if check_file_exists(output_csv_url):
                    log.info(f"Task {task_id} already processed. Skipping.")
                    continue

            # Prepare instruments and queries
            instruments_to_use, instruments_list = prepare_instruments(
                config_instruments_to_use, start_date, end_date
            )
            dc_queries = prepare_queries(instruments_to_use, start_date, end_date)

            # Setup Dask if needed
            client = setup_dask_if_needed()

            # Load data
            ds, source_datasets_uuids = load_data(dc_queries, tile_geobox, dc)

            # Close Dask client if it was created
            if client is not None:
                client.close()

            # Water analysis
            ds = water_analysis(
                ds,
                water_frequency_threshold=WFTH,
                wofs_varname="wofs_ann_freq",
                permanent_water_threshold=PWT,
                sigma_coefficient=SC,
            )

            # Reflectance correction
            ds = R_correction(ds, instruments_to_use, WFTL)

            # Hue calculation for MSI
            if "msi_agm" in instruments_list.keys():
                log.info("Calculating the hue.")
                ds["hue"] = hue_calculation(ds, instrument="msi_agm")

                log.info(
                    "Determining the open water type for each pixel "
                    "using the instrument msi_agm"
                )
                ds["owt_msi"] = OWT_pixel(ds, instrument="msi_agm", resample_rate=3)

            # OWT calculation for OLI
            if "oli_agm" in instruments_list.keys():
                log.info(
                    "Determining the open water type for each pixel "
                    "using the instrument oli_agm"
                )
                ds["owt_oli"] = OWT_pixel(ds, instrument="oli_agm", resample_rate=3)

            # Mask dataset based on water frequency threshold
            mask = (ds.wofs_ann_freq >= WFTL).compute()
            ds_masked = ds.where(mask, drop=True)

            # Run WQ algorithms
            log.info("Applying the WQ algorithms to water areas.")
            ds_out, wq_vars_df = WQ_vars(
                ds_masked,
                instruments_list=instruments_list,
                stack_wq_vars=False
            )

            del ds_masked
            gc.collect()

            # Get list of WQ variables
            wq_vars_list = list(chain.from_iterable([
                wq_vars_df[col].dropna().to_list()
                for col in wq_vars_df.columns
            ]))

            # Define variables to keep
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

            # Create drop list for unused variables
            droplist = []
            for instrument in list(instruments_list.keys()):
                for band in list(instruments_list[instrument].keys()):
                    variable = instruments_list[instrument][band]["varname"]
                    if variable not in initial_keep_list:
                        droplist = np.append(droplist, variable)
                        droplist = np.append(droplist, variable + "r")

            ds_out = ds_out.drop_vars(droplist, errors="ignore")

            # Save each band as COG
            fs = get_filesystem(output_directory, anon=False)
            bands = list(ds_out.data_vars)

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
                da: xr.DataArray = ds_out[band].astype(np.float32)

                # Set attributes
                if band not in wq_vars_list:
                    da.attrs = dict(
                        nodata=np.nan,
                        scales=1,
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

                # Write COG
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

            # Save WQ parameters table
            output_csv_url = get_wq_csv_url(
                output_directory=output_directory,
                tile_id=tile_id,
                temporal_id=temporal_id,
                product_name=product_name,
                product_version=product_version,
            )
            with fs.open(output_csv_url, mode="w") as f:
                wq_vars_df.to_csv(f, index=False)

            # Generate STAC metadata
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                log.info("Creating metadata STAC file ...")
                stac_file_url = prepare_dataset(
                    dataset_path=get_parent_dir(output_csv_url),
                    source_datasets_uuids=source_datasets_uuids,
                )

            log.info(f"Successfully processed task: {task_id}")

        except Exception as error:
            log.exception(error)
            failed_tasks.append(task_id)

    # Handle failed tasks
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
        log.info(f"Worker {worker_idx} completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    cli()