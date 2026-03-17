import gc
import json
import logging
import sys

import click
import dask
import pandas as pd
import rioxarray
import xarray as xr
from datacube import Datacube
from waterbodies.db import get_waterbodies_engine

from water_quality.io import check_directory_exists, get_filesystem, join_url
from water_quality.logs import setup_logging
from water_quality.summaries.summary import (
    add_water_quality_observations_to_db,
)
from water_quality.tasks import split_tasks


@click.command(
    name="process-raster-tasks",
    no_args_is_help=True,
)
@click.argument(
    "tasks",
    type=str,
)
@click.argument(
    "waterbodies-to-filter",
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
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help=(
        "If overwrite is True tasks that have already been processed "
        "will be rerun. "
    ),
)
@click.option(
    "--log",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="WARNING",
    show_default=True,
    help="control the log level, e.g., --log=error",
)
def cli(
    tasks: str,
    waterbodies_to_filter: str,
    max_parallel_steps: int,
    worker_idx: int,
    overwrite: bool,
    log: str,
):
    """
    Generate annual summaries of water quality variables for DE Africa waterbodies
    using raster based processing.

    TASKS: a text file containing a list of the DE Africa Waterbodies Historical
    Extent COGs to be processed.

    WATERBODIES_TO_FILTER: a text file containing a list of waterbodies to be
    excluded from the raster based processing of the water quality annual summaries.
    These are waterbodies that cover multiple tiles and will be processed separately
    using the vector based processing tool.

    MAX_PARALLEL_STEPS: The total number of parallel workers or pods
    expected in the workflow. This value is used to divide the list of
    tasks to be processed among the available workers.

    WORKER_IDX: The sequential index (0-indexed) of the current worker.
    This index determines which subset of tasks the current worker will
    process.
    """
    log_level = getattr(logging, log.upper())
    _log = setup_logging(log_level)

    fs = get_filesystem(waterbodies_to_filter, anon=False)
    with fs.open(waterbodies_to_filter, "r") as file:
        uids_to_exclude = set(sorted(json.load(file)))

    fs = get_filesystem(tasks, anon=False)
    with fs.open(tasks, "r") as file:
        all_tasks = sorted(json.load(file))

    tasks_to_run = split_tasks(all_tasks, max_parallel_steps, worker_idx)

    if not tasks_to_run:
        _log.warning(f"Worker {worker_idx} has no tasks to process. Exiting.")
        sys.exit(0)

    _log.info(f"Worker {worker_idx} processing {len(tasks_to_run)} tasks")

    dc = Datacube(app="process_raster_tasks")
    measurements = [
        "fai",
        "ndvi",
        "hue",
        "owt",
        "chla",
        "tsi",
        "tsm",
        "st_max",
        "st_median",
        "st_min",
        "water_mask",
    ]
    product = "wq_annual"
    dask_chunks = {"x": 3000, "y": 3000}
    # m2_per_km2 = 1_000_000
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    engine = get_waterbodies_engine()

    failed_tasks = []
    for idx, cog_path in enumerate(tasks_to_run):
        _log.info(
            f"Processing historical extent COG {idx + 1} of {len(tasks_to_run)}: {cog_path} "
        )
        try:
            extent_da = rioxarray.open_rasterio(cog_path).squeeze()

            wb_id_to_uid = {
                int(wb_id): uid
                for wb_id, uid in json.loads(
                    extent_da.attrs["WB_ID_to_UID"]
                ).items()
            }
            wb_ids_to_exclude = [
                wb_id
                for wb_id, uid in wb_id_to_uid.items()
                if uid in uids_to_exclude
            ]
            wb_id_to_uid_filtered = {
                wb_id: uid
                for wb_id, uid in wb_id_to_uid.items()
                if wb_id not in wb_ids_to_exclude
            }

            assert len(wb_id_to_uid_filtered) == len(wb_id_to_uid) - len(
                wb_ids_to_exclude
            )

            if len(wb_ids_to_exclude) > 0:
                extent_da = extent_da.where(
                    ~extent_da.isin(wb_ids_to_exclude), other=0
                )

            extent_da = extent_da.where(extent_da != 0)

            ds = dc.load(
                product=product,
                like=extent_da.odc.geobox,
                measurements=measurements,
                dask_chunks=dask_chunks,
            )

            # assert ds.odc.geobox.crs.projected
            # pixel_area_km2 = (
            #    abs(ds.odc.geobox.resolution.x * ds.odc.geobox.resolution.y)
            #    / m2_per_km2
            # )

            _log.info(
                f"Processing per waterbody statistics for {ds.time.size} years for {len(wb_id_to_uid_filtered)} waterbodies ..."
            )
            df_drop_columns = ["band", "spatial_ref"]

            _log.info(
                "Processing water area consistently indicating algae ..."
            )
            water_mask_count = (
                (~ds["water_mask"].isnull()).groupby(extent_da).sum()
            )
            fai_count = (~ds["fai"].isnull()).groupby(extent_da).sum()
            ndvi_count = (~ds["ndvi"].isnull()).groupby(extent_da).sum()

            water_mask_count, fai_count, ndvi_count = dask.compute(
                water_mask_count, fai_count, ndvi_count
            )

            # --- FAI cover (algae) ---
            fai_cover = (
                fai_count / water_mask_count.where(water_mask_count > 0)
            ) * 100

            fai_cover_df = fai_cover.to_dataframe(name="fai_cover").drop(
                columns=df_drop_columns
            )
            del fai_cover
            gc.collect()

            _log.info(
                "Processing water area consistently indicating vegetation ..."
            )
            # --- NDVI cover (vegetation) ---
            ndvi_cover = (
                ndvi_count / water_mask_count.where(water_mask_count > 0)
            ) * 100
            ndvi_cover_df = ndvi_cover.to_dataframe(name="ndvi_cover").drop(
                columns=df_drop_columns
            )
            del water_mask_count, ndvi_cover
            gc.collect()

            annual_quantile_measurements = [
                "hue",
                "owt",
                "chla",
                "tsi",
                "tsm",
                "st_max",
                "st_median",
                "st_min",
            ]

            # Sanity check — single set construction, reused for both assertions
            annual_quantile_set = set(annual_quantile_measurements)
            missing_from_measurements = annual_quantile_set - set(measurements)
            missing_from_ds = annual_quantile_set - set(ds.data_vars)

            assert not missing_from_measurements, (
                f"Measurements missing from `measurements`: {missing_from_measurements}"
            )
            assert not missing_from_ds, (
                f"Measurements missing from `ds.data_vars`: {missing_from_ds}"
            )

            quantiles_df_to_merge = []
            for measurement in annual_quantile_measurements:
                _log.info(
                    f"Processing per waterbody quantiles for the {measurement} variable"
                )
                with xr.set_options(use_flox=False):
                    # groupby egearly computes.
                    quantiles_da = (
                        ds[measurement].groupby(extent_da).quantile(quantiles)
                    )
                quantiles_df = quantiles_da.to_dataframe().unstack("quantile")
                quantiles_df.columns = [
                    f"{measurement}_q{q}".replace(".", "_")
                    for _, q in quantiles_df.columns
                ]
                quantiles_df_to_merge.append(quantiles_df)

            per_waterbody_summaries = pd.concat(
                [*quantiles_df_to_merge, fai_cover_df, ndvi_cover_df], axis=1
            )
            # Dask clean up
            del (
                quantiles_da,
                quantiles_df,
                quantiles_df_to_merge,
                fai_cover_df,
                ndvi_cover_df,
            )
            gc.collect()

            per_waterbody_summaries = (
                per_waterbody_summaries.reset_index().rename(
                    columns={"group": "wb_id"}
                )
            )

            per_waterbody_summaries["uid"] = per_waterbody_summaries[
                "wb_id"
            ].map(wb_id_to_uid_filtered)

            per_waterbody_summaries["obs_id"] = (
                per_waterbody_summaries["time"].dt.strftime("%Y/%m/%d")
                + "_"
                + per_waterbody_summaries["uid"]
            )

            per_waterbody_summaries = per_waterbody_summaries.rename(
                columns={"time": "date"}
            )
            cols = per_waterbody_summaries.columns.tolist()
            cols.insert(0, cols.pop(cols.index("uid")))
            cols.insert(0, cols.pop(cols.index("obs_id")))

            per_waterbody_summaries = per_waterbody_summaries[cols].drop(
                columns=["wb_id"]
            )

            add_water_quality_observations_to_db(
                water_quality_measures=per_waterbody_summaries,
                engine=engine,
                update_rows=overwrite,
            )
            _log.info(
                f"Finished processing water quality summaries for {cog_path}"
            )
        except Exception as error:
            _log.exception(error)
            failed_tasks.append(cog_path)

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
        _log.error(f"Failed tasks: {failed_tasks_json_array}")
        _log.info(f"Failed tasks written to {failed_tasks_output_file}")
        sys.exit(1)
    else:
        _log.info(f"Worker {worker_idx} completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    cli()
