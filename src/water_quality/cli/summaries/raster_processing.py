import gc
import json
import logging
import sys

import click
import dask
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from datacube import Datacube
from waterbodies.db import get_waterbodies_engine

from water_quality.io import check_directory_exists, get_filesystem, join_url
from water_quality.logs import setup_logging
from water_quality.summaries.summary import (
    REQUIRED_COLUMNS,
    add_water_quality_observations_to_db,
    check_obs_ids,
)
from water_quality.tasks import split_tasks


@click.command(
    name="process-raster-tasks",
    no_args_is_help=True,
)
@click.argument(
    "raster-processing-tasks",
    type=str,
)
@click.argument(
    "waterbodies-to-exclude",
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
    raster_processing_tasks: str,
    waterbodies_to_exclude: str,
    max_parallel_steps: int,
    worker_idx: int,
    overwrite: bool,
    log: str,
):
    """
    Generate annual summaries of water quality variables for DE Africa waterbodies
    using raster based processing.

    RASTER_PROCESSING_TASKS: a text file containing a list of the DE Africa Waterbodies Historical
    Extent COGs to be processed.

    WATERBODIES_TO_FILTER: a json file containing a mapping of waterbodies uids
    and their corresponding historical extent COGs that cover multiple tiles.
    The waterbodies in this file will be excluded from the raster based processing
    of the water quality annual summaries and instead will be processed separately
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

    fs = get_filesystem(waterbodies_to_exclude, anon=False)
    with fs.open(waterbodies_to_exclude, "r") as file:
        uids_to_exclude = list(json.load(file).keys())

    fs = get_filesystem(raster_processing_tasks, anon=False)
    with fs.open(raster_processing_tasks, "r") as file:
        all_tasks = file.read().splitlines()

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
    dask_chunks = {"x": 3200, "y": 3200}
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    engine = get_waterbodies_engine()

    failed_tasks = []
    for idx, cog_path in enumerate(tasks_to_run):
        _log.info(
            f"Processing historical extent COG {idx + 1} of {len(tasks_to_run)}: {cog_path} "
        )
        try:
            extent_da = rioxarray.open_rasterio(cog_path).squeeze()

            # During rasterization of the historical extent COGs, the nodata
            # value was set to 0. The processing makes the assumption that this
            # is maintained. If this assumption is violated, summaries produced
            # will be incorrect.
            nodata_val = 0
            assert nodata_val == extent_da.odc.nodata

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

            wb_ids_to_keep = list(wb_id_to_uid_filtered.keys())

            if len(wb_ids_to_keep) > 0:
                extent_da = extent_da.where(
                    extent_da.isin(wb_ids_to_keep), other=nodata_val
                )
            else:
                _log.info(
                    "No waterbodies for raster processing in this COG after filtering. "
                    "Skipping processing for this historical extent COG file."
                )
                continue

            extent_da = extent_da.where(extent_da != nodata_val).astype(
                np.float32
            )

            # Load all years of data available
            ds = dc.load(
                product=product,
                like=extent_da.odc.geobox,
                measurements=measurements,
                dask_chunks=dask_chunks,
            )
            ds = ds.where(extent_da)

            # Edge case: If no wq_annual data is available for the historical extent COG
            if len(list(ds.data_vars)) == 0:
                _log.info(
                    f"No {product} data available for any year for this historical extent COG. "
                    "Skipping processing for this historical extent COG file."
                )
                continue

            years = [
                pd.Timestamp(year).strftime("%Y/%m/%d")
                for year in ds.time.values
            ]
            waterbody_uids = list(wb_id_to_uid_filtered.values())

            possible_obs_ids = [
                f"{year}_{waterbody_uid}"
                for year in years
                for waterbody_uid in waterbody_uids
            ]
            existing_obs_ids = check_obs_ids(possible_obs_ids, engine)

            if overwrite:
                obs_ids = set(existing_obs_ids).union(
                    set(possible_obs_ids) - set(existing_obs_ids)
                )
            else:
                obs_ids = set(possible_obs_ids) - set(existing_obs_ids)

            obs_ids = sorted(list(obs_ids))

            if not obs_ids:
                _log.info(
                    "All possible water quality observations for waterbodies in this historical "
                    "extent COG file already exist in the database and overwrite is set to False. "
                    "Skipping processing for this historical extent COG file."
                )
                continue

            _log.info(
                f"Processing per waterbody statistics for {ds.time.size} years for "
                f"{len(wb_id_to_uid_filtered)} waterbodies ..."
            )
            df_drop_columns = ["band", "spatial_ref"]

            _log.info(
                "Processing water area consistently indicating algae ..."
            )
            water_mask_count = (
                ds["water_mask"]
                .notnull()
                .groupby(extent_da)
                .sum(dim=("x", "y"))
            )
            fai_count = (
                ds["fai"].notnull().groupby(extent_da).sum(dim=("x", "y"))
            )
            ndvi_count = (
                ds["ndvi"].notnull().groupby(extent_da).sum(dim=("x", "y"))
            )

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
                da_rechunked = ds[measurement].chunk({"time": -1})
                try:
                    # This works for most historical extent COG tiles.
                    quantiles_da = (
                        da_rechunked.groupby(extent_da)
                        .quantile(quantiles, dim=("x", "y"))
                        .compute()
                    )
                    quantiles_df = (
                        quantiles_da.to_dataframe()
                        .drop(columns=df_drop_columns)
                        .unstack("quantile")
                    )
                except ValueError:
                    with xr.set_options(use_flox=False):
                        quantiles_da = (
                            da_rechunked.groupby(extent_da)
                            .quantile(quantiles, dim=("stacked_y_x"))
                            .compute()
                        )
                        quantiles_df = quantiles_da.to_dataframe().unstack(
                            "quantile"
                        )

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

            missing = set(REQUIRED_COLUMNS) - set(
                per_waterbody_summaries.columns
            )
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            extra = set(per_waterbody_summaries.columns) - set(
                REQUIRED_COLUMNS
            )
            if extra:
                raise ValueError(f"Found extra columns: {extra}")

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
