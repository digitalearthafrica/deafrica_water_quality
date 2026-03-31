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
from datacube.testutils.io import rio_slurp_xarray
from deafrica_tools.waterbodies import get_waterbody
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.geo.xr import rasterize, xr_reproject
from waterbodies.db import get_waterbodies_engine

from water_quality.grid import get_waterbodies_grid
from water_quality.io import check_directory_exists, get_filesystem, join_url
from water_quality.logs import setup_logging
from water_quality.summaries.summary import (
    REQUIRED_COLUMNS,
    add_water_quality_observations_to_db,
    check_obs_ids,
)
from water_quality.tasks import split_tasks
from water_quality.tiling import reproject_tile_geobox


@click.command(
    name="process-vector-tasks",
    no_args_is_help=True,
)
@click.argument(
    "vector-processing-tasks",
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
    vector_processing_tasks: str,
    max_parallel_steps: int,
    worker_idx: int,
    overwrite: bool,
    log: str,
):
    """
    Generate annual summaries of water quality variables for DE Africa waterbodies
    using vector based processing.

    VECTOR_PROCESSING_TASKS : a json file containing a mapping of waterbodies uids
    and their corresponding historical extent COGs that cover multiple tiles.

    MAX_PARALLEL_STEPS: The total number of parallel workers or pods
    expected in the workflow. This value is used to divide the list of
    tasks to be processed among the available workers.

    WORKER_IDX: The sequential index (0-indexed) of the current worker.
    This index determines which subset of tasks the current worker will
    process.
    """
    log_level = getattr(logging, log.upper())
    _log = setup_logging(log_level)

    fs = get_filesystem(vector_processing_tasks, anon=False)
    with fs.open(vector_processing_tasks, "r") as file:
        uids_to_cogs = json.load(file)

    tasks_to_run = split_tasks(
        list(uids_to_cogs.keys()), max_parallel_steps, worker_idx
    )

    if not tasks_to_run:
        _log.warning(f"Worker {worker_idx} has no tasks to process. Exiting.")
        sys.exit(0)

    _log.info(f"Worker {worker_idx} processing {len(tasks_to_run)} tasks")

    dc = Datacube(app="process_vector_tasks")
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
    for idx, waterbody_uid in enumerate(tasks_to_run):
        _log.info(
            f"Processing waterbody {idx + 1} of {len(tasks_to_run)}: {waterbody_uid} "
        )
        try:
            historical_extent_cogs = uids_to_cogs[waterbody_uid]

            if len(historical_extent_cogs) < 3:
                gridspec = get_waterbodies_grid()
                waterbody_gdf = get_waterbody(waterbody_uid).to_crs(
                    gridspec.crs
                )
                waterbody_wb_id = waterbody_gdf.iloc[0].wb_id
                waterbody_geom = Geometry(
                    geom=waterbody_gdf.iloc[0].geometry, crs=waterbody_gdf.crs
                )
                waterbody_geobox = GeoBox.from_geopolygon(
                    geopolygon=waterbody_geom,
                    resolution=gridspec.resolution,
                    crs=gridspec.crs,
                )
                extent_da = rasterize(
                    poly=waterbody_geom, how=waterbody_geobox
                ).astype("int8")
                # The long way to replace values in order to preserve the raster geobox
                extent_da = extent_da.where(extent_da == 0, waterbody_wb_id)
                extent_da = extent_da.where(extent_da != 0, np.nan).astype(
                    np.float32
                )
            else:
                extent_da = (
                    xr.open_mfdataset(
                        historical_extent_cogs,
                        engine="rasterio",
                        combine="by_coords",
                        join="outer",
                        chunks={
                            "x": 9600,
                            "y": 9600,
                        },
                    )
                    .squeeze()["band_data"]
                    .chunk(dask_chunks)
                )

                if len(historical_extent_cogs) > 3:
                    new_geobox = reproject_tile_geobox(
                        extent_da.odc.geobox, 30
                    )
                    extent_da = xr_reproject(
                        extent_da, how=new_geobox, resampling="nearest"
                    ).chunk(dask_chunks)

                extent_da.name = "band"
                wb_id_to_uid = {
                    int(wb_id): uid
                    for wb_id, uid in json.loads(
                        extent_da.attrs["WB_ID_to_UID"]
                    ).items()
                }
                uid_to_wb_id = {v: k for k, v in wb_id_to_uid.items()}
                waterbody_wb_id = uid_to_wb_id[waterbody_uid]

                extent_da = extent_da.where(
                    extent_da == waterbody_wb_id
                ).astype(np.float32)

            # Load all years available for the wq_annual product.
            ds = dc.load(
                product=product,
                like=extent_da.odc.geobox,
                measurements=measurements,
                dask_chunks=dask_chunks,
                # Resampling set to nearest to account for "water_mask"
                resampling="nearest",
            )
            ds = ds.where(extent_da)

            # Edge case: If no wq_annual data is available for the waterbody.
            if len(list(ds.data_vars)) == 0:
                _log.info(
                    f"No {product} data available for any year for this waterbody. "
                    "Skipping processing for this waterbody."
                )
                continue

            years = [
                pd.Timestamp(year).strftime("%Y/%m/%d")
                for year in ds.time.values
            ]

            possible_obs_ids = [f"{year}_{waterbody_uid}" for year in years]
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
                    "All possible water quality observations for this waterbody "
                    "already exist in the database and overwrite is set to False. "
                    "Skipping processing for this waterbody."
                )
                continue

            _log.info(
                f"Processing per waterbody statistics for {ds.time.size} years for "
                f"waterbody {waterbody_uid} ..."
            )
            df_drop_columns = ["band", "spatial_ref"]

            _log.info(
                "Processing water area consistently indicating algae ..."
            )
            # Note that the .groupby(extent_da) used in the raster processing
            # does not work for vector processing since extent_da is loaded
            # with dask chunks.
            water_mask_count = (
                ds["water_mask"].notnull().astype("int8").sum(dim=("x", "y"))
            )
            fai_count = ds["fai"].notnull().astype("int8").sum(dim=("x", "y"))
            ndvi_count = (
                ds["ndvi"].notnull().astype("int8").sum(dim=("x", "y"))
            )

            water_mask_count, fai_count, ndvi_count = dask.compute(
                water_mask_count, fai_count, ndvi_count
            )

            pixel_area_m2 = abs(
                extent_da.odc.geobox.resolution.x
                * extent_da.odc.geobox.resolution.y
            )
            water_area = (
                (water_mask_count * pixel_area_m2)
                .to_dataframe(name="water_area_m2")
                .drop(columns=df_drop_columns, errors="ignore")
            )

            # --- FAI cover (algae) ---
            fai_cover = (
                fai_count / water_mask_count.where(water_mask_count > 0)
            ) * 100

            fai_cover_df = fai_cover.to_dataframe(name="fai_cover").drop(
                columns=df_drop_columns, errors="ignore"
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
                columns=df_drop_columns, errors="ignore"
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

            # Sanity check
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
                # This works for most waterbodies
                quantiles_da = da_rechunked.quantile(
                    quantiles, dim=("x", "y")
                ).compute()
                quantiles_df = (
                    quantiles_da.to_dataframe()
                    .drop(columns=df_drop_columns, errors="ignore")
                    .unstack("quantile")
                )
                quantiles_df.columns = [
                    f"{measurement}_q{q}".replace(".", "_")
                    for _, q in quantiles_df.columns
                ]
                quantiles_df_to_merge.append(quantiles_df)

            per_waterbody_summaries = pd.concat(
                [
                    *quantiles_df_to_merge,
                    fai_cover_df,
                    ndvi_cover_df,
                    water_area,
                ],
                axis=1,
            )
            # Dask clean up
            del (
                quantiles_da,
                quantiles_df,
                quantiles_df_to_merge,
                fai_cover_df,
                ndvi_cover_df,
                water_area,
            )
            gc.collect()

            per_waterbody_summaries = per_waterbody_summaries.reset_index()
            per_waterbody_summaries["uid"] = waterbody_uid

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
            per_waterbody_summaries = per_waterbody_summaries[cols]

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
                f"Finished processing water quality summaries for the waterbody {waterbody_uid}"
            )

        except Exception as error:
            _log.exception(error)
            failed_tasks.append(waterbody_uid)

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
