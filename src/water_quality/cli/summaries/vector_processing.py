import json
import logging
import sys

import click
import pandas as pd
from datacube import Datacube
from deafrica_tools.waterbodies import get_waterbody
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.geo.xr import rasterize
from waterbodies.db import get_waterbodies_engine

from water_quality.grid import get_waterbodies_grid
from water_quality.io import check_directory_exists, get_filesystem, join_url
from water_quality.logs import setup_logging
from water_quality.summaries.summary import (
    add_water_quality_observations_to_db,
)
from water_quality.tasks import split_tasks


@click.command(
    name="process-vector-tasks",
    no_args_is_help=True,
)
@click.argument(
    "tasks",
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
    max_parallel_steps: int,
    worker_idx: int,
    overwrite: bool,
    log: str,
):
    """
    Generate annual summaries of water quality variables for DE Africa waterbodies
    using vector based processing.

    TASKS: a text file containing a list of DE Africa Waterbodies Historical Extent
    UIDS that were excluded from the raster based processing of the water quality annual summaries.
    These are waterbodies that cover multiple tiles.

    MAX_PARALLEL_STEPS: The total number of parallel workers or pods
    expected in the workflow. This value is used to divide the list of
    tasks to be processed among the available workers.

    WORKER_IDX: The sequential index (0-indexed) of the current worker.
    This index determines which subset of tasks the current worker will
    process.
    """
    log_level = getattr(logging, log.upper())
    _log = setup_logging(log_level)

    fs = get_filesystem(tasks, anon=False)
    with fs.open(tasks, "r") as file:
        all_waterbody_uids = sorted(json.load(file))

    tasks_to_run = split_tasks(
        all_waterbody_uids, max_parallel_steps, worker_idx
    )

    if not tasks_to_run:
        _log.warning(f"Worker {worker_idx} has no tasks to process. Exiting.")
        sys.exit(0)

    _log.info(f"Worker {worker_idx} processing {len(tasks_to_run)} tasks")

    grid = get_waterbodies_grid()
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
    dask_chunks = {"x": 800, "y": 800}
    # m2_per_km2 = 1_000_000
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    engine = get_waterbodies_engine()

    failed_tasks = []
    # Process each task
    for idx, waterbody_uid in enumerate(tasks_to_run):
        _log.info(
            f"Processing waterbody {idx + 1} of {len(tasks_to_run)}: {waterbody_uid} "
        )
        try:
            waterbody = get_waterbody(waterbody_uid)
            waterbody_geom = Geometry(
                waterbody["geometry"].iloc[0], waterbody.crs
            )
            waterbody_geobox = GeoBox.from_geopolygon(
                waterbody_geom, resolution=grid.resolution, crs=grid.crs
            )
            ds = dc.load(
                product=product,
                like=waterbody_geobox,
                measurements=measurements,
                dask_chunks=dask_chunks,
            )

            extent_da = rasterize(waterbody_geom, how=waterbody_geobox)
            ds = ds.where(extent_da)

            _log.info(
                f"Processing per waterbody statistics for {ds.time.size} years for waterbody {waterbody_uid} ..."
            )
            df_drop_columns = ["spatial_ref"]

            _log.info(
                "Processing water area consistently indicating algae ..."
            )
            water_mask_count = (~ds["water_mask"].isnull()).sum(dim=("x", "y"))
            fai_count = (~ds["fai"].isnull()).sum(dim=("x", "y"))
            fai_cover = (
                fai_count / water_mask_count.where(water_mask_count > 0)
            ) * 100
            fai_cover_df = fai_cover.to_dataframe(name="fai_cover").drop(
                columns=df_drop_columns
            )

            _log.info(
                "Processing water area consistently indicating vegetation ..."
            )
            ndvi_count = (~ds["ndvi"].isnull()).sum(dim=("x", "y"))
            ndvi_cover = (
                ndvi_count / water_mask_count.where(water_mask_count > 0)
            ) * 100
            ndvi_cover_df = ndvi_cover.to_dataframe(name="ndvi_cover").drop(
                columns=df_drop_columns
            )

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
            assert set(annual_quantile_measurements).issubset(
                set(measurements)
            )
            assert set(annual_quantile_measurements).issubset(
                set(list(ds.data_vars))
            )

            quantiles_df_to_merge = []
            for measurement in annual_quantile_measurements:
                _log.info(
                    f"Processing per waterbody quantiles for the {measurement} variable"
                )
                quantiles_df = (
                    ds[measurement]
                    .quantile(quantiles, dim=("x", "y"))
                    .to_dataframe()
                    .unstack("quantile")
                )
                quantiles_df.columns = [
                    f"{measurement}_q{q}".replace(".", "_")
                    for _, q in quantiles_df.columns
                ]
                quantiles_df_to_merge.append(quantiles_df)

            per_waterbody_summaries = pd.concat(
                [*quantiles_df_to_merge, fai_cover_df, ndvi_cover_df], axis=1
            )
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

            add_water_quality_observations_to_db(
                water_quality_measures=per_waterbody_summaries,
                engine=engine,
                update_rows=overwrite,
            )
            _log.info(
                f"Finished processing water quality summaries the waterbody {waterbody_uid}"
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
