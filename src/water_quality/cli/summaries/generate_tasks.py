import json
import logging
from collections import defaultdict

import click
import rioxarray
from tqdm import tqdm

from water_quality.io import (
    check_directory_exists,
    find_geotiff_files,
    get_basename,
    get_filesystem,
    join_url,
)
from water_quality.logs import setup_logging


@click.command(
    name="generate-tasks",
    no_args_is_help=True,
)
@click.argument(
    "historical-extent-rasters-dir",
    type=str,
)
@click.option(
    "--output-dir",
    type=str,
    default="/tmp",
    show_default=True,
    help=(
        "The directory to write the text file containing the historical extent rasters file paths "
        "for raster processing and the json file containing the waterbodies uids and their "
        "corresponding COG paths for vector processing. "
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
    historical_extent_rasters_dir: str,
    output_dir: str,
    log: str,
):
    """
    From the DE Africa waterbodies historical extent COGs in the
    HISTORICAL_EXTENT_RASTERS_DIR directory, get a list of the COGs to process
    using the raster processing tool to get the per waterbody annual summaries,
    for the waterbodies that span a single tile.
    Also get a mapping of waterbodies uids and their corresponding historical extent
    COGs for vector processing for the waterbodies that span multiple tiles.
    """
    log_level = getattr(logging, log.upper())
    _log = setup_logging(log_level)

    fs = get_filesystem(path=output_dir, anon=False)
    if not check_directory_exists(path=output_dir):
        fs.mkdirs(path=output_dir, exist_ok=True)

    _log.info(
        f"Searching for waterbodies historical extent cogs in {historical_extent_rasters_dir} ..."
    )
    historical_extent_cogs = find_geotiff_files(historical_extent_rasters_dir)

    cog_path_to_uids = {}
    for cog_path in tqdm(historical_extent_cogs):
        ds = rioxarray.open_rasterio(cog_path)
        wbid_to_uid = json.loads(ds.attrs["WB_ID_to_UID"])
        uids = sorted(list(wbid_to_uid.values()))
        cog_path_to_uids[cog_path] = uids

    # Sort to ensure during parallel processing
    # tasks requiring the shortest time are processed first.
    cog_path_to_uids = dict(
        sorted(cog_path_to_uids.items(), key=lambda x: len(x[1]))
    )

    assert len(cog_path_to_uids) == len(historical_extent_cogs)

    _log.info(
        f"Found {len(historical_extent_cogs)} waterbodies historical extent cogs"
    )

    raster_processing_tasks = list(cog_path_to_uids.keys())

    raster_processing_tasks_fp = join_url(
        output_dir, "raster_processing_tasks.txt"
    )

    with fs.open(raster_processing_tasks_fp, "w") as file:
        file.write("\n".join(raster_processing_tasks))
    _log.info(
        f"Historical extent rasters for raster processing of water quality summaries written to {raster_processing_tasks_fp}"
    )

    _log.info(
        "Searching for waterbodies covered by more than 1 historical extent COG ..."
    )

    uids_to_cog_paths = defaultdict(list)
    for cog_path, uids in cog_path_to_uids.items():
        for uid in uids:
            uids_to_cog_paths[uid].append(cog_path)

    # Sort to ensure during parallel processing
    # tasks requiring the shortest time are processed first.
    uids_to_cog_paths = dict(
        sorted(uids_to_cog_paths.items(), key=lambda x: len(x[1]))
    )

    multi_tile_uids = {}
    for uid, cog_paths in uids_to_cog_paths.items():
        if len(cog_paths) > 1:
            multi_tile_uids[uid] = cog_paths

    # Sort to ensure during parallel processing
    # tasks requiring the shortest time are processed first.
    multi_tile_uids = dict(
        sorted(multi_tile_uids.items(), key=lambda x: len(x[1]))
    )

    _log.info(
        f"Found {len(multi_tile_uids)} waterbodies covered by more than 1 historical extent COG. "
        "These waterbodies will need to be processed seperately during generation of water quality summaries."
    )

    # Previous continental runs show that waterbodies covered by the following
    # Historical Extent COGs meet the criteria of only being covered by a
    # single tile but have difficulty being processed using the raster-processing
    # tool and thus must be procesed using the vector processing tool.
    cog_paths_exceptions_filter = [
        "historical_extent_x177y097.tif",
        "historical_extent_x191y093.tif",
        "historical_extent_x208y056.tif",
        "historical_extent_x214y047.tif",
    ]
    cog_paths_exceptions = [
        i
        for i in historical_extent_cogs
        if get_basename(i) in cog_paths_exceptions_filter
    ]

    _log.info(
        f"Identifying waterbodies in {len(cog_paths_exceptions)} historical extent COGs "
        "that cannot be processed using the raster processing tool"
    )

    uids_exceptions = []
    for cog_path in cog_paths_exceptions:
        ds = rioxarray.open_rasterio(cog_path)
        wbid_to_uid = json.loads(ds.attrs["WB_ID_to_UID"])
        uids = sorted(list(wbid_to_uid.values()))
        uids = [i for i in uids if i not in list(multi_tile_uids.keys())]
        uids_exceptions.extend(uids)

    _log.info(
        f"{len(uids_exceptions)} additional waterbodies identified for vector processing"
    )

    for uid in uids_exceptions:
        multi_tile_uids[uid] = uids_to_cog_paths[uid]

    # Sort to ensure during parallel processing
    # tasks requiring the shortest time are processed first.
    multi_tile_uids = dict(
        sorted(multi_tile_uids.items(), key=lambda x: len(x[1]))
    )

    _log.info(
        f"{len(multi_tile_uids)} waterbodies in total for vector processing"
    )

    waterbodies_uids_fp = join_url(output_dir, "vector_processing_tasks.json")
    with fs.open(waterbodies_uids_fp, "w") as file:
        json.dump(multi_tile_uids, file, indent=2)
    _log.info(
        f"Waterbodies for vector processing of water quality summaries written to {waterbodies_uids_fp}"
    )

    # Can be procesed with resource limits set to 8CPU 60GB memory 100 parallel pods
    uids_2tile = {}
    # Can be procesed with resource limits set to 60CPU 480GB memory 50 parallel pods
    uids_3tile = {}
    # Need to be resampled from 10m to higher to process using nearest neighbour
    # 60CPU 480GB memory 50 parallel pods
    uids_4tile_plus = {}

    for uid, cog_paths in multi_tile_uids.items():
        if len(cog_paths) < 3:
            uids_2tile[uid] = cog_paths
        else:
            if len(cog_paths) == 3:
                uids_3tile[uid] = cog_paths
            else:
                uids_4tile_plus[uid] = cog_paths

    waterbodies_uids_fp = join_url(
        output_dir, "vector_processing_tasks_2tile.json"
    )
    with fs.open(waterbodies_uids_fp, "w") as file:
        json.dump(uids_2tile, file, indent=2)

    _log.info(
        "Waterbodies covered by 2 or less historical extent COGs for vector processing of "
        f"water quality summaries written to {waterbodies_uids_fp}"
    )

    waterbodies_uids_fp = join_url(
        output_dir, "vector_processing_tasks_3tile.json"
    )
    with fs.open(waterbodies_uids_fp, "w") as file:
        json.dump(uids_3tile, file, indent=2)

    _log.info(
        "Waterbodies covered by 3 historical extent COGs for vector processing of water quality "
        f"summaries written to {waterbodies_uids_fp}"
    )

    waterbodies_uids_fp = join_url(
        output_dir, "vector_processing_tasks_4_plustile.json"
    )
    with fs.open(waterbodies_uids_fp, "w") as file:
        json.dump(uids_4tile_plus, file, indent=2)

    _log.info(
        "Waterbodies covered by 4 or more historical extent COGs for vector processing of water "
        f"quality summaries written to {waterbodies_uids_fp}"
    )
