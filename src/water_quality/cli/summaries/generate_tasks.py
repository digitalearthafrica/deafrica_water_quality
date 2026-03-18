import json
import logging
from collections import defaultdict

import click
import rioxarray
from tqdm import tqdm

from water_quality.io import (
    check_directory_exists,
    find_geotiff_files,
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
    Get a list of the DE Africa waterbodies historical extent COGs in the
    HISTORICAL_EXTENT_RASTERS_DIR directory to process for the water quality
    annual summaries using the raster processing tool and a mapping of waterbodies
    uids and their corresponding historical extent COGs for vector processing.
    These waterbodies cover multiple tiles and will be processed separately
    using the vector based processing tool.
    """
    log_level = getattr(logging, log.upper())
    _log = setup_logging(log_level)

    _log.info(
        f"Searching for waterbodies historical extent cogs in {historical_extent_rasters_dir} ..."
    )
    historical_extent_cogs = find_geotiff_files(historical_extent_rasters_dir)
    _log.info(
        f"Found {len(historical_extent_cogs)} waterbodies historical extent cogs"
    )

    _log.info(
        "Searching for waterbodies covered by more than 1 historical extent COG ..."
    )
    cog_path_to_uids = {}
    for cog_path in tqdm(historical_extent_cogs):
        ds = rioxarray.open_rasterio(cog_path)
        wbid_to_uid = json.loads(ds.attrs["WB_ID_to_UID"])
        uids = sorted(list(wbid_to_uid.values()))
        cog_path_to_uids[cog_path] = uids

    assert len(cog_path_to_uids) == len(historical_extent_cogs)

    uids_to_cog_paths = defaultdict(list)
    for cog_path, uids in cog_path_to_uids.items():
        for uid in uids:
            uids_to_cog_paths[uid].append(cog_path)

    multi_tile_uids = {}
    for uid, cog_paths in uids_to_cog_paths.items():
        if len(cog_paths) > 1:
            multi_tile_uids[uid] = cog_paths

    _log.info(
        f"Found {len(multi_tile_uids)} waterbodies covered by more than 1 historical extent COG. "
        "These waterbodies will need to be processed seperately during generation of water quality summaries."
    )

    fs = get_filesystem(path=output_dir, anon=False)
    if not check_directory_exists(path=output_dir):
        fs.mkdirs(path=output_dir, exist_ok=True)

    waterbodies_uids_fp = join_url(
        output_dir, "waterbodies_for_vector_processing"
    )
    with fs.open(waterbodies_uids_fp, "w") as file:
        json.dump(multi_tile_uids, file, indent=2)

    _log.info(
        f"Waterbodies' UIDs for vector processing of water quality summaries written to {waterbodies_uids_fp}"
    )

    historical_extent_rasters_fp = join_url(output_dir, "tasks")
    with fs.open(historical_extent_rasters_fp, "w") as file:
        file.write(json.dumps(historical_extent_cogs) + "\n")
    _log.info(
        f"Historical extent rasters for raster processing of water quality summaries written to {historical_extent_rasters_fp}"
    )
