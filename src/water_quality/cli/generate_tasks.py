from importlib.resources import files
from itertools import chain

import click
import geopandas as gpd
from odc.geo.geom import Geometry

from water_quality.io import (
    check_directory_exists,
    check_file_exists,
    get_filesystem,
)
from water_quality.logs import setup_logging
from water_quality.tasks import create_task_id
from water_quality.tiling import (
    get_africa_tiles,
    get_aoi_tiles,
    parse_region_code,
)


@click.command(
    name="generate-tasks",
    no_args_is_help=True,
)
@click.option(
    "--tile-ids",
    help="Optional list of comma separated tile IDs in the format "
    "x{x:02d}/y{y:02d} to generate tasks for. For example "
    "`x188/y109,x178/y095,x199y/100`",
)
@click.option(
    "--tile-ids-file",
    help="Optional path to text file containing the tile IDs to "
    "generate tasks for. This file can be generated using the command "
    "`wq-generate-tiles`.",
)
@click.option(
    "--place-name",
    type=str,
    help="Optional name of a test area to get the tile IDs to generate "
    "tasks for. To view the names of these predefined test areas, run "
    "the command `wq-list-test-areas`.",
)
@click.argument(
    "start-year",
    type=int,
)
@click.argument(
    "end-year",
    type=int,
)
@click.argument(
    "output-file",
    type=str,
)
def cli(
    tile_ids: str,
    tile_ids_file: str,
    place_name: str,
    start_year: int,
    end_year: int,
    output_file: str,
):
    """
    Prepare tasks for the time range START_YEAR to END_YEAR for
    running the DE Africa Water Quality continental workflow on. If no
    tile IDs are specified, the tasks will be generated for all the
    tiles across Africa. The tasks will be written to the file
    OUTPUT_FILE.
    """
    log = setup_logging()

    if tile_ids and tile_ids_file and place_name:
        raise click.UsageError(
            "Specify exactly one of --tile-ids or --tile-ids-file "
            "or --place-name."
        )

    if tile_ids:
        region_codes = tile_ids.split(",")
        region_codes = [i.strip() for i in region_codes]
        tile_ids_list = [parse_region_code(i) for i in region_codes]
    elif tile_ids_file:
        # Assumption here is the file is public-read.
        if not check_file_exists(tile_ids_file):
            raise FileNotFoundError(f"{tile_ids_file} does not exist!")
        else:
            fs = get_filesystem(tile_ids_file, anon=True)
            with fs.open(tile_ids_file, "r") as f:
                region_codes = f.readlines()
                region_codes = [i.strip() for i in region_codes]
                tile_ids_list = [parse_region_code(i) for i in region_codes]
    else:
        if place_name:
            places_fp = files("water_quality.data").joinpath("places.parquet")
            places_gdf = gpd.read_parquet(places_fp)
            place_name_list = places_gdf["name"].to_list()
            if place_name not in place_name_list:
                raise ValueError(
                    f"{place_name} not in found in test areas file. "
                    f"Expected names include {' ,'.join(place_name_list)}"
                )
            else:
                log.info(f"Getting tiles for test area {place_name}")
                place = places_gdf[places_gdf["name"].isin([place_name])]
                aoi_geom = Geometry(geom=place.iloc[0].geometry, crs=place.crs)
                tiles = get_aoi_tiles(aoi_geom)
        else:
            log.info("Getting tiles for all of Africa for continental run")
            tiles = get_africa_tiles(save_to_disk=False)

        tiles = list(tiles)
        tile_ids_list = [tile[0] for tile in tiles]

    years = list(range(start_year, end_year + 1))

    tasks = []
    for year in years:
        yearly_tasks = [
            create_task_id(year=year, tile_id=tile_id)
            for tile_id in tile_ids_list
        ]
        tasks.append(yearly_tasks)

    tasks = list(chain.from_iterable(tasks))
    tasks.sort()

    log.info(f"Total tasks: {len(tasks)}")

    fs = get_filesystem(path=output_file, anon=False)
    parent_dir = fs._parent(output_file)
    if not check_directory_exists(parent_dir):
        fs.makedirs(parent_dir, exist_ok=True)

    with fs.open(output_file, "w") as file:
        for task_id in tasks:
            file.write(str(task_id) + "\n")
    log.info(f"{len(tasks)} tasks written to {output_file}")
