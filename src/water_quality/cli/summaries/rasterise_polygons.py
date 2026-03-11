import json
import logging
import sys

import click
import geopandas as gpd
from odc.geo.xr import wrap_xr, write_cog
from rasterio.features import rasterize
from tqdm import tqdm

from water_quality.grid import get_waterbodies_grid
from water_quality.io import check_directory_exists, get_filesystem, join_url
from water_quality.logs import setup_logging
from water_quality.tiling import (
    get_africa_tiles,
    parse_region_code,
    tiles_to_gdf,
)


@click.command(
    name="rasterise-polygons",
    no_args_is_help=True,
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
@click.argument(
    "waterbodies-path",
    type=str,
)
@click.argument(
    "output-dir",
    type=str,
)
def cli(
    log: str,
    waterbodies_path: str,
    output_dir: str,
):
    """
    Rasterise the DE Africa Waterbodies Historical Extent polygons at the file
    specified by WATERBODIES_PATH. The polygons are rasterized to the DE Africa
    Water Quality Monitoring Service grid.
    Write the tiled rasters to the OUTPUT_DIR directory.
    """
    log_level = getattr(logging, log.upper())
    _log = setup_logging(log_level)

    fs = get_filesystem(output_dir, anon=False)
    if not check_directory_exists(output_dir):
        fs.mkdirs(output_dir)

    tiles = get_africa_tiles()
    tiles_gdf = tiles_to_gdf(tiles)
    tiles_gdf = tiles_gdf.set_index("region_code")

    waterbodies = gpd.read_file(waterbodies_path).to_crs(tiles_gdf.crs)
    waterbodies.set_index("wb_id", inplace=True)

    inner_join = gpd.sjoin(
        waterbodies, tiles_gdf, how="inner", predicate="intersects"
    )

    region_codes = sorted(inner_join["region_code"].unique().tolist())

    grid = get_waterbodies_grid()
    nodata = 0

    failed_tasks = []
    with tqdm(
        iterable=region_codes,
        desc="Rasterise historical extent polygons by grid tile",
        total=len(region_codes),
    ) as region_codes:
        for region_code in region_codes:
            try:
                intersecting_polygons = inner_join[
                    inner_join["region_code"].isin([region_code])
                ]

                tile_index = parse_region_code(region_code)
                tile_geobox = grid.tile_geobox(tile_index)

                shapes = zip(
                    intersecting_polygons.geometry, intersecting_polygons.index
                )
                tile_raster_np = rasterize(
                    shapes=shapes,
                    out_shape=tile_geobox.shape,
                    fill=nodata,
                    transform=tile_geobox.transform,
                )
                tile_raster_da = wrap_xr(
                    im=tile_raster_np, gbox=tile_geobox, nodata=nodata
                )
                tags = dict(
                    WB_ID_to_UID=json.dumps(
                        dict(
                            zip(
                                intersecting_polygons.index,
                                intersecting_polygons.uid,
                            )
                        )
                    )
                )
                cog_bytes = write_cog(
                    geo_im=tile_raster_da,
                    fname=":mem:",
                    overwrite=True,
                    nodata=nodata,
                    tags=tags,
                )
                output_cog_url = join_url(
                    output_dir, f"historical_extent_{region_code}.tif"
                )
                with fs.open(output_cog_url, "wb") as f:
                    f.write(cog_bytes)

                _log.debug(
                    f"Rasterized historical extent tile {region_code} saved to {output_cog_url}"
                )
            except Exception as error:
                _log.exception(error)
                failed_tasks.append(region_code)

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
        _log.info(
            "Rasterizing of historical extent polygons completed successfully!"
        )
        sys.exit(0)


if __name__ == "__main__":
    cli()
