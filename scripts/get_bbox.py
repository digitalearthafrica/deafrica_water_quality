import argparse
from importlib.resources import files

import geopandas as gpd
from odc.geo.geom import Geometry

from water_quality.logs import setup_logging
from water_quality.tiling import get_aoi_tiles, tiles_to_gdf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--place-name",
    required=True,
    type=str,
    help="Name of the place to get the bounding box for",
)
args = parser.parse_args()

log = setup_logging()

place_name = args.place_name

log.info(f"Getting bounding box for place: {place_name}")

# Get the tiles that cover the area of interest
places_fp = files("water_quality.data").joinpath("places.parquet")
places_gdf = gpd.read_parquet(places_fp)
place_name_list = places_gdf["name"].to_list()
if place_name not in place_name_list:
    raise ValueError(
        f"'{place_name}' not found in test areas file; expected names include {', '.join(place_name_list)}"
    )
else:
    place = places_gdf[places_gdf["name"].isin([place_name])]
    aoi_geom = Geometry(geom=place.iloc[0].geometry, crs=place.crs)
    tiles = get_aoi_tiles(aoi_geom)

tiles_gdf = tiles_to_gdf(tiles).to_crs("EPSG:4326")
log.info(f"Number of tiles covering {place_name}: {len(tiles_gdf)}")

# Comma seperated string
bounds = ",".join(
    [str(element) for element in tiles_gdf.total_bounds.flatten()]
)
log.info(
    f"Bounding box for all {len(tiles_gdf)} tile(s) covering {place_name}: {bounds}"
)

print(bounds, flush=True)
