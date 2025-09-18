"""
This module provides functions to get DE Africa Waterbodies
Historical Extent data for SDG Indicator 6.6.1 water quality reporting.
"""

import io
import zipfile

import geopandas as gpd
import pandas as pd
import requests
import shapely
from deafrica_tools.waterbodies import (
    get_waterbodies as get_deafrica_waterbodies,
)
from shapely.geometry import box

from water_quality.africa_extent import AFRICA_EXTENT_URL


def get_processed_lakes() -> gpd.GeoDataFrame:
    """
    Get a table containing the processed lakes in the Copernicus
    Land Monitoring Service Lake Water Quality products.

    Returns
    -------
    GeoDataFrame
        Table containing the processed lakes in the Copernicus
        Land Monitoring Service Lake Water Quality products.
    """
    url = "https://land.copernicus.eu/en/technical-library/list-of-processed-lakes-lake-water-quality-v1.0/@@download/file"

    response = requests.get(url)

    # Open the zip file.
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)

    geometry = df["repr_point_wkt"].map(shapely.wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    return gdf


def get_africa_processed_lakes() -> gpd.GeoDataFrame:
    """
    Get a table of the processed lakes in the Copernicus
    Land Monitoring Service Lake Water Quality products
    that are in Africa.

    Returns
    -------
    gpd.GeoDataFrame
        Table containing the processed lakes in the Copernicus
        Land Monitoring Service Lake Water Quality products that are
        in Africa.
    """
    all_lakes = get_processed_lakes()
    africa_extent = gpd.read_file(AFRICA_EXTENT_URL).to_crs("EPSG:4326")
    sel_id_str = all_lakes.sjoin(
        africa_extent, how="inner", predicate="intersects"
    )["id_str"].unique()
    africa_lakes = all_lakes[all_lakes["id_str"].isin(sel_id_str)]
    africa_lakes = africa_lakes.reset_index(drop=True)
    return africa_lakes


def get_waterbodies_geoms(
    country_gdf: gpd.GeoDataFrame, buffer_m: int | float = None
) -> gpd.GeoDataFrame:
    """
    Get the geometry for each waterbody from the Waterbodies Historical Extent
    product that intersects with the country geometry and is listed in the
    lakes processed in the CGLS Lake Water Quality products.

    Parameters
    ----------
    country_gdf : gpd.GeoDataFrame
        Country geometry.
    buffer_m : int | float, optional
        Buffer in meters to apply to the geometries for the waterbodies, by default None

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the waterbodies for a country.
    """
    assert len(country_gdf) == 1

    crs = country_gdf.crs
    country_gdf = country_gdf.to_crs("EPSG:4326")

    # Select all water bodies located within the bounding box of the country
    waterbodies = get_deafrica_waterbodies(
        tuple(country_gdf.total_bounds), crs="EPSG:4326"
    )
    # Filter the waterbodies further to get only waterbodies that intersect with the country's boundary
    intersecting_waterbodies_ids = (
        waterbodies.sjoin(country_gdf, how="inner", predicate="intersects")[
            "uid"
        ]
        .unique()
        .tolist()
    )
    waterbodies = waterbodies[
        waterbodies["uid"].isin(intersecting_waterbodies_ids)
    ]

    # Filter the waterbodies further to get only waterbodies that were processed for the Lake Surface Water Quality
    cgls_processed_lakes = get_africa_processed_lakes()
    intersecting_waterbodies_ids = (
        waterbodies.sjoin(
            cgls_processed_lakes, how="inner", predicate="intersects"
        )["uid"]
        .unique()
        .tolist()
    )
    waterbodies = waterbodies[
        waterbodies["uid"].isin(intersecting_waterbodies_ids)
    ]

    # Filter columns
    waterbodies = waterbodies[["wb_id", "uid", "geometry"]]
    waterbodies.set_geometry("geometry", inplace=True)
    waterbodies.reset_index(drop=True, inplace=True)

    if buffer_m:
        waterbodies = waterbodies.to_crs("EPSG:6933")
        waterbodies["geometry"] = waterbodies["geometry"].buffer(buffer_m)
        waterbodies = waterbodies.to_crs("EPSG:4326")

    country_gdf = country_gdf.to_crs(crs)
    return waterbodies
