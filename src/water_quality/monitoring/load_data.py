import logging
from collections import defaultdict
from importlib.resources import files
from itertools import chain

import numpy as np
import pandas as pd
import rioxarray
import toolz
import xarray as xr
from deafrica_tools.waterbodies import get_waterbody
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.geo.xr import assign_crs
from odc.stats._text import split_and_check

from water_quality.dates import year_to_dc_datetime
from water_quality.grid import get_waterbodies_grid
from water_quality.io import (
    find_csv_files,
    find_geotiff_files,
    get_parent_dir,
    join_url,
    parse_wq_cog_url,
)
from water_quality.tiling import (
    get_aoi_tiles,
    get_region_code,
    get_tile_region_codes,
)

log = logging.getLogger(__name__)


def _load_all_waterbodies_uids():
    """
    Load the list of waterbodies uids (geohashes) for all waterbody
    polygons in the DE Africa Historical Extent product.

    > **Note**: This file should be updated with every new version
    release of the DE Africa Historical Extent product.
    """
    file_path = files("water_quality.data").joinpath("waterbodies_uids.txt")
    with open(file_path, "r") as f:
        waterbodies_uids = [line.strip() for line in f]
    return waterbodies_uids


def _verify_waterbody_uid(waterbody_uid: str) -> str:
    """
    Check if a waterbody uid is in the list of waterbody uids for all
    waterbodies in the DE Africa Historical Extent product.

    Parameters
    ----------
    waterbody_uid : str
        Geohash / UID for the waterbody to search for.

    Returns
    -------
    str
        Input waterbody uid if found in the DE Africa Historical Extent
        product.

    """
    all_waterbodies_uids = _load_all_waterbodies_uids()
    if waterbody_uid not in all_waterbodies_uids:
        raise ValueError(f"Waterbody {waterbody_uid} not found")
    else:
        return waterbody_uid


def get_waterbody_geom(waterbody_uid: str) -> Geometry:
    """
    Get the geometry (extent) of a waterbody.

    Parameters
    ----------
    waterbody_uid : str
        Geohash / UID for the waterbody to search for.

    Returns
    -------
    Geometry
        Geometry (extent) of the waterbody.
    """
    waterbody_gdf = get_waterbody(waterbody_uid)
    waterbody_geom = Geometry(
        geom=waterbody_gdf.iloc[0].geometry, crs=waterbody_gdf.crs
    )

    gridspec = get_waterbodies_grid()
    waterbody_geom = waterbody_geom.to_crs(gridspec.crs)
    return waterbody_geom


def get_waterbody_geobox(waterbody_uid: str) -> GeoBox:
    """
    Get the Geobox for a waterbody with the same resolution and CRS
    as the DE Africa Water Quality workflow spatial grid.

    Parameters
    ----------
    waterbody_uid : str
        Geohash / UID for the waterbody to search for.

    Returns
    -------
    GeoBox
        GeoBox of the waterbody.
    """
    waterbody_gdf = get_waterbody(waterbody_uid)
    waterbody_geom = Geometry(
        geom=waterbody_gdf.iloc[0].geometry, crs=waterbody_gdf.crs
    )

    gridspec = get_waterbodies_grid()
    waterbody_geobox = GeoBox.from_geopolygon(
        geopolygon=waterbody_geom,
        resolution=gridspec.resolution,
        crs=gridspec.crs,
    )
    return waterbody_geobox


def get_waterbody_tiles(
    waterbody_uid: str,
) -> list[tuple[tuple[int, int], GeoBox]]:
    """
    Get the DE Africa Water Quality tiles overlapping a waterbody's
    extent.

    Parameters
    ----------
    waterbody_uid : str
        Geohash / UID for the waterbody to search for.

    Returns
    -------
    list[tuple[tuple[int, int], GeoBox]]
        Grid index (tile ID) and corresponding GeoBox overlapping the
        waterbody's extent.
    """
    waterbody_geom = get_waterbody_geom(waterbody_uid)
    tiles = get_aoi_tiles(waterbody_geom)
    tiles = list(tiles)
    return tiles


def get_bands_to_load(wq_vars_csv_url: str) -> list[str]:
    """
    Get the list of all the bands required when loading the  water
    quality variables for a tile.

    Parameters
    ----------
    wq_vars_csv_url : str
        Path to the csv file to load the water quality variables table
        from.

    Returns
    -------
    list[str]
        List of all the bands required when loading the  water quality
        variables for a tile.
    """
    wq_vars_df = pd.read_csv(wq_vars_csv_url)
    bands_to_load = list(
        chain.from_iterable(
            [wq_vars_df[col].dropna().to_list() for col in wq_vars_df.columns]
        )
    )

    other_bands = [
        "wofs_ann_pwater",
        "wofs_ann_wetcount",
        "wofs_ann_clearcount",
        "wofs_ann_freq",
    ]
    bands_to_load.extend(other_bands)
    return bands_to_load


def create_ds_from_cogs(
    cog_urls: str,
    bands_to_load: list[str],
    waterbody_uid: str = None,
    aoi_geopolygon: Geometry = None,
) -> xr.Dataset:
    """
    Given a list of all the COGs for a tile for a specific year, load
    the water quality variables (bands) specified and crop the Dataset to
    the extent of the selected waterbody or area of interest.

    Parameters
    ----------
    cog_urls : str
        List of all the COGs found for a tile for a single year.
    bands_to_load : list[str]
        Water quality variables to load from the list of COGs.
    waterbody_uid : str
        The UID/geohash of the waterbody to crop the data to.
    aoi_geopolygon : Geometry
        Geometry defining the area of interest to crop the data to. This
        is mutually exclusive to the `waterboduy_uid` parameter.
    Returns
    -------
    xr.Dataset
        Dataset containing all the water quality variables required,
        cropped to the extent of the selected waterbody or area of
        interest.
    """
    if waterbody_uid is None and aoi_geopolygon is None:
        raise TypeError(
            "You must provide either 'waterbody_uid' or 'aoi_geopolygon'."
        )

    if waterbody_uid is not None and aoi_geopolygon is not None:
        raise TypeError(
            "You must provide either 'waterbody_uid' or "
            "'aoi_geopolygon', but not both."
        )

    if waterbody_uid:
        geom = get_waterbody_geom(waterbody_uid)
    elif aoi_geopolygon:
        gridspec = get_waterbodies_grid()
        geom = aoi_geopolygon.to_crs(gridspec.crs)

    data_vars = {}
    for cog_url in cog_urls:
        band_name, _, year = parse_wq_cog_url(cog_url)
        if band_name in bands_to_load:
            da = rioxarray.open_rasterio(
                cog_url, chunks={"x": 300, "y": 300}
            ).squeeze()
            if "band" in da.coords:
                da = da.drop_vars("band")
            da = assign_crs(da, da.rio.crs)
            da = da.odc.crop(geom)
            time_coords = np.array(
                [year_to_dc_datetime(int(year))], dtype="datetime64[ns]"
            )
            da = da.expand_dims(time=time_coords)
            data_vars[band_name] = da
    ds = xr.Dataset(data_vars)
    return ds


def _get_cog_year(cog_url: str) -> str:
    """
    Helper function to get the year component
    in the url of an annual water quality variable COG
    """
    _, _, temporal_id, band = parse_wq_cog_url(cog_url)
    year, _ = split_and_check(temporal_id, "--P", 2)
    return year


def load_annual_water_quality_variables(
    water_frequency_threshold: float,
    waterbody_uid: str = None,
    aoi_geopolygon: Geometry = None,
    years: list[int] = None,
    product_location: str = "s3://deafrica-water-quality-dev/mapping/wqs_annual/1-0-0/",
) -> xr.Dataset:
    """
    Load the annual water quality variables for a waterbody or
    defined area of interest.

    Parameters
    ----------
    water_frequency_threshold : float
        Threshold to use when classifying water and non-water pixels
        in the normalization process for the water quality variables.
    waterbody_uid : str, optional
        The UID/geohash of the waterbody to load data for.
    aoi_geopolygon : Geometry, optional
        Geometry defining the area of interest to load data for. This
        is mutually exclusive to the `waterboduy_uid` parameter.
    years : list[int], optional
        List of years to load data for. If None all available data is
        loaded.
    product_location : str, optional
        Directory containing the water quality service annual product,
        by default "s3://deafrica-water-quality-dev/mapping/wqs_annual/1-0-0/"

    Returns
    -------
    xr.Dataset
        Chl-A and TSS water quality variables for the specified area of
        interest.
    """

    if waterbody_uid is None and aoi_geopolygon is None:
        raise TypeError(
            "You must provide either 'waterbody_uid' or 'aoi_geopolygon'."
        )

    if waterbody_uid is not None and aoi_geopolygon is not None:
        raise TypeError(
            "You must provide either 'waterbody_uid' or "
            "'aoi_geopolygon', but not both."
        )

    if waterbody_uid:
        tiles = get_waterbody_tiles(waterbody_uid)
        region_codes = get_tile_region_codes(tiles, sep="")
        log.info(
            f"Found {len(tiles)} tiles covering the "
            f"waterbody {waterbody_uid}: {', '.join(region_codes)}"
        )

    if aoi_geopolygon:
        tiles = get_aoi_tiles(aoi_geopolygon)
        tiles = list(tiles)
        region_codes = get_tile_region_codes(tiles, sep="")
        log.info(
            f"Found {len(tiles)} tiles covering "
            f"the area of interest: {', '.join(region_codes)}"
        )

    grouped_by_year_and_tile = defaultdict(lambda: defaultdict(list))
    for tile_id, _ in tiles:
        region_code = get_region_code(tile_id, sep="/")
        tile_cogs_dir = join_url(product_location, region_code)
        all_tile_cog_urls = find_geotiff_files(tile_cogs_dir)
        for year, cog_urls in toolz.groupby(
            _get_cog_year, all_tile_cog_urls
        ).items():
            grouped_by_year_and_tile[year][tile_id].extend(cog_urls)

    if years is not None:
        years = [str(year) for year in years]
        grouped_by_year_and_tile = {
            k: v for k, v in grouped_by_year_and_tile.items() if k in years
        }

    per_year_ds = []
    for year, grouped_by_tile in grouped_by_year_and_tile.items():
        log.info(f"Loading data for year {year}")
        per_tile_ds = []
        for tile_id, cog_urls in grouped_by_tile.items():
            log.info(f"Loading data for tile {get_region_code(tile_id)}")
            wq_vars_csv_url = find_csv_files(get_parent_dir(cog_urls[0]))[0]
            bands_to_load = get_bands_to_load(wq_vars_csv_url=wq_vars_csv_url)

            ds = create_ds_from_cogs(
                cog_urls=cog_urls,
                bands_to_load=bands_to_load,
                waterbody_uid=waterbody_uid,
                aoi_geopolygon=aoi_geopolygon,
            )
            ds = ds.compute()
            ds = normalise_water_quality_variables(
                ds=ds,
                wq_vars_csv_url=wq_vars_csv_url,
                water_frequency_threshold=water_frequency_threshold,
            )
            per_tile_ds.append(ds)
        ds = xr.merge(per_tile_ds)
        per_year_ds.append(ds)

    ds = xr.concat(per_year_ds, dim="time")
    return ds
