import logging
from collections import defaultdict
from importlib.resources import files

import numpy as np
import pandas as pd
import rioxarray
import toolz
import xarray as xr
from deafrica_tools.waterbodies import get_waterbody
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.geo.xr import assign_crs

from water_quality.dates import year_to_dc_datetime
from water_quality.grid import get_waterbodies_grid
from water_quality.io import (
    find_geotiff_files,
    get_wq_csv_url,
    join_url,
    parse_wq_cog_url,
)
from water_quality.tiling import (
    get_aoi_tiles,
    get_region_code,
    get_tile_region_codes,
)

log = logging.getLogger(__name__)

NORMALISATION_PARAMETERS = {
    "ndssi_rg_msi_agm": {"scale": 83.711, "offset": 56.756},
    "ndssi_rg_oli_agm": {"scale": 45.669, "offset": 45.669},
    "ndssi_rg_tm_agm": {"scale": 149.21, "offset": 57.073},
    "ndssi_bnir_oli_agm": {"scale": 37.125, "offset": 37.125},
    "ti_yu_oli_agm": {"scale": 6.656, "offset": 36.395},
    "ti_yu_tm_agm": {"scale": 8.064, "offset": 42.562},
    "tsm_lym_oli_agm": {"scale": 1.0, "offset": 0.0},
    "tsm_lym_msi_agm": {"scale": 14.819, "offset": -118.137},
    "tsm_lym_tm_agm": {"scale": 1.184, "offset": -2.387},
    "tss_zhang_msi_agm": {"scale": 18.04, "offset": 0.0},
    "tss_zhang_oli_agm": {"scale": 10.032, "offset": 0.0},
    "spm_qiu_oli_agm": {"scale": 1.687, "offset": -0.322},
    "spm_qiu_tm_agm": {"scale": 2.156, "offset": -16.863},
    "spm_qiu_msi_agm": {"scale": 2.491, "offset": -4.112},
    "ndci_msi54_agm": {"scale": 131.579, "offset": 21.737},
    "ndci_msi64_agm": {"scale": 33.153, "offset": 33.153},
    "ndci_msi74_agm": {"scale": 33.516, "offset": 33.516},
    "ndci_tm43_agm": {"scale": 53.157, "offset": 28.088},
    "ndci_oli54_agm": {"scale": 38.619, "offset": 29.327},
    "chla_meris2b_msi_agm": {"scale": 1.148, "offset": -36.394},
    "chla_modis2b_msi_agm": {"scale": 0.22, "offset": 7.139},
    "chla_modis2b_tm_agm": {"scale": 1.209, "offset": -63.141},
    "ndssi_bnir_tm_agm": {"scale": 37.41, "offset": 37.41},
}


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


def _get_cog_year(cog_url: str) -> str:
    _, _, year = parse_wq_cog_url(cog_url)
    return year


def get_wq_measures_cogs(
    waterbody_uid: str, wq_measures_dir: str
) -> dict[str, dict[tuple[int, int], list[str]]]:
    """
    Find all the COGs in the water quality measures directory that
    overlap with the extent of a waterbody and group them by year and
    tile.

    Parameters
    ----------
    waterbody_uid : str
        Geohash / UID for the waterbody to search for.
    wq_measures_dir : str
        Directory containing the water quality measures COGs.

    Returns
    -------
    dict[str, dict[tuple[int, int], list]]
        All the COGs in the water quality measures directory that
        overlap with the extent of a waterbody, grouped by year and
        tile.
    """
    tiles = get_waterbody_tiles(waterbody_uid)
    region_codes = get_tile_region_codes(tiles, sep="")
    log.info(
        f"Found {len(tiles)} tiles covering "
        f"waterbody {waterbody_uid}: {', '.join(region_codes)}"
    )
    grouped_by_year_and_tile = defaultdict(lambda: defaultdict(list))
    for tile_id, _ in tiles:
        region_code = get_region_code(tile_id, "/")
        tile_cogs_dir = join_url(wq_measures_dir, region_code)
        all_tile_cog_urls = find_geotiff_files(tile_cogs_dir)
        for year, cog_urls in toolz.groupby(
            _get_cog_year, all_tile_cog_urls
        ).items():
            grouped_by_year_and_tile[year][tile_id].extend(cog_urls)
    return grouped_by_year_and_tile


def load_wq_measurements_table(wq_parameters_csv_url: str) -> pd.DataFrame:
    """
    Load the water quality measures table from a CSV file.

    Parameters
    ----------
    wq_parameters_csv_url : str
        Path to the csv file to load the water quality measures table
        from.

    Returns
    -------
    pd.DataFrame
        Table containing the water quality measures.
    """
    wq_parameters_df = pd.read_csv(wq_parameters_csv_url)
    assert all(
        col in wq_parameters_df.columns
        for col in ["tss_measure", "chla_measure"]
    ), "Required columns 'tss_measure' and/or 'chla_measure' are missing."
    return wq_parameters_df


def get_bands_to_load(wq_parameters_csv_url: str) -> list[str]:
    """
    Get the list of all the bands required when loading the  water
    quality measures for a tile.

    Parameters
    ----------
    wq_parameters_csv_url : str
        Path to the csv file to load the water quality measures table
        from.

    Returns
    -------
    list[str]
        List of all the bands required when loading the  water quality
        measures for a tile.
    """
    wq_parameters_df = load_wq_measurements_table(wq_parameters_csv_url)
    bands_to_load = []
    for col in wq_parameters_df:
        bands = wq_parameters_df[col].dropna().to_list()
        bands_to_load.extend(bands)

    other_bands = [
        "wofs_ann_pwater",
        "wofs_ann_wetcount",
        "wofs_ann_clearcount",
        "wofs_ann_freq",
    ]
    bands_to_load.extend(other_bands)
    return bands_to_load


def create_ds_from_cogs(
    cog_urls: str, bands_to_load: list[str], waterbody_uid: str
) -> xr.Dataset:
    """
    Given a list of all the COGs for a tile for a specific year, load
    the water quality measures (bands) specified and crop the Dataset to
    the extent of the selected waterbody.

    Parameters
    ----------
    cog_urls : str
        List of all the COGs found for a tile for a single year.
    bands_to_load : list[str]
        Water quality measures (bands) to load from the list of COGs.
    waterbody_uid : str
        The UID/geohash of the waterbody to crop the data to.

    Returns
    -------
    xr.Dataset
        Dataset containing all the water quality measures required,
        cropped to the extent of the selected waterbody.
    """
    waterbody_geom = get_waterbody_geom(waterbody_uid)

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
            da = da.odc.crop(waterbody_geom)
            time_coords = np.array(
                [year_to_dc_datetime(int(year))], dtype="datetime64[ns]"
            )
            da = da.expand_dims(time=time_coords)
            data_vars[band_name] = da
    ds = xr.Dataset(data_vars)
    return ds


def normalise_water_quality_measures(
    ds: xr.Dataset,
    wq_parameters_csv_url: str,
    water_frequency_threshold: float,
) -> xr.Dataset:
    """
    Normalize the water quality measures in a Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the water quality measures to normalize.
    wq_parameters_csv_url : str
        Path to the csv file to load the water quality measures table
        from.
    water_frequency_threshold : float
        Threshold to use when classifying water and non-water pixels
        in the normalization process.

    Returns
    -------
    xr.Dataset
        Water quality measures after normalization.
    """
    wq_parameters_df = load_wq_measurements_table(wq_parameters_csv_url)

    for col in wq_parameters_df.columns:
        bands = wq_parameters_df[col].dropna().to_list()
        ds = ds.assign_coords({col: bands})

        var_name = col.replace("_measure", "")

        da_dims = ["time", "x", "y", col]
        da_coords = {dim: ds.coords[dim] for dim in da_dims}
        da_data_shape = tuple(len(da_coords[dim]) for dim in da_dims)
        da = xr.DataArray(
            data=np.zeros(da_data_shape, dtype=np.float32),
            coords=da_coords,
            dims=da_dims,
            name=var_name,
        )
        log.info(
            f"Applying scale and offset to {var_name} water quality variables"
        )
        for band in bands:
            log.debug(
                f"Applying scale and offset to {var_name} "
                f"water quality variable {band}"
            )
            scale = NORMALISATION_PARAMETERS[band]["scale"]
            offset = NORMALISATION_PARAMETERS[band]["offset"]
            da.sel({col: band})[:] = ds[band] * scale + offset

        ds[var_name] = da
        ds = ds.drop_vars(bands, errors="ignore")

    log.info("Get median of tss and chla measurements for water pixels")
    water_mask = ds["wofs_ann_pwater"] > water_frequency_threshold
    ds["tss_agm_med"] = xr.where(
        water_mask, ds["tss"].median(dim="tss_measure"), np.nan
    )
    ds["chla_agm_med"] = xr.where(
        water_mask, ds["chla"].median(dim="chla_measure"), np.nan
    )
    ds = ds.drop_vars(["tss", "chla"])
    ds = ds.drop_dims(["tss_measure", "chla_measure"])
    return ds


def load_water_quality_measures(
    waterbody_uid: str, wq_measures_dir: str, water_frequency_threshold: float
) -> xr.Dataset:
    """Load the water quality measures for a waterbody.

    Parameters
    ----------
    waterbody_uid : str
        The UID/geohash of the waterbody to load data for.
    wq_measures_dir : str
        Directory containing the water quality measures COGs.
    water_frequency_threshold : float
        Threshold to use when classifying water and non-water pixels
        in the normalization process.

    Returns
    -------
    xr.Dataset
        Chl-A and TSS water quality measures for a waterbody.
    """
    log.info(f"Loading data for waterbody {waterbody_uid}")
    grouped_by_year_and_tile = get_wq_measures_cogs(
        waterbody_uid=waterbody_uid, wq_measures_dir=wq_measures_dir
    )
    per_year_ds = []
    for year, grouped_by_tile in grouped_by_year_and_tile.items():
        log.info(f"Loading data for year {year}")
        per_tile_ds = []
        for tile_id, cog_urls in grouped_by_tile.items():
            log.info(f"Loading data for tile {get_region_code(tile_id)}")
            wq_parameters_csv_url = get_wq_csv_url(
                output_directory=wq_measures_dir, tile_id=tile_id, year=year
            )
            bands_to_load = get_bands_to_load(
                wq_parameters_csv_url=wq_parameters_csv_url
            )

            ds = create_ds_from_cogs(
                cog_urls=cog_urls,
                bands_to_load=bands_to_load,
                waterbody_uid=waterbody_uid,
            )
            ds = ds.compute()
            ds = normalise_water_quality_measures(
                ds=ds,
                wq_parameters_csv_url=wq_parameters_csv_url,
                water_frequency_threshold=water_frequency_threshold,
            )
            per_tile_ds.append(ds)
        ds = xr.merge(per_tile_ds)
        per_year_ds.append(ds)

    ds = xr.concat(per_year_ds, dim="time")
    return ds
