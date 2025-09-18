import logging
from collections import Counter
from typing import Any

import dask
import numpy as np
import xarray as xr
from datacube import Datacube
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject

from water_quality.dates import year_to_dc_datetime
from water_quality.mapping.instruments import INSTRUMENTS_MEASUREMENTS
from water_quality.tiling import reproject_tile_geobox

log = logging.getLogger(__name__)

INSTRUMENTS_PRODUCTS = {
    "tm_agm": ["gm_ls5_ls7_annual"],
    "oli_agm": ["gm_ls8_annual", "gm_ls8_ls9_annual"],
    "msi_agm": ["gm_s2_annual"],
    "tirs": ["ls5_st", "ls7_st", "ls8_st", "ls9_st"],
    "wofs_ann": ["wofs_ls_summary_annual"],
}


def get_dc_products(instrument_name: str) -> list[str]:
    """
    Get the datacube products to load for a given instrument.

    Parameters
    ----------
    instrument_name : str
        Name of the instruments

    Returns
    -------
    list[str]
        Datacube products to load for an instrument

    """
    dc_products = INSTRUMENTS_PRODUCTS.get(instrument_name, None)
    if dc_products is None:
        raise NotImplementedError(
            f"Datacube products for the instrument {instrument_name} "
            "are not defined."
        )
    else:
        return dc_products


def get_dc_measurements(instrument_name: str) -> list[str]:
    """
    Get the datacube measurements to load for a given instrument.

    Parameters
    ----------
    instrument_name : str
        Name of the instrument

    Returns
    -------
    list[str]
        Datacube measurements to load for the instrument

    """
    measurements = INSTRUMENTS_MEASUREMENTS.get(instrument_name, None)
    if measurements is None:
        raise NotImplementedError(
            f"Datacube measurements for the instrument {instrument_name} "
            "are not defined."
        )
    else:
        dc_measurements: list[str] = []
        for measurement_name, measurement_info in measurements.items():
            is_required = measurement_info["parameters"][0]
            assert isinstance(is_required, bool)
            if is_required is True:
                dc_measurements.append(measurement_name)
            else:
                continue
        return dc_measurements


def get_measurements_name_dict(instrument_name: str) -> dict[str, tuple[str]]:
    """
    Get the dictionary for re-naming measurements to have unique
    dataset variable names for the loaded data for an instrument.

    Parameters
    ----------
    instrument_name : str
        Name of the instrument

    Returns
    -------
    dict[str, tuple[str]]
        Dictionary whose keys are the datacube measurements loaded for
        the instrument and whose values are the desired names for the
        measurements.

    """
    measurements = INSTRUMENTS_MEASUREMENTS.get(instrument_name, None)
    if measurements is None:
        raise NotImplementedError(
            f"Datacube measurements for the instrument {instrument_name} "
            "are not defined."
        )
    else:
        measurements_name_dict: dict[str, tuple[str]] = {}
        for measurement_name, measurement_info in measurements.items():
            new_measurement_name: tuple[str] = measurement_info["varname"]
            is_required = measurement_info["parameters"][0]
            assert isinstance(is_required, bool)
            if is_required is True:
                measurements_name_dict[measurement_name] = new_measurement_name
            else:
                continue
        return measurements_name_dict


def build_dc_queries(
    instruments_to_use: dict[str, dict[str, bool]],
    start_date: str,
    end_date: str,
    resampling: str = "bilinear",
) -> dict[str, dict[str, Any]]:
    """
    Build a reusable datacube query for each instrument to load
    data for.

    Parameters
    ----------
    instruments_to_use : dict[str, dict[str, bool]]
        A dictionary of the selected instruments to use for the analysis.
    start_date : str
        The start of the time range to load data for.
    end_date : str
        The end of the time range to load data for.
    resampling : str, optional
        Resampling method to use, by default "bilinear".
    Returns
    -------
    dict[str, dict[str, Any]]
        Datacube query for each instrument.
    """
    dc_queries = {}
    for instrument_name, usage in instruments_to_use.items():
        if usage["use"] is True:
            dc_products = get_dc_products(instrument_name)
            dc_measurements = get_dc_measurements(instrument_name)
            dc_query = dict(
                product=dc_products,
                measurements=dc_measurements,
                time=(start_date, end_date),
                resampling=resampling,
                # align=(0, 0), not supported when using like
            )
            dc_queries[instrument_name] = dc_query
    return dc_queries


def process_st_data_to_annnual(
    ds_tirs: xr.Dataset, ds_wofs_ann: xr.Dataset
) -> xr.Dataset:
    """
    Processes daily surface temperature data into an annual time series,
    masked by water extent derived from WOfS annual statistics.

    The function performs the following steps:
    - Converts raw surface temperature values to degrees Celsius.
    - Filters data by temperature validity, quality assurance threshold,
      and high surface emissivity.
    - Aggregates daily temperature data into annual median, 10th
        percentile (min), and 90th percentile (max) statistics.
    - Masks aggregated temperature data to include only pixels with
      consistent water presence based on the WOfS annual frequency
      threshold.

    Parameters
    ----------
    ds_tirs : xr.Dataset
        An xarray Dataset containing daily surface temperature data
        with bands:
            - 'tirs_st'
            - 'tirs_st_qa'
            - 'tirs_emis'

    ds_wofs_ann : xr.Dataset
        An xarray Dataset of Water Observations from Space (WOfS) annual
        water frequency statistics. It must include the band:
        'wofs_ann_freq'

    Returns
    -------
    xr.Dataset
        An annual xarray Dataset with the following data variables:
            - 'tirs_st_ann_med': Median annual surface temperature (°C).
            - 'tirs_st_ann_min': 10th percentile annual temperature (°C).
            - 'tirs_st_ann_max': 90th percentile annual temperature (°C).

        All layers are masked to include only pixels where the WOfS
        annual water frequency exceeds 0.5.
    """
    # Rescale the daily timeseries to centigrade, remove outliers,
    # apply quality filter and also filter on emissivity > 0.95.
    tirs_st = (ds_tirs["tirs_st"] * 0.00341802 + 149.0) - 273.15
    # -- uncertainty in kelvin
    tirs_st_qa = ds_tirs["tirs_st_qa"] * 0.01
    # -- emissivity fraction
    tirs_emis = ds_tirs["tirs_emis"] * 0.0001

    valid_mask = (tirs_st > 0) & (tirs_st_qa < 5) & (tirs_emis > 0.95)
    ds_tirs["tirs_st"] = tirs_st.where(valid_mask)

    del tirs_st, tirs_st_qa, tirs_emis, valid_mask

    if ds_tirs.chunks is not None:
        # Rechunk so the time dimension has only one chunk
        # for the quantile commputation
        ds_tirs = ds_tirs.chunk({"time": ds_tirs.sizes["time"]})

    annual_ds_tirs = xr.Dataset()

    group = ds_tirs["tirs_st"].groupby("time.year")
    annual_ds_tirs["tirs_st_ann_med"] = group.median(dim="time")
    annual_ds_tirs["tirs_st_ann_min"] = group.quantile(0.1, dim="time")
    annual_ds_tirs["tirs_st_ann_max"] = group.quantile(0.9, dim="time")

    # Replace the year coordinate with datetime64[ns] time coordinate
    annual_ds_tirs = annual_ds_tirs.rename({"year": "time"})
    time_values = np.array(
        [year_to_dc_datetime(i) for i in annual_ds_tirs.time.values],
        dtype="datetime64[ns]",
    )
    annual_ds_tirs = annual_ds_tirs.assign_coords(time=time_values)

    # Restrict values to areas of water
    water_frequency_threshold = 0.5
    water_mask = (
        ds_wofs_ann["wofs_ann_freq"].sel(time=time_values)
        > water_frequency_threshold
    )

    for var in ["tirs_st_ann_med", "tirs_st_ann_min", "tirs_st_ann_max"]:
        annual_ds_tirs[var] = annual_ds_tirs[var].where(water_mask)

    return annual_ds_tirs


def build_wq_agm_dataset(
    dc_queries: dict[str, dict[str, Any]],
    tile_geobox: GeoBox,
    dc: Datacube = None,
) -> xr.Dataset:
    """Build a combined annual dataset from loading data
    for each instrument using the datacube queries provided.

    Parameters
    ----------
    dc_queries : dict[str, dict[str, Any]]
        Datacube query to use to load data for each instrument.

    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it’s crs.

    dc: Datacube
        Datacube connection to use when loading data.

    Returns
    -------
    xr.Dataset
        A single dataset containing all the data found for each
        instrument in the datacube.
    """
    if dc is None:
        dc = Datacube(app="Build_wq_agm_dataset")

    # Get default resolution to load data in from the tile Geobox
    # should be 10m.
    default_res = int(abs(tile_geobox.resolution.x))

    # Load Landsat surface temperature, surface reflectance, and
    # derivatives (e.g. WOfS, GeoMADs) data in the native resolution
    # of 30m .
    log.info("Loading data using datacube queries ...")
    loaded_data: dict[str, xr.Dataset] = {}
    for instrument_name, dc_query in dc_queries.items():
        if instrument_name != "msi_agm" and default_res != 30:
            like = reproject_tile_geobox(
                tile_geobox=tile_geobox, resolution_m=30
            )
        else:
            like = tile_geobox

        xy_chunk_size = int(like.shape.x / 5)
        dask_chunks = {"x": xy_chunk_size, "y": xy_chunk_size}

        ds = dc.load(**dc_query, like=like, dask_chunks=dask_chunks)
        ds = ds.rename(get_measurements_name_dict(instrument_name))
        loaded_data[instrument_name] = ds

    log.info("Processing surface temperature data to annual composite ...")
    if "tirs" in loaded_data.keys():
        if "wofs_ann" not in loaded_data.keys():
            raise ValueError(
                "Data for the wofs_ann instrument is required to process "
                "daily surface temperature data for the tirs instrument into "
                "an annual timeseries."
            )
        else:
            loaded_data["tirs"] = process_st_data_to_annnual(
                ds_tirs=loaded_data["tirs"],
                ds_wofs_ann=loaded_data["wofs_ann"],
            )

    log.info("Computing instrument datasets ...")
    loaded_data = dict(
        zip(loaded_data.keys(), dask.compute(*loaded_data.values()))
    )

    log.info("Reprojecting instrument datasets ...")
    for instrument_name, ds in loaded_data.items():
        if ds.odc.geobox.resolution != tile_geobox.resolution:
            loaded_data[instrument_name] = xr_reproject(
                loaded_data[instrument_name],
                how=tile_geobox,
                resampling=dc_queries[instrument_name]["resampling"],
            )

    combined = xr.merge(list(loaded_data.values()))
    combined = combined.drop_vars("quantile", errors="ignore")

    return combined
