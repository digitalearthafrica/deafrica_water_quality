import copy
import logging
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
from datacube import Datacube
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject

from water_quality.dates import (
    validate_end_date,
    validate_start_date,
    year_to_dc_datetime,
)
from water_quality.mapping.instruments import INSTRUMENTS_MEASUREMENTS
from water_quality.tiling import reproject_tile_geobox

log = logging.getLogger(__name__)

COMPOSITE_INSTRUMENTS = {
    "tm_agm": ["gm_ls5_ls7_annual"],
    "oli_agm": ["gm_ls8_annual", "gm_ls8_ls9_annual"],
    "msi_agm": ["gm_s2_annual"],
    "wofs_ann": ["wofs_ls_summary_annual"],
    "tirs": ["ls5_st", "ls7_st", "ls8_st", "ls9_st"],
}
SINGLE_DAY_INSTRUMENTS = {
    "tm": ["ls5_sr", "ls7_sr"],
    "oli": ["ls8_sr", "ls9_sr"],
    "msi": ["s2_l2a"],
}
INSTRUMENTS_PRODUCTS = {**COMPOSITE_INSTRUMENTS, **SINGLE_DAY_INSTRUMENTS}


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
        Datacube queries for each instrument.
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
            )
            if instrument_name in SINGLE_DAY_INSTRUMENTS:
                dc_query.update({"group_by": "solar_day"})
            dc_queries[instrument_name] = dc_query
    return dc_queries


def load_oli_agm_data(
    dc_query: dict[str, Any], tile_geobox: GeoBox, compute: bool, dc: Datacube
) -> xr.Dataset:
    """Load and process data for the `oli_agm` instrument.

    Parameters
    ----------
    dc_query : dict[str, Any]
        Datacube query to use to load data for the instrument `oli_agm`.
    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it's crs.
    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.
    dc : Datacube
        Datacube connection to use when loading data.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the processed data for the
        instrument oli_agm.

    """
    log.info("Loading data for the instrument `oli_agm` ...")
    query = copy.deepcopy(dc_query)

    if dc is None:
        dc = Datacube(app="LoadOliAgm")

    dask_chunks = {"x": 3200, "y": 3200}
    ds = dc.load(**query, like=tile_geobox, dask_chunks=dask_chunks)
    ds = ds.rename(get_measurements_name_dict("oli_agm"))

    # For each band mask no data values to np.nan
    for band in ds.data_vars:
        nodata = ds[band].attrs["nodata"]
        ds[band] = ds[band].where(ds[band] != nodata)

    if compute:
        log.info("Computing oli_agm dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_oli_data(
    dc_query: dict[str, Any],
    tile_geobox: GeoBox,
    compute: bool = True,
    dc: Datacube = None,
) -> xr.Dataset:
    """Load and process data for the `oli` instrument.

    Parameters
    ----------
    dc_query : dict[str, Any]
        Datacube query to use to load data for the instrument `oli`.
    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it's crs.
    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.
    dc : Datacube
        Datacube connection to use when loading data, by default None.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the processed data for the
        instrument oli.
    """
    log.info("Loading data for the instrument `oli` ...")
    query = copy.deepcopy(dc_query)

    if dc is None:
        dc = Datacube(app="LoadOli")

    dask_chunks = {"x": 3200, "y": 3200}
    ds = dc.load(**query, like=tile_geobox, dask_chunks=dask_chunks)
    ds = ds.rename(get_measurements_name_dict("oli"))

    # For each band mask no data values to np.nan
    for band in ds.data_vars:
        if band != "oli_pq":
            # Mask no data (0)
            # ? pq/pixel quality band no data value is 1
            ds[band] = ds[band].where(ds[band] > 0)

            # Rescale and multiply by 10,000 to match range of data
            # for the msi instrument.
            ds[band] = (2.75e-5 * ds[band] - 0.2) * 10000

    if compute:
        log.info("Computing oli dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_msi_agm_data(
    dc_query: dict[str, Any], tile_geobox: GeoBox, compute: bool, dc: Datacube
) -> xr.Dataset:
    """Load and process data for the `msi_agm` instrument.

    Parameters
    ----------
    dc_query : dict[str, Any]
        Datacube query to use to load data for the instrument `msi_agm`.
    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it's crs.
    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.
    dc : Datacube
        Datacube connection to use when loading data.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the processed data for the
        instrument `msi_agm`.
    """
    log.info("Loading data for the instrument `msi_agm` ...")
    query = copy.deepcopy(dc_query)

    if dc is None:
        dc = Datacube(app="LoadMsiAgm")

    dask_chunks = {"x": 3200, "y": 3200}
    ds = dc.load(**query, like=tile_geobox, dask_chunks=dask_chunks)
    ds = ds.rename(get_measurements_name_dict("msi_agm"))

    # For each band mask no data values to np.nan
    for band in ds.data_vars:
        nodata = ds[band].attrs["nodata"]
        ds[band] = ds[band].where(ds[band] != nodata)

    if compute:
        log.info("Computing msi_agm dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_msi_data(
    dc_query: dict[str, Any], tile_geobox: GeoBox, compute: bool, dc: Datacube
) -> xr.Dataset:
    """Load and process data for the `msi` instrument.

    Parameters
    ----------
    dc_query : dict[str, Any]
        Datacube query to use to load data for the instrument `msi`.
    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it's crs.
    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.
    dc : Datacube
        Datacube connection to use when loading data.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the processed data for the
        instrument `msi`.
    """
    log.info("Loading data for the instrument `msi` ...")
    query = copy.deepcopy(dc_query)

    if dc is None:
        dc = Datacube(app="LoadMsi")

    dask_chunks = {"x": 3200, "y": 3200}
    ds = dc.load(**query, like=tile_geobox, dask_chunks=dask_chunks)
    ds = ds.rename(get_measurements_name_dict("msi"))

    # For each band mask no data values to np.nan
    for band in ds.data_vars:
        # Nodata value for all bands is 0
        ds[band] = ds[band].where(ds[band] > 0)
        # TODO: Add rescaling when switching to s2_l2a_c1

    if compute:
        log.info("Computing msi dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_tm_agm_data(
    dc_query: dict[str, Any], tile_geobox: GeoBox, compute: bool, dc: Datacube
) -> xr.Dataset:
    """Load and process data for the `tm_agm` instrument.

    Parameters
    ----------
    dc_query : dict[str, Any]
        Datacube query to use to load data for the instrument `tm_agm`.
    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it's crs.
    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.
    dc : Datacube
        Datacube connection to use when loading data.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the processed data for the
        instrument `tm_agm`.
    """
    log.info("Loading data for the instrument `tm_agm` ...")
    query = copy.deepcopy(dc_query)

    if dc is None:
        dc = Datacube(app="LoadTmAgm")

    dask_chunks = {"x": 3200, "y": 3200}
    ds = dc.load(**query, like=tile_geobox, dask_chunks=dask_chunks)
    ds = ds.rename(get_measurements_name_dict("tm_agm"))

    # For each band mask no data values to np.nan
    for band in ds.data_vars:
        nodata = ds[band].attrs["nodata"]
        ds[band] = ds[band].where(ds[band] != nodata)

    if compute:
        log.info("Computing tm_agm dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_tm_data(
    dc_query: dict[str, Any], tile_geobox: GeoBox, compute: bool, dc: Datacube
) -> xr.Dataset:
    """Load and process data for the `tm` instrument.

    Parameters
    ----------
    dc_query : dict[str, Any]
        Datacube query to use to load data for the instrument `tm`.
    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it's crs.
    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.
    dc : Datacube
        Datacube connection to use when loading data.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the processed data for the
        instrument `tm`.
    """
    log.info("Loading data for the instrument `tm` ...")
    query = copy.deepcopy(dc_query)

    if dc is None:
        dc = Datacube(app="LoadTm")

    dask_chunks = {"x": 3200, "y": 3200}
    ds = dc.load(**query, like=tile_geobox, dask_chunks=dask_chunks)
    ds = ds.rename(get_measurements_name_dict("tm"))

    # For each band mask no data values to np.nan
    for band in ds.data_vars:
        # Nodata value for surface reflectance bands is 0
        # ? pq/pixel quality band no data value is 1
        ds[band] = ds[band].where(ds[band] > 0)

        if band != "tm_pq":
            # Rescale and multiply by 10,000 to match range of data
            # for the msi instrument.
            ds[band] = (2.75e-5 * ds[band] - 0.2) * 10000

    if compute:
        log.info("Computing tm dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_tirs_data(
    dc_query: dict[str, Any], tile_geobox: GeoBox, compute: bool, dc: Datacube
) -> xr.Dataset:
    """Load and process data for the `tirs` instrument.

    Parameters
    ----------
    dc_query : dict[str, Any]
        Datacube query to use to load data for the instrument `tirs`.
    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it's crs.
    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.
    dc : Datacube
        Datacube connection to use when loading data.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the processed data for the
        instrument `tirs`.
    """
    log.info("Loading data for the instrument `tirs` ...")
    query = copy.deepcopy(dc_query)

    if dc is None:
        dc = Datacube(app="LoadTirs")

    dask_chunks = {"x": 3200, "y": 3200}
    ds = dc.load(**query, like=tile_geobox, dask_chunks=dask_chunks)
    ds = ds.rename(get_measurements_name_dict("tirs"))

    # For each band mask no data values to np.nan
    # tirs_st_qa -9999, tirs_emis -9999, tirs_st 0
    for band in ds.data_vars:
        if band != "tirs_st":
            nodata = ds[band].attrs["nodata"]
            ds[band] = ds[band].where(ds[band] != nodata)
        else:
            ds["tirs_st"] = ds["tirs_st"].where(ds["tirs_st"] > 0)

    # Rescale data
    ds["tirs_st_qa"] = 0.01 * ds["tirs_st_qa"]
    ds["tirs_emis"] = 0.0001 * ds["tirs_emis"]
    ds["tirs_st"] = ds["tirs_st"] * 0.00341802 + 149.0

    # Convert surface temperature from Kelvin to centrigrade.
    ds["tirs_st"] = ds["tirs_st"] - 273.15

    if compute:
        log.info("Computing tm dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_tirs_annual_composite_data(
    dc_query: dict[str, Any],
    tile_geobox: GeoBox,
    compute: bool,
    dc: Datacube,
) -> xr.Dataset:
    """Load and process data for the `tirs` instrument to produce an
    annual composite.

    Parameters
    ----------
    dc_query: dict[str, Any]
        Datacube query to use to load data for the instrument `tirs`.

    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it's crs.

    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.

    dc : Datacube
        Datacube connection to use when loading data.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the surface temperature annnual
        composite produced from data for the instrument `tirs`.
    """
    query = copy.deepcopy(dc_query)

    # Due to memory constraints tirs data must be loaded in its native
    # resolution of 30 m and later reprojected to the target tile geobox
    # if upsampling the data.
    if abs(tile_geobox.resolution.y) < 30:
        native_tirs_geobox = reproject_tile_geobox(
            tile_geobox=tile_geobox, resolution_m=30
        )
        ds_tirs = load_tirs_data(
            dc_query=query,
            tile_geobox=native_tirs_geobox,
            compute=False,
            dc=dc,
        )
    else:
        native_tirs_geobox = None
        ds_tirs = load_tirs_data(
            dc_query=query,
            tile_geobox=tile_geobox,
            compute=False,
            dc=dc,
        )

    # Remove outliers (no data value for surface temp is 0),
    # apply quality filter
    # and also filter on emissivity > 0.95
    valid_mask = (
        (ds_tirs["tirs_st"] > 0)
        & (ds_tirs["tirs_st_qa"] < 5)
        & (ds_tirs["tirs_emis"] > 0.95)
    )
    ds_tirs["tirs_st"] = ds_tirs["tirs_st"].where(valid_mask)

    if ds_tirs.chunks is not None:
        # Rechunk so the time dimension has only one chunk
        # for the quantile commputation
        ds_tirs = ds_tirs.chunk({"time": ds_tirs.sizes["time"]})

    annual_ds_tirs = xr.Dataset()

    group = ds_tirs["tirs_st"].groupby("time.year")
    annual_ds_tirs["tirs_st_ann_med"] = group.median(dim="time")

    quantiles = [0.1, 0.9]
    quantile_results = group.quantile(quantiles, dim="time")
    annual_ds_tirs["tirs_st_ann_min"] = quantile_results.sel(quantile=0.1)
    annual_ds_tirs["tirs_st_ann_max"] = quantile_results.sel(quantile=0.9)

    # Replace the year coordinate with datetime64[ns] time coordinate
    annual_ds_tirs = annual_ds_tirs.rename({"year": "time"})
    time_values = np.array(
        [year_to_dc_datetime(i) for i in annual_ds_tirs.time.values],
        dtype="datetime64[ns]",
    )
    annual_ds_tirs = annual_ds_tirs.assign_coords(time=time_values)

    if native_tirs_geobox is not None:
        # Reproject to target tile geobox
        annual_ds_tirs = xr_reproject(
            annual_ds_tirs,
            how=tile_geobox,
            resampling=query["resampling"],
        )

    if compute:
        log.info("Computing tirs annual composite dataset ...")
        annual_ds_tirs = annual_ds_tirs.compute()
        log.info("Done.")
    return annual_ds_tirs


def load_wofs_ann_data(
    dc_query: dict[str, Any],
    tile_geobox: GeoBox,
    compute: bool = True,
    dc: Datacube = None,
) -> xr.Dataset:
    """Load and process data for the `wofs_ann` instrument.

    Parameters
    ----------
    dc_query : dict[str, Any]
        Datacube query to use to load data for the instrument `wofs_ann`.
    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it's crs.
    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.
    dc : Datacube
        Datacube connection to use when loading data, by default None.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the processed data for the
        instrument `wofs_ann`.
    """
    log.info("Loading data for the instrument wofs_ann ...")
    query = copy.deepcopy(dc_query)

    if dc is None:
        dc = Datacube(app="LoadWofsAnn")

    dask_chunks = {"x": 3200, "y": 3200}
    ds = dc.load(**query, like=tile_geobox, dask_chunks=dask_chunks)
    ds = ds.rename(get_measurements_name_dict("wofs_ann"))

    # For each band mask no data values to np.nan
    for band in ds.data_vars:
        nodata = ds[band].attrs["nodata"]
        ds[band] = ds[band].where(ds[band] != nodata)

    if compute:
        log.info("Computing wofs_ann dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_water_mask(
    dc_query: dict[str, Any],
    tile_geobox: GeoBox,
    compute: bool = True,
    dc: Datacube = None,
) -> xr.DataArray:
    """Load and process 5 years data for the `wofs_ann` instrument to
    generate a water mask.

    Parameters
    ----------
    dc_query : dict[str, Any]
        Datacube query to use to load data for the instrument `wofs_ann`.
    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it's crs.
    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.
    dc : Datacube
        Datacube connection to use when loading data, by default None.

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the water mask derived from the
        wofs_ann data.
    """
    log.info("Loading data for the instrument wofs_ann ...")
    query = copy.deepcopy(dc_query)

    # Assumption here is that start and end date will
    # always be covering a single year.
    year_start = validate_start_date(dc_query["time"][0])
    year_end = validate_end_date(dc_query["time"][1])
    delta = (year_end - year_start).days + 1
    assert delta in [365, 366], (
        f"Expected time in query to cover a single year (365 or 366 days), not {delta} days."
    )

    # Expand date range to cover 5 years
    five_year_start = f"{year_end.year - 4}-01-01"
    query.update({"time": (five_year_start, dc_query["time"][1])})

    if dc is None:
        dc = Datacube(app="LoadWofsAnn")

    dask_chunks = {"x": 3200, "y": 3200}
    ds = dc.load(**query, like=tile_geobox, dask_chunks=dask_chunks)
    ds = ds.rename(get_measurements_name_dict("wofs_ann"))

    # For each band mask no data values to np.nan
    for band in ds.data_vars:
        nodata = ds[band].attrs["nodata"]
        ds[band] = ds[band].where(ds[band] != nodata)

    # Calculate the ratio of clear wet observations to total clear
    # observations for each pixel over the 5 year period.
    clear_count_sum = ds["wofs_ann_clearcount"].sum(dim="time")
    wet_count_sum = ds["wofs_ann_wetcount"].sum(dim="time")

    frequency = da.divide(
        wet_count_sum,
        clear_count_sum,
        out=da.full_like(wet_count_sum, np.nan, dtype=np.float32),
        where=(clear_count_sum > 0),
    )
    frequency.name = "frequency"

    # Generate a water mask  by thresholding the frequency.
    water_mask = frequency > 0.45
    water_mask = water_mask.where(~np.isnan(frequency), other=np.nan).astype(
        np.float32
    )
    water_mask.name = "water_mask"

    # Add a time coordinate for compatibility
    time_values = np.array(
        [year_to_dc_datetime(year_start.year)],
        dtype="datetime64[ns]",
    )
    water_mask = water_mask.expand_dims(time=time_values)

    if compute:
        log.info("Computing wofs_ann dataset ...")
        water_mask = water_mask.compute()
        log.info("Done.")
    else:
        water_mask = water_mask.persist()
    del ds, clear_count_sum, wet_count_sum, frequency

    # Add attributes
    water_mask.attrs = dict(
        nodata=np.nan,
        scales=1,
        offsets=0,
    )

    return water_mask


def build_wq_agm_dataset(
    dc_queries: dict[str, dict[str, Any]],
    tile_geobox: GeoBox,
    dc: Datacube = None,
    compute: bool = False,
) -> xr.Dataset:
    """Build a combined annual dataset from loading data
    for each composite products instrument using the datacube queries
    provided.

    Parameters
    ----------
    dc_queries : dict[str, dict[str, Any]]
        Datacube query to use to load data for each instrument.

    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including it's crs.

    dc: Datacube
        Datacube connection to use when loading data, by default None.

    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.

    Returns
    -------
    xr.Dataset
        A single dataset containing all the data found for each
        instrument in the datacube.
    """
    if dc is None:
        dc = Datacube(app="BuildAnnualDataset")

    loaded_datasets: dict[str, xr.DataArray | xr.Dataset] = {}

    if "wofs_ann" not in dc_queries:
        raise ValueError(
            "Datacube query for the instrument `wofs_ann` must be provided."
        )
    else:
        loaded_datasets["wofs_ann"] = load_wofs_ann_data(
            dc_query=dc_queries["wofs_ann"],
            tile_geobox=tile_geobox,
            compute=False,
            dc=dc,
        )
        loaded_datasets["water_mask"] = load_water_mask(
            dc_query=dc_queries["wofs_ann"],
            tile_geobox=tile_geobox,
            compute=False,
            dc=dc,
        )

    if "oli_agm" in dc_queries:
        loaded_datasets["oli_agm"] = load_oli_agm_data(
            dc_query=dc_queries["oli_agm"],
            tile_geobox=tile_geobox,
            compute=False,
            dc=dc,
        )
    if "msi_agm" in dc_queries:
        loaded_datasets["msi_agm"] = load_msi_agm_data(
            dc_query=dc_queries["msi_agm"],
            tile_geobox=tile_geobox,
            compute=False,
            dc=dc,
        )
    if "tm_agm" in dc_queries:
        loaded_datasets["tm_agm"] = load_tm_agm_data(
            dc_query=dc_queries["tm_agm"],
            tile_geobox=tile_geobox,
            compute=False,
            dc=dc,
        )
    if "tirs" in dc_queries and "wofs_ann" in dc_queries:
        loaded_datasets["tirs_ann"] = load_tirs_annual_composite_data(
            tirs_dc_query=dc_queries["tirs"],
            wofs_ann_dc_query=dc_queries["wofs_ann"],
            tile_geobox=tile_geobox,
            compute=False,
            dc=dc,
        )

    # Merge while still lazy
    log.info("Merging instrument datasets (lazy) ...")
    combined = xr.merge(list(loaded_datasets.values()), compat="no_conflicts")
    combined = combined.drop_vars("quantile", errors="ignore")

    if compute:
        # Compute only once at the very end
        log.info("Computing final merged dataset ...")
        combined = combined.compute()
        log.info("Done.")

    return combined
