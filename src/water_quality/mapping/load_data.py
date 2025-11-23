import logging
from itertools import chain
from typing import Any, Callable
from uuid import UUID

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

    ds_tirs = process_tirs_data(ds_tirs)

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

    # Restrict values to areas of water
    water_frequency_threshold = 0.5
    water_mask = (
        ds_wofs_ann["wofs_ann_freq"].sel(time=time_values)
        > water_frequency_threshold
    )

    for var in ["tirs_st_ann_med", "tirs_st_ann_min", "tirs_st_ann_max"]:
        annual_ds_tirs[var] = annual_ds_tirs[var].where(water_mask)

    return annual_ds_tirs


def process_oli_data(ds_oli: xr.Dataset) -> xr.Dataset:
    """
    Process data loaded for the instrument `oli` by:
    * Setting no data values for each band to nan
    * Rescaling the loaded surface reflectance data

    Parameters
    ----------
    ds_oli : xr.Dataset
        Data loaded for the instrument `oli`.

    Returns
    -------
    Dataset
        Data loaded for the instrument `oli` that has the zeros set to
        nans and has been rescaled.
    """

    for band in list(ds_oli.data_vars):
        if band != "oli_pq":
            # Mask no data
            # ? pq/pixel quality band no data value is 1
            ds_oli[band] = ds_oli[band].where(ds_oli[band] > 0)

            # Rescale and multiply by 10,000 to match range of data
            # for the msi instrument.
            ds_oli[band] = (2.75e-5 * ds_oli[band] - 0.2) * 10000

    return ds_oli


def process_msi_data(ds_msi: xr.Dataset) -> xr.Dataset:
    """
    Process data loaded for the instrument `msi`by:
    * Setting no data values for each band to nan
    * Rescaling the loaded surface reflectance data

    Parameters
    ----------
    ds_msi : xr.Dataset
        Data loaded for the instrument `msi`.

    Returns
    -------
    Dataset
        Data loaded for the instrument `msi` that has the zeros set to
        nans and the has been rescaled.
    """
    for band in list(ds_msi.data_vars):
        # Nodata value for all bands is 0
        ds_msi[band] = ds_msi[band].where(ds_msi[band] > 0)
        # Add rescaling when switching to s2_l2a_c1
    return ds_msi


def process_tm_data(ds_tm: xr.Dataset) -> xr.Dataset:
    """
    Process data loaded for the instrument `tm` by:
    * Setting no data values for each band to nan
    * Rescaling the loaded surface reflectance data

    Parameters
    ----------
    ds_tm : xr.Dataset
        Data loaded for the instrument `tm`.

    Returns
    -------
    Dataset
        Data loaded for the instrument `tm` that has the zeros set to
        nans and has been rescaled.
    """
    for band in list(ds_tm.data_vars):
        # Nodata value for surface reflectance bands is 0
        # ? pq/pixel quality band no data value is 1
        ds_tm[band] = ds_tm[band].where(ds_tm[band] > 0)

        if band != "tm_pq":
            # Rescale and multiply by 10,000 to match range of data
            # for the msi instrument.
            ds_tm[band] = (2.75e-5 * ds_tm[band] - 0.2) * 10000

    return ds_tm


def process_tirs_data(ds_tirs: xr.Dataset) -> xr.Dataset:
    """
    Process data loaded for the instrument `tirs` by:
    * Setting no data values for each band to nan
    * Rescaling the loaded surface temperature, surface temperature
        quality (uncertainty in kelvin), and emissivity fraction.
    * Converting the surface temperature from Kelvin to Centrigrade.

    Parameters
    ----------
    ds_tirs : xr.Dataset
        Data loaded for the instrument `tirs`.

    Returns
    -------
    Dataset
        Data loaded for the instrument `tirs` that has the zeros set to
        nans and has been rescaled.
    """
    # TODO: Does setting zeros to nans produce the same results
    # as setting no data values for each band to nan?
    # tirs_st_qa -9999, tirs_emis -9999, tirs_st 0
    ds_tirs["tirs_st_qa"] = ds_tirs["tirs_st_qa"].where(
        ds_tirs["tirs_st_qa"] != -9999
    )
    ds_tirs["tirs_emis"] = ds_tirs["tirs_emis"].where(
        ds_tirs["tirs_emis"] != -9999
    )
    ds_tirs["tirs_st"] = ds_tirs["tirs_st"].where(ds_tirs["tirs_st"] > 0)

    # Set zeros to nans
    # for band in list(ds_tirs.data_vars):
    #    ds_tirs[band] = ds_tirs[band].where(ds_tirs[band] > 0)

    # Rescale data
    ds_tirs["tirs_st_qa"] = 0.01 * ds_tirs["tirs_st_qa"]
    ds_tirs["tirs_emis"] = 0.0001 * ds_tirs["tirs_emis"]
    ds_tirs["tirs_st"] = ds_tirs["tirs_st"] * 0.00341802 + 149.0

    # Convert surface temperature from Kelvin to centrigrade.
    ds_tirs["tirs_st"] = ds_tirs["tirs_st"] - 273.15

    return ds_tirs


def _load_and_reproject_instrument_data(
    dc_queries: dict[str, dict[str, Any]],
    tile_geobox: GeoBox,
    instruments_to_filter: list[str],
    dc: Datacube,
    proccess_loaded_data_func: Callable[
        [dict[str, xr.Dataset]], dict[str, xr.Dataset]
    ],
) -> tuple[dict[str, xr.Dataset], dict[str, list[UUID]]]:
    """
    Helper function to load, compute, and reproject instrument data from
    the datacube.

    This function encapsulates the common logic for filtering queries,
    determining the appropriate loading resolution, loading data,
    triggering Dask computation, and reprojecting the datasets to the
    target geobox.

    Parameters
    ----------
    dc_queries : dict[str, Dict[str, Any]]
        Datacube query to use to load data for each instrument.
    tile_geobox : GeoBox
        Defines the location and resolution of a rectangular grid of
        data, including its crs.
    instruments_to_filter : set[str]
        A set of instrument names (e.g., {"oli", "msi"}) to filter
        the `dc_queries` by, ensuring only relevant instruments are loaded.
    dc : Datacube
        Datacube connection to use when loading data.

    Returns
    -------
    tuple[dict[str, xr.Dataset], dict[str, list[UUID]]]
        A tuple containing:
            - A dictionary mapping the instrument name to a **computed and
                reprojected** xarray Dataset containing data found for the
                instrument in the datacube.
            - A dictionary mapping the instrument name to a list of
            UUIDs for the datasets loaded for the instrument.
    """
    # Determine the default resolution from the tile GeoBox (e.g., 10m, 30m)
    default_res = int(abs(tile_geobox.resolution.x))

    # Filter the provided datacube queries based on the specified instruments
    queries = {
        k: v for k, v in dc_queries.items() if k in instruments_to_filter
    }

    log.info("Loading data using datacube queries...")

    input_datasets: dict[str, list[UUID]] = {}
    loaded_data: dict[str, xr.Dataset] = {}
    for instrument_name, dc_query in queries.items():
        # Find datasets.
        datasets = dc.find_datasets(
            product=dc_query["product"],
            time=dc_query["time"],
            # resolution and crs does not matter at this point
            # just need the spatial boundaries
            like=tile_geobox,
        )
        input_datasets[instrument_name] = [dataset.id for dataset in datasets]

        # Load datasets.

        # Load Landsat products and derivatives at their native 30m resolution
        # if the default output resolution is not already 30m.
        if "msi" not in instrument_name and default_res != 30:
            like = reproject_tile_geobox(
                tile_geobox=tile_geobox, resolution_m=30
            )
        else:
            like = tile_geobox

        xy_chunk_size = int(like.shape.x / 5)
        dask_chunks = {"x": xy_chunk_size, "y": xy_chunk_size}

        ds = dc.load(
            datasets=datasets,
            measurements=dc_query["measurements"],
            like=like,
            resampling=dc_query["resampling"],
            dask_chunks=dask_chunks,
            group_by=dc_query.get("group_by", None),
        )
        ds = ds.rename(get_measurements_name_dict(instrument_name))
        loaded_data[instrument_name] = ds

    loaded_data = proccess_loaded_data_func(loaded_data)

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

    return loaded_data, input_datasets


def load_single_day_instruments_data(
    dc_queries: dict[str, dict[str, Any]],
    tile_geobox: GeoBox,
    dc: Datacube = None,
) -> tuple[dict[str, xr.Dataset], dict[str, list[UUID]]]:
    """
    Load data for each single day instrument using the datacube queries
    provided.

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
    tuple[dict[str, xr.Dataset], dict[str, list[UUID]]]
        A tuple containing:
            - A dictionary mapping the instrument name to the a Dataset
                containing data found for the instrument in the datacube.
            - A dictionary mapping the instrument name to a list of
            UUIDs for the datasets loaded for the instrument.
    """
    if dc is None:
        dc = Datacube(app="LoadSingleDayInstruments")

    def _process_single_day_instrument_data(
        loaded_data: dict[str, xr.Dataset],
    ) -> dict[str, xr.Dataset]:
        """Processes single-day satellite instrument data for oli, msi, and
        tm instruments."""
        if "oli" in loaded_data.keys():
            loaded_data["oli"] = process_oli_data(ds_oli=loaded_data["oli"])

        if "msi" in loaded_data.keys():
            loaded_data["msi"] = process_msi_data(ds_msi=loaded_data["msi"])

        if "tm" in loaded_data.keys():
            loaded_data["tm"] = process_tm_data(ds_tm=loaded_data["tm"])

        return loaded_data

    loaded_data, input_datasets = _load_and_reproject_instrument_data(
        dc_queries=dc_queries,
        tile_geobox=tile_geobox,
        instruments_to_filter=list(SINGLE_DAY_INSTRUMENTS.keys()),
        dc=dc,
        proccess_loaded_data_func=_process_single_day_instrument_data,
    )
    return (loaded_data, input_datasets)


def load_composite_instruments_data(
    dc_queries: dict[str, dict[str, Any]],
    tile_geobox: GeoBox,
    dc: Datacube = None,
) -> tuple[dict[str, xr.Dataset], dict[str, list[UUID]]]:
    """
    Load data for each composite instrument using the datacube queries
    provided.

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
    tuple[dict[str, xr.Dataset], dict[str, list[UUID]]]
        A tuple containing:
            - A dictionary mapping the instrument name to the a Dataset
                containing data found for the instrument in the datacube.
            - A dictionary mapping the instrument name to a list of
            uuids for the datasets loaded for the instrument.
    """
    if dc is None:
        dc = Datacube(app="LoadCompositeInstruments")

    def _process_composite_instrument_data(
        loaded_data: dict[str, xr.Dataset],
    ) -> dict[str, xr.Dataset]:
        """Process composite sallelite instrument data."""
        if "tirs" in loaded_data.keys():
            log.info(
                "Processing surface temperature data to annual composite ..."
            )
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
        return loaded_data

    loaded_data, input_datasets = _load_and_reproject_instrument_data(
        dc_queries=dc_queries,
        tile_geobox=tile_geobox,
        instruments_to_filter=list(COMPOSITE_INSTRUMENTS.keys()),
        dc=dc,
        proccess_loaded_data_func=_process_composite_instrument_data,
    )

    return loaded_data, input_datasets


def build_wq_agm_dataset(
    dc_queries: dict[str, dict[str, Any]],
    tile_geobox: GeoBox,
    dc: Datacube = None,
) -> tuple[xr.Dataset, list[UUID]]:
    """Build a combined annual dataset from loading data
    for each composite products instrument using the datacube queries
    provided.

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
    tuple[xr.Dataset, list[UUID]]
        A tuple containing:
        - A single dataset containing all the data found for each
        instrument in the datacube.
        - A list of the UUIDs for all the datasets found for each
        instrument in the datacube.

    """
    if dc is None:
        dc = Datacube(app="Build_wq_agm_dataset")
    loaded_data, input_datasets = load_composite_instruments_data(
        dc_queries=dc_queries, tile_geobox=tile_geobox, dc=dc
    )
    combined = xr.merge(list(loaded_data.values()))
    combined = combined.drop_vars("quantile", errors="ignore")

    source_datasets_uuids = list(chain.from_iterable(input_datasets.values()))
    return combined, source_datasets_uuids
