import logging

import numpy as np
import xarray as xr
from datacube import Datacube
from datacube.model import Dataset
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject

from water_quality.dates import year_to_dc_datetime
from water_quality.mapping.instruments import INSTRUMENTS_MEASUREMENTS
from water_quality.tiling import reproject_tile_geobox

log = logging.getLogger(__name__)


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


def _mask_nodata(da: xr.DataArray) -> xr.DataArray:
    """Mask no data values in an xarray DataArray to np.nan."""
    nodata_val = da.attrs.get("nodata", None)
    if nodata_val is None:
        return da
    if np.issubdtype(da.dtype, np.integer):
        da = da.astype("float32")
    if isinstance(nodata_val, float) and np.isnan(nodata_val):
        return da.where(~np.isnan(da))
    else:
        return da.where(da != nodata_val, np.nan)


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


def load_wofs_ann_data(
    dss: dict[str, list[Dataset]],
    tile_geobox: GeoBox,
    compute: bool,
    dc: Datacube,
) -> xr.Dataset:
    """Load and process data for the `wofs_ann` instrument.

    Parameters
    ----------
    dss: dict[str, list[Dataset]]
        A dictionary mapping instruments to a list of datacube datasets
        available.
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
        instrument wofs_ann.

    """
    inst = "wofs_ann"
    log.info(f"Loading data for the instrument '{inst}' ...")
    if inst not in list(dss.keys()):
        error = (
            f"No datasets found for instrument '{inst}'. ",
            "Returning empty dataset.",
        )
        log.error(error)
        return xr.Dataset()

    datasets = dss[inst]
    # TODO: Set a global dask chunk size configuration
    # Expected tile size is 9600 x 9 600 at 10 m resolution
    dask_chunks = {"x": 4800, "y": 4800, "time": -1}
    measurements = get_measurements_name_dict(inst)
    # For int data nearest is preferred
    # bilinear for float data.
    resampling = "nearest"

    if dc is None:
        dc = Datacube(app=f"Load_{inst}")

    ds = dc.load(
        datasets=datasets,
        measurements=list(measurements.keys()),
        like=tile_geobox,
        resampling=resampling,
        dask_chunks=dask_chunks,
    )
    ds = ds.rename(measurements)
    ds = ds.map(_mask_nodata)
    if compute:
        log.info(f"Computing {inst} dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_oli_agm_data(
    dss: dict[str, list[Dataset]],
    tile_geobox: GeoBox,
    compute: bool,
    dc: Datacube,
) -> xr.Dataset:
    """Load and process data for the `oli_agm` instrument.

    Parameters
    ----------
    dss: dict[str, list[Dataset]]
        A dictionary mapping instruments to a list of datacube datasets available.
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
    inst = "oli_agm"
    log.info(f"Loading data for the instrument '{inst}' ...")
    if inst not in list(dss.keys()):
        error = (
            f"No datasets found for instrument '{inst}'. ",
            "Returning empty dataset.",
        )
        log.error(error)
        return xr.Dataset()

    datasets = dss[inst]
    # TODO: Set a global dask chunk size configuration
    # Expected tile size is 9600 x 9 600 at 10 m resolution
    dask_chunks = {"x": 4800, "y": 4800, "time": -1}
    measurements = get_measurements_name_dict(inst)
    # For int data nearest is preferred
    # bilinear for float data.
    resampling = "bilinear"

    if dc is None:
        dc = Datacube(app=f"Load_{inst}")

    ds = dc.load(
        datasets=datasets,
        measurements=list(measurements.keys()),
        like=tile_geobox,
        resampling=resampling,
        dask_chunks=dask_chunks,
    )
    ds = ds.rename(measurements)
    ds = ds.map(_mask_nodata)

    if compute:
        log.info(f"Computing {inst} dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_oli_data(
    datasets: list[Dataset],
    tile_geobox: GeoBox,
    compute: bool = True,
    dc: Datacube = None,
) -> xr.Dataset:
    """Load and process data for the `oli` instrument.

    Parameters
    ----------
    datasets : list[Dataset]
        List of datasets to load data for the instrument `oli`.
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

    if dc is None:
        dc = Datacube(app="LoadOli")

    dask_chunks = {"x": 4800, "y": 4800}
    ds = dc.load(
        datasets=datasets,
        like=tile_geobox,
        resampling="bilinear",
        dask_chunks=dask_chunks,
    )
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
    dss: dict[str, list[Dataset]],
    tile_geobox: GeoBox,
    compute: bool,
    dc: Datacube,
) -> xr.Dataset:
    """Load and process data for the `msi_agm` instrument.

    Parameters
    ----------
    dss: dict[str, list[Dataset]]
        A dictionary mapping instruments to a list of datacube datasets available.
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
    inst = "msi_agm"
    log.info(f"Loading data for the instrument '{inst}' ...")
    if inst not in list(dss.keys()):
        error = (
            f"No datasets found for instrument '{inst}'. ",
            "Returning empty dataset.",
        )
        log.error(error)
        return xr.Dataset()

    datasets = dss[inst]
    # TODO: Set a global dask chunk size configuration
    # Expected tile size is 9600 x 9 600 at 10 m resolution
    dask_chunks = {"x": 4800, "y": 4800, "time": -1}
    measurements = get_measurements_name_dict(inst)
    # For int data nearest is preferred
    # bilinear for float data.
    resampling = "bilinear"

    if dc is None:
        dc = Datacube(app=f"Load_{inst}")

    ds = dc.load(
        datasets=datasets,
        measurements=list(measurements.keys()),
        like=tile_geobox,
        resampling=resampling,
        dask_chunks=dask_chunks,
    )
    ds = ds.rename(measurements)
    ds = ds.map(_mask_nodata)

    if compute:
        log.info(f"Computing {inst} dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_msi_data(
    datasets: list[Dataset], tile_geobox: GeoBox, compute: bool, dc: Datacube
) -> xr.Dataset:
    """Load and process data for the `msi` instrument.

    Parameters
    ----------
    datasets : list[Dataset]
        List of datasets to load data for the instrument `msi`.
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

    if dc is None:
        dc = Datacube(app="LoadMsi")

    dask_chunks = {"x": 4800, "y": 4800}
    ds = dc.load(
        datasets=datasets,
        like=tile_geobox,
        resampling="bilinear",
        dask_chunks=dask_chunks,
    )
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
    dss: dict[str, list[Dataset]],
    tile_geobox: GeoBox,
    compute: bool,
    dc: Datacube,
) -> xr.Dataset:
    """Load and process data for the `tm_agm` instrument.

    Parameters
    ----------
    dss: dict[str, list[Dataset]]
        A dictionary mapping instruments to a list of datacube datasets available.
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
    inst = "tm_agm"

    log.info(f"Loading data for the instrument '{inst}' ...")
    if inst not in list(dss.keys()):
        error = (
            f"No datasets found for instrument '{inst}'. ",
            "Returning empty dataset.",
        )
        log.error(error)
        return xr.Dataset()

    datasets = dss[inst]
    # TODO: Set a global dask chunk size configuration
    # Expected tile size is 9600 x 9 600 at 10 m resolution
    dask_chunks = {"x": 4800, "y": 4800, "time": -1}
    measurements = get_measurements_name_dict(inst)
    # For int data nearest is preferred
    # bilinear for float data.
    resampling = "bilinear"

    if dc is None:
        dc = Datacube(app=f"Load_{inst}")

    ds = dc.load(
        datasets=datasets,
        measurements=list(measurements.keys()),
        like=tile_geobox,
        resampling=resampling,
        dask_chunks=dask_chunks,
    )
    ds = ds.rename(measurements)
    ds = ds.map(_mask_nodata)

    if compute:
        log.info(f"Computing {inst} dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_tm_data(
    datasets: list[Dataset], tile_geobox: GeoBox, compute: bool, dc: Datacube
) -> xr.Dataset:
    """Load and process data for the `tm` instrument.

    Parameters
    ----------
    datasets : list[Dataset]
        List of datasets to load data for the instrument `tm`.
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

    if dc is None:
        dc = Datacube(app="LoadTm")

    dask_chunks = {"x": 4800, "y": 4800}
    ds = dc.load(
        datasets=datasets,
        like=tile_geobox,
        resampling="bilinear",
        dask_chunks=dask_chunks,
    )
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
    dss: dict[str, list[Dataset]],
    tile_geobox: GeoBox,
    compute: bool,
    dc: Datacube,
) -> xr.Dataset:
    """Load and process data for the `tirs` instrument.

    Parameters
    ----------
    dss: dict[str, list[Dataset]]
        A dictionary mapping instruments to a list of datacube datasets available.
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

    inst = "tirs"
    log.info(f"Loading data for the instrument '{inst}' ...")
    if inst not in list(dss.keys()):
        error = (
            f"No datasets found for instrument '{inst}'. ",
            "Returning empty dataset.",
        )
        log.error(error)
        return xr.Dataset()

    datasets = dss[inst]
    # TODO: Set a global dask chunk size configuration
    # Expected tile size is 9600 x 9 600 at 10 m resolution
    dask_chunks = {"x": 4800, "y": 4800, "time": -1}
    measurements = get_measurements_name_dict(inst)
    # For int data nearest is preferred
    # bilinear for float data.
    resampling = "bilinear"

    if dc is None:
        dc = Datacube(app=f"Load_{inst}")

    ds = dc.load(
        datasets=datasets,
        measurements=list(measurements.keys()),
        like=tile_geobox,
        resampling=resampling,
        dask_chunks=dask_chunks,
    )
    ds = ds.rename(measurements)
    ds = ds.map(_mask_nodata)

    # Rescale data
    ds["tirs_st_qa"] = 0.01 * ds["tirs_st_qa"]
    ds["tirs_emis"] = 0.0001 * ds["tirs_emis"]
    ds["tirs_st"] = ds["tirs_st"] * 0.00341802 + 149.0

    # Convert surface temperature from Kelvin to centrigrade.
    ds["tirs_st"] = ds["tirs_st"] - 273.15

    if compute:
        log.info(f"Computing {inst} dataset ...")
        ds = ds.compute()
        log.info("Done.")

    return ds


def load_tirs_annual_composite_data(
    datasets: list[Dataset],
    tile_geobox: GeoBox,
    compute: bool,
    dc: Datacube,
) -> xr.Dataset:
    """Load and process data for the `tirs` instrument to produce an
    annual composite.

    Parameters
    ----------
    datasets : list[Dataset]
        List of datasets to load data for the instrument `tirs`.

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
        An xarray Dataset containing the surface temperature annual
        composite produced from data for the instrument `tirs`.
    """
    # Due to memory constraints tirs data must be loaded in its native
    # resolution of 30 m and later reprojected to the target tile geobox
    # if upsampling the data.
    if abs(tile_geobox.resolution.y) < 30:
        native_tirs_geobox = reproject_tile_geobox(
            tile_geobox=tile_geobox, resolution_m=30
        )
        ds_tirs = load_tirs_data(
            datasets=datasets,
            tile_geobox=native_tirs_geobox,
            compute=False,
            dc=dc,
        )
    else:
        native_tirs_geobox = None
        ds_tirs = load_tirs_data(
            datasets=datasets,
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
            resampling="bilinear",
        )

    if compute:
        log.info("Computing tirs annual composite dataset ...")
        annual_ds_tirs = annual_ds_tirs.compute()
        log.info("Done.")
    return annual_ds_tirs


def load_annual_data(
    dss: dict[str, list[Dataset]],
    tile_geobox: GeoBox,
    dc: Datacube = None,
    compute: bool = False,
) -> dict[str, xr.Dataset]:
    """
    Load yearly data for the instruments "wofs_ann", "oli_agm", "msi_agm", "tm_agm" and "tirs". Note, for the "wofs_ann" instrument,
    5 years of data is loaded.

    Parameters
    ----------
    dss : dict[str, list[Dataset]]
        A dictionary mapping instruments to a list of datacube datasets
        available.

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
    dict[str, xr.Dataset]
        A dictionary mapping each instrument to the xr.Dataset of the loaded datasets for that instrument.
    """
    if dc is None:
        dc = Datacube(app="LoadAnnualData")

    loaded_datasets: dict[str, xr.Dataset] = {}

    instruments = list(dss.keys())

    if "wofs_ann" in instruments:
        loaded_datasets["wofs_ann"] = load_wofs_ann_data(
            dss=dss,
            tile_geobox=tile_geobox,
            compute=compute,
            dc=dc,
        )
    if "oli_agm" in instruments:
        loaded_datasets["oli_agm"] = load_oli_agm_data(
            dss=dss,
            tile_geobox=tile_geobox,
            compute=compute,
            dc=dc,
        )
    if "msi_agm" in instruments:
        loaded_datasets["msi_agm"] = load_msi_agm_data(
            dss=dss,
            tile_geobox=tile_geobox,
            compute=compute,
            dc=dc,
        )
    if "tm_agm" in instruments:
        loaded_datasets["tm_agm"] = load_tm_agm_data(
            dss=dss,
            tile_geobox=tile_geobox,
            compute=compute,
            dc=dc,
        )
    if "tirs" in instruments:
        loaded_datasets["tirs"] = load_tirs_data(
            dss=dss,
            tile_geobox=tile_geobox,
            compute=compute,
            dc=dc,
        )

    for inst, ds in loaded_datasets.items():
        is_empty = not ds.data_vars
        if is_empty:
            del loaded_datasets[inst]

    return loaded_datasets
