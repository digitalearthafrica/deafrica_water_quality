"""
This module provides functions to implement water detection using DE
Africa Water Observations from Space (WOfS) products.
"""

import logging

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)


def five_year_water_mask(
    annual_data: dict[str, xr.Dataset],
    compute: bool = True,
) -> xr.DataArray:
    """Process 5 years of data for the `wofs_ann` instrument to
    generate a water mask.

    Parameters
    ----------
    annual_data : dict[str, xr.Dataset]
        A dictionary mapping instruments to the xr.Dataset of the loaded
        annual (geomedian) datacube datasets available for that
        instrument.
    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the water mask derived from the
        wofs_ann data.
    """
    log.info("Generating 5 year water mask ...")

    inst = "wofs_ann"
    if inst not in list(annual_data.keys()):
        error = (
            f"No datasets found for instrument '{inst}'. ",
            "Cannot generate water mask. Returning empty DataArray.",
        )
        log.error(error)
        return xr.DataArray(data=[], dims=["time"], coords={"time": []})

    inst_ds = annual_data[inst]
    # Calculate the ratio of clear wet observations to total clear
    # observations for each pixel over the 5 year period.
    clearcount_sum = inst_ds["wofs_ann_clearcount"].sum(
        dim="time", skipna=True
    )
    wet_count_sum = inst_ds["wofs_ann_wetcount"].sum(dim="time", skipna=True)

    frequency = xr.where(
        clearcount_sum > 0,
        (wet_count_sum / clearcount_sum),
        np.nan,
    )
    # TODO: Do we need to preserve land pixels as 0, together
    # with nodata (np.nan)?
    water_mask_da = xr.where(frequency > 0.45, 1.0, np.nan).astype("float32")
    water_mask_da.name = "water_mask"
    water_mask_da.attrs = dict(
        nodata=np.nan,
        scales=1,
        offsets=0,
    )
    # Add a time coordinate for compatibility with other annual datasets.
    # Use the last year in the 5 year period
    # based on how the tasks were generated in `wq-generate-tasks`.
    water_mask_da = water_mask_da.expand_dims(
        time=[inst_ds.isel(time=-1).time.values]
    )

    if compute:
        log.info("\tComputing water mask ...")
        water_mask_da = water_mask_da.compute()
    else:
        log.info("\tPersisting water mask ...")
        water_mask_da = water_mask_da.persist()
    del inst_ds, clearcount_sum, wet_count_sum, frequency
    log.info("Processing complete for water mask.")
    return water_mask_da


def clear_water_mask(
    annual_data: dict[str, xr.Dataset],
    water_frequency_threshold: float,
    water_mask: xr.DataArray,
    agm_fai: xr.DataArray,
    compute: bool = True,
):
    """Calculate the clear water mask using the Geomedian FAI
    and the 5 year water mask.

    Parameters
    ----------
    annual_data : dict[str, xr.Dataset]
        A dictionary mapping instruments to the xr.Dataset of the loaded
        annual (geomedian) datacube datasets available for that
        instrument.
    water_frequency_threshold : float
        The frequency threshold above which a pixel is classified as water.
    water_mask : xr.DataArray
        Water mask to apply for masking non-water pixels, where 1
        indicates water.
    agm_fai : xr.DataArray
        An xarray DataArray containing the geomedian FAI.
    compute : bool
        Whether to compute the dask arrays immediately, by default True.
        Set to False to keep datasets lazy for memory efficiency.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the following data variables:
        - `clear_water_mask` (float): A mask showing where clear water is
            detected.
    """
    log.info("Calculating clear water mask ...")

    inst = "wofs_ann"
    if inst not in list(annual_data.keys()):
        error = (
            f"No datasets found for instrument '{inst}'. ",
            "Cannot generate clear water mask. Returning empty DataArray.",
        )
        log.error(error)
        return xr.DataArray(data=[], dims=["time"], coords={"time": []})

    inst_ds = annual_data[inst]

    # Get the last year of data since wofs_ann data covers 5 years
    ds = inst_ds.isel(time=-1).expand_dims(time=1)
    # Boolean instead of:
    # ds["wofs_ann_water"] = ds["wofs_ann_freq"].where(ds["wofs_ann_freq"] >
    # water_frequency_threshold, 0)
    wofs_ann_water = ds["wofs_ann_freq"] > water_frequency_threshold
    # TODO: Do we need to preserve land pixels as 0, together with
    # nodata (np.nan)?
    clear_water_mask = xr.where(
        np.isnan(agm_fai) & (water_mask == 1) & wofs_ann_water, 1.0, np.nan
    ).astype("float32")
    clear_water_mask.name = "clear_water"
    clear_water_mask.attrs = dict(
        nodata=np.nan,
        scales=1,
        offsets=0,
    )
    if compute:
        log.info("\tComputing clear water mask ...")
        clear_water_mask = clear_water_mask.compute()
    del inst_ds, ds, wofs_ann_water
    log.info("Processing complete for clear water mask.")
    return clear_water_mask


def water_analysis(
    wofs_ann_ds: xr.Dataset,
    water_frequency_threshold: float = 0.5,
):
    """Performs water detection analysis on DE Africa WOfS Annual
    Summary data.

    Parameters
    ----------
    wofs_ann_ds : Dataset
        An xarray Dataset containing the processed data for the
        instrument `wofs_ann` for a single year.
    water_frequency_threshold : float, optional
        The frequency threshold above which a pixel is classified as
        general water, by default 0.5

    Returns
    -------
    xarray.Dataset
        The input Dataset with the following new data variables added:
        - `wofs_ann_watermask` (float): A mask showing where general water is detected.

    """

    # Standard deviation of the annual frequency at each pixel
    # should really be dividing by n-1 but then I would need to
    # change SC
    wofs_ann_freq_sigma = (
        (wofs_ann_ds.wofs_ann_freq * (1 - wofs_ann_ds.wofs_ann_freq))
        / wofs_ann_ds.wofs_ann_clearcount
    ) ** 0.5
    # A variable called watermask is used in places.
    # I set the value of the mask as sigma or nan
    # Renamed this from watermask to wofs_ann_watermask to prevent
    # confusion with the 5 year summary watermask
    wofs_ann_ds["wofs_ann_watermask"] = wofs_ann_freq_sigma.where(
        wofs_ann_ds["wofs_ann_freq"] > water_frequency_threshold
    )

    return wofs_ann_ds
