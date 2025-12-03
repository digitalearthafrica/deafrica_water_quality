import logging

import numpy as np
import xarray as xr
from xarray import Dataset

log = logging.getLogger(__name__)

# Parameters for dark pixel correction
DP_ADJUST = {
    "msi": {
        "ref_var": "msi12",
        "var_list": [
            "msi04",
            "msi03",
            "msi02",
            "msi01",
            "msi05",
            "msi06",
            "msi07",
        ],
    },
    "oli": {
        "ref_var": "oli07",
        "var_list": ["oli04", "oli03", "oli02", "oli01"],
    },
    "tm": {"ref_var": "tm07", "var_list": ["tm04", "tm03", "tm02", "tm01"]},
}


def R_correction(
    ds: xr.Dataset,
    water_mask: xr.DataArray,
    instrument: str,
    drop: bool = False,
) -> xr.Dataset:
    """
    Applies atmospheric dark pixel Rayleigh correction (R) correction
    to specified remote sensing bands within an xarray Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        An xarray Dataset containing remote sensing bands
        (e.g., 'msi04_agm', 'oli07_agm') and 'water_mask' for
        water masking.
    water_mask : xr.DataArray
        Water mask to apply for masking non-water pixels, where 1
        indicates water.
    instrument : str
        The instrument name indicating which set of bands to correct.
    drop : bool, optional
        If True, the original uncorrected bands will be dropped from the
        Dataset after correction, by default False.
    Returns
    -------
    xarray.Dataset
        The input `ds` Dataset with new atmospherically corrected bands
        added. Each corrected band will have its original name appended
        with 'r' (e.g., 'msi04_agm' becomes 'msi04_agmr').
    """
    if instrument.endswith("_agm"):
        inst = instrument.split("_")[0]
        ref_var = DP_ADJUST[inst]["ref_var"] + "_agm"
        target_vars = [var + "_agm" for var in DP_ADJUST[inst]["var_list"]]
    else:
        inst = instrument
        ref_var = DP_ADJUST[inst]["ref_var"]
        target_vars = DP_ADJUST[inst]["var_list"]

    for target_var in target_vars:
        if target_var not in list(ds.data_vars):
            error = ValueError(
                f"Variable {target_var} expected but not found in the "
                f"dataset - (non-fatal error) for instrument {instrument}",
            )
            log.warning(error)
            continue
        else:
            new_var = f"{target_var}r"
            ds[new_var] = xr.where(
                water_mask == 1,
                xr.where(
                    ds[target_var] > ds[ref_var],
                    ds[target_var] - ds[ref_var],
                    xr.where(ds[target_var] > 0, ds[target_var], np.nan),
                ),
                ds[target_var],
            )
            if drop:
                ds = ds.drop_vars(target_var)
                ds = ds.rename({new_var: target_var})
    return ds


def apply_R_correction(
    instrument_data: dict[str, xr.Dataset],
    water_mask: xr.DataArray,
    compute: bool = False,
    drop: bool = True,
) -> dict[str, xr.Dataset]:
    """
    Wrapper function to apply Rayleigh correction across
    multiple instruments

    Parameters
    ----------
    instrument_data : dict[str, xr.Dataset]
        A dictionary mapping instruments to the xr.Dataset of the
        loaded datacube datasets available for that instrument.
    water_mask : xr.DataArray
        Water mask to apply for masking non-water pixels, where 1
        indicates water.
    drop : bool, optional
        If True, the original uncorrected bands will be dropped from the
        Dataset after correction, by default False.
    compute : bool
        Whether to compute the dask arrays immediately, by default False.
        Set to False to keep datasets lazy for memory efficiency.

    Returns
    -------
    instrument_data : dict[str, xr.Dataset]
        A dictionary mapping instruments to the xr.Dataset of the loaded datacube datasets available for that
        instrument, with new atmospherically corrected bands
        added. Each corrected band will have its original name appended
        with 'r' (e.g., 'msi04_agm' becomes 'msi04_agmr').
    """
    # Get the instruments in the dataset that the pixel correction
    # will be applied.

    for instrument, ds in instrument_data.items():
        if instrument not in [
            "msi",
            "msi_agm",
            "oli",
            "oli_agm",
            "tm",
            "tm_agm",
        ]:
            error = (
                f"R correction not implemented for instrument: {instrument}"
            )
            log.warning(error)
            continue
        else:
            log.info(f"Applying R correction for instrument: {instrument}")
            instrument_data[instrument] = R_correction(
                ds=ds, water_mask=water_mask, instrument=instrument, drop=drop
            )
        if compute:
            log.info(
                "\tComputing Rayleigh corrected data for "
                f"instrument {instrument} ..."
            )
            instrument_data[instrument] = instrument_data[instrument].compute()

        log.info(f"R correction complete for {instrument}.")
    return instrument_data
