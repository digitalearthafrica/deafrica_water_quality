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
    ds: Dataset,
    instruments_to_use: dict[str, dict[str, bool]],
    drop: bool = False,
) -> Dataset:
    """
    Applies atmospheric dark pixel Rayleigh correction (R) correction
    to specified remote sensing bands within an xarray Dataset.

    This function iterates through defined sensors and, if enabled,
    performs a dark pixel subtraction.

    Parameters
    ----------
    ds : xarray.Dataset
        An xarray Dataset containing remote sensing bands
        (e.g., 'msi04_agm', 'oli07_agm') and 'water_mask' for
        water masking.
    instruments_to_use : dict[str, dict[str, bool]]
        A dictionary of the instruments used to get the remote sensing
        bands.
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
    # Get the instruments in the dataset that the pixel correction
    # will be applied.

    for inst in instruments_to_use:
        usage = instruments_to_use[inst]["use"]
        if usage is True:
            # Get the pixel correction parameters for
            # the instrument.
            if "agm" in inst:
                inst_prefix = inst.split("_")[0]
                if inst_prefix not in DP_ADJUST.keys():
                    continue
                else:
                    ref_var = f"{DP_ADJUST[inst_prefix]['ref_var']}_agm"
                    target_vars = [
                        f"{i}_agm" for i in DP_ADJUST[inst_prefix]["var_list"]
                    ]
            else:
                if inst not in DP_ADJUST.keys():
                    continue
                else:
                    ref_var = DP_ADJUST[inst]["ref_var"]
                    target_vars = DP_ADJUST[inst]["var_list"]

            log.info(
                f"Performing dark pixel correction for instrument {inst} ..."
            )
            if ref_var not in ds.data_vars:
                raise ValueError(
                    f"Variable {ref_var} expected  but not found in the "
                    f"dataset - correction FAILING for instrument {inst}",
                )
            else:
                for target_var in target_vars:
                    if target_var not in ds.data_vars:
                        log.error(
                            ValueError(
                                f"Variable {target_var} expected  but not found in the "
                                f"dataset - (non-fatal error) for instrument {inst}",
                            )
                        )
                    else:
                        new_var = f"{target_var}r"
                        # Prevent overwriting.
                        if new_var in ds.data_vars:
                            ds = ds.drop_vars(new_var)

                        # Initialize new_var with zeros and the same
                        # dimensions as ref_var
                        ds[new_var] = ds[ref_var] * 0.0

                        # Calculate a modified value:
                        ds[new_var] = xr.where(
                            ds[target_var] > 0,
                            xr.where(
                                ds[target_var] > ds[ref_var],
                                ds[target_var] - ds[ref_var],
                                ds[target_var],
                            ),
                            np.nan,
                        )
                        ds[new_var] = ds[new_var].where(
                            ds["water_mask"] == 1, ds[target_var]
                        )

                        if drop:
                            # Rename the modified variable to replace the original
                            ds = ds.drop_vars(target_var)
                            ds = ds.rename({new_var: target_var})

    return ds
