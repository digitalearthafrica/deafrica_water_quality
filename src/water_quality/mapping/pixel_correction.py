import logging

from xarray import Dataset

log = logging.getLogger(__name__)

# Parameters for dark pixel correction
DP_ADJUST = {
    "msi_agm": {
        "ref_var": "msi12_agm",
        "var_list": [
            "msi04_agm",
            "msi03_agm",
            "msi02_agm",
            "msi05_agm",
            "msi06_agm",
            "msi07_agm",
        ],
    },
    "oli_agm": {
        "ref_var": "oli07_agm",
        "var_list": ["oli04_agm", "oli03_agm", "oli02_agm"],
    },
    "tm_agm": {
        "ref_var": "tm07_agm",
        "var_list": ["tm04_agm", "tm03_agm", "tm02_agm", "tm01_agm"],
    },
}

# def R_correction(
#     ds: Dataset,
#     instruments_to_use: dict[str, dict[str, bool]],
#     water_frequency_threshold: float = 0.9,
# ) -> Dataset:
#     """
#     Applies atmospheric dark pixel (R) correction to specified
#     remote sensing bands within an xarray Dataset.

#     Skips gracefully if expected variables are missing.
#     """

#     for sensor, config in instruments_to_use.items():
#         if not config.get("use", False):
#             log.info(f"Skipping dark pixel correction for {sensor}: not enabled")
#             continue

#         if sensor not in DP_ADJUST:
#             log.warning(f"{sensor} not in DP_ADJUST — skipping")
#             continue

#         log.info(f"Performing dark pixel correction for sensor {sensor} ...")

#         ref_var = DP_ADJUST[sensor]["ref_var"]
#         if ref_var not in ds.data_vars:
#             log.warning(
#                 f"Skipping {sensor}: reference variable {ref_var} not found in dataset"
#             )
#             continue

#         for target_var in DP_ADJUST[sensor]["var_list"]:
#             if target_var not in ds.data_vars:
#                 log.warning(
#                     f"Skipping {sensor}: target variable {target_var} not found in dataset"
#                 )
#                 continue

#             new_var = target_var + "r"
#             ds[new_var] = (
#                 (ds[target_var] - ds[ref_var])
#                 .where(ds[target_var] > ds[ref_var], 0)
#                 .where(ds[target_var] > 0)
#                 .where(
#                     ds.wofs_ann_freq > water_frequency_threshold,
#                     ds[target_var],
#                 )
#             )

#     return ds
def R_correction(
    ds: Dataset,
    instruments_to_use: dict[str, dict[str, bool]],
    water_frequency_threshold: float = 0.9,
) -> Dataset:
    """
    Applies atmospheric dark pixel (R) correction to specified
    remote sensing bands within an xarray Dataset.ary_

    This function iterates through defined sensors and, if enabled,
    performs a dark pixel subtraction.

    Parameters
    ----------
    ds : xarray.Dataset
        An xarray Dataset containing remote sensing bands
        (e.g., 'msi04_agm', 'oli07_agm') and 'wofs_ann_freq' for
        water masking.
    instruments_to_use : dict[str, dict[str, bool]]
        A dictionary of the instruments used to get the remote sensing
        bands.
    water_frequency_threshold : float, optional
        The water frequency threshold. Pixels with 'wofs_ann_freq'
        values below this threshold will have their original
        (uncorrected) band values retained, effectively excluding
        water bodies from the dark pixel correction. Defaults to 0.9.


    Returns
    -------
    xarray.Dataset
        The input `ds` Dataset with new atmospherically corrected bands
        added. Each corrected band will have its original name appended
        with 'r' (e.g., 'msi04_agm' becomes 'msi04_agmr').
    """
    for sensor in DP_ADJUST.keys():
        if sensor in instruments_to_use.keys():
            usage = instruments_to_use[sensor]["use"]
            if usage:
                log.info(
                    f"Performing dark pixel correction for sensor {sensor} ..."
                )
                ref_var = DP_ADJUST[sensor]["ref_var"]
                if ref_var not in ds.data_vars:
                    raise ValueError(
                        f"Variable {ref_var} expected  but not found in the "
                        f"dataset - correction FAILING for sensor {sensor}",
                    )
                else:
                    for target_var in DP_ADJUST[sensor]["var_list"]:
                        if target_var not in ds.data_vars:
                            raise ValueError(
                                f"Variable {target_var} expected  but not "
                                "found in the dataset; terminating the "
                                "R_correction"
                            )
                        else:
                            new_var = str(target_var + "r")
                            ds[new_var] = (
                                (ds[target_var] - ds[ref_var])
                                .where(ds[target_var] > ds[ref_var], 0)
                                .where(ds[target_var] > 0)
                                .where(
                                    ds.wofs_ann_freq
                                    > water_frequency_threshold,
                                    ds[target_var],
                                )
                            )

            else:
                log.info(
                    f"Dark pixel correction requested for sensor {sensor}, "
                    "but sensor is not used in analysis."
                )
        else:
            raise ValueError(
                f"Dark pixel correction requested for sensor {sensor}, but "
                "sensor is not listed in instruments to use."
            )
    return ds
