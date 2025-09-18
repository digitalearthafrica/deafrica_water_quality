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


def R_correction(
    ds: Dataset,
    instruments_to_use: dict[str, dict[str, bool]],
    water_frequency_threshold: float = 0.9,
):
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
