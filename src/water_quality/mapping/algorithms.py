"""
This module provides functions to apply various water quality algorithms
to EO data from a set of instruments.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


# =============================================================================
# Water Quality Algorithms
# =============================================================================
def ChlA_Toming(
    dataset: xr.Dataset, band5: str, band4: str, band6: str
) -> xr.DataArray:
    # ---- Function to calculate ndci from input nir and red bands ----
    return dataset[band5] - 0.5 * (dataset[band4] / dataset[band6])


def ChlA_3BDA(
    dataset: xr.Dataset, blue_band: str, green_band: str, red_band: str
) -> xr.DataArray:
    # as described in Byrne et al 2024:  LAQUA: a LAndsat water QUality
    # retrieval tool for east African lakes

    # The reflectances should be less than one for this function
    # to make sense
    scale = 1 / 10000.0
    return (dataset[blue_band] * scale) - (
        dataset[red_band] * scale * dataset[green_band] * scale
    )


def ChlA_Tebbs(
    dataset: xr.Dataset, NIR_band: str, Red_band: str
) -> xr.DataArray:
    # ---- the paramters are optional but for completeness ... ----
    a0 = -135
    a1 = 451
    return a0 + a1 * (dataset[NIR_band] / dataset[Red_band])


def NDCI_NIR_R(
    dataset: xr.Dataset, NIR_band: str, red_band: str
) -> xr.DataArray:
    """Normalized Difference Chlorophyll Index (NDCI)."""
    return (dataset[NIR_band] - dataset[red_band]) / (
        dataset[NIR_band] + dataset[red_band]
    )


def ChlA_MERIS2B(
    dataset: xr.Dataset, band_708: str, band_665: str
) -> xr.DataArray:
    """MERIS two-band chlorophyll-a estimation."""
    X = dataset[band_708] / dataset[band_665]
    return (25.28 * (X**2)) + 14.85 * X - 15.18


def ChlA_MODIS2B(
    dataset: xr.Dataset, band_748: str, band_667: str
) -> xr.DataArray:
    """MODIS two-band chlorophyll-a estimation."""
    X = dataset[band_748] / dataset[band_667]
    return (190.34 * X) - 32.45


def NDSSI_RG(
    dataset: xr.Dataset, red_band: str, green_band: str
) -> xr.DataArray:
    """Normalized Difference Suspended Sediment Index (Red-Green)."""
    return (dataset[red_band] - dataset[green_band]) / (
        dataset[red_band] + dataset[green_band]
    )


def NDSSI_BNIR(
    dataset: xr.Dataset, blue_band: str, NIR_band: str
) -> xr.DataArray:
    """Normalized Difference Suspended Sediment Index (Blue-NIR)."""
    return (dataset[blue_band] - dataset[NIR_band]) / (
        dataset[blue_band] + dataset[NIR_band]
    )


def TI_yu(
    dataset: xr.Dataset,
    NIR: str,
    Red: str,
    Green: str,
    scalefactor: float = 0.01,
) -> xr.DataArray:
    """Turbidity Index of Yu et al. 2019.
    An empirical algorithm to seamlessly retrieve the concentration of
    suspended particulate matter from water color across ocean to turbid
    river mouths. Remote Sens. Environ. 235, 111491 (2019).
    Used in screening turbid waters for mapping floating algal blooms
    Initially developed with TM
    -TI = ((Red − green) − (NIR − Rgreen)) ^ 0.5
    """
    return scalefactor * (
        ((dataset[Red] - dataset[Green]) - (dataset[NIR] - dataset[Green]))
        ** 0.5
    )


def TSM_LYM_ETM(
    dataset: xr.Dataset,
    green_band: str,
    red_band: str,
    scale_factor: float = 0.0001,
) -> xr.DataArray:
    """
    Lymburner Total Suspended Matter (TSM)
    Paper: [Lymburner et al. 2016](https://www.sciencedirect.com/science/article/abs/pii/S0034425716301560)
    Units of mg/L concentration. Variants for ETM and OLT, slight
    difference in parameters.
    These models, developed by leo lymburner and arnold dekker, are simple,
    stable, and produce credible results over a range of observations.

    """
    return 3983 * (
        ((dataset[green_band] + dataset[red_band]) * scale_factor / 2)
        ** 1.6246
    )


def TSM_LYM_OLI(
    dataset: xr.Dataset,
    green_band: str,
    red_band: str,
    scale_factor: float = 0.0001,
) -> xr.DataArray:
    return 3957 * (
        ((dataset[green_band] + dataset[red_band]) * scale_factor / 2)
        ** 1.6436
    )


def SPM_QIU(
    dataset: xr.Dataset, green_band: str, red_band: str
) -> xr.DataArray:
    X = dataset[red_band] / dataset[green_band]
    return 10.0 ** (2.26 * (X**3) - 5.42 * (X**2) + 5.58 * X - 0.72)


def TSS_QUANG8(dataset: xr.Dataset, red_band: str) -> xr.DataArray:
    """Quang et al. 2017 TSS estimation."""
    return 380.32 * dataset[red_band] * 0.0001 - 1.7826


def TSS_Zhang21(
    dataset: xr.Dataset,
    G: str,
    R: str,
    scale_factor: float = 0.0001,
    with_model: bool = True,
) -> xr.DataArray:
    """Zhang et al. 2023 TSS estimation (stable version)."""
    Green = dataset[G].where(~(dataset[G] > 0), 1) * scale_factor
    Red = dataset[R] * scale_factor
    if with_model:
        Green = Green / (2 * np.pi)
        Red = Red / (2 * np.pi)
        return 0.71 * np.e ** (21.31 * (Green + Red) * (Red / Green))
    else:
        return (Green + Red) * (Red / Green)


def TSS_GreenRed(
    dataset: xr.Dataset,
    G: str,
    R: str,
    scale_factor: float = 0.0001,
):
    return TSS_Zhang21(
        dataset,
        G=G,
        R=R,
        scale_factor=scale_factor,
        with_model=False,
    )


def TSS_Zhang23(
    dataset: xr.Dataset,
    B: str,
    G: str,
    R: str,
    scale_factor: float = 0.0001,
    with_model=True,
):
    """Model of Zhang 2023"""
    Blue = dataset[B].where(~(dataset[B] > 0), 1) * scale_factor
    Red = dataset[R] * scale_factor
    Green = dataset[G] * scale_factor
    if with_model:
        Green = Green / (2 * np.pi)
        Red = Red / (2 * np.pi)
        Blue = Blue / (2 * np.pi)

        return 1.20 * np.e ** (14.44 * (Green + Red) * (Red / Blue))
    else:
        return (Green + Red) * (Red / Blue)


def TSS_GreenRedBlue(
    dataset: xr.Dataset,
    G: str,
    R: str,
    B: str,
    scale_factor: float = 0.0001,
):
    """Zhang23 model without the exponential model fit"""
    return TSS_Zhang23(
        dataset=dataset,
        G=G,
        R=R,
        B=B,
        scale_factor=scale_factor,
        with_model=False,
    )


# =============================================================================
# Normalization Parameters
# =============================================================================
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


# =============================================================================
# Algorithm Dictionaries
# =============================================================================
def set_wq_algorithms(suffix=""):
    s = suffix
    ndci_nir_r = {
        "msi" + s: {
            "54": {
                "func": NDCI_NIR_R,
                "wq_varname": "ndci_msi54",
                "args": {"NIR_band": "msi05" + s, "red_band": "msi04" + s},
            },
            "64": {
                "func": NDCI_NIR_R,
                "wq_varname": "ndci_msi64",
                "args": {"NIR_band": "msi06" + s, "red_band": "msi04" + s},
            },
            "74": {
                "func": NDCI_NIR_R,
                "wq_varname": "ndci_msi74",
                "args": {"NIR_band": "msi07" + s, "red_band": "msi04" + s},
            },
        },
        "tm" + s: {
            "func": NDCI_NIR_R,
            "wq_varname": "ndci_tm43",
            "args": {"NIR_band": "tm04" + s, "red_band": "tm03" + s},
        },
        "oli" + s: {
            "func": NDCI_NIR_R,
            "wq_varname": "ndci_oli54",
            "args": {"NIR_band": "oli05" + s, "red_band": "oli04" + s},
        },
    }

    chla_toming = {  # looks more like a tss indicator!
        "msi" + s: {
            "func": ChlA_Toming,
            "wq_varname": "chla_toming_msi",
            "args": {
                "band5": "msi05" + s,
                "band4": "msi04" + s,
                "band6": "msi06" + s,
            },
        }
    }
    chla_3bda = {
        "msi" + s: {
            "func": ChlA_3BDA,
            "wq_varname": "chla_3bda_msi",
            "args": {
                "blue_band": "msi02" + s,
                "red_band": "msi04" + s,
                "green_band": "msi03" + s,
            },
        },
        "oli" + s: {
            "func": ChlA_3BDA,
            "wq_varname": "chla_3bda_oli",
            "args": {
                "blue_band": "oli02" + s,
                "red_band": "oli04" + s,
                "green_band": "oli03" + s,
            },
        },
        "tm" + s: {
            "func": ChlA_3BDA,
            "wq_varname": "chla_3bda_tm",
            "args": {
                "blue_band": "tm01" + s,
                "red_band": "tm03" + s,
                "green_band": "tm02" + s,
            },
        },
    }
    chla_tebbs = {  # this is just a ratio of two bands
        "msi" + s: {
            "func": ChlA_Tebbs,
            "wq_varname": "chla_tebbs_msi",
            "args": {"NIR_band": "msi8a" + s, "Red_band": "msi04" + s},
        },
        "oli" + s: {
            "func": ChlA_Tebbs,
            "wq_varname": "chla_tebbs_oli",
            "args": {"NIR_band": "oli05" + s, "Red_band": "oli04" + s},
        },
        "tm" + s: {
            "func": ChlA_Tebbs,
            "wq_varname": "chla_tebbs_tm",
            "args": {"NIR_band": "tm04" + s, "Red_band": "tm03" + s},
        },
    }
    chla_meris2b = {
        "msi" + s: {
            "func": ChlA_MERIS2B,
            "wq_varname": "chla_meris2b_msi",
            "args": {"band_708": "msi05" + s, "band_665": "msi04" + s},
        }
    }

    chla_modis2b = {
        "msi" + s: {
            "func": ChlA_MODIS2B,
            "wq_varname": "chla_modis2b_msi",
            "args": {"band_748": "msi06" + s, "band_667": "msi04" + s},
        },
        "tm" + s: {
            "func": ChlA_MODIS2B,
            "wq_varname": "chla_modis2b_tm",
            "args": {"band_748": "tm04" + s, "band_667": "tm03" + s},
        },
    }

    ndssi_rg = {
        "msi" + s: {
            "func": NDSSI_RG,
            "wq_varname": "ndssi_rg_msi",
            "args": {"red_band": "msi04" + s, "green_band": "msi03" + s},
        },
        "oli" + s: {
            "func": NDSSI_RG,
            "wq_varname": "ndssi_rg_oli",
            "args": {"red_band": "oli04" + s, "green_band": "oli03" + s},
        },
        "tm" + s: {
            "func": NDSSI_RG,
            "wq_varname": "ndssi_rg_tm",
            "args": {"red_band": "tm03" + s, "green_band": "tm02" + s},
        },
    }

    ndssi_bnir = {
        "oli" + s: {
            "func": NDSSI_BNIR,
            "wq_varname": "ndssi_bnir_oli",
            "args": {"NIR_band": "oli06" + s, "blue_band": "oli02" + s},
        },
    }

    ti_yu = {
        "oli" + s: {
            "func": TI_yu,
            "wq_varname": "ti_yu_oli",
            "args": {
                "NIR": "oli06" + s,
                "Red": "oli04" + s,
                "Green": "oli03" + s,
            },
        },
        "tm" + s: {
            "func": TI_yu,
            "wq_varname": "ti_yu_tm",
            "args": {
                "NIR": "tm04" + s,
                "Red": "tm03" + s,
                "Green": "tm02" + s,
            },
        },
    }

    tsm_lym = {
        "oli" + s: {
            "func": TSM_LYM_OLI,
            "wq_varname": "tsm_lym_oli",
            "args": {"red_band": "oli04" + s, "green_band": "oli03" + s},
        },
        "msi" + s: {
            "func": TSM_LYM_OLI,
            "wq_varname": "tsm_lym_msi",
            "args": {"red_band": "msi04" + s, "green_band": "msi03" + s},
        },
        "tm" + s: {
            "func": TSM_LYM_ETM,
            "wq_varname": "tsm_lym_tm",
            "args": {"red_band": "tm03" + s, "green_band": "tm02" + s},
        },
    }

    spm_qiu = {
        "tm" + s: {
            "func": SPM_QIU,
            "wq_varname": "spm_qiu_tm",
            "args": {"red_band": "tm03" + s, "green_band": "tm02" + s},
        },
        "msi" + s: {
            "func": SPM_QIU,
            "wq_varname": "spm_qiu_msi",
            "args": {"red_band": "msi04" + s, "green_band": "msi03" + s},
        },
    }

    tss_zhang23 = {
        "msi" + s: {
            "func": TSS_Zhang23,
            "wq_varname": "tss_zhang23_msi",
            "args": {"G": "msi03" + s, "R": "msi04" + s, "B": "msi02" + s},
        },
        "oli" + s: {
            "func": TSS_Zhang23,
            "wq_varname": "tss_zhang23_oli",
            "args": {"G": "oli03" + s, "R": "oli04" + s, "B": "oli02" + s},
        },
        "tm" + s: {
            "func": TSS_Zhang23,
            "wq_varname": "tss_zhang23_tm",
            "args": {"G": "tm02" + s, "R": "tm03" + s, "B": "tm01" + s},
        },
    }
    tss_GRB = {
        "msi" + s: {
            "func": TSS_GreenRedBlue,
            "wq_varname": "tss_grb_msi",
            "args": {"G": "msi03" + s, "R": "msi04" + s, "B": "msi02" + s},
        },
        "oli" + s: {
            "func": TSS_GreenRedBlue,
            "wq_varname": "tss_grb_oli",
            "args": {"G": "oli03" + s, "R": "oli04" + s, "B": "oli02" + s},
        },
        "tm" + s: {
            "func": TSS_GreenRedBlue,
            "wq_varname": "tss_grb_tm",
            "args": {"G": "tm02" + s, "R": "tm03" + s, "B": "tm01" + s},
        },
    }
    tss_zhang21 = {
        "msi" + s: {
            "func": TSS_Zhang21,
            "wq_varname": "tss_zhang21_msi",
            "args": {"G": "msi03" + s, "R": "msi04" + s},
        },
        "oli" + s: {
            "func": TSS_Zhang21,
            "wq_varname": "tss_zhang21_oli",
            "args": {"G": "oli03" + s, "R": "oli04" + s},
        },
        "tm" + s: {
            "func": TSS_Zhang21,
            "wq_varname": "tss_zhang21_tm",
            "args": {"G": "tm02" + s, "R": "tm03" + s},
        },
    }
    tss_GR = {
        "msi" + s: {
            "func": TSS_GreenRed,
            "wq_varname": "tss_gr_msi",
            "args": {"G": "msi03" + s, "R": "msi04" + s},
        },
        "oli" + s: {
            "func": TSS_GreenRed,
            "wq_varname": "tss_gr_oli",
            "args": {"G": "oli03" + s, "R": "oli04" + s},
        },
        "tm" + s: {
            "func": TSS_GreenRed,
            "wq_varname": "tss_gr_tm",
            "args": {"G": "tm02" + s, "R": "tm03" + s},
        },
    }

    # ---- algorithms are grouped into two over-arching dictionaries ----
    algorithms_chla = {
        "ndci_nir_r": ndci_nir_r,
        "chla_toming": chla_toming,
        "chla_3bda": chla_3bda,
        "chla_tebbs": chla_tebbs,
        "chla_meris2b": chla_meris2b,
        "chla_modis2b": chla_modis2b,
    }
    algorithms_tsm = {
        "ndssi_rg": ndssi_rg,
        "ndssi_bnir": ndssi_bnir,
        "ti_yu": ti_yu,
        "tsm_lym": tsm_lym,
        # "tss_zhang23"  : tss_zhang23 ,
        # "tss_zhang21"  : tss_zhang21 ,
        "tss_grb": tss_GRB,
        "tss_gr": tss_GR,
        "spm_qiu": spm_qiu,
    }
    return (algorithms_chla, algorithms_tsm)


# =============================================================================
# Functions to Run Algorithms on Datasets
# =============================================================================
def run_wq_algorithms(
    instrument_data: dict[str, xr.Dataset],
    algorithms_group: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> xr.Dataset:
    """Run a group of water quality algorithms on a dataset."""
    wq_vars_ds = xr.Dataset()
    for algorithm_name, algorithm_app in algorithms_group.items():
        for instrument_name, inst_app in algorithm_app.items():
            if instrument_name in list(instrument_data.keys()):
                # Check if there are multiple implementations of the
                # algorithm for an instrument
                inst_ds = instrument_data[instrument_name]
                # Single implementation
                if "func" in list(inst_app.keys()):
                    inst_alg_apps = [inst_app]
                else:
                    # Multiple implementations
                    inst_alg_apps = list(inst_app.values())

                for alg_entry in inst_alg_apps:
                    func = alg_entry["func"]
                    args = alg_entry["args"]
                    wq_varname = alg_entry["wq_varname"]
                    wq_vars_ds[wq_varname] = func(inst_ds, **args)
    return wq_vars_ds


def compute_trophic_state_index(chla_da: xr.DataArray) -> xr.DataArray:
    """
    Compute the Trophic State Index from the Chlorophyll-a (µg/l) values.
    The Chlorophyll-a (µg/l) values are classified into Trophic State
    Index values based on the table of Trophic State Index and related
    chlorophyll concentration classes according to Carlson (1977)).

    Parameters
    ----------
    chla_da : xr.DataArray
        DataArray containing the Chlorophyll-a (µg/l) values to derive
        the Trophic State Index from.

    Returns
    -------
    xr.DataArray
        DataArray containing the Trophic State Index values.
    """
    chla_bins = np.array(
        [0.04, 0.12, 0.34, 0.94, 2.6, 6.4, 20, 56, 154, 427, 1183]
    )
    tsi_labels = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    slope, intercept = np.polyfit(np.log10(chla_bins), tsi_labels, 1)
    min_chla_val = 0.01

    # The 0.53 ensures that the lowest values match
    tsi = (
        np.log10(chla_da.where(chla_da >= min_chla_val, min_chla_val)) * slope
        + intercept
        + 0.53
    )
    tsi = tsi.where(tsi >= 0, 0).where(tsi <= 100, 100)
    tsi = tsi.where(~np.isnan(chla_da), np.nan)

    return tsi


def WQ_vars(
    annual_data: dict[str, xr.Dataset],
    water_mask: xr.DataArray,
    compute: bool,
    stack_wq_vars: bool = True,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """
    Compute Chlorophyll-A (ChlA) and Total Suspended
    Matter (TSM) water quality variables.

    Parameters
    ----------
    annual_data : dict[str, xr.Dataset]
        A dictionary mapping instruments to the xr.Dataset of the loaded
        annual (geomedian) datacube datasets available for that
        instrument.
    water_mask : xr.DataArray
        Water mask to apply for masking non-water pixels, where 1
        indicates water.
    stack_wq_vars : bool, optional
        If True, normalize the water quality variables then then stack them
        into the variables "tsm" and "chla" in the output dataset.
        Finally, compute the Trophic State Index from the Chlorophyll-a
        (µg/l) values and add it to the output dataset as the variable "tsi".
        If False, return all computed water quality variables as separate
        variables in the output dataset and a DataFrame listing the
        variables that should be stacked to "tsm" and "chla", by default True.
    compute : bool, optional
        Whether to compute the dask arrays immediately after stacking,
        by default False. Set to False to keep datasets lazy for memory
        efficiency.

    Returns
    -------
    tuple[xr.Dataset, pd.DataFrame]:
            If `stack_wq_vars` is False, returns a tuple containing:
            - xr.Dataset: Dataset containing all computed water quality
              variables as separate variables.
            - pd.DataFrame: DataFrame listing the water quality variables
              that should be stacked to "tsm" and "chla".
            If `stack_wq_vars` is True, returns:
            - xr.Dataset: a Dataset containing the stacked water quality
              variables "tsm" and "chla", and the computed Trophic State
              Index "tsi".
            - pd.DataFrame: DataFrame listing the water quality variables
              that were stacked to "tsm" and "chla".
    """
    suffix = "_agm"

    ALGORITHMS_CHLA, ALGORITHMS_TSM = set_wq_algorithms(suffix)
    log.info("Running TSM water quality algorithms ...")
    tsm_ds = run_wq_algorithms(
        instrument_data=annual_data, algorithms_group=ALGORITHMS_TSM
    )
    log.info("Running Chla water quality algorithms ...")
    chla_ds = run_wq_algorithms(
        instrument_data=annual_data, algorithms_group=ALGORITHMS_CHLA
    )

    tsm_df = pd.DataFrame({"tsm_measures": list(tsm_ds.data_vars)})
    chla_df = pd.DataFrame({"chla_measures": list(chla_ds.data_vars)})
    all_wq_vars_df = pd.concat([tsm_df, chla_df], axis=1)

    if stack_wq_vars:
        log.info(
            "Applying normalisation parameters to water quality variables"
        )
        for ds in [tsm_ds, chla_ds]:
            for band in list(ds.data_vars):
                if band in NORMALISATION_PARAMETERS.keys():
                    scale = NORMALISATION_PARAMETERS[band]["scale"]
                    offset = NORMALISATION_PARAMETERS[band]["offset"]
                    ds[band] = ds[band] * scale + offset

        log.info("Stacking water quality variables ...")
        # Stack the TSM water quality variables.
        tsm_da = tsm_ds.to_stacked_array(
            new_dim="tsm_measures",
            sample_dims=list(tsm_ds.dims),
            variable_dim="tsm_wq_vars",
            name="tsm",
        )
        tsm_da.attrs = {}

        # Stack the Chla water quality variables.
        chla_da = chla_ds.to_stacked_array(
            new_dim="chla_measures",
            sample_dims=list(chla_ds.dims),
            variable_dim="chla_wq_vars",
            name="chla",
        )
        chla_da.attrs = {}
        log.info("Get median of tss and chla measurements for water pixels")
        # Since median is a non local operation, a compute is triggered here
        # which takes approximately 30 minutes for a single tile
        tsm_da = tsm_da.median(dim="tsm_measures").where(water_mask == 1)
        chla_da = chla_da.median(dim="chla_measures").where(water_mask == 1)
        tsi_da = compute_trophic_state_index(chla_da)

        ds = xr.Dataset({"tsm": tsm_da, "chla": chla_da, "tsi": tsi_da})

        if compute:
            log.info("\tComputing TSM, Chla, and TSI dataset ...")
            ds = ds.compute()
        log.info(
            "TSM, Chla, and TSI water quality variables computation done."
        )
        return ds, all_wq_vars_df
    else:
        # Add the normalisation parameters as attributes to each variable
        ds = xr.merge([chla_ds, tsm_ds])
        for band in list(ds.data_vars):
            if band in NORMALISATION_PARAMETERS.keys():
                scale = NORMALISATION_PARAMETERS[band]["scale"]
                offset = NORMALISATION_PARAMETERS[band]["offset"]
                ds[band].attrs["normalisation_scale"] = scale
                ds[band].attrs["normalisation_offset"] = offset

        return ds, all_wq_vars_df
