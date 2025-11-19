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
# Helper to create algorithm entries
# =============================================================================
def create_wq_entry(func, wq_varname, **band_args):
    """Helper to create a dictionary entry for a water quality algorithm."""
    return {"func": func, "wq_varname": wq_varname, "args": band_args}


# =============================================================================
# Water Quality Algorithms
# =============================================================================
def NDCI_NIR_R(dataset: xr.Dataset, NIR_band: str, red_band: str) -> xr.DataArray:
    """Normalized Difference Chlorophyll Index (NDCI)."""
    return (dataset[NIR_band] - dataset[red_band]) / (dataset[NIR_band] + dataset[red_band])

def ChlA_MERIS2B(dataset: xr.Dataset, band_708: str, band_665: str) -> xr.DataArray:
    """MERIS two-band chlorophyll-a estimation."""
    X = dataset[band_708] / dataset[band_665]
    return (25.28 * (X ** 2)) + 14.85 * X - 15.18

def ChlA_MODIS2B(dataset: xr.Dataset, band_748: str, band_667: str) -> xr.DataArray:
    """MODIS two-band chlorophyll-a estimation."""
    X = dataset[band_748] / dataset[band_667]
    return 190.34 * X - 32.45

def NDSSI_RG(dataset: xr.Dataset, red_band: str, green_band: str) -> xr.DataArray:
    """Normalized Difference Suspended Sediment Index (Red-Green)."""
    return (dataset[red_band] - dataset[green_band]) / (dataset[red_band] + dataset[green_band])

def NDSSI_BNIR(dataset: xr.Dataset, blue_band: str, NIR_band: str) -> xr.DataArray:
    """Normalized Difference Suspended Sediment Index (Blue-NIR)."""
    return (dataset[blue_band] - dataset[NIR_band]) / (dataset[blue_band] + dataset[NIR_band])

def TI_yu(dataset: xr.Dataset, NIR: str, Red: str, Green: str, scalefactor: float = 0.01) -> xr.DataArray:
    """Turbidity Index of Yu et al. 2019."""
    delta = ((dataset[Red] - dataset[Green]) - (dataset[NIR] - dataset[Green]))
    delta = xr.where(delta < 0, 0, delta)
    return scalefactor * np.sqrt(delta)

def TSM_LYM_ETM(dataset: xr.Dataset, green_band: str, red_band: str, scale_factor: float = 0.0001) -> xr.DataArray:
    return 3983 * (((dataset[green_band] + dataset[red_band]) * scale_factor / 2) ** 1.6246)

def TSM_LYM_OLI(dataset: xr.Dataset, green_band: str, red_band: str, scale_factor: float = 0.0001) -> xr.DataArray:
    return 3957 * (((dataset[green_band] + dataset[red_band]) * scale_factor / 2) ** 1.6436)

def SPM_QIU(dataset: xr.Dataset, green_band: str, red_band: str) -> xr.DataArray:
    X = dataset[red_band] / dataset[green_band]
    return 10.0 ** (2.26 * (X**3) - 5.42 * (X**2) + 5.58 * X - 0.72)

def TSS_QUANG8(dataset: xr.Dataset, red_band: str) -> xr.DataArray:
    """Quang et al. 2017 TSS estimation."""
    return 380.32 * dataset[red_band] * 0.0001 - 1.7826

def TSS_Zhang(dataset: xr.Dataset, blue_band: str, green_band: str, red_band: str, scale_factor: float = 0.0001) -> xr.DataArray:
    """Zhang et al. 2023 TSS estimation (stable version)."""
    abovezero = 1e-5
    GplusR = dataset[green_band] + dataset[red_band]
    RdivB = dataset[red_band] / (dataset[blue_band] + abovezero)
    X = GplusR * RdivB * scale_factor
    return 14.44 * X


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
# === Water Quality Algorithm Dictionaries (cleaned with create_wq_entry) ===

ndci_nir_r = {
    "msi_agm": {
        "54": create_wq_entry(NDCI_NIR_R, "ndci_msi54_agm", NIR_band="msi05_agmr", red_band="msi04_agmr"),
        "64": create_wq_entry(NDCI_NIR_R, "ndci_msi64_agm", NIR_band="msi06_agmr", red_band="msi04_agmr"),
        "74": create_wq_entry(NDCI_NIR_R, "ndci_msi74_agm", NIR_band="msi07_agmr", red_band="msi04_agmr"),
    },
    "tm_agm": create_wq_entry(NDCI_NIR_R, "ndci_tm43_agm", NIR_band="tm04_agm", red_band="tm03_agmr"),
    "oli_agm": create_wq_entry(NDCI_NIR_R, "ndci_oli54_agm", NIR_band="oli05_agm", red_band="oli04_agmr"),
}

chla_meris2b = {
    "msi_agm": create_wq_entry(ChlA_MERIS2B, "chla_meris2b_msi_agm", band_708="msi05_agmr", band_665="msi04_agmr"),
    "msi": create_wq_entry(ChlA_MERIS2B, "chla_meris2b_msi", band_708="msi05", band_665="msi04"),
}

chla_modis2b = {
    "msi_agm": create_wq_entry(ChlA_MODIS2B, "chla_modis2b_msi_agm", band_748="msi06_agmr", band_667="msi04_agmr"),
    "msi": create_wq_entry(ChlA_MODIS2B, "chla_modis2b_msi", band_748="msi06", band_667="msi04"),
    "tm_agm": create_wq_entry(ChlA_MODIS2B, "chla_modis2b_tm_agm", band_748="tm04_agmr", band_667="tm03_agmr"),
}

ndssi_rg = {
    "msi_agm": create_wq_entry(NDSSI_RG, "ndssi_rg_msi_agm", red_band="msi04_agmr", green_band="msi03_agmr"),
    "msi": create_wq_entry(NDSSI_RG, "ndssi_rg_msi", red_band="msi04r", green_band="msi03_agmr"),
    "oli_agm": create_wq_entry(NDSSI_RG, "ndssi_rg_oli_agm", red_band="oli04_agmr", green_band="oli03_agmr"),
    "oli": create_wq_entry(NDSSI_RG, "ndssi_rg_oli", red_band="oli04r", green_band="oli03r"),
    "tm_agm": create_wq_entry(NDSSI_RG, "ndssi_rg_tm_agm", red_band="tm03_agmr", green_band="tm02_agmr"),
    "tm": create_wq_entry(NDSSI_RG, "ndssi_rg_tm", red_band="tm03r", green_band="tmi02r"),
}

ndssi_bnir = {
    "msi": create_wq_entry(NDSSI_BNIR, "ndssi_bnir_msi", NIR_band="msi08", blue_band="msi02_agmr"),
    "oli_agm": create_wq_entry(NDSSI_BNIR, "ndssi_bnir_oli_agm", NIR_band="oli06_agm", blue_band="oli02_agmr"),
    "oli": create_wq_entry(NDSSI_BNIR, "ndssi_bnir_oli", NIR_band="oli06", blue_band="oli02r"),
    "tm": create_wq_entry(NDSSI_BNIR, "ndssi_bnir_tm", NIR_band="tm04", blue_band="tm01r"),
}

ti_yu = {
    "msi": create_wq_entry(TI_yu, "ti_yu_msi", NIR="msi08", Red="msi04r", Green="msi03_agmr"),
    "oli_agm": create_wq_entry(TI_yu, "ti_yu_oli_agm", NIR="oli06_agm", Red="oli04_agmr", Green="oli03_agmr"),
    "oli": create_wq_entry(TI_yu, "ti_yu_oli", NIR="oli06", Red="oli04r", Green="oli03r"),
    "tm_agm": create_wq_entry(TI_yu, "ti_yu_tm_agm", NIR="tm04_agm", Red="tm03_agmr", Green="tm02_agmr"),
    "tm": create_wq_entry(TI_yu, "ti_yu_tm", NIR="tm04", Red="tm03r", Green="tmi02r"),
}

tsm_lym = {
    "oli_agm": create_wq_entry(TSM_LYM_OLI, "tsm_lym_oli_agm", red_band="oli04_agmr", green_band="oli03_agmr"),
    "oli": create_wq_entry(TSM_LYM_OLI, "tsm_lym_oli", red_band="oli04r", green_band="oli03r"),
    "msi_agm": create_wq_entry(TSM_LYM_OLI, "tsm_lym_msi_agm", red_band="msi04_agmr", green_band="msi03_agmr"),
    "msi": create_wq_entry(TSM_LYM_OLI, "tsm_lym_msi", red_band="msi04r", green_band="msi03r"),
    "tm_agm": create_wq_entry(TSM_LYM_ETM, "tsm_lym_tm_agm", red_band="tm03_agmr", green_band="tm02_agmr"),
    "tm": create_wq_entry(TSM_LYM_ETM, "tsm_lym_tm", red_band="tm03r", green_band="tm02r"),
}

spm_qiu = {
    "oli_agm": create_wq_entry(SPM_QIU, "spm_qiu_oli_agm", red_band="oli04_agmr", green_band="oli03_agmr"),
    "oli": create_wq_entry(SPM_QIU, "spm_qiu_oli", red_band="oli04r", green_band="oli03r"),
    "tm_agm": create_wq_entry(SPM_QIU, "spm_qiu_tm_agm", red_band="tm03_agmr", green_band="tm02_agmr"),
    "tm": create_wq_entry(SPM_QIU, "spm_qiu_tm", red_band="tm03r", green_band="tm02r"),
    "msi_agm": create_wq_entry(SPM_QIU, "spm_qiu_msi_agm", red_band="msi04_agmr", green_band="msi03_agmr"),
    "msi": create_wq_entry(SPM_QIU, "spm_qiu_msi", red_band="msi04r", green_band="msi03r"),
}

tss_zhang = {
    "msi_agm": create_wq_entry(TSS_Zhang, "tss_zhang_msi_agm", blue_band="msi02_agmr", red_band="msi04_agmr", green_band="msi03_agmr"),
    "msi": create_wq_entry(TSS_Zhang, "tss_zhang_msi", blue_band="msi02r", red_band="msi04r", green_band="msi03_agmr"),
    "oli_agm": create_wq_entry(TSS_Zhang, "tss_zhang_oli_agm", blue_band="oli02_agmr", red_band="oli04_agmr", green_band="oli03_agmr"),
    "oli": create_wq_entry(TSS_Zhang, "tss_zhang_oli", blue_band="oli02r", red_band="oli04r", green_band="oli03r"),
}


# =============================================================================
# Algorithm Groups
# =============================================================================
ALGORITHMS_CHLA = {
    "ndci_nir_r": ndci_nir_r,
    "chla_meris2b": chla_meris2b,
    "chla_modis2b": chla_modis2b,
}

ALGORITHMS_TSS = {
    "ndssi_rg": ndssi_rg,
    "ndssi_bnir": ndssi_bnir,
    "ti_yu": ti_yu,
    "tsm_lym": tsm_lym,
    "tss_zhang": tss_zhang,
    "spm_qiu": spm_qiu,
}

# =============================================================================
# Functions to Run Algorithms on Datasets
# =============================================================================
def run_wq_algorithms(
    ds: xr.Dataset,
    instruments_list: dict[str, dict[str, dict[str, str | tuple]]],
    algorithms_group: dict[str, dict[str, dict[str, Any]]],
) -> tuple[xr.Dataset, list[str]]:
    """Run a group of water quality algorithms on a dataset."""
    wq_varlist = []
    for algorithm_name, algorithm_apps in algorithms_group.items():
        log.info(f"Running algorithm group: {algorithm_name}")
        for instrument_name, inst_app in algorithm_apps.items():
            if instrument_name not in instruments_list:
                continue
            instrument_alg_apps = (
                list(inst_app.values()) if isinstance(inst_app, dict) and "func" not in inst_app else [inst_app]
            )
            for alg_entry in instrument_alg_apps:
                func = alg_entry["func"]
                args = alg_entry["args"]
                wq_varname = alg_entry["wq_varname"]
                ds[wq_varname] = func(ds, **args)

                scale_offset = NORMALISATION_PARAMETERS.get(wq_varname, {"scale": 1, "offset": 0})
                ds[wq_varname].attrs = {"nodata": np.nan, "scales": scale_offset["scale"], "offsets": scale_offset["offset"]}
                wq_varlist.append(wq_varname)
    return ds, wq_varlist


def WQ_vars(
    ds: xr.Dataset,
    instruments_list: dict[str, dict[str, dict[str, str | tuple]]],
    stack_wq_vars: bool = False,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Compute Chlorophyll-A (ChlA) and Total Suspended Solids (TSS) water quality variables."""
    ds, tss_vars = run_wq_algorithms(ds, instruments_list, ALGORITHMS_TSS)
    ds, chla_vars = run_wq_algorithms(ds, instruments_list, ALGORITHMS_CHLA)

    tss_df = pd.DataFrame({"tss_measures": tss_vars})
    chla_df = pd.DataFrame({"chla_measures": chla_vars})
    all_wq_vars_df = pd.concat([tss_df, chla_df], axis=1)

    if stack_wq_vars:
        original_dims = list(ds.dims)
        ds["tss"] = ds[tss_vars].to_stacked_array(
            new_dim="tss_measures",
            sample_dims=original_dims,
            variable_dim="tss_wq_vars",
            name="tss",
        )
        ds["tss"].attrs = {var: ds[var].attrs for var in tss_vars}

        ds["chla"] = ds[chla_vars].to_stacked_array(
            new_dim="chla_measures",
            sample_dims=original_dims,
            variable_dim="chla_wq_vars",
            name="chla",
        )
        ds["chla"].attrs = {var: ds[var].attrs for var in chla_vars}

        ds = ds.drop_vars(tss_vars + chla_vars)

    return ds, all_wq_vars_df

