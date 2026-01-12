"""
This module provides functions to apply various water quality algorithms
to EO data from a set of instruments.
"""

import logging
from itertools import chain
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def FAI(ds: xr.Dataset, instrument: str) -> xr.DataArray:
    """
    Calculate the Floating Algae Index (FAI) for a given instrument.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the spectral bands.
    instrument : str
        Name of the satellite/instrument (e.g., 'msi', 'oli', 'tm', 'msi_agm', etc.).

    Returns
    -------
    xr.DataArray
        Floating Algae Index values.
    """
    # Central wavelengths (nm) for the red, NIR, and SWIR bands
    CENTRAL_WAVELENGTHS = {
        "msi": {
            "red": ("msi04", 665),
            "nir": ("msi8a", 864),
            "swir": ("msi11", 1612),
        },
        "msi_agm": {
            "red": ("msi04_agm", 665),
            "nir": ("msi8a_agm", 864),
            "swir": ("msi11_agm", 1612),
        },
        "oli": {
            "red": ("oli04", 655.0),
            "nir": ("oli05", 865.0),
            "swir": ("oli06", 1610.0),
        },
        "oli_agm": {
            "red": ("oli04_agm", 655.0),
            "nir": ("oli05_agm", 865.0),
            "swir": ("oli06_agm", 1610.0),
        },
        "tm": {
            "red": ("tm03", 660.0),
            "nir": ("tm04", 830.0),
            "swir": ("tm05", 1650.0),
        },
        "tm_agm": {
            "red": ("tm03_agm", 660.0),
            "nir": ("tm04_agm", 830.0),
            "swir": ("tm05_agm", 1650.0),
        },
    }
    if instrument not in CENTRAL_WAVELENGTHS.keys():
        log.error(
            f"Invalid instrument '{instrument}'. FAI will be calculated as zero."
        )
        return xr.zeros_like(ds[list(ds.data_vars)[0]]).rename("FAI")
    else:
        # Extract band names (vars) and central wavelengths (l) for the
        # selected instrument
        inst_bands = CENTRAL_WAVELENGTHS[instrument]
        red_band, l_red = inst_bands["red"]
        nir_band, l_nir = inst_bands["nir"]
        swir_band, l_swir = inst_bands["swir"]

        # Compute FAI
        # FAI = Observed_NIR - Interpolated_NIR
        # Observerd NIR is the Rayleigh-corrected reflectance in the NIR
        # Interpolated_NIR = Red + (SWIR - Red) * (NIR_wl - Red_wl) / (SWIR_wl - Red_wl)
        # Interpolated_NIR is a linear baseline formed by the red and SWIR bands.
        # It follows the convention Slope-intercept form:
        # y = mx+b, where m is the slope of the line and b is the y-intercept.

        # Calculate the slope coefficient (m) for the baseline
        # m = (NIR_wl - Red_wl) / (SWIR_wl - Red_wl)
        m = (l_nir - l_red) / (l_swir - l_red)

        # Calculate the baseline reflectance at NIR wavelength
        interpolated_nir = ds[red_band] + (ds[swir_band] - ds[red_band]) * m
        # Scale by 10000 assuming input reflectance is in 0-10000 range
        fai = ds[nir_band] - interpolated_nir
        fai.name = "FAI"
        return fai / 10000


def geomedian_FAI(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the FAI (Floating Algae Index) across multiple instruments
    and produce a combined weighted mean FAI for water pixels.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the spectral bands.

    Returns
    -------
    xr.DataArray
        The updated input xarray Dataset with new bands:
        - '{instrument}_fai' for each processed instrument (masked).
        - 'agm_fai' (the final weighted average FAI, masked).
    """

    reference_mean = {
        "fai": {"msi_agm": 0.0970, "oli_agm": 0.1015, "tm_agm": 0.0962},
    }
    # threshold = {
    #     'fai' : {'msi_agm': 0.05   , 'oli_agm' : 0.05   , 'tm_agm': 0.05},
    #    }
    fai_threshold = 0.05
    # Keep this order for consistent processing.
    geomedian_instruments = ["msi_agm", "oli_agm", "tm_agm"]
    # The instrument to use as the scaling reference
    reference_inst = "msi_agm"

    # Initialization for Weighted Average
    mean_fai_weighted_sum = None
    agm_count_total = None

    fai_bands = []
    for inst_agm in geomedian_instruments:
        # Use the smad band as an indicator that data for the geomedian
        # instrument exists in the dataset.
        count_band = f"{inst_agm}_count"
        if count_band in ds.data_vars:
            # Scale factor based on the reference mean of the current instrument
            # relative to the reference mean of the reference instrument.
            scale = (
                reference_mean["fai"][reference_inst]
                / reference_mean["fai"][inst_agm]
            )
            # Calculate the FAI for the instrument and scale
            fai_data = FAI(ds, inst_agm) * scale
            # Replace all NaN values with 0s in FAI
            # fai_data = fai_data.fillna(0)
            # Replace all NaN values with 0s in count band.
            inst_count = ds[count_band].fillna(0)

            weighted_fai = fai_data * inst_count

            # Aggregate the weighted FAI and the total count
            if mean_fai_weighted_sum is None:
                mean_fai_weighted_sum = weighted_fai
                agm_count_total = inst_count
            else:
                mean_fai_weighted_sum += weighted_fai
                agm_count_total += inst_count

            # Add the instrument-specific masked FAI to the Dataset
            ds[f"{inst_agm}_fai"] = fai_data
            fai_bands.append(f"{inst_agm}_fai")

    # Trim the fai values back to relevant areas and values
    ds[fai_bands] = ds[fai_bands].where(ds[fai_bands] > fai_threshold)
    # Mask to only include water pixels.
    ds[fai_bands] = ds[fai_bands].where(ds["water_mask"] == 1)

    if mean_fai_weighted_sum is not None and agm_count_total is not None:
        # Avoid division by zero:
        agm_count_total = agm_count_total.where(agm_count_total != 0)
        mean_fai = mean_fai_weighted_sum / agm_count_total

        # Trim the fai values back to relevant areas and values
        mean_fai = mean_fai.where(mean_fai > fai_threshold)
        # Mask to only include water pixels.
        mean_fai = mean_fai.where(ds["water_mask"] == 1)

        ds["agm_fai"] = mean_fai

    return ds


def geomedian_NDVI(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the NDVI across multiple instruments
    and produce a combined weighted mean NDVI for water pixels.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the spectral bands.

    Returns
    -------
    xr.DataArray
        The updated input xarray Dataset with new bands:
        - '{instrument}_ndvi' for each processed instrument (masked).
        - 'agm_ndvi' (the final weighted average NDVI, masked).
    """

    NDVI_BANDS = {
        "tm_agm": {"red": "tm03_agm", "nir": "tm04_agm"},
        "oli_agm": {"red": "oli04_agm", "nir": "oli05_agm"},
        "msi_agm": {"red": "msi04_agm", "nir": "msi8a_agm"},
    }

    # These values are used as a simple adjustment of the NDVI values to maximise comparability
    reference_mean = {
        "ndvi": {"msi_agm": 0.2335, "oli_agm": 0.2225, "tm_agm": 0.2000},
    }
    # threshold = {
    #     'ndvi' : {'msi_agm': 0.05, 'oli_agm' : 0.05, 'tm_agm': 0.05},
    #     }
    ndvi_threshold = 0.05

    # Keep this order for consistent processing.
    geomedian_instruments = ["msi_agm", "oli_agm", "tm_agm"]
    # The instrument to use as the scaling reference
    reference_inst = "msi_agm"

    # Initialization for Weighted Average
    mean_ndvi_weighted_sum = None
    agm_count_total = None

    ndvi_bands = []
    for inst_agm in geomedian_instruments:
        # Use the smad band as an indicator that data for the geomedian
        # instrument exists in the dataset.
        count_band = f"{inst_agm}_count"
        if count_band in ds.data_vars:
            # Scale factor based on the reference mean of the current instrument
            # relative to the reference mean of the reference instrument.
            scale = (
                reference_mean["ndvi"][reference_inst]
                / reference_mean["ndvi"][inst_agm]
            )
            # Calculate the NDVI for the instrument and scale
            inst_bands = NDVI_BANDS[inst_agm]
            red_band = inst_bands["red"]
            nir_band = inst_bands["nir"]
            ndvi_data = (ds[nir_band] - ds[red_band]) / (
                ds[nir_band] + ds[red_band]
            )
            ndvi_data = ndvi_data * scale

            # Replace all NaN values with 0s in NDVI
            # ndvi_data = ndvi_data.fillna(0)
            # Replace all NaN values with 0s in count band.
            inst_count = ds[count_band].fillna(0)

            weighted_ndvi = ndvi_data * inst_count

            # Aggregate the weighted NDVI and the total count
            if mean_ndvi_weighted_sum is None:
                mean_ndvi_weighted_sum = weighted_ndvi
                agm_count_total = inst_count
            else:
                mean_ndvi_weighted_sum += weighted_ndvi
                agm_count_total += inst_count

            # Add the instrument-specific masked NDVI to the Dataset
            ds[f"{inst_agm}_ndvi"] = ndvi_data
            ndvi_bands.append(f"{inst_agm}_ndvi")

    # Trim the ndvi values back to relevant areas and values
    ds[ndvi_bands] = ds[ndvi_bands].where(ds[ndvi_bands] > ndvi_threshold)
    # Mask to only include water pixels.
    ds[ndvi_bands] = ds[ndvi_bands].where(ds["water_mask"] == 1)

    if mean_ndvi_weighted_sum is not None and agm_count_total is not None:
        # Avoid division by zero:
        agm_count_total = agm_count_total.where(agm_count_total != 0)
        mean_ndvi = mean_ndvi_weighted_sum / agm_count_total

        # Trim the fai values back to relevant areas and values
        mean_ndvi = mean_ndvi.where(mean_ndvi > ndvi_threshold)
        # Mask to only include water pixels.
        mean_ndvi = mean_ndvi.where(ds["water_mask"] == 1)

        ds["agm_ndvi"] = mean_ndvi
    return ds


# =============================================================================
# Helper to create algorithm entries
# =============================================================================
def create_wq_entry(func, wq_varname, **band_args):
    """Helper to create a dictionary entry for a water quality algorithm."""
    return {"func": func, "wq_varname": wq_varname, "args": band_args}


# =============================================================================
# Water Quality Algorithms
# =============================================================================
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
    return 190.34 * X - 32.45


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
    """Turbidity Index of Yu et al. 2019."""
    delta = (dataset[Red] - dataset[Green]) - (dataset[NIR] - dataset[Green])
    delta = xr.where(delta < 0, 0, delta)
    return scalefactor * np.sqrt(delta)


def TSM_LYM_ETM(
    dataset: xr.Dataset,
    green_band: str,
    red_band: str,
    scale_factor: float = 0.0001,
) -> xr.DataArray:
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


def TSS_Zhang(
    dataset: xr.Dataset,
    blue_band: str,
    green_band: str,
    red_band: str,
    scale_factor: float = 0.0001,
) -> xr.DataArray:
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
        "54": create_wq_entry(
            NDCI_NIR_R,
            "ndci_msi54_agm",
            NIR_band="msi05_agmr",
            red_band="msi04_agmr",
        ),
        "64": create_wq_entry(
            NDCI_NIR_R,
            "ndci_msi64_agm",
            NIR_band="msi06_agmr",
            red_band="msi04_agmr",
        ),
        "74": create_wq_entry(
            NDCI_NIR_R,
            "ndci_msi74_agm",
            NIR_band="msi07_agmr",
            red_band="msi04_agmr",
        ),
    },
    "tm_agm": create_wq_entry(
        NDCI_NIR_R, "ndci_tm43_agm", NIR_band="tm04_agm", red_band="tm03_agmr"
    ),
    "oli_agm": create_wq_entry(
        NDCI_NIR_R,
        "ndci_oli54_agm",
        NIR_band="oli05_agm",
        red_band="oli04_agmr",
    ),
}

chla_meris2b = {
    "msi_agm": create_wq_entry(
        ChlA_MERIS2B,
        "chla_meris2b_msi_agm",
        band_708="msi05_agmr",
        band_665="msi04_agmr",
    ),
    "msi": create_wq_entry(
        ChlA_MERIS2B, "chla_meris2b_msi", band_708="msi05", band_665="msi04"
    ),
}

chla_modis2b = {
    "msi_agm": create_wq_entry(
        ChlA_MODIS2B,
        "chla_modis2b_msi_agm",
        band_748="msi06_agmr",
        band_667="msi04_agmr",
    ),
    "msi": create_wq_entry(
        ChlA_MODIS2B, "chla_modis2b_msi", band_748="msi06", band_667="msi04"
    ),
    "tm_agm": create_wq_entry(
        ChlA_MODIS2B,
        "chla_modis2b_tm_agm",
        band_748="tm04_agmr",
        band_667="tm03_agmr",
    ),
}

ndssi_rg = {
    "msi_agm": create_wq_entry(
        NDSSI_RG,
        "ndssi_rg_msi_agm",
        red_band="msi04_agmr",
        green_band="msi03_agmr",
    ),
    "msi": create_wq_entry(
        NDSSI_RG, "ndssi_rg_msi", red_band="msi04r", green_band="msi03_agmr"
    ),
    "oli_agm": create_wq_entry(
        NDSSI_RG,
        "ndssi_rg_oli_agm",
        red_band="oli04_agmr",
        green_band="oli03_agmr",
    ),
    "oli": create_wq_entry(
        NDSSI_RG, "ndssi_rg_oli", red_band="oli04r", green_band="oli03r"
    ),
    "tm_agm": create_wq_entry(
        NDSSI_RG,
        "ndssi_rg_tm_agm",
        red_band="tm03_agmr",
        green_band="tm02_agmr",
    ),
    "tm": create_wq_entry(
        NDSSI_RG, "ndssi_rg_tm", red_band="tm03r", green_band="tmi02r"
    ),
}

ndssi_bnir = {
    "msi": create_wq_entry(
        NDSSI_BNIR, "ndssi_bnir_msi", NIR_band="msi08", blue_band="msi02_agmr"
    ),
    "oli_agm": create_wq_entry(
        NDSSI_BNIR,
        "ndssi_bnir_oli_agm",
        NIR_band="oli06_agm",
        blue_band="oli02_agmr",
    ),
    "oli": create_wq_entry(
        NDSSI_BNIR, "ndssi_bnir_oli", NIR_band="oli06", blue_band="oli02r"
    ),
    "tm": create_wq_entry(
        NDSSI_BNIR, "ndssi_bnir_tm", NIR_band="tm04", blue_band="tm01r"
    ),
}

ti_yu = {
    "msi": create_wq_entry(
        TI_yu, "ti_yu_msi", NIR="msi08", Red="msi04r", Green="msi03_agmr"
    ),
    "oli_agm": create_wq_entry(
        TI_yu,
        "ti_yu_oli_agm",
        NIR="oli06_agm",
        Red="oli04_agmr",
        Green="oli03_agmr",
    ),
    "oli": create_wq_entry(
        TI_yu, "ti_yu_oli", NIR="oli06", Red="oli04r", Green="oli03r"
    ),
    "tm_agm": create_wq_entry(
        TI_yu,
        "ti_yu_tm_agm",
        NIR="tm04_agm",
        Red="tm03_agmr",
        Green="tm02_agmr",
    ),
    "tm": create_wq_entry(
        TI_yu, "ti_yu_tm", NIR="tm04", Red="tm03r", Green="tmi02r"
    ),
}

tsm_lym = {
    "oli_agm": create_wq_entry(
        TSM_LYM_OLI,
        "tsm_lym_oli_agm",
        red_band="oli04_agmr",
        green_band="oli03_agmr",
    ),
    "oli": create_wq_entry(
        TSM_LYM_OLI, "tsm_lym_oli", red_band="oli04r", green_band="oli03r"
    ),
    "msi_agm": create_wq_entry(
        TSM_LYM_OLI,
        "tsm_lym_msi_agm",
        red_band="msi04_agmr",
        green_band="msi03_agmr",
    ),
    "msi": create_wq_entry(
        TSM_LYM_OLI, "tsm_lym_msi", red_band="msi04r", green_band="msi03r"
    ),
    "tm_agm": create_wq_entry(
        TSM_LYM_ETM,
        "tsm_lym_tm_agm",
        red_band="tm03_agmr",
        green_band="tm02_agmr",
    ),
    "tm": create_wq_entry(
        TSM_LYM_ETM, "tsm_lym_tm", red_band="tm03r", green_band="tm02r"
    ),
}

spm_qiu = {
    "oli_agm": create_wq_entry(
        SPM_QIU,
        "spm_qiu_oli_agm",
        red_band="oli04_agmr",
        green_band="oli03_agmr",
    ),
    "oli": create_wq_entry(
        SPM_QIU, "spm_qiu_oli", red_band="oli04r", green_band="oli03r"
    ),
    "tm_agm": create_wq_entry(
        SPM_QIU, "spm_qiu_tm_agm", red_band="tm03_agmr", green_band="tm02_agmr"
    ),
    "tm": create_wq_entry(
        SPM_QIU, "spm_qiu_tm", red_band="tm03r", green_band="tm02r"
    ),
    "msi_agm": create_wq_entry(
        SPM_QIU,
        "spm_qiu_msi_agm",
        red_band="msi04_agmr",
        green_band="msi03_agmr",
    ),
    "msi": create_wq_entry(
        SPM_QIU, "spm_qiu_msi", red_band="msi04r", green_band="msi03r"
    ),
}

tss_zhang = {
    "msi_agm": create_wq_entry(
        TSS_Zhang,
        "tss_zhang_msi_agm",
        blue_band="msi02_agmr",
        red_band="msi04_agmr",
        green_band="msi03_agmr",
    ),
    "msi": create_wq_entry(
        TSS_Zhang,
        "tss_zhang_msi",
        blue_band="msi02r",
        red_band="msi04r",
        green_band="msi03_agmr",
    ),
    "oli_agm": create_wq_entry(
        TSS_Zhang,
        "tss_zhang_oli_agm",
        blue_band="oli02_agmr",
        red_band="oli04_agmr",
        green_band="oli03_agmr",
    ),
    "oli": create_wq_entry(
        TSS_Zhang,
        "tss_zhang_oli",
        blue_band="oli02r",
        red_band="oli04r",
        green_band="oli03r",
    ),
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
                list(inst_app.values())
                if isinstance(inst_app, dict) and "func" not in inst_app
                else [inst_app]
            )
            for alg_entry in instrument_alg_apps:
                func = alg_entry["func"]
                args = alg_entry["args"]
                wq_varname = alg_entry["wq_varname"]
                ds[wq_varname] = func(ds, **args)

                scale_offset = NORMALISATION_PARAMETERS.get(
                    wq_varname, {"scale": 1, "offset": 0}
                )
                ds[wq_varname].attrs = {
                    "nodata": np.nan,
                    "scales": scale_offset["scale"],
                    "offsets": scale_offset["offset"],
                }
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


def classify_chla_values(chla_values: np.ndarray) -> np.ndarray:
    """
    Classify Chlorophyll-a (µg/l) values into Trophic State Index
    values based on the table of Trophic State Index and related
    chlorophyll concentration classes (according to Carlson (1977)).

    Parameters
    ----------
    chla_values : np.ndarray
        Chlorophyll-a (µg/l) values to classify

    Returns
    -------
    np.ndarray
        Corresponding Trophic State Index values.
    """
    # Chlorophyll-a (µg/l) (upper limit)
    conditions = [
        (chla_values <= 0.04),
        (chla_values > 0.04) & (chla_values <= 0.12),
        (chla_values > 0.12) & (chla_values <= 0.34),
        (chla_values > 0.34) & (chla_values <= 0.94),
        (chla_values > 0.94) & (chla_values <= 2.6),
        (chla_values > 2.6) & (chla_values <= 6.4),
        (chla_values > 6.4) & (chla_values <= 20),
        (chla_values > 20) & (chla_values <= 56),
        (chla_values > 56) & (chla_values <= 154),
        (chla_values > 154) & (chla_values <= 427),
        (chla_values > 427) & (chla_values <= 1183),
    ]
    # Trophic State Index, CGLOPS TSI values
    choices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    tsi_values = np.select(conditions, choices, default=np.nan)
    return tsi_values


def compute_trophic_state_index(
    ds: xr.Dataset, chla_variable: str
) -> xr.Dataset:
    """
    Compute the Trophic State Index from the Chlorophyll-a (µg/l) values
    and add the Trophic State Index as the variable "tsi" to the input
    dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to add the Trophic State Index to.
    chla_variable : str
        Variable in input dataset containing the Chlorophyll-a (µg/l)
        values to derive the Trophic State Index from.

    Returns
    -------
    xr.Dataset
        Input dataset with the Trophic State Index added as the
        variable "tsi".
    """
    ds["tsi"] = (tuple(ds.dims), classify_chla_values(ds[chla_variable]))
    return ds


def normalise_and_stack_wq_vars(
    ds: xr.Dataset,
    wq_vars_table: pd.DataFrame,
    water_frequency_threshold: float,
) -> xr.Dataset:
    """
    Normalize the water quality variables in the input dataset `ds`,
    then stack them into the variables "tss" and "chla".
    Finally, compute the Trophic State Index from the Chlorophyll-a
    (µg/l) values and add it to the dataset as the variable "tsi".

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the water quality variables to normalize.
    wq_vars_table : pd.DataFrame
        DataFrame containing the water quality variables table.
    water_frequency_threshold : float
        Threshold to use when classifying water and non-water pixels
        in the normalization process.

    Returns
    -------
    xr.Dataset
        3D dataset containing the water quality variables
        after normalization and stacking.
    """
    tss_wq_vars = wq_vars_table["tss_measures"].dropna().to_list()
    chla_wq_vars = wq_vars_table["chla_measures"].dropna().to_list()
    all_wq_vars = list(chain(tss_wq_vars, chla_wq_vars))

    # Apply normalization parameters
    for band in list(ds.data_vars):
        if band in NORMALISATION_PARAMETERS.keys():
            scale = NORMALISATION_PARAMETERS[band]["scale"]
            offset = NORMALISATION_PARAMETERS[band]["offset"]
            ds[band] = ds[band] * scale + offset

    log.info("Stack the water quality variables")
    # Keep the  dimensions of the 3D dataset
    original_ds_dims = list(ds.dims)

    # Stack the TSS water quality variables.
    tss_da = ds[tss_wq_vars].to_stacked_array(
        new_dim="tss_measures",
        sample_dims=original_ds_dims,
        variable_dim="tss_wq_vars",
        name="tss",
    )
    tss_da.attrs = {}

    # Stack the Chla water quality variables.
    chla_da = ds[chla_wq_vars].to_stacked_array(
        new_dim="chla_measures",
        sample_dims=original_ds_dims,
        variable_dim="chla_wq_vars",
        name="chla",
    )
    chla_da.attrs = {}

    # Drop the original water quality variables
    ds = ds.drop_vars(all_wq_vars)

    log.info("Get median of tss and chla measurements for water pixels")
    ds["tss"] = xr.where(
        ds["water_mask"] == 1, tss_da.median(dim="tss_measures"), np.nan
    )
    ds["chla"] = xr.where(
        ds["water_mask"] == 1, chla_da.median(dim="chla_measures"), np.nan
    )
    # ds = ds.drop_dims(["tss_measures", "chla_measures"], errors="ignore")

    # Compute the Trophic State Index from the Chlorophyll-a (µg/l) values
    ds = compute_trophic_state_index(ds, chla_variable="chla")

    return ds
