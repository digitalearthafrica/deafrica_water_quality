"""
This module provides functions to apply various water quality algorithms
to EO data from a set of instruments.
"""

import logging
from typing import Any

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)


def WQ_vars(
    ds: xr.Dataset,
    algorithms: dict[str, dict[str, dict[str, Any]]],
    instruments_list: dict[str, dict[str, dict[str, str | tuple]]],
    new_dimension_name: str | None = None,
    new_varname: str | None = None,
) -> tuple[list[str], xr.Dataset]:
    """
    Run the TSS/TSP algorithms applying each algorithm to the
    instruments and band combinations set in the `algorithms`
    dictionary, checking that the necessary instruments are in the
    dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with surface reflectance data.
    algorithms : dict[str, dict[str, dict[str, Any]]]
        Water Quality TSS/TSP algorithms, instruments, and bands
    instruments_list :  dict[str, dict[str, dict[str, str | tuple]]],
        Instruments that have been used to build the dataset.
    new_dimension_name : str | None, optional
        'tss_measure', or 'chla_measure', by default None
    new_varname : str | None, optional
        'tss' or 'chla', by default None

    Returns
    -------
    tuple[list[str], xr.Dataset]
        _description_
    """
    log.info(
        f"Running  WQ algorithms for: {', '.join(list(algorithms.keys()))}"
    )
    wq_varlist = []
    for alg in algorithms.keys():
        log.info(f"Running  WQ algorithm {alg}")
        for inst in list(algorithms[alg].keys()):
            params = algorithms[alg][inst]
            if inst in list(instruments_list.keys()):
                log.info(f"Instrument: {inst}")
                # Special case here as options are possible.
                if inst == "msi_agm" and alg == "ndci_nir_r":
                    for option in params.keys():
                        opparams = params[option]
                        function = opparams["func"]
                        ds[opparams["wq_varname"]] = function(
                            ds, **opparams["args"]
                        )
                        wq_varlist = np.append(
                            wq_varlist, opparams["wq_varname"]
                        )
                else:
                    function = params["func"]
                    ds[params["wq_varname"]] = function(ds, **params["args"])
                    wq_varlist = np.append(wq_varlist, params["wq_varname"])
            else:
                # Instrument is not used to build ds
                # skip
                pass

    # If relevant arguments are provided, then create a dimension for
    # the data variables and move them into it
    if new_dimension_name is not None and new_varname is not None:
        if new_dimension_name not in ["tss_measure", "chla_measure"]:
            raise ValueError()
        if new_varname not in ["tss", "chla"]:
            raise ValueError()

        new_dim_labels = list(wq_varlist)
        ds = ds.assign_coords({new_dimension_name: new_dim_labels})

        da_dims = ["time", "x", "y", new_dimension_name]
        da_coords = {dim: ds.coords[dim] for dim in da_dims}
        da_data_shape = tuple(len(da_coords[dim]) for dim in da_dims)
        da = xr.DataArray(
            data=np.zeros(da_data_shape, dtype=np.float32),
            coords=da_coords,
            dims=da_dims,
            name=new_varname,
        )

        for name in new_dim_labels:
            da.sel({new_dimension_name: name})[:] = ds[name]

        ds[new_varname] = da
        ds = ds.drop_vars(new_dim_labels, errors="ignore")

    return ds, wq_varlist


def NDCI_NIR_R(
    dataset: xr.Dataset, NIR_band: str, red_band: str
) -> xr.DataArray:
    """
    Function to calculate ndci from input nir and red bands ----
    """
    return (dataset[NIR_band] - dataset[red_band]) / (
        dataset[NIR_band] + dataset[red_band]
    )


# Functions to estimate ChlA using the MERIS and
# MODIS 2-Band models, using closest bands from MSI (6, 5, 4) or other
def ChlA_MERIS2B(
    dataset: xr.Dataset,
    band_708: str,
    band_665: str,
) -> xr.DataArray:
    """
    MERIS two band:
    meris2b = 25.28 x MERIS_2BM^2 + 14:85 x / MERIS_2BM – 15.18
    where MERIS_2BM = band9(703.75–713.75 nm) / band7(660–670 nm)
    closest S2 bands are : B5 (703.9) and B4(664) respectively
    may need to divide by pi

    MERIS2B = 25.28 * ((ds.msi05 / ds.msi04r)*2) + 14.85 * (ds.msi05 / ds.msi04r) - 15.18
    """
    # matching MSI bands are 5 and 4
    log.info("ChlA_MERIS two-band model")
    X = dataset[band_708] / dataset[band_665]
    return (25.28 * (X) * 2) + 14.85 * (X) - 15.18


def ChlA_MODIS2B(
    dataset: xr.Dataset, band_748: str, band_667: str
) -> xr.DataArray:
    """
    MODIS
    Chla = 190.34 x MODIS_2BM – 32.45
    where: MODIS_2BM = Band15(743–753 nm) / Band13(662–672 nm)
    closest S2 bands are : B6(740) and B4(664)

    MODIS2B = 190.34 * (ds.msi06/ ds.msi04r) - 32.45
    """
    # matching MSI bands are 6 and 4
    log.info("ChlA_MODIS two-band model")
    X = dataset[band_748] / dataset[band_667]
    return (190.34 * X) - 32.45


# ---- Normalised Difference Suspeneded Sediment Index NDSSI ---
#    These are essentially
#        red-green / red_+green, or,
#        blue-NIR / blue + NIR
def NDSSI_RG(
    dataset: xr.Dataset, red_band: str, green_band: str
) -> xr.DataArray:
    log.info("NDSSI_RG")
    return (dataset[red_band] - dataset[green_band]) / (
        dataset[red_band] + dataset[green_band]
    )


def NDSSI_BNIR(
    dataset: xr.Dataset, blue_band: str, NIR_band: str
) -> xr.DataArray:
    log.info("NDSSI_BNIR")
    return (dataset[blue_band] - dataset[NIR_band]) / (
        dataset[blue_band] + dataset[NIR_band]
    )


# ---- Turbidity index of Yu, X. et al.
#    An empirical algorithm to seamlessly retrieve the concentration
#    of suspended particulate matter from water color across ocean to
#    turbid river mouths. Remote Sens. Environ. 235, 111491 (2019).
#    Used in screening turbid waters for mapping floating algal blooms
#    Initially developed with TM
#    -TI = ((Red − green) − (NIR − Rgreen)) ^ 0.5
def TI_yu(
    dataset: xr.Dataset,
    NIR: str,
    Red: str,
    Green: str,
    scalefactor: float = 0.01,
) -> xr.DataArray:
    log.info("TI_yu")
    # TODO check scalefactor * ((dataset[Red] - dataset[Green])
    # - ((dataset[NIR] - dataset[Green]) * 0.5)) correction!
    return scalefactor * (
        ((dataset[Red] - dataset[Green]) - (dataset[NIR] - dataset[Green]))
        ** 0.5
    )


# ---- Lymburner Total Suspended Matter (TSM)
# Paper: [Lymburner et al. 2016] (https://www.sciencedirect.com/science/article/abs/pii/S0034425716301560)
# Units of mg/L concentration. Variants for ETM and OLT, slight
# difference in parameters.
# These models, developed by leo lymburner and arnold dekker, are
# simple, stable, and produce credible results over a range of
# observations
def TSM_LYM_ETM(
    dataset: xr.Dataset,
    green_band: str,
    red_band: str,
    scale_factor: float = 0.0001,
) -> xr.DataArray:
    log.info("TSM_LYM_ETM")
    return (
        3983
        * ((dataset[green_band] + dataset[red_band]) * scale_factor / 2)
        ** 1.6246
    )


def TSM_LYM_OLI(
    dataset: xr.Dataset,
    green_band: str,
    red_band: str,
    scale_factor: float = 0.0001,
) -> xr.DataArray:
    log.info("TSM_LYM_OLI")
    return (
        3957
        * ((dataset[green_band] + dataset[red_band]) * scale_factor / 2)
        ** 1.6436
    )


# Qui Function to calculate Suspended Particulate Model value
# Paper: Zhongfeng Qiu et.al. 2013 - except it's not.
# This model seems to discriminate well although the scaling is
# questionable and it goes below zero due to the final subtraction.
# (The final subtraction seems immaterial in the context of our work
# (overly precise) and I skip it.)
def SPM_QIU(
    dataset: xr.Dataset, green_band: str, red_band: str
) -> xr.DataArray:
    log.info("SPM_QIU")
    return (
        10.0
        ** (
            2.26 * ((dataset[red_band] / dataset[green_band]) ** 3)
            - 5.42 * ((dataset[red_band] / dataset[green_band]) ** 2)
            + 5.58 * (dataset[red_band] / dataset[green_band])
            - 0.72
        )
        # - 1.43
    )


# ---- Quang Total Suspended Solids (TSS)
# Paper: Quang et al. 2017
# Units of mg/L concentration
def TSS_QUANG8(dataset: xr.Dataset, red_band: str) -> xr.DataArray:
    # ---- Function to calculate quang8 value ----
    log.info("TSS_QUANG8")
    return 380.32 * (dataset.red_band) * 0.0001 - 1.7826


# ---- Model of Zhang et al 2023: ----
#      This model seems to be funamentally unstable, using band ratios as an exponent
#      is asking for extreme values and numerical overflows. It runs to inf all over the place.
#      Green = B3 520-600; Blue = B2 450-515 Red = B4 630-680
#  The function is not scale-less.
#  The factor of 0.0001 is not part of the forumula but  scales back from 10000 range which is clearly
#  ridiculous (exp(10000) etc. is not a good number). This therefore avoids overflow.
# This model can only be used together with other models and indices; it may handle some situations well...
def TSS_Zhang(
    dataset: xr.Dataset,
    blue_band: str,
    green_band: str,
    red_band: str,
    scale_factor: float = 0.0001,
) -> xr.DataArray:
    log.info("TSS_Zhang")
    abovezero = 0.00001  # avoids div by zero if blue is zero
    GplusR = dataset[green_band] + dataset[red_band]
    RdivB = dataset[red_band] / (dataset[blue_band] + abovezero)
    X = (GplusR * RdivB) * scale_factor
    # return(10**(14.44*X))*1.20
    # return  np.exp(14.44* X)*1.20
    return (
        14.44 * X
    )  # the distribution of results is exponential; this measure will be more stable without raising to the power.


# ----  dictionary of instruments, bands, algorithms, and  functions -----------------------
ndci_nir_r = {
    "msi_agm": {
        "54": {
            "func": NDCI_NIR_R,
            "wq_varname": "ndci_msi54_agm",
            "args": {"NIR_band": "msi05_agmr", "red_band": "msi04_agmr"},
        },
        "64": {
            "func": NDCI_NIR_R,
            "wq_varname": "ndci_msi64_agm",
            "args": {"NIR_band": "msi06_agmr", "red_band": "msi04_agmr"},
        },
        "74": {
            "func": NDCI_NIR_R,
            "wq_varname": "ndci_msi74_agm",
            "args": {"NIR_band": "msi07_agmr", "red_band": "msi04_agmr"},
        },
    },
    "tm_agm": {
        "func": NDCI_NIR_R,
        "wq_varname": "ndci_tm43_agm",
        "args": {"NIR_band": "tm04_agm", "red_band": "tm03_agmr"},
    },
    "oli_agm": {
        "func": NDCI_NIR_R,
        "wq_varname": "ndci_oli54_agm",
        "args": {"NIR_band": "oli05_agm", "red_band": "oli04_agmr"},
    },
}

chla_meris2b = {
    "msi_agm": {
        "func": ChlA_MERIS2B,
        "wq_varname": "chla_meris2b_msi_agm",
        "args": {"band_708": "msi05_agmr", "band_665": "msi04_agmr"},
    },
    "msi": {
        "func": ChlA_MERIS2B,
        "wq_varname": "chla_meris2b_msi",
        "args": {"band_708": "msi05", "band_665": "msi04"},
    },
}

chla_modis2b = {
    "msi_agm": {
        "func": ChlA_MODIS2B,
        "wq_varname": "chla_modis2b_msi_agm",
        "args": {"band_748": "msi06_agmr", "band_667": "msi04_agmr"},
    },
    "msi": {
        "func": ChlA_MODIS2B,
        "wq_varname": "chla_modis2b_msi",
        "args": {"band_748": "msi06", "band_667": "msi04"},
    },
    "tm_agm": {
        "func": ChlA_MODIS2B,
        "wq_varname": "chla_modis2b_tm_agm",
        "args": {"band_748": "tm04_agmr", "band_667": "tm03_agmr"},
    },
}

ndssi_rg = {
    "msi_agm": {
        "func": NDSSI_RG,
        "wq_varname": "ndssi_rg_msi_agm",
        "args": {"red_band": "msi04_agmr", "green_band": "msi03_agmr"},
    },
    "msi": {
        "func": NDSSI_RG,
        "wq_varname": "ndssi_rg_msi",
        "args": {"red_band": "msi04r", "green_band": "msi03_agmr"},
    },
    "oli_agm": {
        "func": NDSSI_RG,
        "wq_varname": "ndssi_rg_oli_agm",
        "args": {"red_band": "oli04_agmr", "green_band": "oli03_agmr"},
    },
    "oli": {
        "func": NDSSI_RG,
        "wq_varname": "ndssi_rg_oli",
        "args": {"red_band": "oli04r", "green_band": "oli03r"},
    },
    "tm_agm": {
        "func": NDSSI_RG,
        "wq_varname": "ndssi_rg_tm_agm",
        "args": {"red_band": "tm03_agmr", "green_band": "tm02_agmr"},
    },
    "tm": {
        "func": NDSSI_RG,
        "wq_varname": "ndssi_rg_tm",
        "args": {"red_band": "tm03r", "green_band": "tmi02r"},
    },
}

ndssi_bnir = {  # "msi_agm" : {'func': NDSSI_BNIR,   "wq_varname" : 'ndssi_bnir_msi_agm'     ,'args' : { "NIR_band":'msi08_agmr', "blue_band":'msi02_agmr'}},
    "msi": {
        "func": NDSSI_BNIR,
        "wq_varname": "ndssi_bnir_msi",
        "args": {"NIR_band": "msi08", "blue_band": "msi02_agmr"},
    },
    "oli_agm": {
        "func": NDSSI_BNIR,
        "wq_varname": "ndssi_bnir_oli_agm",
        "args": {"NIR_band": "oli06_agm", "blue_band": "oli02_agmr"},
    },
    "oli": {
        "func": NDSSI_BNIR,
        "wq_varname": "ndssi_bnir_oli",
        "args": {"NIR_band": "oli06", "blue_band": "oli02r"},
    },
    # "tm_agm"  : {'func': NDSSI_BNIR,   "wq_varname" : 'ndssi_bnir_tm_agm'      ,'args' : { "NIR_band":'tm04_agm'  , "blue_band":'tm01_agmr' }},
    "tm": {
        "func": NDSSI_BNIR,
        "wq_varname": "ndssi_bnir_tm",
        "args": {"NIR_band": "tm04", "blue_band": "tm01r"},
    },
}


ti_yu = {  # "msi_agm" : {'func': TI_yu,        "wq_varname" : 'ti_yu_msi_agm'        ,'args' : {"NIR" : 'msi08_agmr', "Red":'msi04_agmr', "Green":'msi03_agmr'}},
    "msi": {
        "func": TI_yu,
        "wq_varname": "ti_yu_msi",
        "args": {"NIR": "msi08", "Red": "msi04r", "Green": "msi03_agmr"},
    },
    "oli_agm": {
        "func": TI_yu,
        "wq_varname": "ti_yu_oli_agm",
        "args": {
            "NIR": "oli06_agm",
            "Red": "oli04_agmr",
            "Green": "oli03_agmr",
        },
    },
    "oli": {
        "func": TI_yu,
        "wq_varname": "ti_yu_oli",
        "args": {"NIR": "oli06", "Red": "oli04r", "Green": "oli03r"},
    },
    "tm_agm": {
        "func": TI_yu,
        "wq_varname": "ti_yu_tm_agm",
        "args": {"NIR": "tm04_agm", "Red": "tm03_agmr", "Green": "tm02_agmr"},
    },
    "tm": {
        "func": TI_yu,
        "wq_varname": "ti_yu_tm",
        "args": {"NIR": "tm04", "Red": "tm03r", "Green": "tmi02r"},
    },
}

tsm_lym = {
    "oli_agm": {
        "func": TSM_LYM_OLI,
        "wq_varname": "tsm_lym_oli_agm",
        "args": {"red_band": "oli04_agmr", "green_band": "oli03_agmr"},
    },
    "oli": {
        "func": TSM_LYM_OLI,
        "wq_varname": "tsm_lym_oli",
        "args": {"red_band": "oli04r", "green_band": "oli03r"},
    },
    "msi_agm": {
        "func": TSM_LYM_OLI,
        "wq_varname": "tsm_lym_msi_agm",
        "args": {"red_band": "msi04_agmr", "green_band": "msi03_agmr"},
    },
    "msi": {
        "func": TSM_LYM_OLI,
        "wq_varname": "tsm_lym_msi",
        "args": {"red_band": "msi04r", "green_band": "msi03r"},
    },
    "tm_agm": {
        "func": TSM_LYM_ETM,
        "wq_varname": "tsm_lym_tm_agm",
        "args": {"red_band": "tm03_agmr", "green_band": "tm02_agmr"},
    },
    "tm": {
        "func": TSM_LYM_ETM,
        "wq_varname": "tsm_lym_tm",
        "args": {"red_band": "tm03r", "green_band": "tm02r"},
    },
}

spm_qiu = {
    "oli_agm": {
        "func": SPM_QIU,
        "wq_varname": "spm_qiu_oli_agm",
        "args": {"red_band": "oli04_agmr", "green_band": "oli03_agmr"},
    },
    "oli": {
        "func": SPM_QIU,
        "wq_varname": "spm_qiu_oli",
        "args": {"red_band": "oli04r", "green_band": "oli03r"},
    },
    "tm_agm": {
        "func": SPM_QIU,
        "wq_varname": "spm_qiu_tm_agm",
        "args": {"red_band": "tm03_agmr", "green_band": "tm02_agmr"},
    },
    "tm": {
        "func": SPM_QIU,
        "wq_varname": "spm_qiu_tm",
        "args": {"red_band": "tm03r", "green_band": "tm02r"},
    },
    "msi_agm": {
        "func": SPM_QIU,
        "wq_varname": "spm_qiu_msi_agm",
        "args": {"red_band": "msi04_agmr", "green_band": "msi03_agmr"},
    },
    "msi": {
        "func": SPM_QIU,
        "wq_varname": "spm_qiu_msi",
        "args": {"red_band": "msi04r", "green_band": "msi03r"},
    },
}

tss_zhang = {
    "msi_agm": {
        "func": TSS_Zhang,
        "wq_varname": "tss_zhang_msi_agm",
        "args": {
            "blue_band": "msi02_agmr",
            "red_band": "msi04_agmr",
            "green_band": "msi03_agmr",
        },
    },
    "msi": {
        "func": TSS_Zhang,
        "wq_varname": "tss_zhang_msi",
        "args": {
            "blue_band": "msi02r",
            "red_band": "msi04r",
            "green_band": "msi03_agmr",
        },
    },
    "oli_agm": {
        "func": TSS_Zhang,
        "wq_varname": "tss_zhang_oli_agm",
        "args": {
            "blue_band": "oli02_agmr",
            "red_band": "oli04_agmr",
            "green_band": "oli03_agmr",
        },
    },
    "oli": {
        "func": TSS_Zhang,
        "wq_varname": "tss_zhang_oli",
        "args": {
            "blue_band": "oli02r",
            "red_band": "oli04r",
            "green_band": "oli03r",
        },
    },
}

# ---- algorithms are grouped into two over-arching dictionaries ----
ALGORITHMS_CHLA = {
    "ndci_nir_r": ndci_nir_r,
    "chla_meris2b": chla_meris2b,
    "chla_modis2b": chla_modis2b,
}
ALGORITHMS_TSM = {
    "ndssi_rg": ndssi_rg,
    "ndssi_bnir": ndssi_bnir,
    "ti_yu": ti_yu,
    "tsm_lym": tsm_lym,
    "tss_zhang": tss_zhang,
    "spm_qiu": spm_qiu,
}
