import logging
from typing import Any

from water_quality.dates import validate_end_date, validate_start_date

log = logging.getLogger(__name__)

INSTRUMENTS_DATES = {
    "oli_agm": [2013, 2024],
    "oli": [2013, 2025],
    "msi_agm": [2017, 2024],
    "msi": [2017, 2025],
    "wofs_ann": [1990, 2024],
    "tm_agm": [1990, 2012],
    "tm": [1990, 2023],
    "tirs": [2000, 2025],
}

# Here is where to turn a particular band on or off, using the 'parameters' entry
INSTRUMENTS_MEASUREMENTS = {
    "wofs_ann": {
        "frequency": {
            "varname": ("wofs_ann_freq"),
            "parameters": (True, "other"),
        },
        "count_clear": {
            "varname": ("wofs_ann_clearcount"),
            "parameters": (True,),
        },
        "count_wet": {"varname": ("wofs_ann_wetcount"), "parameters": (True,)},
    },
    "wofs_all": {
        "frequency": {
            "varname": ("wofs_all_freq"),
            "parameters": (True, "other"),
        },
        "count_clear": {
            "varname": ("wofs_all_clearcount"),
            "parameters": (True,),
        },
        "count_wet": {"varname": ("wofs_all_wetcount"), "parameters": (True,)},
    },
    "oli_agm": {
        "SR_B2": {"varname": ("oli02_agm"), "parameters": (True, "450-510")},
        "SR_B3": {"varname": ("oli03_agm"), "parameters": (True, "530-590")},
        "SR_B4": {"varname": ("oli04_agm"), "parameters": (True, "640-670")},
        "SR_B5": {"varname": ("oli05_agm"), "parameters": (True, "850-880")},
        "SR_B6": {"varname": ("oli06_agm"), "parameters": (True, "1570-1650")},
        "SR_B7": {"varname": ("oli07_agm"), "parameters": (True, "2110-2290")},
        "smad": {"varname": ("oli_agm_smad"), "parameters": (True,)},
        "emad": {"varname": ("oli_agm_emad"), "parameters": (True,)},
        "bcmad": {"varname": ("oli_agm_bcmad"), "parameters": (True,)},
        "count": {"varname": ("oli_agm_count"), "parameters": (True,)},
    },
    "oli": {
        "SR_B1": {"varname": ("oli01"), "parameters": (True, "450-510")},
        "SR_B2": {"varname": ("oli02"), "parameters": (True, "450-510")},
        "SR_B3": {"varname": ("oli03"), "parameters": (True, "530-590")},
        "SR_B4": {"varname": ("oli04"), "parameters": (True, "640-670")},
        "SR_B5": {"varname": ("oli05"), "parameters": (True, "850-880")},
        "SR_B6": {"varname": ("oli06"), "parameters": (True, "1570-1650")},
        "SR_B7": {"varname": ("oli07"), "parameters": (True, "2110-2290")},
        "pq": {"varname": ("oli_pq"), "parameters": (True,)},
    },
    "msi_agm": {
        "B02": {"varname": ("msi02_agm"), "parameters": (True, "460-525")},
        "B03": {"varname": ("msi03_agm"), "parameters": (True,)},
        "B04": {"varname": ("msi04_agm"), "parameters": (True,)},
        "B05": {"varname": ("msi05_agm"), "parameters": (True,)},
        "B06": {"varname": ("msi06_agm"), "parameters": (True,)},
        "B07": {"varname": ("msi07_agm"), "parameters": (True,)},
        "B08": {
            "varname": ("msi08_agm"),
            "parameters": (
                False,
                "uint16 	1 	0.0 	[band_08, nir, nir_1] 	NaN",
            ),
        },
        "B8A": {
            "varname": ("msi8a_agm"),
            "parameters": (
                True,
                "uint16 	1 	0.0 	[band_8a, nir_narrow, nir_2] 	NaN",
            ),
        },
        "B11": {
            "varname": ("msi11_agm"),
            "parameters": (
                True,
                "uint16 	1 	0.0 	[band_11, swir_1, swir_16] 	NaN",
            ),
        },
        "B12": {
            "varname": ("msi12_agm"),
            "parameters": (
                True,
                "uint16 	1 	0.0 	[band_12, swir_2, swir_22] 	NaN",
            ),
        },
        "smad": {"varname": ("msi_agm_smad"), "parameters": (True,)},
        "emad": {"varname": ("msi_agm_emad"), "parameters": (True,)},
        "bcmad": {"varname": ("msi_agm_bcmad"), "parameters": (True,)},
        "count": {"varname": ("msi_agm_count"), "parameters": (True,)},
    },
    "msi": {
        "B01": {
            "varname": ("msi01"),
            "parameters": (
                False,
                " 442 bandwidth 20 nm, spatial 60 m, Coastal aerosol",
            ),
        },
        "B02": {
            "varname": ("msi02"),
            "parameters": (True, " 493 bandwidth 65 nm, spatial 10 m,"),
        },
        "B03": {
            "varname": ("msi03"),
            "parameters": (True, " 559 bandwidth 35 nm, spatial 10 m,"),
        },
        "B04": {
            "varname": ("msi04"),
            "parameters": (True, " 665 bandwidth 31 nm, spatial 10 m,"),
        },
        "B05": {
            "varname": ("msi05"),
            "parameters": (True, " 704 bandwidth 14 nm, spatial 20 m"),
        },
        "B06": {
            "varname": ("msi06"),
            "parameters": (True, " 741 bandwidth 13 nm, spatial 20 m,"),
        },
        "B07": {
            "varname": ("msi07"),
            "parameters": (True, " 780 bandwidth 19 nm, spatial 20 m,"),
        },
        "B08": {
            "varname": ("msi08"),
            "parameters": (
                False,
                " 833 bandwidth 104 nm, spatial 10 m, uint16 	1 	0.0 	[band_08, nir, nir_1] 	NaN",
            ),
        },
        "B8A": {
            "varname": ("msi8a"),
            "parameters": (
                True,
                " 864 bandwidth 21 nm, spatial 20 m, uint16 	1 	0.0 	[band_8a, nir_narrow, nir_2] 	NaN",
            ),
        },
        "B11": {
            "varname": ("msi11"),
            "parameters": (
                True,
                "1612 bandwidth 92 nm, spatial 20 m, uint16 	1 	0.0 	[band_11, swir_1, swir_16] 	NaN",
            ),
        },
        "B12": {
            "varname": ("msi12"),
            "parameters": (
                True,
                "2193 bandwidth 180 nm, spatial 20 m, uint16 	1 	0.0 	[band_12, swir_2, swir_22] 	NaN",
            ),
        },
        "qa": {"varname": ("msi_qa"), "parameters": (True,)},
    },
    "tm_agm": {
        "SR_B1": {
            "varname": ("tm01_agm"),
            "parameters": (True, "blue 450-520"),
        },
        "SR_B2": {
            "varname": ("tm02_agm"),
            "parameters": (True, "green 520-600"),
        },
        "SR_B3": {
            "varname": ("tm03_agm"),
            "parameters": (True, "red   630-690"),
        },
        "SR_B4": {
            "varname": ("tm04_agm"),
            "parameters": (True, "nir   760-900"),
        },
        "SR_B5": {
            "varname": ("tm05_agm"),
            "parameters": (True, "swir1 1550-1750"),
        },
        "SR_B7": {
            "varname": ("tm07_agm"),
            "parameters": (True, "swir2 2080-2350"),
        },
        "smad": {"varname": ("tm_agm_smad"), "parameters": (True,)},
        "emad": {"varname": ("tm_agm_emad"), "parameters": (True,)},
        "bcmad": {"varname": ("tm_agm_bcmad"), "parameters": (True,)},
        "count": {"varname": ("tm_agm_count"), "parameters": (True,)},
    },
    "tm": {
        "SR_B1": {"varname": ("tm01"), "parameters": (True, "blue 450-520")},
        "SR_B2": {"varname": ("tm02"), "parameters": (True, "green 520-600")},
        "SR_B3": {"varname": ("tm03"), "parameters": (True, "red   630-690")},
        "SR_B4": {"varname": ("tm04"), "parameters": (True, "nir   760-900")},
        "SR_B5": {
            "varname": ("tm05"),
            "parameters": (True, "swir1 1550-1750"),
        },
        "SR_B7": {
            "varname": ("tm07"),
            "parameters": (True, "swir2 2080-2350"),
        },
        "pq": {"varname": ("tm_pq"), "parameters": (True,)},
    },
    "tirs": {
        "st": {
            "varname": ("tirs_st"),
            "parameters": (
                True,
                "ST_B10, uint16 	Kelvin 	0.0 	[band_10, st, surface_temperature]",
            ),
        },
        "ST_TRAD": {
            "varname": ("tirs_trad"),
            "parameters": (
                False,
                "ST_TRAD, int16 	W/(m2.sr.μm) 	-9999.0 	[trad, thermal_radiance]",
            ),
        },
        "ST_URAD": {
            "varname": ("tirs_urad"),
            "parameters": (
                False,
                "ST_URAD 	int16 	W/(m2.sr.μm) 	-9999.0 	[urad, upwell_radiance]",
            ),
        },
        "ST_DRAD": {
            "varname": ("tirs_drad"),
            "parameters": (
                False,
                "ST_DRAD 	int16 	W/(m2.sr.μm) 	-9999.0 	[drad, downwell_radiance]",
            ),
        },
        "ST_ATRAN": {
            "varname": ("tirs_atran"),
            "parameters": (
                False,
                "ST_ATRAN 	int16 	1 	-9999.0 	[atran, atmospheric_transmittance]",
            ),
        },
        "emis": {
            "varname": ("tirs_emis"),
            "parameters": (
                True,
                "ST_EMIS 	int16 	1 	-9999.0 	[emis, emissivity]",
            ),
        },
        "emisd": {
            "varname": ("tirs_emsd"),
            "parameters": (
                False,
                "ST_EMSD 	int16 	1 	-9999.0 	[emsd, emissivity_stddev]",
            ),
        },
        "ST_CDIST": {
            "varname": ("tirs_cdist"),
            "parameters": (
                False,
                "ST_CDIST 	int16 	Kilometers 	-9999.0 	[cdist, cloud_distance]",
            ),
        },
        "QA_PIXEL": {
            "varname": ("tirs_qa_pixel"),
            "parameters": (
                False,
                "QA_PIXEL 	uint16 	bit_index 	1.0 	[pq, pixel_quality]",
            ),
        },
        "QA_RADSAT": {
            "varname": ("tirs_radsat"),
            "parameters": (
                False,
                "QA_RADSAT 	uint16 	bit_index 	0.0 	[radsat, radiometric_saturation]",
            ),
        },
        "st_qa": {
            "varname": ("tirs_st_qa"),
            "parameters": (
                True,
                "ST_QA 	int16 	Kelvin 	-9999.0 	[st_qa, surface_temperature_quality]",
            ),
        },
    },
}


def check_instrument_dates(
    instruments_to_use: dict[str, dict[str, bool]],
    start_date: str,
    end_date: str,
):
    """
    Cross check the years data is available for each instrument against
    the years for which the analysis is going to be run.

    Parameters
    ----------
    instruments_to_use : dict[str, dict[str, bool]]
        A dictionary of the selected instruments to use for the analysis and
        a usage parameter showing whether it will be used in the analysis.
    start_date : str
        Start date of the analysis period.
    end_date : str
        End date of the analysis period.

    Returns
    -------
    dict[str, dict[str, bool]]
        Updated `instruments_to_use` where instruments where data is not available for
        the analysis period have their usage parameter set to False.
    """
    start_date = validate_start_date(start_date)
    end_date = validate_end_date(end_date)

    valid_instruments_to_use: dict[str, dict[str, bool]] = {}
    for instrument_name, usage in instruments_to_use.items():
        if usage["use"] is True:
            instruments_data_date_range = INSTRUMENTS_DATES.get(
                instrument_name, None
            )
            if instruments_data_date_range is None:
                valid_instruments_to_use[instrument_name] = {"use": False}
                log.error(
                    f"Valid data date range for instrument {instrument_name} has not been set"
                )
            else:
                instrument_data_start_date = validate_start_date(
                    str(min(instruments_data_date_range))
                )
                instrument_data_end_date = validate_end_date(
                    str(max(instruments_data_date_range))
                )
                # Check for overlap in requested date range and the
                # date range that data is available for the instrument, if
                # there is no overlap disable instrument.
                if (
                    instrument_data_end_date >= start_date
                    and instrument_data_start_date <= end_date
                ):
                    valid_instruments_to_use[instrument_name] = {"use": True}
                else:
                    valid_instruments_to_use[instrument_name] = {"use": False}
                    log.error(
                        f"Instrument {instrument_name} has data for the date range "
                        f"{instrument_data_start_date} to {instrument_data_end_date} which is outside"
                        f" the requested date range of {start_date} to {end_date}."
                    )
        else:
            valid_instruments_to_use[instrument_name] = usage
    return valid_instruments_to_use


def get_instruments_list(
    instruments_to_use: dict[str, dict[str, bool]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Primary list of instruments, measurements, and interoperable variable names

    Parameters
    ----------
    instruments_to_use : dict[str, dict[str, bool]]
        A dictionary of the selected instruments to use for the analysis and
        a usage parameter showing whether it will be used in the analysis.

    Returns
    -------
    dict
        A dictionary containing :
        - 'instruments', the 'master list' of instruments being used in the analysis.
            This is a subsetof the full list available.
    """
    instruments_list = {}
    for instrument_name, instrument_usage in instruments_to_use.items():
        if instrument_usage["use"] is True:
            instrument_info = INSTRUMENTS_MEASUREMENTS[instrument_name]
            instruments_list[instrument_name] = instrument_info
    return instruments_list
