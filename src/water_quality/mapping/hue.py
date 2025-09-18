import logging

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# the x,y,z msi chromatic coefficients
CHROM_COEFFS = chrom_coeffs = {
    "X": {
        "msi01": 8.356,
        "msi02": 12.040,
        "msi03": 53.696,
        "msi04": 32.028,
        "msi05": 0.529,
    },  # x msi chromaticity
    "Y": {
        "msi01": 0.993,
        "msi02": 23.122,
        "msi03": 65.702,
        "msi04": 16.808,
        "msi05": 0.192,
    },
    "Z": {
        "msi01": 43.487,
        "msi02": 61.055,
        "msi03": 1.778,
        "msi04": 0.015,
        "msi05": 0.000,
    },
}
INSTRUMENT_HUE_BANDS = {
    "msi_agm": [
        # "msi01_agmr",
        "msi02_agmr",
        "msi03_agmr",
        "msi04_agmr",
        "msi05_agmr",
    ],
    "msi": ["msi01r", "msi02r", "msi03r", "msi04r", "msi05r"],
}


def hue_adjust(dataset: xr.Dataset) -> xr.Dataset:
    """Make adjustments to the hue to produce the final
    hue values using sensor specific hue adjustment coefficients.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset with the `hue` variable.

    Returns
    -------
    xr.Dataset
        Input dataset with the hue values adjusted.
    """
    if "hue" not in list(dataset.data_vars):
        raise KeyError("The input dataset does not contain a 'hue' variable.")

    # Hue adjustment coefficients for MSI
    deltahuemsi = (-161.23, 1117.08, -2950.14, 3612.17, -1943.57, 364.28)
    deltahueoli = (-52.16, 373.81, -981.83, 1134.19, -533.61, 76.72)
    deltahueetm = (-84.94, 594.17, -1559.86, 1852.50, -918.11, 151.49)
    dataset["hue_delta"] = (
        (dataset["hue"] / 100) ** 5 * deltahuemsi[0]
        + (dataset["hue"] / 100) ** 4 * deltahuemsi[1]
        + (dataset["hue"] / 100) ** 3 * deltahuemsi[2]
        + (dataset["hue"] / 100) ** 2 * deltahuemsi[3]
        + (dataset["hue"] / 100) ** 1 * deltahuemsi[4]
        + (dataset["hue"] / 100) ** 0 * deltahuemsi[5]
    )
    dataset["hue"] = dataset["hue"] + dataset["hue_delta"]
    dataset = dataset.drop_vars("hue_delta")
    return dataset


def hue_calculation(dataset: xr.Dataset, instrument: str) -> xr.DataArray:
    """Calculate the hue by converstion of the wavelengths
    to chromatic coordinates using sensor-specific coefficients.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset to calculate the hue for.
    instrument : str
        Sensor to calculate the hue for.

    Returns
    -------
    xr.Dataset
        The derived hue.

    """
    band_list = INSTRUMENT_HUE_BANDS.get(instrument, None)

    # Verify bands
    if band_list is None:
        raise NotImplementedError(
            f"Hue calculation for the instrument {instrument} is not available"
        )
    else:
        missing_bands = [
            i for i in band_list if i not in list(dataset.data_vars)
        ]
        if missing_bands:
            raise KeyError(
                f"Bands {', '.join(missing_bands)} missing in dataset for "
                "hue calculation."
            )

    # Hue calculation

    # Initiate two Datasets with no variables:
    Cdata = xr.zeros_like(dataset).drop_vars(dataset.data_vars)
    # Summary is not required in the per-pixel processing
    Cdata_summary = dataset.drop_dims(["x", "y"])

    n = 1
    s = np.array([], dtype=np.int8)

    for d in Cdata.dims:
        n = n * Cdata[d].size
        s = np.append(s, Cdata[d].size)

    for XYZ in chrom_coeffs.keys():
        Cdata[XYZ] = Cdata.dims, np.zeros(n).reshape(s)
        for var in band_list:
            var_shortname = var[0:5]
            Cdata[XYZ] = (
                Cdata[XYZ] + dataset[var] * chrom_coeffs[XYZ][var_shortname]
            )

    # ---- normalise the X and Y parameters
    Cdata["Xn"] = Cdata["X"] / (Cdata["X"] + Cdata["Y"] + Cdata["Z"])
    Cdata["Yn"] = Cdata["Y"] / (Cdata["X"] + Cdata["Y"] + Cdata["Z"])
    Xwhite = Ywhite = 1 / 3.0
    # ---- calculate the delta to white ----
    Cdata["Xnd"] = Cdata["Xn"] - Xwhite
    Cdata["Ynd"] = Cdata["Yn"] - Ywhite
    # ---- convert vector to angle ----
    Cdata["hue"] = np.mod(
        np.arctan2(Cdata["Ynd"], Cdata["Xnd"]) * (180.00 / np.pi) + 360.0, 360
    )

    # ---- this gives the correct mathematical angle, ie. from 0 (=east),
    # counter-clockwise as a positive number.
    # ---- note the 'arctan2' function, and that x and y are switched
    # compared to expectations

    # ---- code below is not used for pixel level processing, but is
    # used by others / later!
    Cdata_summary["Xnd"] = (
        Cdata["Xnd"]
        .where(dataset["wofs_ann_freq"] > 0.9)
        .median(dim=("x", "y"))
    )
    Cdata_summary["Ynd"] = (
        Cdata["Ynd"]
        .where(dataset["wofs_ann_freq"] > 0.9)
        .median(dim=("x", "y"))
    )
    Cdata_summary["hue"] = np.mod(
        np.arctan2(Cdata_summary["Ynd"], Cdata_summary["Xnd"])
        * (180.00 / np.pi)
        + 360.0,
        360,
    )
    # apply the hue adjustment - only do it once!
    log.info(
        "Average Hue values pre-adjustment: "
        f"{Cdata_summary['hue'].values.round(1)}"
    )
    Cdata = hue_adjust(Cdata)
    Cdata_summary = hue_adjust(Cdata_summary)
    log.info(
        "Average Hue values post-ajustment: "
        f"{Cdata_summary['hue'].values.round(1)}"
    )
    # The summary output is not required for pixel-level processing,
    # but could be used later.
    # Cdata_summary["hue"]

    return Cdata["hue"]
