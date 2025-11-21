import logging

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def hue_adjust_parameters(instrument: str = None) -> pd.DataFrame:
    """A function to set the hue adjustment parameters for a
    instrument; a quintic polynomial model.

    Returns
    -------
    pd.DataFrame
        DataFrame of hue adjustment parameters for the instrument if instrument
        is specified, else a DataFrame of all instruments and their
        parameters.
    """
    df = pd.DataFrame(
        data={
            "label": ["Resolution", "a5", "a4", "a3", "a2", "a", "offset"],
            "msi": [20, -161.23, 1117.08, -2950.14, 3612.17, -1943.57, 364.28],
            "oli": [30, -52.16, 373.81, -981.83, 1134.19, -533.61, 76.72],
            "tm": [30, -84.94, 594.17, -1559.86, 1852.50, -918.11, 151.49],
        }
    )
    if instrument is not None:
        return df[["label", instrument]]
    else:
        return df


def hue_adjust(ds: xr.Dataset, instrument: str) -> xr.DataArray:
    """Make adjustments to the hue to produce the final
    hue values using sensor specific hue adjustment coefficients.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with the `hue` variable.

    Returns
    -------
    xr.Dataset
        Input dataset with the hue values adjusted.
    """
    if "hue" not in list(ds.data_vars):
        raise KeyError(
            "The input dataset does not contain the 'hue' variable."
        )

    # Keep this order for consistent processing.
    geomedian_instruments = ["msi_agm", "oli_agm", "tm_agm"]
    if instrument in geomedian_instruments:
        inst = instrument.split("_")[0]
    else:
        inst = instrument

    hap = hue_adjust_parameters(inst)
    labels = ["a5", "a4", "a3", "a2", "a", "offset"]
    coefficients = hap.loc[hap.label.isin(labels), inst].values

    ds["hue_delta"] = (
        (ds["hue"] / 100) ** 5 * coefficients[0]
        + (ds["hue"] / 100) ** 4 * coefficients[1]
        + (ds["hue"] / 100) ** 3 * coefficients[2]
        + (ds["hue"] / 100) ** 2 * coefficients[3]
        + (ds["hue"] / 100) ** 1 * coefficients[4]
        + (ds["hue"] / 100) ** 0 * coefficients[5]
    )
    ds["hue"] = ds["hue"] + ds["hue_delta"]
    ds = ds.drop_vars("hue_delta")

    return ds


def chromatic_coefficient_parameters(instrument: str = None) -> pd.DataFrame:
    """
    A function to set the chromatic coefficient parameters for a instrument.

    Parameters
    ----------
    instrument : str
        Name of instrument to retrieve parameters for.

    Returns
    -------
    pd.DataFrame
        Chromatic coefficient parameters for the instrument if instrument
        is specified, else a dictionary of all instruments and their
        parameters.
    """
    msi = pd.DataFrame(
        {
            "nm": ["R400", "R490", "R560", "R665", "R705", "R710"],
            "band": ["", "2", "3", "4", "5", ""],
            "name": ["", "msi02", "msi03", "msi04", "msi05", ""],
            "X": [8.356, 12.040, 53.696, 32.028, 0.529, 0.016],
            "Y": [0.993, 23.122, 65.702, 16.808, 0.192, 0.006],
            "Z": [43.487, 61.055, 1.778, 0.015, 0.000, 0.000],
        },
    )

    oli = pd.DataFrame(
        {
            "nm": ["R400", "R443", "R482", "R561", "R655", "R710"],
            "band": ["", "1", "2", "3", "4", ""],
            "name": ["", "oli01", "oli02", "oli03", "oli04", ""],
            "X": [2.217, 11.053, 6.950, 51.135, 34.457, 0.852],
            "Y": [0.082, 1.320, 21.053, 66.023, 18.034, 0.311],
            "Z": [10.745, 58.038, 34.931, 2.606, 0.016, 0.000],
        }
    )

    tm = pd.DataFrame(
        {
            "nm": ["R400", "R485", "R565", "R660", "R710"],
            "band": ["", "1", "2", "3", ""],
            "name": ["", "tm01", "tm02", "tm03", ""],
            "X": [7.8195, 13.104, 53.791, 31.304, 0.6463],
            "Y": [0.807, 24.097, 65.801, 15.883, 0.235],
            "Z": [40.336, 63.845, 2.142, 0.013, 0.000],
        }
    )
    all_coeffs = {"msi": msi, "oli": oli, "tm": tm}
    if instrument is not None:
        return all_coeffs[instrument]
    else:
        return all_coeffs


def hue_calculation(
    ds: xr.Dataset, instrument: str, rayleigh_corrected_data: bool = True
) -> xr.DataArray:
    """Calculate the hue by conversion of the wavelengths
    to chromatic coordinates using sensor-specific coefficients.
    Method is as per Van Der Woerd 2018.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to calculate the hue for.
    instrument : str
        Sensor to calculate the hue for.
    rayleigh_corrected_data : bool
        Whether the input data has been Rayleigh corrected.
    Returns
    -------
    xr.DataArray
        The derived hue.
    """
    # Notes on hue: More accurate hue angles are retrieved if more bands
    # are used - but the visible bands are most important results for
    # ETM+ are therefore  less accurate than for MSI and OLI?
    # The OLI geomedian lacks band 1, so it cannot be used. This leaves
    # a gap in the data. Examiation of a time series shows clear patterns.
    # Oli data give lower values than msi and tm, which are in good agreement

    log.info(f"Calculating the hue for the instrument: {instrument}")
    # Keep this order for consistent processing.
    geomedian_instruments = ["msi_agm", "oli_agm", "tm_agm"]

    if instrument in geomedian_instruments:
        inst = instrument.split("_")[0]
        agm = True
    else:
        inst = instrument
        agm = False

    ccs = chromatic_coefficient_parameters(inst)
    # Get the required bands to calculate hue
    required_bands = [i for i in ccs["name"].to_list() if i != ""]

    # Determine the available bands
    ds_bands = []
    for band_name in required_bands:
        if agm:
            band_name = f"{band_name}_agm"
        if rayleigh_corrected_data:
            band_name = f"{band_name}r"
        if band_name in ds.data_vars:
            ds_bands.append(band_name)

    if len(ds_bands) != len(required_bands):
        log.error(
            f"Aborting hue calculation for instrument {instrument} "
            "due to lack of necessary data bands"
        )
        hue = xr.full_like(
            ds[list(ds.data_vars)[0]], fill_value=np.nan
        ).rename("hue")
        return hue

    Cdata = xr.Dataset()
    for XYZ in ["X", "Y", "Z"]:
        Cdata[XYZ] = xr.zeros_like(ds[list(ds.data_vars)[0]]).rename("XYZ")
        for band in required_bands:
            ds_band = ds_bands[required_bands.index(band)]
            coeff = ccs[ccs.name == band][XYZ].values
            Cdata[XYZ] = ds[ds_band] * coeff + Cdata[XYZ]

    Cdata["XYZ"] = Cdata["X"] + Cdata["Y"] + Cdata["Z"]
    Xwhite = Ywhite = 1 / 3.0

    # Normalise the X and Y parameters and conver to a delta from white:
    Cdata["X"] = Cdata["X"] / Cdata["XYZ"] - Xwhite
    Cdata["Y"] = Cdata["Y"] / Cdata["XYZ"] - Ywhite
    Cdata["Z"] = Cdata["Z"] / Cdata["XYZ"]

    # Convert vector to angle
    Cdata["hue"] = np.mod(
        np.arctan2(Cdata["Y"], Cdata["X"]) * (180.00 / np.pi) + 360.0, 360
    )
    Cdata = hue_adjust(Cdata, instrument)

    # This gives the correct mathematical angle, ie. from 0 (=east),
    # counter-clockwise as a positive number.
    # Note the 'arctan2' function, and that x and y are switched
    # compared to expectations
    return Cdata.hue


def geomedian_hue(ds: xr.Dataset) -> xr.Dataset:
    """
    Generate the mean weighted geomedian hue from the
    available geomedian instruments in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to calculate the geomedian hue for.

    Returns
    -------
    xr.Dataset
        Input dataset with the geomedian hue added.
    """
    # Keep this order for consistent processing.
    geomedian_instruments = ["msi_agm", "oli_agm", "tm_agm"]

    mean_hue_weighted_sum = None
    agm_count_total = None

    for inst_agm in geomedian_instruments:
        # Use the smad band as an indicator that data for the geomedian
        # instrument exists in the dataset.
        smad_band = f"{inst_agm}_smad"
        if smad_band in ds.data_vars:
            # Calculate the hue for this sensor
            hue_data = hue_calculation(ds, inst_agm)

            count_band = f"{inst_agm}_count"
            # Replace all NaN values with 0s in count band.
            inst_count = ds[count_band].fillna(0)
            # Negate the counts where there is no data produced
            inst_count = inst_count.where(~np.isnan(hue_data), 0)
            # Replace all NaN values with 0s in hue data
            hue_data = hue_data.fillna(0)

            weighted_hue = hue_data * inst_count

            # Aggregate the weighted hue and the total count
            if mean_hue_weighted_sum is None:
                mean_hue_weighted_sum = weighted_hue
                agm_count_total = inst_count
            else:
                mean_hue_weighted_sum += weighted_hue
                agm_count_total += inst_count

            # Mask to only include water pixels.
            hue_data = hue_data.where(ds["water_mask"] == 1)
            # Add the instrument-specific masked hue to the Dataset
            ds[f"{inst_agm}_hue"] = hue_data

    if mean_hue_weighted_sum is not None and agm_count_total is not None:
        # Avoid division by zero:
        agm_count_total = agm_count_total.where(agm_count_total != 0)
        mean_hue = mean_hue_weighted_sum / agm_count_total
        # Mask to only include water pixels.
        mean_hue = mean_hue.where(ds["water_mask"] == 1)
        # Trim extreme values that can arise
        ds["agm_hue"] = xr.where(
            mean_hue > 25, xr.where(mean_hue < 100, mean_hue, np.nan), np.nan
        )
    return ds
