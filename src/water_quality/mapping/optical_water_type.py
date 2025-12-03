import logging
from importlib.resources import files

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def get_OWT_description(owt):
    """
    Get description of a OWT value
    """
    if not (1 <= owt <= 13):
        raise ValueError("OWT value must be between 1 and 13")

    data = [
        [3, "oligotrophic (clear)"],
        [9, "oligotrophic (clear)"],
        [13, "oligotrophic (clear)"],
        [1, "eutrophic and blue-green"],
        [2, "eutrophic and blue-green"],
        [4, "eutrophic and blue-green"],
        [5, "eutrophic and blue-green"],
        [11, "eutrophic and blue-green"],
        [12, "eutrophic and blue-green"],
        [6, "hyper-eutrophic and green-brown"],
        [7, "hyper-eutrophic and green-brown"],
        [8, "hyper-eutrophic and green-brown"],
        [10, "hyper-eutrophic and green-brown"],
    ]
    data.sort()
    columns = ["OWT", "description"]

    df = pd.DataFrame(data=data, columns=columns)
    desc = df[df["OWT"] == owt]["description"].item()
    return desc


def create_OWT_response_model() -> dict[str, pd.DataFrame]:
    """
    Create Optical Water Type (OWT) response models and write to
    csv files for all supported instruments.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary of DataFrames containing OWT response models for each
        instrument
    """
    sensors = [
        "msi",
        "tm",
        "oli",
    ]

    OWT_spectra_fp = files("water_quality.data").joinpath(
        "Vagelis_OWT_allwaters_mean_standardised.csv"
    )
    OWT_spectra = pd.read_csv(OWT_spectra_fp)
    # Limit to inland water types
    OWT_spectra = OWT_spectra[OWT_spectra.index < 13]

    data = {}

    # Read in the wavelengths for the bands in oli, tm and msi
    # (spectral response models)
    for sensor in sensors:
        # (spectral response models)
        sensor_data_fp = files("water_quality.data").joinpath(
            f"sensor bands-{sensor}.csv"
        )
        sensor_data = pd.read_csv(sensor_data_fp)
        sensor_data["band_name"] = sensor_data["band_name"]

        # Rename columns to avoid clashes with reserved words
        sensor_data.rename(
            columns={"max": "l_max", "min": "l_min"}, inplace=True
        )

        # set up a  dataframe to take results for this instrument
        owt_values = OWT_spectra["wl"].to_list()
        owt_labels = [f"OWT-{i}" for i in owt_values]
        inst_OWT = pd.DataFrame(data={"OWT": owt_values}, index=owt_labels)

        # list the bands  that are going to be relevant , ie., less than 800nm
        band_list = sensor_data[sensor_data["l_max"] < 800][
            "band_name"
        ].to_list()

        for band_name in band_list:
            band_data = sensor_data[
                sensor_data["band_name"] == band_name
            ].iloc[0]

            # Determine the integration interval based on the central
            # wavelength and the width
            delta = band_data["width"] * 0.8 * 0.5
            central = band_data["central"]

            # Find start and end column numbers in the response functions
            # and average over those for each OWT, must allow for the
            # spectral band width.
            start = str(int(central - delta))
            end = str(int(central + delta))
            inst_OWT.insert(
                inst_OWT.columns.size,
                band_name,
                OWT_spectra.loc[:, start:end].T.mean().to_list(),
            )
        inst_OWT.to_csv(
            files("water_quality.data").joinpath(f"{sensor}_OWT_vectors.csv")
        )
        log.info(f"Written OWT response model for {sensor} to csv file.")
        data[sensor] = inst_OWT
    return data


def OWT(
    ds: xr.Dataset,
    instrument: str,
    OWT_vectors: pd.DataFrame,
    agm: bool = False,
    check_type: bool = False,
) -> xr.DataArray:
    """
    Calculate per-pixel Optical Water Type by comparing to OWT
    spectral vectors.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing spectral band data.

    instrument: str
        Name of instrument.

    OWT_vectors: pd.DataFrame
        OWT spectral vectors, expected to have an index column formated
        as OWT-index.

    agm: bool, optional
        Whether the calculation is for geomedian instruments.
        Default is False.

    check_type: bool, optional
        Whether to check the prevailing water type in intermediate steps.
        Default is False.

    Returns
    -------
    xr.DataArray
        DataArray containing OWT estimate.
    """

    # --- this version uses a long-hand approach to calculating the dot product. Inelegant but simple.
    # --- identify the instrument bands that are relevant and in the dataset
    log.info(
        f"Calculating Optical Water Type (OWT) for instrument: {instrument}"
    )
    suffix = ""
    if agm:
        suffix = suffix + "_agm"

    # This loop renames the columns so that we can match them with the
    # data variables
    for col in OWT_vectors.columns.values:
        if col.find(instrument) > -1:
            OWT_vectors = OWT_vectors.rename(columns={col: col + suffix})
    band_list = list(set(ds.data_vars) & set(OWT_vectors.columns.values))
    band_list.sort()

    # Ditch unnecessary columns and rows in the vectors table and
    # calculate the magnitude of each vector
    OWT_vectors = OWT_vectors.drop(
        columns=list(set(OWT_vectors.columns.values) - set(band_list))
    )
    OWT_vectors["length"] = (
        (OWT_vectors**2).sum(axis=1)
    ) ** 0.5  # calculate the length of each vector

    # --- loop through the Optical water types ---

    # initiate variables to hold current best OWT match & largest cos
    varname = band_list[0]
    owt_cos_max = xr.full_like(
        ds[varname], dtype=float, fill_value=-2
    )  # a number smaller than any cosine
    owt_closest = xr.full_like(
        ds[varname],
        dtype="uint8",
        fill_value=int(OWT_vectors.index[0].split("-")[1]),
    )

    for idx, vec in OWT_vectors.iterrows():
        OWT_index = int(idx.split("-")[1])  # the OWT number
        # print(OWT_index)
        OWT_index = int(idx.split("-")[1])  # the OWT number
        # turn into a data arrays
        da = ds[band_list].to_array(dim="band")
        vec_da = xr.DataArray(vec, dims=["band"]).sel(band=band_list)
        # vectorised computation
        self_product = (da**2).sum(dim="band") ** 0.5
        vector_product = (da * vec_da).sum(dim="band")
        owt_cos = vector_product / (self_product * vec["length"])

        # now find the closest with the largest cos
        owt_closest = xr.where(
            owt_cos > owt_cos_max,
            OWT_index,
            owt_closest,
        )

        owt_cos_max = xr.where(
            owt_cos > owt_cos_max,
            owt_cos,
            owt_cos_max,
        )

        if check_type:
            # only use for debugging
            owt = owt_closest.median().item()
            desc = get_OWT_description(owt)
            print("Prevailng water type is ", owt, " :  ", desc)
    return owt_closest


def run_OWT(
    instrument_data: dict[str, xr.Dataset],
    clear_water_mask: xr.DataArray,
    compute: bool = False,
) -> xr.Dataset:
    """
    A function to run the Optical Water Type (OWT) classification on a dataset.

    Parameters
    ----------
    annual_data : dict[str, xr.Dataset]
        A dictionary mapping instruments to the xr.Dataset of the loaded
        annual (geomedian) datacube datasets available for that
        instrument.
    clear_water_mask : xr.DataArray
        Water mask to apply for masking non-water pixels, where 1
        indicates water.
    compute : bool
        Whether to compute the dask arrays immediately, by default False.
        Set to False to keep datasets lazy for memory efficiency.

    Returns
    -------
    xr.Dataset
        Dataset with OWT classification results.
    """
    owt_results = xr.Dataset()
    loaded_instruments = list(instrument_data.keys())
    for instrument in loaded_instruments:
        # Prevent attempts to run OWT on wofs_ann and tirs
        if instrument in ["msi", "msi_agm", "tm", "tm_agm", "oli", "oli_agm"]:
            log.info(
                f"Running OWT classification for instrument: {instrument} ..."
            )
            ds = instrument_data[instrument]
            if instrument.endswith("_agm"):
                inst = instrument.split("_")[0]
                agm = True
                varname = inst + "_agm_owt"
            else:
                inst = instrument
                agm = False
                varname = inst + "_owt"

            OWT_vectors = pd.read_csv(
                files("water_quality.data").joinpath(
                    f"{inst}_OWT_vectors.csv"
                ),
                index_col=0,
            )
            owt_results[varname] = OWT(
                ds.where(clear_water_mask == 1),
                inst,
                OWT_vectors,
                agm=agm,
            )

    if list(owt_results.data_vars):
        # Keep this order.
        geomedian_instruments = ["tm_agm", "oli_agm", "msi_agm"]
        agm_owt = None
        for inst in geomedian_instruments:
            if inst in loaded_instruments:
                if agm_owt is None:
                    agm_owt = owt_results[inst + "_owt"]
                else:
                    agm_owt = xr.where(
                        ~np.isnan(owt_results[inst + "_owt"]),
                        owt_results[inst + "_owt"],
                        agm_owt,
                    )

        owt_results["agm_owt"] = agm_owt
        # TODO: Add agm_owt calculation for non-agm instruments?

        # Mask again using clear water mask because OWT calculation
        # may assign values to non-water pixels.
        # TODO: Check why this is necessary.
        owt_results = owt_results.where(clear_water_mask == 1)

        if compute:
            log.info("\tComputing OWT  ...")
            owt_results = owt_results.compute()
        log.info("OWT classification complete.")

    return owt_results
