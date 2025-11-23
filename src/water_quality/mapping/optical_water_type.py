import gc

import numpy as np
import xarray as xr

# Estimated spectra for each optical water type for each sensor (MSI, OLI, TM)
# calculated from full spectra table provided by Vagelis Spyrakos
# columns are bands, rows are OWT
OWT_LIST = [
    "owt1",
    "owt2",
    "owt3",
    "owt4",
    "owt5",
    "owt6",
    "owt7",
    "owt8",
    "owt9",
    "owt10",
    "owt11",
    "owt12",
    "owt13",
]

# msi bands: 1, 2, 3, 4, 5, 6, 7
OWT_DATA_MSI = np.asarray(
    [
        0.000518281,
        0.002873764,
        0.003495403,
        0.001515508,
        0.002204739,
        0.003842756,
        0.005026098,
        0.001310923,
        0.007407241,
        0.007373993,
        0.003239791,
        0.001309939,
        0.000351881,
        0.00044227,
        0.002720793,
        0.012390417,
        0.00737146,
        0.001458816,
        0.000422552,
        0.000142766,
        0.00018326,
        0.001011837,
        0.006135417,
        0.006416972,
        0.00401114,
        0.001825363,
        0.000454228,
        0.000554565,
        0.001368776,
        0.005729538,
        0.004259349,
        0.00409354,
        0.001965091,
        0.001276867,
        0.001526747,
        0.000947881,
        0.005756157,
        0.006822492,
        0.003394953,
        0.002302141,
        0.000694368,
        0.000822148,
        0.000720924,
        0.003734303,
        0.004729343,
        0.002251117,
        0.003381287,
        0.002206305,
        0.002708644,
        0.000830314,
        0.004805815,
        0.005933108,
        0.003108917,
        0.002882319,
        0.001181092,
        0.001454993,
        0.001877709,
        0.009255961,
        0.007552662,
        0.002441745,
        0.000847245,
        0.000279124,
        0.000375539,
        0.000842475,
        0.00240552,
        0.002787709,
        0.005383652,
        0.003098927,
        0.001542782,
        0.002194582,
        0.000746178,
        0.004633521,
        0.005087786,
        0.00500395,
        0.002445958,
        0.000762379,
        0.000962499,
        0.001439293,
        0.006599122,
        0.005582806,
        0.003590672,
        0.001796926,
        0.000768249,
        0.000934859,
        0.006322908,
        0.014258851,
        0.002314198,
        0.000275429,
        8.56521e-05,
        6.41172e-05,
        0.000110569,
    ]
).reshape(len(OWT_LIST), 7)

# oli bands: 1,2,3,4
OWT_DATA_OLI = np.asarray(
    [
        0.000536567,
        0.000860402,
        0.002160328,
        0.001178919,
        0.001346095,
        0.002267667,
        0.004757344,
        0.002459032,
        0.002818286,
        0.0041035,
        0.004739951,
        0.001088903,
        0.001034619,
        0.001850067,
        0.004198279,
        0.003021194,
        0.001429095,
        0.001886083,
        0.002822574,
        0.00284571,
        0.000971,
        0.001697917,
        0.004364393,
        0.002620484,
        0.000754857,
        0.001114315,
        0.002976344,
        0.001750548,
        0.000856452,
        0.00142315,
        0.003775541,
        0.002414677,
        0.001945476,
        0.002931367,
        0.004860984,
        0.001833516,
        0.000936538,
        0.000772627,
        0.001869377,
        0.003541968,
        0.000764871,
        0.001385817,
        0.00335777,
        0.003580806,
        0.001499333,
        0.002115267,
        0.003644426,
        0.00264,
        0.006800333,
        0.00588125,
        0.001539375,
        0.000199513,
    ]
).reshape(len(OWT_LIST), 4)

# Zhang et al.
OWT_GROUPS = {
    "oligotrophic": [3, 9, 13],
    "eutropic and blue-green": [1, 2, 4, 5, 11, 12],
    "hypereutrophic and green-brown": [6, 7, 8, 10],
}

# 6 of 7 useful bands (missing band 1)
EXPECTED_MSI_BANDS = ["msi02", "msi03", "msi04", "msi05", "msi06", "msi07"]
# (2,3,4) - missing band 1
EXPECTED_OLI_BANDS = ["oli02", "oli03", "oli04"]
# TM has 4 (all of 1,2,3,4)
# EXPECTED_TM_BANDS = ["tm01", "tm02" , "tm03", "tm04"]


def get_owt_reference_data(instrument: str) -> xr.DataArray:
    # Make a DataArray containing the OWT reference data
    # OWT types are stored in their own dimension to support vector multiplication
    if "msi" in instrument:
        msi_bands = [
            "msi01",
            "msi02",
            "msi03",
            "msi04",
            "msi05",
            "msi06",
            "msi07",
        ]
        owt_msi = xr.DataArray(
            OWT_DATA_MSI,
            dims=("owt", "msi_band"),
            coords={
                "owt": OWT_LIST,
                "msi_band": msi_bands,
            },
            attrs={
                "desc": "optical water types - characteristic reflectances for msi"
            },
        )
        return owt_msi
    elif "oli" in instrument:
        oli_bands = ["oli01", "oli02", "oli03", "oli04"]
        owt_oli = xr.DataArray(
            OWT_DATA_OLI,
            dims=("owt", "oli_band"),
            coords={
                "owt": OWT_LIST,
                "oli_band": oli_bands,
            },
            attrs={
                "desc": "optical water types - characteristic reflectances for oli"
            },
        )
        return owt_oli
    else:
        raise NotImplementedError(
            f"OWT reference data for the instrument {instrument} is not available"
        )


def OWT_pixel(
    ds: xr.Dataset,
    instrument: str,
    resample_rate: int | None = None,
) -> xr.DataArray:
    """Determine the open water type for each pixel, over areas
    that are usually water.

    Parameters
    ----------
    ds : xr.Dataset
        _description_
    instrument : str
        Selected instrument established while building the dataset.

    Returns
    -------
    xr.DataArray
        Open Water Type for each pixel, over areas that are usually water.
    """
    shortname = instrument[0:3]

    OWT = get_owt_reference_data(instrument)

    if instrument == "msi_agm":
        # Select the bands that have undergone dark pixel correction
        # i.e. have the suffix "r".
        suffix = "_agmr"
        band_list = EXPECTED_MSI_BANDS
        dim_name = "msi_band"
        var_name = "msi_vals"
    elif instrument == "oli_agm":
        suffix = "_agmr"
        band_list = EXPECTED_OLI_BANDS
        dim_name = "oli_band"
        var_name = "oli_vals"
    else:
        raise NotImplementedError(
            f"OWT for the instrument {instrument} is not available"
        )

    # Create a dataset whose coordinates match the input dataset.
    resampled_ds = xr.Dataset(coords=ds.coords)

    # Resample the dataset if rate provided.
    if resample_rate is not None:
        resampled_ds = resampled_ds.isel(
            y=slice(None, None, resample_rate),
            x=slice(None, None, resample_rate),
        )

    # Create a dimension for the surface reflectance data
    resampled_ds = resampled_ds.assign_coords({dim_name: band_list})
    data_stack_shape = (
        resampled_ds.time.size,
        resampled_ds.y.size,
        np.size(band_list),
        resampled_ds.x.size,
    )
    data_stack_bands = [i + suffix for i in band_list]
    # TODO: Check on memory consumption at this point
    data_stack = np.dstack(
        [
            # Select to cater for resampling
            ds[band].sel(
                x=resampled_ds.coords["x"], y=resampled_ds.coords["y"]
            )
            for band in data_stack_bands
        ]
    ).reshape(data_stack_shape)
    resampled_ds[var_name] = (("time", "y", dim_name, "x"), data_stack)
    # To allow the matrix multiplication the bands dimension needs to be
    # transposed to the end
    resampled_ds = resampled_ds.transpose("time", "y", "x", dim_name)

    # add a dimension for the OWT type
    OWT = OWT.sel({dim_name: resampled_ds[dim_name].values})
    resampled_ds = resampled_ds.assign_coords(owt=OWT.owt.values)

    # Calculate the dot product between each pixel vector and each
    # owt type vector
    resampled_ds[shortname + "_x_owt"] = resampled_ds[var_name].dot(
        OWT.T, dim=dim_name
    )
    # The result now has for every pixel, the dot product with the
    # spectral reference vector.

    # Calculate the self product (the scale) of each of the OWT
    # reference vectors
    owt_scale = np.sqrt(np.square(OWT).T.sum(dim=dim_name))

    # Calculate the scale of each pixel vector
    resampled_ds[shortname + "_scale"] = np.sqrt(
        np.square(resampled_ds[var_name]).sum(dim=dim_name)
    )

    # Get the cosine value
    resampled_ds[shortname + "_owt_cosine"] = (
        resampled_ds[shortname + "_x_owt"] / resampled_ds[shortname + "_scale"]
    ) / owt_scale

    # Find the owt closest to the msi vector (the largest cosine)
    # to avoid nan problems use np.argmax, then the array is  brought
    # back into the dataset
    resampled_ds[shortname + "_owt"] = (
        ("time", "y", "x"),
        np.argmax(resampled_ds[shortname + "_owt_cosine"].values, axis=3) + 1,
    )

    # replace zeros (where the scale of the pixel vector is zero)
    # with nodata
    resampled_ds[shortname + "_owt"] = xr.where(
        resampled_ds[shortname + "_scale"] > 0,
        resampled_ds[shortname + "_owt"],
        np.nan,
    )

    da = resampled_ds[shortname + "_owt"].copy(deep=True)
    del resampled_ds

    if resample_rate is not None:
        # Interpolate back to original grid
        fill_value = 14
        da = xr.where(np.isnan(da), fill_value, da)

        # Interpolate to the original coordinates,
        # and fill out any gaps including for years prior to the msi
        # sensor being available
        da = da.interp(
            coords={"time": ds.time, "x": ds.x, "y": ds.y},
            method="nearest",
            kwargs={"bounds_error": False, "fill_value": fill_value},
        )

        # Reduce coverage to water areas
        da = xr.where(~np.isnan(ds.watermask), da, np.nan)

        # Replace the fill values with median values for the pixel,
        # or year
        # glob_med = da.where(da != fill_value).median()
        pixel_med = da.where(da != fill_value).median(dim="time")
        annual_med = da.where(da != fill_value).median(dim=("x", "y"))
        da = xr.where(da == fill_value, pixel_med, da)
        da = xr.where(da == fill_value, annual_med, da)
    else:
        # Reduce coverage to water areas
        da = xr.where(~np.isnan(ds.watermask), da, np.nan)

    gc.collect()

    return da
