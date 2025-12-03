import logging

import xarray as xr

log = logging.getLogger(__name__)

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

REFERENCE_MEAN = {"msi_agm": 0.0970, "oli_agm": 0.1015, "tm_agm": 0.0962}
THRESHOLD = {"msi_agm": 0.05, "oli_agm": 0.05, "tm_agm": 0.05}


def FAI(ds: xr.Dataset, instrument: str) -> xr.DataArray:
    """
    Calculate the Floating Algae Index (FAI) for a given instrument.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the spectral bands.
    instrument : str
        Name of the satellite/instrument (e.g., 'msi', 'oli', 'tm',
        'msi_agm', etc.).

    Returns
    -------
    xr.DataArray
        Floating Algae Index values.
    """
    if instrument not in CENTRAL_WAVELENGTHS.keys():
        log.error(
            f"Invalid instrument '{instrument}'. Returning empty DataArray."
        )
        return xr.DataArray(data=[], dims=["time"], coords={"time": []})
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
        m = float(m)

        # Calculate the baseline reflectance at NIR wavelength
        interpolated_nir = ds[red_band] + (ds[swir_band] - ds[red_band]) * m
        # Scale by 10000 assuming input reflectance is in 0-10000 range
        scale = float(10000)
        fai = (ds[nir_band] - interpolated_nir) / scale
        fai.name = "FAI"
        return fai


def geomedian_FAI(
    annual_data: dict[str, xr.Dataset],
    water_mask: xr.DataArray,
    compute: bool = False,
) -> xr.Dataset:
    """
    Calculate the FAI (Floating Algae Index) across multiple instruments
    and produce a combined weighted mean FAI for water pixels.

    Parameters
    ----------
    annual_data : dict[str, xr.Dataset]
        A dictionary mapping instruments to the xr.Dataset of the loaded
        annual (geomedian) datacube datasets available for that
        instrument.
    water_mask : xr.DataArray
        Water mask to apply for masking non-water pixels, where 1
        indicates water.
    compute : bool
        Whether to compute the dask arrays immediately, by default False.
        Set to False to keep datasets lazy for memory efficiency.
    Returns
    -------
    xr.Dataset
        A Dataset with the following bands:
        - '{instrument}_fai' for each geomedian instrument available.
        - 'agm_fai' (the final weighted average FAI, masked).
    """
    # Keep this order for consistent processing.
    geomedian_fai_instruments = ["msi_agm", "oli_agm", "tm_agm"]
    loaded_instruments = list(annual_data.keys())

    if set(geomedian_fai_instruments).isdisjoint(loaded_instruments) is True:
        error = (
            "The Geomedian FAI requires data for at least one instrument "
            f"from: {', '.join(geomedian_fai_instruments)} .",
            "Returning an empty Dataset.",
        )
        log.error(error)
        return xr.Dataset()

    log.info("Calculating Geomedian FAI for available instruments ...")
    fai_ds = xr.Dataset()
    all_inst_fai_list = []
    all_inst_count_list = []

    for inst in geomedian_fai_instruments:
        if inst in loaded_instruments:
            log.info(f"\tCalculating FAI for instrument: {inst} ...")
            inst_ds = annual_data[inst]
            count_band = f"{inst}_count"
            scale = REFERENCE_MEAN["msi_agm"] / REFERENCE_MEAN[inst]

            fai_da = FAI(inst_ds, inst) * scale
            fai_ds[f"{inst}_fai"] = fai_da
            all_inst_fai_list.append(fai_da)
            all_inst_count_list.append(inst_ds[count_band])

    all_inst_fai = xr.concat(all_inst_fai_list, dim="instrument")
    all_inst_count = xr.concat(all_inst_count_list, dim="instrument")
    weighted_fai_sum = (all_inst_fai * all_inst_count).sum(dim="instrument")
    all_inst_count_total = all_inst_count.sum(dim="instrument")
    mean_fai = (
        weighted_fai_sum.where(all_inst_count_total != 0)
        / all_inst_count_total
    )

    # Trim the fai values back to relevant areas and values
    fai_ds["agm_fai"] = mean_fai
    fai_ds = fai_ds.where(fai_ds > THRESHOLD["msi_agm"]).where(water_mask == 1)
    del (
        all_inst_fai_list,
        all_inst_count_list,
        all_inst_fai,
        all_inst_count,
        weighted_fai_sum,
        all_inst_count_total,
        mean_fai,
    )
    if compute:
        log.info("\tComputing FAI dataset ...")
        fai_ds = fai_ds.compute()
    log.info("Geomedian FAI calculation complete.")
    return fai_ds
