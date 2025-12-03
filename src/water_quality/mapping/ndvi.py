import logging

import xarray as xr

log = logging.getLogger(__name__)

NDVI_BANDS = {
    "tm_agm": {"red": "tm03_agm", "nir": "tm04_agm"},
    "oli_agm": {"red": "oli04_agm", "nir": "oli05_agm"},
    "msi_agm": {"red": "msi04_agm", "nir": "msi8a_agm"},
}

REFERENCE_MEAN = {"msi_agm": 0.2335, "oli_agm": 0.2225, "tm_agm": 0.2000}
THRESHOLD = {"msi_agm": 0.05, "oli_agm": 0.05, "tm_agm": 0.05}


def geomedian_NDVI(
    annual_data: dict[str, xr.Dataset],
    water_mask: xr.DataArray,
    compute: bool = False,
) -> xr.DataArray:
    """
    Calculate the NDVI (Normalized Difference Vegetation Index) across
    multiple instruments and produce a combined weighted mean NDVI
    for water pixels.

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
    xr.DataArray
        The updated input xarray Dataset with new bands:
        - '{instrument}_ndvi' for each processed instrument (masked).
        - 'agm_ndvi' (the final weighted average NDVI, masked).
    """

    # Keep this order for consistent processing.
    geomedian_ndvi_instruments = ["msi_agm", "oli_agm", "tm_agm"]
    loaded_instruments = list(annual_data.keys())

    if set(geomedian_ndvi_instruments).isdisjoint(loaded_instruments) is True:
        error = (
            "The Geomedian NDVI requires data for at least one instrument "
            f"from: {', '.join(geomedian_ndvi_instruments)} .",
            "Returning an empty Dataset.",
        )
        log.error(error)
        return xr.Dataset()

    log.info("Calculating Geomedian NDVI for available instruments ...")
    ndvi_ds = xr.Dataset()
    all_inst_ndvi_list = []
    all_inst_count_list = []

    for inst in geomedian_ndvi_instruments:
        if inst in loaded_instruments:
            log.info(f"\tCalculating NDVI for instrument: {inst} ...")
            # Calculate the NDVI for the instrument and scale
            inst_ds = annual_data[inst]
            count_band = f"{inst}_count"
            inst_bands = NDVI_BANDS[inst]
            red_band = inst_bands["red"]
            nir_band = inst_bands["nir"]
            ndvi_da = (inst_ds[nir_band] - inst_ds[red_band]) / (
                inst_ds[nir_band] + inst_ds[red_band]
            )
            scale = REFERENCE_MEAN["msi_agm"] / REFERENCE_MEAN[inst]
            ndvi_da = ndvi_da * scale

            ndvi_ds[f"{inst}_ndvi"] = ndvi_da
            all_inst_ndvi_list.append(ndvi_da)
            all_inst_count_list.append(inst_ds[count_band])

    all_inst_ndvi = xr.concat(all_inst_ndvi_list, dim="instrument")
    all_inst_count = xr.concat(all_inst_count_list, dim="instrument")
    weighted_ndvi_sum = (all_inst_ndvi * all_inst_count).sum(dim="instrument")
    all_inst_count_total = all_inst_count.sum(dim="instrument")
    mean_ndvi = (
        weighted_ndvi_sum.where(all_inst_count_total != 0)
        / all_inst_count_total
    )
    # Trim the ndvi values back to relevant areas and values
    ndvi_ds["agm_ndvi"] = mean_ndvi
    ndvi_ds = ndvi_ds.where(ndvi_ds > THRESHOLD["msi_agm"]).where(
        water_mask == 1
    )
    del (
        all_inst_ndvi_list,
        all_inst_count_list,
        all_inst_ndvi,
        all_inst_count,
        weighted_ndvi_sum,
        all_inst_count_total,
        mean_ndvi,
    )
    if compute:
        log.info("\tComputing NDVI dataset ...")
        ndvi_ds = ndvi_ds.compute()
    log.info("Geomedian NDVI calculation complete.")
    return ndvi_ds
