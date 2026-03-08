import logging

import numpy as np
import xarray as xr
from odc.geo.xr import xr_reproject

from water_quality.dates import year_to_dc_datetime
from water_quality.utils import enforce_float32

log = logging.getLogger(__name__)


def water_temperature(
    annual_data: dict[str, xr.Dataset],
    water_mask: xr.DataArray,
    compute: bool,
) -> xr.Dataset:
    """Load and process data for the `tirs` instrument to produce an
    annual composite of surface temperature for water pixels.

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
        An xarray Dataset containing the surface temperature annual
        composite produced from data for the instrument `tirs`.
    """
    log.info("Processing water temperature annual composite ...")

    inst = "tirs"
    if inst not in list(annual_data.keys()):
        error = (
            f"No datasets found for instrument '{inst}'. "
            "Cannot generate water temperature annual composite. "
            "Returning nan filled xarray.Dataset."
        )

        log.error(error)

        empty_da = xr.full_like(
            water_mask, fill_value=np.nan, dtype=np.float32
        )
        attrs = {
            "nodata": np.nan,
            "scales": 1,
            "offsets": 0,
            "units": "Celsius",
        }
        empty_da.attrs.update(**attrs)

        annual_ds_tirs = xr.Dataset(
            {
                "tirs_st_ann_med": empty_da,
                "tirs_st_ann_min": empty_da,
                "tirs_st_ann_max": empty_da,
            }
        )
        return annual_ds_tirs

    inst_ds = annual_data[inst]
    native_tirs_geobox = inst_ds.odc.geobox

    # Remove outliers (no data value for surface temp is 0),
    # apply quality filter
    # and also filter on emissivity > 0.95
    valid_mask = (
        (inst_ds["tirs_st"] > 0)
        & (inst_ds["tirs_st_qa"] < 5)
        & (inst_ds["tirs_emis"] > 0.95)
    )
    inst_ds["tirs_st"] = inst_ds["tirs_st"].where(valid_mask)

    attrs = {
        "nodata": np.nan,
        "scales": 1,
        "offsets": 0,
        "units": "Celsius",
    }

    annual_ds_tirs = xr.Dataset()

    group = inst_ds["tirs_st"].groupby("time.year")
    annual_ds_tirs["tirs_st_ann_med"] = group.median(dim="time")
    annual_ds_tirs["tirs_st_ann_med"].attrs.update(**attrs)

    quantiles = [0.1, 0.9]
    quantile_results = group.quantile(quantiles, dim="time")
    annual_ds_tirs["tirs_st_ann_min"] = quantile_results.sel(quantile=0.1)
    annual_ds_tirs["tirs_st_ann_min"].attrs.update(**attrs)

    annual_ds_tirs["tirs_st_ann_max"] = quantile_results.sel(quantile=0.9)
    annual_ds_tirs["tirs_st_ann_max"].attrs.update(**attrs)

    # Replace the year coordinate with datetime64[ns] time coordinate
    annual_ds_tirs = annual_ds_tirs.rename({"year": "time"})
    time_values = np.array(
        [year_to_dc_datetime(i) for i in annual_ds_tirs.time.values],
        dtype="datetime64[ns]",
    )
    annual_ds_tirs = annual_ds_tirs.assign_coords(time=time_values)

    if native_tirs_geobox.resolution != water_mask.odc.geobox.resolution:
        # Reproject to target tile geobox
        annual_ds_tirs = xr_reproject(
            annual_ds_tirs,
            how=water_mask.odc.geobox,
            resampling="bilinear",
        )

    # Mask to water pixels only.
    annual_ds_tirs = annual_ds_tirs.where(water_mask == 1)

    if compute:
        log.info("\tComputing water temperature dataset ...")
        annual_ds_tirs = annual_ds_tirs.compute()

    log.info("Processing complete for water temperature dataset.")
    return enforce_float32(annual_ds_tirs)
