import numpy as np
import xarray as xr


def _enforce_float32(da: xr.DataArray) -> xr.DataArray:
    """
    Enforce float32 data type for an xarray DataArray.

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray to be converted to float32.

    Returns
    -------
    xr.DataArray
        The input DataArray with data type enforced as float32.
    """
    if da.dtype == np.float32:
        return da
    else:
        return da.astype(np.float32)


def enforce_float32(
    ds: xr.DataArray | xr.Dataset,
) -> xr.DataArray | xr.Dataset:
    """
    Enforce float32 data type for an xarray DataArray or Dataset.

    Parameters
    ----------
    ds : xr.DataArray or xr.Dataset
        The input DataArray or Dataset to be converted to float32.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The input DataArray or Dataset with data type enforced as float32.
    """
    if isinstance(ds, xr.DataArray):
        return _enforce_float32(ds)
    elif isinstance(ds, xr.Dataset):
        for var in list(ds.data_vars):
            ds[var] = _enforce_float32(ds[var])
        return ds
