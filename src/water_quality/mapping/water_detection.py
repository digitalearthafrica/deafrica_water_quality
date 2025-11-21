"""
This module provides functions to implement water detection using DE
Africa Water Observations from Space (WOfS) products.
"""

import logging

import xarray as xr

log = logging.getLogger(__name__)


def water_analysis(
    wofs_ann_ds: xr.Dataset,
    water_frequency_threshold: float = 0.5,
):
    """Performs water detection analysis on DE Africa WOfS Annual
    Summary data.

    Parameters
    ----------
    wofs_ann_ds : Dataset
        An xarray Dataset containing the processed data for the
        instrument `wofs_ann` for a single year.
    water_frequency_threshold : float, optional
        The frequency threshold above which a pixel is classified as
        general water, by default 0.5

    Returns
    -------
    xarray.Dataset
        The input Dataset with the following new data variables added:
        - `wofs_ann_watermask` (float): A mask showing where general water is detected.

    """

    # Standard deviation of the annual frequency at each pixel
    # should really be dividing by n-1 but then I would need to
    # change SC
    wofs_ann_freq_sigma = (
        (wofs_ann_ds.wofs_ann_freq * (1 - wofs_ann_ds.wofs_ann_freq))
        / wofs_ann_ds.wofs_ann_clearcount
    ) ** 0.5
    # A variable called watermask is used in places.
    # I set the value of the mask as sigma or nan
    # Renamed this from watermask to wofs_ann_watermask to prevent
    # confusion with the 5 year summary watermask
    wofs_ann_ds["wofs_ann_watermask"] = wofs_ann_freq_sigma.where(
        wofs_ann_ds["wofs_ann_freq"] > water_frequency_threshold
    )

    return wofs_ann_ds
