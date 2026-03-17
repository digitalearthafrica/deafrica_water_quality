from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from datacube import Datacube
from odc.geo.geom import Geometry
from odc.geo.xr import rasterize

LWQ_PRODUCTS = [
    "cgls_lwq300_2002_2012",
    "cgls_lwq300_2016_2024",
    "cgls_lwq300_2024_nrt",
]
LWQ_MEASUREMENTS = ["TSI", "turbidity"]
DEVIATION_CATEGORIES = {"low": 1, "medium": 2, "high": 3, "extreme": 4}


def get_monthly_timeseries(
    time_range: tuple[str], waterbody_geom: Geometry, dc: Datacube = None
) -> xr.Dataset:
    """
    Retrieve the 300m resolution monthly Lake Water Quality time series
    for a specified waterbody and time range.

    Parameters
    ----------
    time_range : tuple[str]
        Tuple representing the start and end of the time range
        (e.g., ("2015", "2020")) to load data for.
    waterbody_geom : Geometry
        Geometry defining the spatial extent of the waterbody.
    dc : Datacube, optional
        Datacube connection, by default None

    Returns
    -------
    xr.Dataset
        A dataset containing the 300m resolution monthly time series of
        Lake Water Quality data for the specified waterbody and time range.
    """
    if dc is None:
        dc = Datacube()

    # Load data
    ds = dc.load(
        product=LWQ_PRODUCTS,
        measurements=LWQ_MEASUREMENTS,
        geopolygon=waterbody_geom,
        output_crs="EPSG:6933",
        resolution=(-300, 300),
        time=time_range,
    )

    # Mask the data to the waterbody.
    waterbody_xr = rasterize(waterbody_geom, how=ds.odc.geobox)
    ds = ds.where(waterbody_xr)

    # Mask the no data value
    for var in LWQ_MEASUREMENTS:
        ds[var] = ds[var].where(ds[var] != float(ds[var].nodata))

    # Resample decadal timeseries to monthly timeseries
    monthly_timeseries = ds.resample(time="1ME").mean()

    return monthly_timeseries


def get_monthly_multiannual_baseline(
    baseline_period: tuple[str], waterbody_geom: Geometry, dc: Datacube = None
) -> xr.Dataset:
    """
    Derive the 12 monthly averages (monthly multiannual baseline) for both trophic
    state and turbidity from the 300m resolution Lake Water Quality monthly
    time series, for a given waterbody and baseline reference period.

    Parameters
    ----------
    baseline_period : tuple[str]
        Tuple representing the baseline time range (e.g., ("2006", "2010")) to load data for.
    waterbody_geom : Geometry
        Bounding box defining the spatial extent of the waterbody.
    dc : Datacube, optional
        Datacube connection, by default None

    Returns
    -------
    xr.Dataset
        A dataset containing the 12 monthly multiannual baseline (one average per month)
        for a waterbody.
    """
    if dc is None:
        dc = Datacube()

    baseline_monthly_timeseries = get_monthly_timeseries(
        time_range=baseline_period, waterbody_geom=waterbody_geom, dc=dc
    )

    # Get the long term average for each month
    baseline_monthly_long_term_mean = baseline_monthly_timeseries.groupby(
        "time.month"
    ).mean()

    return baseline_monthly_long_term_mean


def get_target_years_monthly_avgs(
    target_period: tuple[str],
    waterbody_geom: Geometry,
    dc: Datacube = None,
) -> xr.Dataset:
    """
    Load the 300m resolution monthly time series of Lake Water Quality data
    for a specified waterbody and target years.

    Parameters
    ----------
    target_period : tuple[str]
        Tuple of target years (e.g., ("2017", "2021")) to load trophic state
        and turbidity data for the waterbody.
    waterbody_geom : Geometry
        Bounding box defining the spatial extent of the waterbody.
    dc : Datacube, optional
        Datacube connection, by default None

    Returns
    -------
    xr.Dataset
        A dataset containing the 300m resolution monthly time series of Lake
        Water Quality data for the specified
        waterbody and target years.
    """
    if dc is None:
        dc = Datacube()

    monthly_timeseries = get_monthly_timeseries(
        time_range=target_period, waterbody_geom=waterbody_geom, dc=dc
    )
    return monthly_timeseries


def get_monthly_deviations(
    target_years_monthly_avgs: xr.Dataset,
    monthly_multiannual_baseline: xr.Dataset,
) -> xr.Dataset:
    """
    Produce a monthly deviation synthesis for a waterbody using the
    monthly time series of target years and a multiannual monthly baseline.

    Parameters
    ----------
    target_years_monthly_avgs : xr.Dataset
        Monthly time series of observations for the target years for a waterbody.
    monthly_multiannual_baseline : xr.Dataset
        Multiannual monthly baseline dataset for the same waterbody.

    Returns
    -------
    xr.Dataset
        A dataset representing the monthly deviation synthesis computed using the equation:
        ((month_average-month_baseline)/month_baseline) x 100
    """

    def _get_deviations(year_data, baseline):
        deviations = year_data.groupby("time.month").apply(
            lambda x: (
                (
                    (x - baseline.sel(month=x["time.month"]))
                    / baseline.sel(month=x["time.month"])
                )
                * 100
            )
        )

        return deviations.drop_vars("month")

    monthly_deviations = target_years_monthly_avgs.groupby("time.year").map(
        lambda year_data: _get_deviations(
            year_data, monthly_multiannual_baseline
        )
    )

    return monthly_deviations


def get_deviation_categories(
    var: str, deviations_ds: xr.Dataset
) -> np.ndarray:
    """
    Classify pixels in a monthly or annual deviation synthesis data variable
    into categorical deviation levels.

    Pixels are classified based on their deviation values into the following categories:
        - 1: Low (< 25%)
        - 2: Medium (25–50%)
        - 3: High (50–100%)
        - 4: Extreme (> 100%)

    Parameters
    ----------
    var : str
       Name of the data variable within the dataset to classify.
    deviations_ds : xr.Dataset
        Monthly or annual deviation synthesis dataset for a waterbody.

    Returns
    -------
    np.ndarray
        Array of classified pixel values, where each pixel is assigned:
        1 (low), 2 (medium), 3 (high), or 4 (extreme).
    """
    conditions = [
        (deviations_ds[var] <= 25),
        ((deviations_ds[var] > 25) & (deviations_ds[var] <= 50)),
        ((deviations_ds[var] > 50) & (deviations_ds[var] <= 100)),
        (deviations_ds[var] > 100),
    ]
    classes = list(DEVIATION_CATEGORIES.values())
    data = np.select(conditions, classes, default=np.nan)
    return data


def add_deviation_categories_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Add deviation categories as attributes to a dataset.


    Parameters
    ----------
    ds : xr.Dataset
        The dataset to which deviation categories metadata will be added.

    Returns
    -------
    xr.Dataset
        The input dataset with deviation categories included
        as attributes.
    """

    attrs = ds.attrs
    if attrs:
        if "deviation_categories" not in list(attrs.keys):
            attrs["deviation_categories"] = DEVIATION_CATEGORIES
    else:
        attrs = {"deviation_categories": DEVIATION_CATEGORIES}

    ds.attrs = attrs
    return ds


def classify_deviations(deviations_ds: xr.Dataset) -> xr.Dataset:
    """
    Classify pixels in a monthly or annual deviation synthesis.

    This function assigns each pixel in the deviation synthesis to a
    category.

    Parameters
    ----------
    deviations_ds : xr.Dataset
        Monthly or annual deviation synthesis for a waterbody. Expected to
        include one or more data variables with deviation values.

    Returns
    -------
    xr.Dataset
        A dataset of the same shape as `deviations_ds`, where each pixel
        is classified into one of the following categories:

        - 1: Low
        - 2: Medium
        - 3: High
        - 4: Extreme
    """
    deviation_classes = xr.full_like(deviations_ds, fill_value=np.nan)

    for var in LWQ_MEASUREMENTS:
        deviation_classes[var].data = get_deviation_categories(
            var=var, deviations_ds=deviations_ds
        )

    deviation_classes = add_deviation_categories_attrs(deviation_classes)

    return deviation_classes


def get_annual_deviations(monthly_deviations: xr.Dataset) -> xr.Dataset:
    """
    Produce an annual deviation synthesis from a monthly deviation sysnthess

    Parameters
    ----------
    monthly_deviations : xr.Dataset
        Monthly deviation synthesis for a waterbody.

    Returns
    -------
    xr.Dataset
        Annual deviation synthesis for the same waterbody.
    """
    annual_deviations = monthly_deviations.groupby("time.year").mean()
    return annual_deviations


def waterbody_is_affected(
    annual_deviation_classes: xr.Dataset,
) -> pd.DataFrame:
    """
    Determine whether a waterbody is affected based on annual deviation classifications.

    Compares the number of 'high' and 'extreme' deviation pixels to the number of
    'low' and 'medium' deviation pixels for each time step. A waterbody is considered
    affected if the number of high and extreme pixels exceeds the number of low and
    medium pixels.

    Parameters
    ----------
    annual_deviation_classes : xr.Dataset
        Dataset containing classified annual deviation values, where deviation
        categories (e.g., "low", "medium", "high", "extreme") are stored in the
        dataset attributes under the key `"deviation_categories"`.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by time, with a boolean column indicating whether the
        waterbody is considered affected (`True`) or not (`False`) for each year.
    """

    attrs = annual_deviation_classes.attrs
    categories = attrs.get("deviation_categories", None)
    if not categories:
        raise ValueError(
            "Deviation categories missing from `annual_deviation_categories` attrs"
        )

    count = {}
    for label, value in categories.items():
        count[label] = (
            (annual_deviation_classes == value).astype(int).sum(dim=["x", "y"])
        )

    affected = (count["high"] + count["extreme"]) > (
        count["low"] + count["medium"]
    )
    affected_df = affected.to_dataframe().drop(columns=["spatial_ref"])
    return affected_df


def get_turbidity_and_tsi_summary_tables(
    waterbodies_info: list[dict[str, Any]],
) -> tuple[pd.DataFrame]:
    # Check expected info in each item
    for waterbody_info in waterbodies_info:
        # Identifier for the waterbody
        assert "wb_id" or "uid" in list(waterbody_info.keys())
        # Info of interest.
        assert "affected" in list(waterbody_info.keys())
        # Check contents of affected dictionary
        affected = waterbody_info["affected"]
        for year in list(affected.keys()):
            assert "TSI" and "turbidity" in list(affected[year].keys())

    # Get info into a table
    turbidity_affected_lakes_count = defaultdict(
        lambda: {"no_of_not_affected_lakes": 0, "no_of_affected_lakes": 0}
    )
    tsi_affected_lakes_count = defaultdict(
        lambda: {"no_of_not_affected_lakes": 0, "no_of_affected_lakes": 0}
    )

    for waterbody in waterbodies_info:
        affected = waterbody["affected"]
        for year, indicators in affected.items():
            if indicators["turbidity"] is True:
                turbidity_affected_lakes_count[year][
                    "no_of_affected_lakes"
                ] += 1
            else:
                turbidity_affected_lakes_count[year][
                    "no_of_not_affected_lakes"
                ] += 1
            if indicators["TSI"] is True:
                tsi_affected_lakes_count[year]["no_of_affected_lakes"] += 1
            else:
                tsi_affected_lakes_count[year]["no_of_not_affected_lakes"] += 1

    turbidity_affected_lakes_count = pd.DataFrame.from_dict(
        dict(turbidity_affected_lakes_count), orient="index"
    )
    turbidity_affected_lakes_count["total_no_of_lakes"] = len(waterbodies_info)
    turbidity_affected_lakes_count.index.name = "year"
    turbidity_affected_lakes_count["type"] = "turbidity"

    tsi_affected_lakes_count = pd.DataFrame.from_dict(
        dict(tsi_affected_lakes_count), orient="index"
    )
    tsi_affected_lakes_count["total_no_of_lakes"] = len(waterbodies_info)
    tsi_affected_lakes_count.index.name = "year"
    tsi_affected_lakes_count["type"] = "trophic state"

    # Get the proportion of waterbodies affected
    turbidity_affected_lakes_count["EN_LKW_QLTRB %"] = (
        turbidity_affected_lakes_count.apply(
            lambda x: (
                (x["no_of_affected_lakes"] / x["total_no_of_lakes"]) * 100
            ),
            axis=1,
        )
    )
    tsi_affected_lakes_count["EN_LKW_QLTRST %"] = (
        tsi_affected_lakes_count.apply(
            lambda x: (
                (x["no_of_affected_lakes"] / x["total_no_of_lakes"]) * 100
            ),
            axis=1,
        )
    )
    return turbidity_affected_lakes_count, tsi_affected_lakes_count
