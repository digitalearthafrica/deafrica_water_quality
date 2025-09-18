import numpy as np
import pandas as pd
import xarray as xr


def classify_chla_values(chla_values: np.ndarray) -> np.ndarray:
    """
    Classify Chlorophyll-a (µg/l) values into Trophic State Index
    values based on the table of Trophic State Index and related
    chlorophyll concentration classes (according to Carlson (1977)).

    Parameters
    ----------
    chla_values : np.ndarray
        Chlorophyll-a (µg/l) values to classify

    Returns
    -------
    np.ndarray
        Corresponding Trophic State Index values.
    """
    # Chlorophyll-a (µg/l) (upper limit)
    conditions = [
        (chla_values <= 0.04),
        (chla_values > 0.04) & (chla_values <= 0.12),
        (chla_values > 0.12) & (chla_values <= 0.34),
        (chla_values > 0.34) & (chla_values <= 0.94),
        (chla_values > 0.94) & (chla_values <= 2.6),
        (chla_values > 2.6) & (chla_values <= 6.4),
        (chla_values > 6.4) & (chla_values <= 20),
        (chla_values > 20) & (chla_values <= 56),
        (chla_values > 56) & (chla_values <= 154),
        (chla_values > 154) & (chla_values <= 427),
        (chla_values > 427) & (chla_values <= 1183),
    ]
    # Trophic State Index, CGLOPS TSI values
    choices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    tsi_values = np.select(conditions, choices, default=np.nan)
    return tsi_values


def classify_tsi_values(tsi_values: np.ndarray) -> np.ndarray:
    """
    Classify Trophic State Index values into the Trophic Classification
    classes base on the table of Trophic State Index and related
    chlorophyll concentration classes (according to Carlson (1977)).

    Parameters
    ----------
    tsi_values : np.ndarray
        Trophic State Index values

    Returns
    -------
    np.ndarray
        Corresponding Trophic Classification classes.
    """
    conditions = [
        (tsi_values >= 0) & (tsi_values < 40),
        (tsi_values >= 40) & (tsi_values < 60),
        (tsi_values >= 60) & (tsi_values < 80),
        (tsi_values >= 80) & (tsi_values <= 100),
    ]
    choices = ["Oligotrophic", "Mesotrophic", "Eutrophic", "Hypereutrophic"]
    trophic_classification = np.select(conditions, choices, default=np.nan)
    return trophic_classification


def get_trophic_classification_table():
    chla_values = np.array(
        [0.04, 0.12, 0.34, 0.94, 2.6, 6.4, 20, 56, 154, 427, 1183]
    )
    tsi_values = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    trophic_classification = classify_tsi_values(tsi_values)
    df = pd.DataFrame(
        data={
            "Trophic classification": trophic_classification,
            "Trophic State Index, Copernicus Global Land Service TSI values": tsi_values,
            "Chlorophyll-a (µg/l) (upper limit)": chla_values,
        }
    )

    return df


def compute_trophic_state_index(
    ds: xr.Dataset, chla_variable: str
) -> xr.Dataset:
    """
    Compute the Trophic State Index from the Chlorophyll-a (µg/l) values
    and add the Trophic State Index as the variable "TSI" to the input
    dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to add the Trophic State Index to.
    chla_variable : str
        Variable in input dataset containing the Chlorophyll-a (µg/l)
        values to derive the Trophic State Index from.

    Returns
    -------
    xr.Dataset
        Input dataset with the Trophic State Index added as the
        variable "TSI".
    """
    ds["TSI"] = (tuple(ds.dims), classify_chla_values(ds[chla_variable]))
    return ds
