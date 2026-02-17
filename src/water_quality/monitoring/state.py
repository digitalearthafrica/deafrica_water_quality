import numpy as np
import pandas as pd


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
