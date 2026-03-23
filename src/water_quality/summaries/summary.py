import logging

import pandas as pd
from sqlalchemy import insert, select, update
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import Table
from waterbodies.db import create_table

from water_quality.summaries.db_models import (
    WaterbodyWaterQuality,
    WaterbodyWaterQualityPercentiles,
)

_log = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "obs_id",
    "uid",
    "date",
    "hue_q0_1",
    "hue_q0_2",
    "hue_q0_3",
    "hue_q0_4",
    "hue_q0_5",
    "hue_q0_6",
    "hue_q0_7",
    "hue_q0_8",
    "hue_q0_9",
    "owt_q0_1",
    "owt_q0_2",
    "owt_q0_3",
    "owt_q0_4",
    "owt_q0_5",
    "owt_q0_6",
    "owt_q0_7",
    "owt_q0_8",
    "owt_q0_9",
    "chla_q0_1",
    "chla_q0_2",
    "chla_q0_3",
    "chla_q0_4",
    "chla_q0_5",
    "chla_q0_6",
    "chla_q0_7",
    "chla_q0_8",
    "chla_q0_9",
    "tsi_q0_1",
    "tsi_q0_2",
    "tsi_q0_3",
    "tsi_q0_4",
    "tsi_q0_5",
    "tsi_q0_6",
    "tsi_q0_7",
    "tsi_q0_8",
    "tsi_q0_9",
    "tsm_q0_1",
    "tsm_q0_2",
    "tsm_q0_3",
    "tsm_q0_4",
    "tsm_q0_5",
    "tsm_q0_6",
    "tsm_q0_7",
    "tsm_q0_8",
    "tsm_q0_9",
    "st_max_q0_1",
    "st_max_q0_2",
    "st_max_q0_3",
    "st_max_q0_4",
    "st_max_q0_5",
    "st_max_q0_6",
    "st_max_q0_7",
    "st_max_q0_8",
    "st_max_q0_9",
    "st_median_q0_1",
    "st_median_q0_2",
    "st_median_q0_3",
    "st_median_q0_4",
    "st_median_q0_5",
    "st_median_q0_6",
    "st_median_q0_7",
    "st_median_q0_8",
    "st_median_q0_9",
    "st_min_q0_1",
    "st_min_q0_2",
    "st_min_q0_3",
    "st_min_q0_4",
    "st_min_q0_5",
    "st_min_q0_6",
    "st_min_q0_7",
    "st_min_q0_8",
    "st_min_q0_9",
    "fai_cover",
    "ndvi_cover",
]


def create_water_quality_table(engine: Engine) -> Table:
    """
    Create the waterbodies_water_quality table if it does not exist.

    Parameters
    ----------
    engine : Engine

    Returns
    -------
    Table
        The waterbodies_water_quality table.
    """
    table = create_table(engine=engine, db_model=WaterbodyWaterQuality)
    return table


def add_water_quality_observations_to_db(
    water_quality_measures: pd.DataFrame,
    engine: Engine,
    update_rows: bool = True,
):
    """
    Add waterbody water quality observations to the water quality table.

    Parameters
    ----------
    water_quality_measures : pd.DataFrame
        Table containing the water quality observations to add to the database
    engine : Engine
    update_rows : bool, optional
        If True if the a water quality observation id already exists in the table, the row
        will be updated else it will be skipped, by default True
    """
    table = create_water_quality_table(engine=engine)

    Session = sessionmaker(bind=engine)

    # Note: Doing it this way because drill outputs can be millions of rows.
    # Its best to do it in small batches.
    obs_ids_to_check = water_quality_measures["obs_id"].to_list()
    with Session.begin() as session:
        obs_ids_exist = session.scalars(
            select(table.c.obs_id).where(table.c.obs_id.in_(obs_ids_to_check))
        ).all()
        _log.info(
            f"Found {len(obs_ids_exist)} out of {len(obs_ids_to_check)} waterbody "
            f"observations already in the {table.name} table"
        )

    update_statements = []
    insert_parameters = []

    expected_columns = [
        "obs_id",
        "uid",
        "date",
        "hue_q0_1",
        "hue_q0_2",
        "hue_q0_3",
        "hue_q0_4",
        "hue_q0_5",
        "hue_q0_6",
        "hue_q0_7",
        "hue_q0_8",
        "hue_q0_9",
        "owt_q0_1",
        "owt_q0_2",
        "owt_q0_3",
        "owt_q0_4",
        "owt_q0_5",
        "owt_q0_6",
        "owt_q0_7",
        "owt_q0_8",
        "owt_q0_9",
        "chla_q0_1",
        "chla_q0_2",
        "chla_q0_3",
        "chla_q0_4",
        "chla_q0_5",
        "chla_q0_6",
        "chla_q0_7",
        "chla_q0_8",
        "chla_q0_9",
        "tsi_q0_1",
        "tsi_q0_2",
        "tsi_q0_3",
        "tsi_q0_4",
        "tsi_q0_5",
        "tsi_q0_6",
        "tsi_q0_7",
        "tsi_q0_8",
        "tsi_q0_9",
        "tsm_q0_1",
        "tsm_q0_2",
        "tsm_q0_3",
        "tsm_q0_4",
        "tsm_q0_5",
        "tsm_q0_6",
        "tsm_q0_7",
        "tsm_q0_8",
        "tsm_q0_9",
        "st_max_q0_1",
        "st_max_q0_2",
        "st_max_q0_3",
        "st_max_q0_4",
        "st_max_q0_5",
        "st_max_q0_6",
        "st_max_q0_7",
        "st_max_q0_8",
        "st_max_q0_9",
        "st_median_q0_1",
        "st_median_q0_2",
        "st_median_q0_3",
        "st_median_q0_4",
        "st_median_q0_5",
        "st_median_q0_6",
        "st_median_q0_7",
        "st_median_q0_8",
        "st_median_q0_9",
        "st_min_q0_1",
        "st_min_q0_2",
        "st_min_q0_3",
        "st_min_q0_4",
        "st_min_q0_5",
        "st_min_q0_6",
        "st_min_q0_7",
        "st_min_q0_8",
        "st_min_q0_9",
        "fai_cover",
        "ndvi_cover",
    ]
    update_columns = [
        col for col in expected_columns if col != "obs_id"
    ]  # exclude PK

    for row in water_quality_measures.itertuples():
        row_dict = {col: getattr(row, col) for col in expected_columns}

        if row.obs_id not in obs_ids_exist:
            insert_parameters.append(row_dict)
        else:
            if update_rows:
                update_statements.append(
                    update(table)
                    .where(table.c.obs_id == row.obs_id)
                    .values({col: getattr(row, col) for col in update_columns})
                )
            else:
                continue

    if update_statements:
        _log.info(
            f"Updating {len(update_statements)} water quality observations in the {table.name} table"
        )
        with Session.begin() as session:
            for statement in update_statements:
                session.execute(statement)
    else:
        _log.info(
            f"No water quality observations to update in the {table.name} table"
        )

    if insert_parameters:
        _log.info(
            f"Inserting {len(insert_parameters)} water quality observations in the {table.name} table"
        )
        with Session.begin() as session:
            session.execute(insert(table), insert_parameters)
    else:
        _log.error(
            f"No water quality observations to insert into the {table.name} table"
        )


def check_obs_ids(observation_ids: list[str], engine: Engine) -> list[str]:
    """
    Check if the water_quality_table contains the waterbody observation IDs in
    `observation_ids`. Returns only the observation IDs found in the database
    table.

    Parameters
    ----------
    observation_ids : list[str]
        A list of waterbody observation IDs to check in the database table.
    engine : Engine
        The SQLAlchemy engine connected to the database.

    Returns
    -------
    list[str]
        A list of waterbody observation IDs found in the database table.
    """
    if isinstance(observation_ids, str):
        observation_ids = [observation_ids]

    Session = sessionmaker(bind=engine)
    table = create_water_quality_table(engine=engine)

    with Session.begin() as session:
        results = session.scalars(
            select(table.c.obs_id).where(table.c.obs_id.in_(observation_ids))
        ).all()
    return results


def create_water_quality_percentiles_table(engine: Engine) -> Table:
    """
    Create the waterbodies_water_quality_percentiles table if it does not exist.

    Parameters
    ----------
    engine : Engine

    Returns
    -------
    Table
        The waterbodies_water_quality_percentiles table.
    """
    table = create_table(
        engine=engine, db_model=WaterbodyWaterQualityPercentiles
    )
    return table
