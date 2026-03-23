import logging

import click
import numpy as np
import pandas as pd
from odc.stats.model import DateTimeRange
from tqdm import tqdm
from waterbodies.db import get_existing_table_names, get_waterbodies_engine

from water_quality.logs import setup_logging
from water_quality.summaries.summary import (
    add_water_quality_percentiles_to_db,
    create_water_quality_table,
)


@click.command(
    name="rank-waterbodies",
    no_args_is_help=True,
)
@click.option(
    "--temporal-range",
    default="2020--P6Y",
    type=str,
    help="The temporal range over which to average water quality variables for ranking waterbodies.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help=(
        "If overwrite is True tasks that have already been processed "
        "will be rerun. "
    ),
)
@click.option(
    "--log",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="WARNING",
    show_default=True,
    help="control the log level, e.g., --log=error",
)
def cli(
    temporal_range: str,
    overwrite: bool,
    log: str,
):
    """
    Rank waterbodies based on water quality observations.
    """
    log_level = getattr(logging, log.upper())
    _log = setup_logging(log_level)

    try:
        temporal_range = DateTimeRange(temporal_range)
    except ValueError:
        e = ValueError(
            f"Failed to parse supplied temporal_range: '{temporal_range}'",
        )
        _log.error(e)
        raise e

    start_date = temporal_range.start.strftime("%Y-%m-%d")
    end_date = temporal_range.end.strftime("%Y-%m-%d")

    engine = get_waterbodies_engine()

    wq_table = create_water_quality_table(engine)
    assert wq_table.name in get_existing_table_names(engine)

    all_waterbodies_uids = pd.read_sql(
        "SELECT uid from waterbodies_historical_extent", con=engine
    )

    required_columns = [
        "fai_cover",
        "ndvi_cover",
        "hue_q0_5",
        "owt_q0_5",
        "chla_q0_5",
        "tsi_q0_5",
        "tsm_q0_5",
        "st_max_q0_5",
        "st_median_q0_5",
        "st_min_q0_5",
    ]

    _log.info(
        f"Getting the average of water quality variables for {len(all_waterbodies_uids)} "
        f"waterbodies from {start_date} to {end_date} ..."
    )
    avg_summaries = []
    for uid in tqdm(
        iterable=all_waterbodies_uids["uid"].values,
        total=len(all_waterbodies_uids),
    ):
        all_waterbody_summaries = pd.read_sql(
            f"SELECT * FROM waterbodies_water_quality WHERE uid = '{uid}' AND date BETWEEN '{start_date}' AND '{end_date}' ORDER BY date",
            con=engine,
        )
        avg_summaries_df = (
            all_waterbody_summaries[["uid", *required_columns]]
            .groupby("uid")
            .mean(numeric_only=True)
        )
        avg_summaries.append(avg_summaries_df)

    per_waterbody_avg = pd.concat(avg_summaries, axis=0)
    assert len(all_waterbodies_uids) == len(per_waterbody_avg)

    _log.info("Calculating percentiles for each water quality variable ...")
    quantiles = per_waterbody_avg.quantile(
        q=np.linspace(0, 1, 101), numeric_only=True
    )
    assert list(quantiles.columns) == list(per_waterbody_avg.columns)

    for col in per_waterbody_avg.select_dtypes("number").columns:
        bins = quantiles[col].unique()
        per_waterbody_avg[f"{col}_percentile"] = pd.cut(
            per_waterbody_avg[col],
            bins=bins,
            labels=range(1, len(bins)),
            include_lowest=True,
        )

    final_cols = [f"{col}_percentile" for col in required_columns]
    water_quality_percentiles_df = per_waterbody_avg[final_cols]

    add_water_quality_percentiles_to_db(
        water_quality_percentiles=water_quality_percentiles_df,
        engine=engine,
        update_rows=overwrite,
    )

    _log.info(
        f"Finished processing water quality percentiles for {len(all_waterbodies_uids)} "
        f"waterbodies from {start_date} to {end_date}."
    )
