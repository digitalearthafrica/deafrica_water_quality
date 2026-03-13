from importlib.resources import files

import click
import geopandas as gpd
import pandas as pd


@click.command(
    name="list-test-areas",
)
def cli():
    """
    View the names of defined test areas available to run the DE Africa Water Quality workflow on.
    """
    places_fp = files("water_quality.data").joinpath("places.parquet")
    places_gdf = gpd.read_parquet(places_fp)
    columns = ["name", "description"]
    with pd.option_context("display.max_rows", None):
        print(places_gdf[columns])


if __name__ == "__main__":
    cli()
