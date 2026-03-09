from importlib.metadata import version

import click

from water_quality.cli.variables.list_test_areas import (
    cli as list_test_areas_cli,
)

PKG_NAME = "deafrica-water-quality"


@click.group()
@click.version_option(
    version=version(PKG_NAME),
    package_name=PKG_NAME,
    message="%(package)s, version %(version)s",
)
def wqms_variables():
    """
    A collection of tools for processing water quality variables for the
    Digital Earth Africa Water Quality Monitoring System (WQMS).
    """
    pass


@click.group()
@click.version_option(
    version=version(PKG_NAME),
    package_name=PKG_NAME,
    message="%(package)s, version %(version)s",
)
def wqms_summaries():
    """
    A collection of tools for per waterbody water quality summaries for the
    Digital Earth Africa Water Quality Monitoring System (WQMS).
    """
    pass


wqms_summaries.add_command(list_test_areas_cli)
