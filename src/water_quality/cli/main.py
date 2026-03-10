from importlib.metadata import version

import click

from water_quality.cli.summaries.generate_tasks import (
    cli as summaries_generate_tasks_cli,
)
from water_quality.cli.summaries.raster_processing import (
    cli as process_raster_tasks_cli,
)
from water_quality.cli.variables.generate_tasks import (
    cli as generate_tasks_cli,
)
from water_quality.cli.variables.generate_tiles import (
    cli as generate_tiles_cli,
)
from water_quality.cli.variables.list_test_areas import (
    cli as list_test_areas_cli,
)
from water_quality.cli.variables.metadata_generator import (
    cli as metadata_generator_cli,
)
from water_quality.cli.variables.process_annual_tasks import (
    cli as process_annual_tasks_cli,
)
from water_quality.cli.variables.update_stac_files import (
    cli as update_stac_files_cli,
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


wqms_variables.add_command(list_test_areas_cli)
wqms_variables.add_command(generate_tiles_cli)
wqms_variables.add_command(generate_tasks_cli)
wqms_variables.add_command(process_annual_tasks_cli)
wqms_variables.add_command(metadata_generator_cli)
wqms_variables.add_command(update_stac_files_cli)


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


wqms_summaries.add_command(summaries_generate_tasks_cli)
wqms_summaries.add_command(process_raster_tasks_cli)
