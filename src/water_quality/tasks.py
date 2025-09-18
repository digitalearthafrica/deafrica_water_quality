"""
This module provides functions to create tasks used as
as building blocks in the DE Africa Water Quality workflow.
"""

import re

from odc.stats._text import split_and_check

from water_quality.tiling import get_region_code, parse_region_code


def create_task_id(year: str | int, tile_id: tuple[int, int] | str) -> str:
    """Create  a task given a year and a tile id.

    Parameters
    ----------
    year : str | int
        Year to create the task for.
    tile_id : tuple[int, int] | str
        Tile ID for the tile to create the task for.

    Returns
    -------
    str
        Task ID
    """
    if isinstance(year, int):
        year = str(year)
    # task id format "{year}/x{x:03d}/y{y:03d}"
    region_code = get_region_code(tile_id, sep="/")
    task_id = f"{year}/{region_code}"
    return task_id


def parse_task_id(task_id: str) -> tuple[int, tuple[int, int]]:
    """
    Parse a task ID into the year and tile ID it was created from.

    Parameters
    ----------
    task_id : str
        Task ID to parse.

    Returns
    -------
    tuple[int, tuple[int, int]]
        Year and tile ID components of the task.
    """
    # Check Task id has only 3 parts
    sep = "/"  # based on seperator used in create_task_id
    _ = split_and_check(task_id, sep, 3)

    # Get the tile ID
    tile_id = parse_region_code(task_id)

    # Get the year
    year_pattern = re.compile(r"\d{4}")
    year_str = re.search(year_pattern, task_id).group(0)
    year = int(year_str)
    return year, tile_id
