"""
This module provides functions to create tasks used as
as building blocks in the DE Africa Water Quality workflow.
"""

import logging
import math
from itertools import batched
from typing import Any

from odc.stats._text import split_and_check

from water_quality.tiling import get_region_code, parse_region_code

log = logging.getLogger(__name__)


def create_task_id(temporal_id: str, tile_id: tuple[int, int] | str) -> str:
    """Create  a task given a temporal range and a tile id.

    Parameters
    ----------
    temporal_id : str
        Temproal range to create the task for.
    tile_id : tuple[int, int] | str
        Tile ID for the tile to create the task for.

    Returns
    -------
    str
        Task ID
    """
    # task id format "{temporal_id}/x{x:03d}/y{y:03d}"
    region_code = get_region_code(tile_id, sep="/")
    task_id = f"{temporal_id}/{region_code}"
    return task_id


def parse_task_id(task_id: str) -> tuple[str, int, int]:
    """
    Parse a task ID into the temporal ID and tile ID it was created from.

    Parameters
    ----------
    task_id : str
        Task ID to parse.

    Returns
    -------
    tuple[str, tuple[int, int]]
        Temporal ID and tile ID components of the task.
    """
    # Check Task id has only 3 parts
    sep = "/"  # based on seperator used in create_task_id
    parts = split_and_check(task_id, sep, 3)

    # Get the tile ID
    tile_id = parse_region_code(task_id)

    # Get the temporal ID
    temporal_id = [p for p in parts if not p.startswith(("x", "y"))]
    # Should always be one item in the temporal_id.
    temporal_id = temporal_id[0]
    return temporal_id, *tile_id


def split_tasks(
    all_tasks: list[Any], max_parallel_steps: int, worker_idx: int
) -> list[str]:
    """Divide tasks across workers."""
    chunk_size = math.ceil(len(all_tasks) / max_parallel_steps)
    try:
        task_chunks = list(batched(all_tasks, chunk_size))
    except ValueError as e:
        log.error(f"Error batching tasks: {e}")
        return []
    else:
        if len(task_chunks) <= worker_idx:
            return []
        else:
            return task_chunks[worker_idx]


def filter_tasks_by_task_id(
    all_task_ids: list[tuple[str, int, int]], task_ids: str | list[str]
) -> list[tuple[str, int, int]]:
    """Filter tasks by task ID."""
    if isinstance(task_ids, str):
        task_filter = [parse_task_id(i.strip()) for i in task_ids.split(",")]
    else:
        task_filter = [parse_task_id(i) for i in task_ids]

    filtered_tasks = list(set(task_filter).intersection(set(all_task_ids)))

    return filtered_tasks


def check_task_id_for_tile_id(task_id: str, tile_id: str) -> bool:
    if set(tile_id).issubset(task_id):
        if tile_id[-1] == task_id[-1]:
            return True
        else:
            return False
    else:
        return False


def filter_tasks_by_tile_id(
    all_task_ids: list[tuple[str, int, int]], tile_ids: str | list[str]
) -> list[tuple[str, int, int]]:
    """Filter tasks by tile ID."""
    if isinstance(tile_ids, str):
        tile_filter = [
            parse_region_code(i.strip()) for i in tile_ids.split(",")
        ]
    else:
        tile_filter = [parse_region_code(i) for i in tile_ids]

    filtered_tasks = list(
        filter(
            lambda task_id: any(
                check_task_id_for_tile_id(task_id, tile_id)
                for tile_id in tile_filter
            ),
            all_task_ids,
        )
    )
    return filtered_tasks
