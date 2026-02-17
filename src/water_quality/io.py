"""
Utilities for interacting with local, cloud (S3, GCS), and HTTP filesystems
"""

import logging
import os
import posixpath
import re
from pathlib import Path

import fsspec
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs.core import S3FileSystem
from yarl import URL

from water_quality.tiling import get_region_code, parse_region_code

log = logging.getLogger(__name__)


def is_s3_path(path: str) -> bool:
    """
    Checks if a given path is an s3 URI.
    """
    fs, _ = fsspec.core.url_to_fs(path)
    return isinstance(fs, S3FileSystem)


def is_gcsfs_path(path: str) -> bool:
    """
    Checks if a given path is a gsutil URI.
    """
    fs, _ = fsspec.core.url_to_fs(path)
    return isinstance(fs, GCSFileSystem)


def is_http_url(path: str) -> bool:
    """
    Checks if a given path is a http(s) URL.
    """
    fs, _ = fsspec.core.url_to_fs(path)
    return isinstance(fs, HTTPFileSystem)


def is_local_path(path: str) -> bool:
    """
    Checks if a given path is a local storage path.
    """
    fs, _ = fsspec.core.url_to_fs(path)
    return isinstance(fs, LocalFileSystem)


def join_url(base, *paths) -> str:
    """
    Join two or more pathname components, inserting '/' as needed.
    """
    if is_local_path(base):
        return os.path.join(base, *paths)
    else:
        # Ensure urls join correctly
        return posixpath.join(base, *paths)


def get_basename(path: str):
    """
    Returns the final component of a pathname
    """
    if is_local_path(path):
        return os.path.basename(path)
    else:
        return posixpath.basename(path)


def get_parent_dir(path: str):
    """
    Returns the logical parent of the path.
    """
    if is_local_path(path):
        return str(Path(path).resolve().parent)
    else:
        return str(URL(path).parent)


def get_filesystem(
    path: str,
    anon: bool = True,
) -> S3FileSystem | LocalFileSystem | GCSFileSystem:
    """
    Instantiate a file-system based on the input path type.
    """
    if is_s3_path(path=path):
        fs = S3FileSystem(
            anon=anon,
            # Use profile only on sandbox
            # profile="default",
            s3_additional_kwargs={"ACL": "bucket-owner-full-control"},
        )
    elif is_gcsfs_path(path=path):
        if anon:
            fs = GCSFileSystem(token="anon")
        else:
            fs = GCSFileSystem()
    elif is_http_url(path):
        fs = HTTPFileSystem()
    elif is_local_path(path=path):
        fs = LocalFileSystem()
    return fs


def check_file_exists(path: str) -> bool:
    """
    Checks if a given path exists and is a file.
    """
    fs = get_filesystem(path=path, anon=True)
    if fs.exists(path) and fs.isfile(path):
        return True
    else:
        return False


def check_directory_exists(path: str) -> bool:
    """
    Checks if a given path exists and is a directory.
    """
    fs = get_filesystem(path=path, anon=True)
    if fs.exists(path) and fs.isdir(path):
        return True
    else:
        return False


def check_file_extension(
    path: str, accepted_file_extensions: list[str]
) -> bool:
    """Check if the file extension for a given path is among the list
    of allowed file extensions"""
    _, file_extension = os.path.splitext(path)
    if file_extension.lower() in accepted_file_extensions:
        return True
    else:
        return False


def find_files_by_extension(
    directory_path: str,
    accepted_file_extensions: list[str],
    file_name_pattern: str = ".*",
) -> list[str]:
    """
    Recursively find files matching extensions and optional filename pattern.

    Parameters
    ----------
    directory_path : str
         Path to search (local, S3, or GCS).
    accepted_file_extensions : list[str]
        List of extensions to include (e.g., ['.tif', '.csv']).
    file_name_pattern : str, optional
        Regex pattern to filter file names (default: match all).

    Returns
    -------
    list[str]
        List of matching file paths.
    """
    file_name_pattern = re.compile(file_name_pattern)

    fs = get_filesystem(path=directory_path, anon=True)

    matched_files = []

    for root, dirs, files in fs.walk(directory_path):
        for file_name in files:
            if check_file_extension(
                path=file_name,
                accepted_file_extensions=accepted_file_extensions,
            ):
                if re.search(file_name_pattern, file_name):
                    matched_files.append(os.path.join(root, file_name))
                else:
                    continue
            else:
                continue

    if is_s3_path(path=directory_path):
        matched_files = [f"s3://{file}" for file in matched_files]
    elif is_gcsfs_path(path=directory_path):
        matched_files = [f"gs://{file}" for file in matched_files]
    return matched_files


def find_geotiff_files(
    directory_path: str, file_name_pattern: str = ".*"
) -> list[str]:
    """
    Recursively find geotiff files matching an optional filename pattern.
    """
    geotiff_file_extensions = [".tif", ".tiff", ".gtiff"]
    geotiff_file_paths = find_files_by_extension(
        directory_path=directory_path,
        accepted_file_extensions=geotiff_file_extensions,
        file_name_pattern=file_name_pattern,
    )
    return geotiff_file_paths


def find_json_files(
    directory_path: str, file_name_pattern: str = ".*"
) -> list[str]:
    """
    Recursively find json files matching an optional filename pattern.
    """
    json_file_extensions = [".json"]
    json_file_paths = find_files_by_extension(
        directory_path=directory_path,
        accepted_file_extensions=json_file_extensions,
        file_name_pattern=file_name_pattern,
    )
    return json_file_paths


def find_csv_files(
    directory_path: str, file_name_pattern: str = ".*"
) -> list[str]:
    """
    Recursively find csv files matching an optional filename pattern.
    """
    csv_file_extensions = [".csv"]
    csv_file_paths = find_files_by_extension(
        directory_path=directory_path,
        accepted_file_extensions=csv_file_extensions,
        file_name_pattern=file_name_pattern,
    )
    return csv_file_paths


def get_wq_dataset_path(
    output_directory: str,
    tile_id: tuple[int, int],
    temporal_id: str,
    product_name: str,
    product_version: str,
) -> str:
    """
    Return the dataset path for a task.

    The dataset path is the directory where all water quality variables
    and associated metadata for the task are stored.
    """
    product_version_dashed = product_version.replace(".", "-")

    region_code = get_region_code(tile_id, sep="/")

    dataset_path = join_url(
        output_directory,
        product_name,
        product_version_dashed,
        region_code,
        temporal_id,
    )

    if not check_directory_exists(dataset_path):
        fs = get_filesystem(dataset_path, anon=False)
        fs.makedirs(dataset_path, exist_ok=True)

    return dataset_path


def get_wq_dataset_id(dataset_path: str) -> str:
    """
    Return a unique dataset ID from a dataset path.

    e.g. wqs_annual_x217y077_2024--P1Y
    """
    dataset_path = dataset_path.rstrip("/")
    # Parse the dataset path
    if is_local_path(dataset_path):
        (
            product_name,
            product_version_dashed,
            region_code_x,
            region_code_y,
            temporal_id,
        ) = Path(dataset_path).resolve().parts[-5:]
    else:
        (
            product_name,
            product_version_dashed,
            region_code_x,
            region_code_y,
            temporal_id,
        ) = URL(dataset_path).parts[-5:]

    dataset_id = f"{product_name}_{region_code_x}{region_code_y}_{temporal_id}"
    return dataset_id


def get_wq_stac_url(dataset_path: str) -> str:
    """
    Return the file path for writing the dataset metadata STAC file.
    """
    dataset_id = get_wq_dataset_id(dataset_path)
    output_path = join_url(dataset_path, f"{dataset_id}.stac-item.json")
    return output_path


def get_wq_cog_url(
    output_directory: str,
    tile_id: tuple[int, int],
    temporal_id: str,
    band_name: str,
    product_name: str,
    product_version: str,
) -> str:
    """
    Return the path or URL for writing a water variable COG file.
    """
    parent_dir = get_wq_dataset_path(
        output_directory=output_directory,
        tile_id=tile_id,
        temporal_id=temporal_id,
        product_name=product_name,
        product_version=product_version,
    )

    # f"{product_name}_{region_code}_{temporal_id}_{band}.tif"
    region_code = get_region_code(tile_id, sep="")
    file_name = f"{product_name}_{region_code}_{temporal_id}_{band_name}.tif"
    cog_url = join_url(parent_dir, file_name)
    return cog_url


def parse_wq_cog_url(cog_url: str) -> tuple[str, str, str, str]:
    """
    Parse the filename of a water quality variable COG URL.

    Extracts the product name, region code, temporal ID, and band name.

    Parameters
    ----------
    cog_url : str
        Path or URL to a water quality variable COG.

    Returns
    -------
    tuple[str, str, str, str]
        A tuple containing the product name, region code, temporal ID,
        and band name.
    """
    # f"{product_name}_{region_code}_{temporal_id}_{band}.tif"
    base = get_basename(cog_url)
    base = os.path.splitext(base)[0]
    parts = base.split("_")
    if len(parts) < 4:
        raise ValueError("Filename does not contain enough parts")

    region_code = get_region_code(parse_region_code(base), sep="")
    temporal_id = [p for p in parts if "--P" in p][0]

    product_name = "_".join(parts[: parts.index(region_code)])
    band = "_".join(parts[parts.index(temporal_id) + 1 :])

    return product_name, region_code, temporal_id, band


def get_wq_csv_url(
    output_directory: str,
    tile_id: tuple[int, int],
    temporal_id: str,
    product_name: str,
    product_version: str,
) -> str:
    """
    Return the path or URL for writing a table with TSS and Chl-a water
    quality variables for a task.
    """
    parent_dir = get_wq_dataset_path(
        output_directory=output_directory,
        tile_id=tile_id,
        temporal_id=temporal_id,
        product_name=product_name,
        product_version=product_version,
    )
    # f"{product_name}_{region_code}_{temporal_id}_{band}.tif"
    region_code = get_region_code(tile_id, sep="")
    file_name = f"{product_name}_{region_code}_{temporal_id}_water_quality_variables.csv"
    csv_url = join_url(parent_dir, file_name)
    return csv_url


def get_gdal_vsi_prefix(file_path) -> str:
    # Based on file extension
    _, file_extension = os.path.splitext(file_path)
    if file_extension in [".zip"]:
        vsi_prefix_1 = "vsizip"
    elif file_extension in [".gz"]:
        vsi_prefix_1 = "vsigzip"
    elif file_extension in [".tar", ".tgz"]:
        vsi_prefix_1 = "vsitar"
    elif file_extension in [".7z"]:
        vsi_prefix_1 = "vsi7z"
    elif file_extension in [".rar"]:
        vsi_prefix_1 = "vsirar"
    else:
        vsi_prefix_1 = ""

    if vsi_prefix_1:
        vsi_prefix_1_file_path = f"/{vsi_prefix_1}/{file_path}"
    else:
        vsi_prefix_1_file_path = file_path

    # Network based
    if is_local_path(file_path):
        return vsi_prefix_1_file_path
    elif is_http_url(file_path):
        return f"/vsicurl/{vsi_prefix_1_file_path}"
    elif is_s3_path(file_path):
        return f"/vsis3/{vsi_prefix_1_file_path}"
    elif is_gcsfs_path(file_path):
        return f"/vsigs/{vsi_prefix_1_file_path}"
    else:
        NotImplementedError()
