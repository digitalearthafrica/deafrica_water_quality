"""
Create per dataset metadata (stac files) for the Water Quality Service
products
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

import rioxarray
import toolz
import yaml
from eodatasets3.images import ValidDataMethod
from odc.apps.dc_tools._docs import odc_uuid
from odc.stats.model import DateTimeRange

from water_quality.io import (
    find_geotiff_files,
    get_filesystem,
    get_parent_dir,
    get_wq_dataset_id,
    get_wq_stac_url,
    parse_wq_cog_url,
)
from water_quality.mapping.algorithms import NORMALISATION_PARAMETERS
from water_quality.metadata.easi_assemble import EasiPrepare

log = logging.getLogger(__name__)


def generate_annual_product_template(tile_directory: str) -> dict[str, Any]:
    """
    Generate a ODC product template file for the annual water
    quality variables, based on all the data available for the years
    2000-2024.

    Parameters
    ----------
    tile_directory : str
        S3 or local directory containing all the COG files for a single
        tile across multiple years.

    Returns
    -------
    dict[str, Any]
        ODC product template for the product wq_annual_v1 product.
    """
    all_tile_geotifs = find_geotiff_files(tile_directory)

    datasets = toolz.groupby(lambda f: get_parent_dir(f), all_tile_geotifs)

    # Compile a list of unique measurements and their metadata
    #  across the years of data available for the tile.
    measurements = []
    for dataset_path, dataset_measurements in datasets.items():
        for measurement_path in dataset_measurements:
            measurement_info = {}
            product_name, region_code, temporal_id, band = parse_wq_cog_url(
                measurement_path
            )

            scale_and_offset = NORMALISATION_PARAMETERS.get(band, None)
            if scale_and_offset is not None:
                scale_factor = scale_and_offset["scale"]
                add_offset = scale_and_offset["offset"]
            else:
                scale_factor = 1.0
                add_offset = 0.0

            exists = any(item.get("name") == band for item in measurements)
            if exists:
                continue
            else:
                measurement_info["name"] = band
                measurement_info["dtype"] = "float32"
                measurement_info["nodata"] = "NaN"
                measurement_info["units"] = "1"
                measurement_info["scale_factor"] = scale_factor
                measurement_info["add_offset"] = add_offset

                measurements.append(measurement_info)

    measurements = sorted(measurements, key=lambda x: x["name"])
    # Create the product config.
    description = "DE Africa Water Quality Service annual water quality variables at 10m resolution - version 1."
    metadata_type = "eo3"
    license = "CC-BY-4.0"
    load = dict(crs="EPSG:6933", resolution=dict(x=10, y=-10))
    metadata = dict(product_name=product_name)

    product_config = dict(
        name=product_name,
        description=description,
        metadata_type=metadata_type,
        license=license,
        load=load,
        metadata=metadata,
        measurements=measurements,
    )

    output_yaml = "wq_annual.odc-product.yaml"

    fs = get_filesystem(output_yaml, anon=False)
    with fs.open(output_yaml, "w") as stream:
        yaml.dump_all(
            [product_config],
            stream,
            sort_keys=False,
            explicit_start=True,
            explicit_end=True,
            indent=2,
        )
    log.info(f"{product_name} ODC product template written to {output_yaml} ")
    return product_config


def get_dummy_product_yaml(
    dataset_path: str,
) -> dict[str, Any]:
    """
    Generate an ODC product definition file for a dataset, with only
    the measurements (bands) available for the dataset. This is to
    account for the fact not all datasets for a water quality service
    product will have the same measurements due to varying availability
    of instrument data across the years.

    Parameters
    ----------
    dataset_path : str
        Path or URI to the dataset directory

    Returns
    -------
    dict[str, Any]
        ODC product definition for the product a dataset belongs to.
    """
    measurements = []
    dataset_measurements = find_geotiff_files(dataset_path)
    for measurement_path in dataset_measurements:
        measurement_info = {}
        product_name, region_code, temporal_id, band = parse_wq_cog_url(
            measurement_path
        )

        scale_and_offset = NORMALISATION_PARAMETERS.get(band, None)
        if scale_and_offset is not None:
            scale_factor = scale_and_offset["scale"]
            add_offset = scale_and_offset["offset"]
        else:
            scale_factor = 1.0
            add_offset = 0.0

        exists = any(item.get("name") == band for item in measurements)
        if exists:
            continue
        else:
            measurement_info["name"] = band
            measurement_info["dtype"] = "float32"
            measurement_info["nodata"] = "NaN"
            measurement_info["units"] = "1"
            measurement_info["scale_factor"] = scale_factor
            measurement_info["add_offset"] = add_offset

            measurements.append(measurement_info)

    measurements = sorted(measurements, key=lambda x: x["name"])

    # Create a dummy product config with the measurements
    # specific to a dataset.
    description = f"DE Africa Water Quality Service {product_name}."
    metadata_type = "eo3"
    license = "CC-BY-4.0"
    load = dict(crs="EPSG:6933", resolution=dict(x=10, y=-10))
    metadata = dict(product_name=product_name)

    product_config = dict(
        name=product_name,
        description=description,
        metadata_type=metadata_type,
        license=license,
        load=load,
        metadata=metadata,
        measurements=measurements,
    )

    return product_config


def get_common_attrs(dataset_measurement_url: str) -> dict:
    """Get the attributes from a single dataset measurement that are
    expected to be common to all measurements for a dataset."""

    # Attributes located in the metadata of the measurement GeoTIFF file
    common_attrs = rioxarray.open_rasterio(dataset_measurement_url).attrs

    if "product_version" not in common_attrs:
        common_attrs["product_version"] = "1.0.0"
    # Attributes from the file name of the measurement GeoTIFF file
    _, region_code, temporal_id, _ = parse_wq_cog_url(dataset_measurement_url)
    temporal_range = DateTimeRange(temporal_id)

    common_attrs.update(
        dict(
            region_code=region_code,
            date_time=temporal_range.start,
            date_time_range=(temporal_range.start, temporal_range.end),
        )
    )
    return common_attrs


def prepare_dataset(
    dataset_path: str,
    source_datasets_uuids: list[UUID] = None,
) -> str:
    """Prepares a STAC dataset metadata file for a data product.

    Parameters
    ----------
    dataset_path : str
        Directory of the dataset. The dataset path is the directory
        where all water quality variables and associated metadata for
        a processed task are stored.
    source_datasets_uuids : list[UUID]
        Option list of the UUIDs for datasets that were used in the
        calculation of this dataset.
    Returns
    -------
    str
        Path to odc dataset STAC file
    """
    dataset_id = get_wq_dataset_id(dataset_path)
    output_path = get_wq_stac_url(dataset_path)

    product_definition = get_dummy_product_yaml(dataset_path)

    ## Initialise and validate inputs
    # Creates variables (see EasiPrepare for others):
    # - p.dataset_path
    # - p.product_name
    p = EasiPrepare(
        dataset_path=dataset_path,
        product_yaml=product_definition,
        output_path=output_path,
        input_source_datasets=source_datasets_uuids,
    )

    # Find all measurement paths for a dataset
    dataset_id_regex = rf"{dataset_id}_(.*?)\.tif$"
    measurement_map = p.map_measurements_to_paths(dataset_id_regex)

    # Get attrs from one of the measurement files
    common_attrs = get_common_attrs(list(measurement_map.values())[0])

    ## IDs and Labels
    # The version of the source dataset
    p.dataset_version = f"{common_attrs['product_version']}"
    p.dataset_id = odc_uuid(p.product_name, p.dataset_version, [dataset_id])
    # product_name is added by EasiPrepare().init()
    p.product_uri = (
        f"https://explorer.digitalearth.africa/product/{p.product_name}"
    )

    ## Satellite, Instrument and Processing level
    # High-level name for the source data (satellite platform or project name).
    # Comma-separated for multiple platforms.
    p.platform = "WaterQualityService"
    #  Instrument name, optional
    # p.instrument = 'SAMPLETYPE'
    # Organisation that produces the data
    # URI domain format containing a '.'
    p.producer = "digitalearthafrica.org"
    # ODC/EASI identifier for this "family" of products, optional
    p.product_family = "water_quality_service"
    # Helpful but not critical
    p.properties["odc:file_format"] = "GeoTIFF"
    p.properties["odc:product"] = p.product_name

    ## Scene capture and Processing
    # Searchable datetime of the dataset, datetime object
    p.datetime = common_attrs["date_time"]
    # Searchable start and end datetimes of the dataset, datetime objects
    p.datetime_range = common_attrs["date_time_range"]
    # When the source dataset was created by the producer, datetime object
    p.processed = datetime.now()

    ## Geometry
    # Geometry adds a "valid data" polygon for the scene, which helps bounding box searching in ODC
    # Either provide a "valid data" polygon or calculate it from all bands in the dataset
    # ValidDataMethod.thorough = Vectorize the full valid pixel mask as-is
    # ValidDataMethod.filled = Fill holes in the valid pixel mask before vectorizing
    # ValidDataMethod.convex_hull = Take convex-hull of valid pixel mask before vectorizing
    # ValidDataMethod.bounds = Use the image file bounds, ignoring actual pixel values
    # p.geometry = Provide a "valid data" polygon rather than read from the file, shapely.geometry.base.BaseGeometry()
    # p.crs = Provide a CRS string if measurements GridSpec.crs is None, "epsg:*" or WKT
    p.valid_data_method = ValidDataMethod.bounds

    ## Scene metrics, as available
    # The "region" of acquisition, if applicable
    p.region_code = common_attrs["region_code"]

    # Add measurement paths
    for measurement_name, file_location in measurement_map.items():
        p.note_measurement(
            measurement_name=measurement_name,
            file_path=file_location,
            expand_valid_data=True,
            relative_to_metadata=False,
        )
    ## Complete validate and return
    # validation is against the eo3 specification and your product/dataset definitions
    try:
        dataset_uuid, output_path = p.write_stac(
            validate_correctness=True, sort_measurements=True
        )
    except Exception as error:
        raise error
    log.info(f"Wrote dataset {dataset_uuid} to {output_path}")
    return output_path
