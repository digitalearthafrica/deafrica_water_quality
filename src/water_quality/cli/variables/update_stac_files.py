import json
import logging
import sys

import click
from odc.aws import s3_url_parse
from s3fs.core import S3FileSystem

from water_quality.io import check_directory_exists, get_filesystem, join_url
from water_quality.logs import setup_logging


def self_link_is_dev_bucket(content: str, bucket: str) -> bool:
    """
    Check if the file path of a STAC object's own metadata file
    contains the specified bucket. This assumes if the bucket is
    contained in the metadata file path, it is also contained in the
    file paths for the assets.

    Parameters
    ----------
    content : str
        STAC file content as a string.
    bucket : str
        The bucket name to check for in the self link.

    Returns
    -------
    bool
        True if the self link contains the specified bucket, False otherwise.

    """
    stac = json.loads(content)

    links = stac.get("links", None)

    if not links:
        raise ValueError("No links found in message...")

    for link in links:
        if link["rel"] == "self":
            if f"s3://{bucket}/" in link["href"]:
                return True

    # Assume it's fine, so more likely to not process it.
    return False


@click.command(
    "update-stac-files",
    no_args_is_help=True,
)
@click.option(
    "--source-bucket",
    type=str,
    required=True,
    help="The s3 bucket containing the dataset STAC metadata files.",
)
@click.option(
    "--dev-bucket",
    type=str,
    required=True,
    help="The s3 bucket to replace in the self link and asset paths of the STAC files.",
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
@click.argument(
    "stac-path-template",
    type=str,
)
def cli(
    source_bucket: str,
    dev_bucket: str,
    log: str,
    stac_path_template: str,
):
    """
    Update the s3 bucket name in the STAC files matching the STAC_PATH_TEMPLATE
    pattern if the self link in a STAC file contains the dev bucket name.

    This is to ensure that the STAC files created in the DEV environment when copied over
    to the PDS bucket reference the correct file paths.
    """
    log_level = getattr(logging, log.upper())
    _log = setup_logging(log_level)

    fs = S3FileSystem(
        anon=False,
        # Use profile only on sandbox
        # profile="default",
        s3_additional_kwargs={"ACL": "bucket-owner-full-control"},
    )

    stac_files = fs.glob(stac_path_template)
    stac_files = {f"s3://{file}" for file in set(stac_files)}

    if len(stac_files) == 0:
        _log.info(
            f"Found {len(stac_files)} files matching the pattern {stac_path_template}"
        )
        sys.exit(0)
    else:
        _log.info(
            f"Found {len(stac_files)} matching the pattern {stac_path_template}"
        )

    failed_tasks = []
    for idx, file_path in enumerate(stac_files):
        _log.info(
            f"Processing file {idx + 1} of {len(stac_files)}: {file_path} "
        )
        try:
            expected_bucket = s3_url_parse(file_path)[0]
            assert expected_bucket == source_bucket

            with fs.open(file_path) as f:
                content = json.dumps(json.load(f))

            if self_link_is_dev_bucket(content, dev_bucket):
                _log.info(
                    f"Updating dev bucket {dev_bucket} with source {expected_bucket}"
                )
                content = content.replace(dev_bucket, expected_bucket)

                with fs.open(file_path, "w") as file:
                    json.dump(
                        json.loads(content), file, indent=2
                    )  # `indent=4` makes it human-readable

                _log.info(f"Successfully updated file: {file_path}")
            else:
                _log.info(
                    f"Self links and asset paths in {file_path} do not contain"
                    f" dev bucket {dev_bucket}, so skipping update for this file."
                )
        except Exception as error:
            _log.exception(error)
            failed_tasks.append(file_path)

    # Handle failed tasks
    if failed_tasks:
        failed_tasks_json_array = json.dumps(failed_tasks)

        tasks_directory = "/tmp/"
        failed_tasks_output_file = join_url(tasks_directory, "failed_tasks")

        fs = get_filesystem(path=tasks_directory, anon=False)
        if not check_directory_exists(path=tasks_directory):
            fs.mkdirs(path=tasks_directory, exist_ok=True)

        with fs.open(failed_tasks_output_file, "a") as file:
            file.write(failed_tasks_json_array + "\n")
        _log.error(f"Failed tasks: {failed_tasks_json_array}")
        _log.info(f"Failed tasks written to {failed_tasks_output_file}")
        sys.exit(1)
    else:
        _log.info("Worker completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    cli()
