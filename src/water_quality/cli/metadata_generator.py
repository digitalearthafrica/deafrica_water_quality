import json
import sys
import warnings

import click

from water_quality.cli.common import split_tasks
from water_quality.io import (
    check_directory_exists,
    check_file_exists,
    find_geotiff_files,
    get_filesystem,
    get_parent_dir,
    get_wq_stac_url,
    join_url,
)
from water_quality.logs import setup_logging
from water_quality.metadata.prepare_metadata import prepare_dataset


@click.command(
    "create-stac-files",
    no_args_is_help=True,
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
@click.argument(
    "datasets-dir",
    type=str,
)
@click.argument(
    "max-parallel-steps",
    type=int,
)
@click.argument(
    "worker-idx",
    type=int,
)
def cli(
    overwrite: bool,
    datasets_dir: str,
    max_parallel_steps: int,
    worker_idx: int,
):
    """Generate STAC metadata files for water quality datasets in DATASETS_DIR.

    MAX_PARALLEL_STEPS indicates the total number of parallel workers
    processing tasks, and WORKER_IDX indicates the index of this worker
    (0-indexed).
    """
    log = setup_logging()

    all_geotiffs = find_geotiff_files(datasets_dir)
    all_dataset_paths = list(set(get_parent_dir(i) for i in all_geotiffs))
    all_dataset_paths.sort()
    log.info(f"Found {len(all_dataset_paths)} datasets")

    datasets_to_run = split_tasks(
        all_dataset_paths, max_parallel_steps, worker_idx
    )

    if not datasets_to_run:
        log.warning(
            f"Worker {worker_idx} has no datasets to process. Exiting."
        )
        sys.exit(0)

    log.info(f"Worker {worker_idx} processing {len(datasets_to_run)} datasets")

    failed_tasks = []
    for idx, dataset_path in enumerate(datasets_to_run):
        log.info(
            f"Generating stac file for {dataset_path} {idx + 1}/{len(datasets_to_run)}"
        )
        output_stac_url = get_wq_stac_url(dataset_path)
        exists = check_file_exists(output_stac_url)
        if not overwrite and exists:
            log.info(
                f"{output_stac_url} exists! Skipping processing dataset {dataset_path}"
            )
            continue
        else:
            try:
                # Generate STAC metadata
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    log.info("Creating metadata STAC file ...")
                    stac_file_url = prepare_dataset(
                        dataset_path=dataset_path,
                        source_datasets_uuids=None,
                    )
            except Exception as e:
                log.error(
                    f"Failed to generate STAC file for dataset {dataset_path}: {e}"
                )
                failed_tasks.append(dataset_path)
                continue

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
        log.error(f"Failed tasks: {failed_tasks_json_array}")
        log.info(f"Failed tasks written to {failed_tasks_output_file}")
        sys.exit(1)
    else:
        log.info(f"Worker {worker_idx} completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    cli()
