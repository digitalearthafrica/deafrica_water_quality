# deafrica_water_quality

repo for code development to produce a water quality service organized by work packages, at least in part.

## wq command line tools

### Installation

Clone this repository and `cd` into the repository folder `cd deafrica_water_quality/`.
Run the following command to install the python package:
`pip install -e .`

### Usage

The `water-quality` packages provides a set of tools to generate water quality variables from a variety of Earth Observation sensors.

#### Steps to run wq

#### 1. Save tasks

From your Sandbox (or a machine that has access to your ODC database), run:

```bash
wq-generate-tasks tasks.txt
```

The output of this command is a text file containing a list of task ids for the prospective run.

``` bash
Usage: wq-generate-tasks [OPTIONS] START_YEAR END_YEAR OUTPUT_FILE

  Prepare tasks for the time range START_YEAR to END_YEAR for running the DE
  Africa Water Quality continental workflow on. If no tile IDs are specified,
  the tasks will be generated for all the tiles across Africa. The tasks will
  be written to the file OUTPUT_FILE.

Options:
  --tile-ids TEXT       Optional list of comma separated tile IDs in the
                        format x{x:02d}/y{y:02d} to generate tasks for. For
                        example `x188/y109,x178/y095,x199y/100`
  --tile-ids-file TEXT  Optional path to text file containing the tile IDs to
                        generate tasks for. This file can be generated using
                        the command `wq-generate-tiles`.
  --place-name TEXT     Optional name of a test area to get the tile IDs to
                        generate tasks for. To view the names of these
                        predefined test areas, run the command `wq-list-test-
                        areas`.
  --help                Show this message and exit.


```

#### 2. Process the tasks

To process the tasks generated from the previous step, the command `wq-process-tasks` requires:

1. The file containing the task IDs for the tasks to process, generated from the previous step.
2. The output directory to write the COG fileS containing the water quality variables generated for each processed task.
3. The total number of parallel workers or pods expected in
  the workflow. This value is used to divide the list of input tiles among the
  available workers.
4. The sequential index (0-indexed) of the current worker. This
  index determines which subset of tiles the current worker will process.
5. A YAML configuration file describing the time range for the analysis, the instruments to use, water frequency thresholds, etc.

``` bash
Usage: wq-process-tasks [OPTIONS] OUTPUT_DIRECTORY MAX_PARALLEL_STEPS
                        WORKER_IDX

  Get the Water Quality variables for the input tasks and write the resulting
  water quality variables to COG files.

  OUTPUT_DIRECTORY: The directory to write the water quality variables COG
  files to for each task.

  MAX_PARALLEL_STEPS: The total number of parallel workers or pods expected in
  the workflow. This value is used to divide the list of tasks to be processed
  among the available workers.

  WORKER_IDX: The sequential index (0-indexed) of the current worker. This
  index determines which subset of tasks the current worker will process.

Options:
  --tasks TEXT                  List of comma separated tasks in the
                                formatyear/x{x:02d}/y{y:02d} to generate water
                                quality variables for. For example
                                `2015/x200/y34,x178/y095,x199y/100`
  --tasks-file TEXT             Optional path to a text file containing the
                                tasks to generatewater quality variables for.
                                This file can be generated using the command
                                `wq-generate-tiles`.
  --analysis-config TEXT        Config for the analysis parameters in yaml
                                format, file or text  [required]
  --overwrite / --no-overwrite  If overwrite is True tasks that have already
                                been processed will be rerun.   [default: no-
                                overwrite]
  --help                        Show this message and exit.


```

Sample `cfg.yaml`:

````yaml
resolution: 30
instruments_to_use:
  oli_agm:
    use: true
  oli:
    use: false
  msi_agm:
    use: true
  msi:
    use: false
  tm_agm:
    use: true
  tm:
    use: false
  tirs:
    use: true
  wofs_ann:
    use: true
  wofs_all:
    use: false
water_frequency_threshold_high: 0.5
water_frequency_threshold_low: 0.1
permanent_water_threshold : 0.0875
sigma_coefficient : 1.2
````

## Water Quality

The file `"wq_tile_ids_and_waterbodies_uids.parquet"` in the data folder matches the `uids` for all waterbodies in the DE Africa Historical Extent product with the tile IDs  of the tiles each waterbody intersects with.
This file is to be updated with each new release of the DE Africa Historical Extent product using the script `data/tiles_and_waterbody_uids.py`.
