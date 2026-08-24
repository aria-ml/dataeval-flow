# Run workflows in containers

Deploy DataEval Flow workflows as a container — pull a pre-built image, write a
config file, and launch with bind-mounted data.

## Used in these tutorials

Every DataEval Flow workflow can be run from a container, so this guide is
referenced from all of the tutorials:

- {doc}`Run a full evaluation pipeline end to end <../notebooks/end_to_end>`
- {doc}`Clean a dataset <../notebooks/data_cleaning>`
- {doc}`Analyze dataset quality across splits <../notebooks/data_analysis>`
- {doc}`Assess dataset coverage <../notebooks/data_coverage>`
- {doc}`Split a dataset <../notebooks/dataset_splitting>`
- {doc}`Monitor incoming data for drift <../notebooks/drift_monitoring>`
- {doc}`Detect classwise drift <../notebooks/classwise_drift>`
- {doc}`Detect out-of-distribution samples <../notebooks/ood_detection>`
- {doc}`Prioritize unlabeled data for labeling <../notebooks/data_prioritization>`
- {doc}`Parameter Sweep for Data Cleaning <../notebooks/parameter_sweep>`

## Prerequisites

- Docker Engine 20.10+ (or Docker Desktop)
- Your dataset saved to disk in a [supported format](#dataset-formats)
- For GPU variants: NVIDIA Container Toolkit (`nvidia-container-toolkit`)

## 1. Pull the Docker image

Pre-built images are published to the JATIC Harbor registry:

```bash
# CPU-only
docker pull harbor.jatic.net/aria/dataeval:cpu

# GPU (CUDA 13.0 — recommended for modern GPUs)
docker pull harbor.jatic.net/aria/dataeval:cu130
```

Available variants:

| Tag | Base | Use case |
| --- | --- | --- |
| `cpu` | Ubuntu 24.04 | Machines without NVIDIA GPU |
| `cu126` | Ubuntu 24.04 | Older GPUs / CUDA 12.6 drivers |
| `cu130` | Ubuntu 24.04 | Modern GPUs (RTX 50 series) / CUDA 13.0 drivers |

All GPU variants bundle their own CUDA runtime libraries via PyTorch — the
host only needs the NVIDIA driver and Container Toolkit.

## 2. Prepare the host directories

Create the directory layout the container expects:

```bash
mkdir -p workspace/{config,output,cache}
```

By default the container looks for config files inside the data mount
(`/dataeval`). You can also mount a config directory independently — see
[Specifying a config file](#specifying-a-config-file) for examples of both
approaches.

### File permissions

The container runs as a non-root user (`dataeval`). Mounted directories for
`/output` and `/cache` must be writable by the container process. Two options:

#### Option 1: Pass your host UID (recommended)

Use `--user` to run the container as your host user, so mounted directories
are naturally writable:

```bash
docker run --user "$(id -u):$(id -g)" ...
```

#### Option 2: Open directory permissions

Make the output and cache directories world-writable on the host:

```bash
chmod 777 workspace/output workspace/cache
```

## 3. Write the configuration file

Create `params.yaml` in your config directory (e.g. `workspace/config/params.yaml`)
or inside your data directory. The config follows a **define-once,
reference-by-name** pattern with these sections:

| Section | Required | Purpose |
| --- | --- | --- |
| `datasets` | Yes | Named dataset definitions |
| `views` | No | Named view pipelines (dataset operations) |
| `sources` | Yes | Bundles a dataset with an optional view |
| `preprocessors` | No | Named preprocessing pipelines (torchvision transforms) |
| `extractors` | No | Model + optional preprocessor + batch size |
| `metadata` | No | Named metadata policies (encoding, vocabularies, exclusions), referenced by workflows |
| `workflows` | Yes | Named workflow instances (type + parameters) |
| `tasks` | Yes | Lightweight composition — references a workflow, sources, and optional extractor |
| `seed` | No | Seed for every stochastic component of the run |
| `deterministic` | No | Force PyTorch deterministic algorithms (only meaningful alongside `seed`) |
| `logging` | No | App and library log levels |

The legacy `selections` / `selection` / `steps` keys are still accepted as deprecated
aliases for `views` / `view` / `operations`; new configs should use the current names.

(dataset-formats)=

### Dataset formats

The `datasets` section supports four formats:

```yaml
datasets:
  # HuggingFace arrow format
  - name: hf_train
    format: huggingface
    path: my-dataset       # relative to the data mount (/dataeval)
    split: train
    task: image_classification   # image_classification | object_detection

  # Local image directory
  - name: photos
    format: image_folder
    path: raw-photos
    recursive: false       # default: false
    infer_labels: false    # infer class labels from subdirectory names

  # COCO format
  - name: coco_train
    format: coco
    path: coco-data
    annotations_file: annotations.json
    images_dir: images

  # YOLO format
  - name: yolo_train
    format: yolo
    path: yolo-data        # dataset root: data.yaml + image/label trees
    split: train           # train | val | test; omit to load every split
```

Each object-detection format selects a split its own way. COCO has one
annotation file per split, so `annotations_file` (with `images_dir`, when the
file names are not relative to the dataset root) picks one:

```yaml
datasets:
  - name: coco_train
    format: coco
    path: coco
    annotations_file: annotations/instances_train2017.json
    images_dir: train2017

  - name: coco_val
    format: coco
    path: coco
    annotations_file: annotations/instances_val2017.json
    images_dir: val2017
```

YOLO keeps every split under one root, so `split` picks one — in either
Ultralytics arrangement (`images/train/` + `labels/train/`, or `train/images/` +
`train/labels/`):

```yaml
datasets:
  - name: yolo_train
    format: yolo
    path: yolo-data
    split: train

  - name: yolo_val
    format: yolo
    path: yolo-data
    split: val          # "validation" is accepted and normalizes to "val"
```

Keep `path` on the dataset root rather than pointing it at a split
subdirectory: the root is where `data.yaml` lives, and without it class names
fall back to the numeric ids from the label files. Two more optional YOLO
fields cover non-standard layouts — `yaml_file` for a config that is not at the
root under a conventional name, and `ann_dir` for labels kept outside the
`labels/` sibling of `images/`. Both are relative to `path`.

### Sources, extractors, and views

Sources bundle a dataset with an optional view. Extractors bundle a model
with an optional preprocessor. Tasks reference these by name.

```yaml
views:
  - name: first_5k
    operations:
      - type: Limit
        params:
          size: 5000

sources:
  - name: train_full
    dataset: hf_train

  - name: train_subset
    dataset: hf_train
    view: first_5k

extractors:
  - name: bovw_extractor
    model: bovw
    vocab_size: 2048       # 256–4096
    batch_size: 32

  - name: resnet_extractor
    model: onnx
    model_path: "./resnet50-v2-7.onnx"
    output_name: "resnetv24_flatten0_reshape0"
    preprocessor: resnet_preprocess   # references a preprocessors entry
    batch_size: 64
```

### Workflow types

Eight workflow types are available. Define named instances in the `workflows`
section, then reference them from tasks.

`````{tab-set}
````{tab-item} data-cleaning
Outlier and duplicate detection with configurable thresholds.
See the {doc}`Data Cleaning tutorial <../notebooks/data_cleaning>` for a full walkthrough.

```yaml
workflows:
  - name: standard_clean
    type: data-cleaning
    outlier_method: adaptive       # adaptive | zscore | modzscore | iqr
    outlier_flags:
      - dimension
      - pixel
      - visual
    outlier_threshold: 3.5         # optional custom threshold
    duplicate_cluster_sensitivity: 0.5
    duplicate_cluster_algorithm: hdbscan
    health_thresholds:
      exact_duplicates: 0.0
      near_duplicates: 5.0
      image_outliers: 5.0
    mode: advisory
```
````
````{tab-item} data-analysis
Statistical analysis including outliers, duplicates, diversity, and bias.
See the {doc}`Data Analysis tutorial <../notebooks/data_analysis>` for a full walkthrough.

```yaml
workflows:
  - name: full_analysis
    type: data-analysis
    outlier_method: adaptive
    outlier_flags: [dimension, pixel, visual]
    balance: true
    diversity_method: simpson       # simpson | shannon
    include_image_stats: true
    divergence_method: mst          # mst | fnn (cross-split)
```
````
````{tab-item} data-splitting
Partition a dataset into train/val/test splits.

```yaml
workflows:
  - name: stratified_split
    type: data-splitting
    test_frac: 0.2
    val_frac: 0.1
    stratify: true
    num_folds: 1
```
````
````{tab-item} drift-monitoring
Detect distribution drift between reference and test datasets.
See the {doc}`Drift Monitoring tutorial <../notebooks/drift_monitoring>` for a full walkthrough.

```yaml
workflows:
  - name: ks_drift
    type: drift-monitoring
    detectors:
      - method: univariate         # univariate | mmd | domain_classifier | kneighbors
        test: ks                   # ks | cvm | mwu | anderson | bws
        p_val: 0.05
        correction: bonferroni
```
````
````{tab-item} ood-detection
Identify out-of-distribution samples.
See the {doc}`OOD Detection tutorial <../notebooks/ood_detection>` for a full walkthrough.

```yaml
workflows:
  - name: ood_knn
    type: ood-detection
    detectors:
      - method: kneighbors         # kneighbors | domain_classifier
        k: 5
        distance_metric: cosine    # cosine | euclidean
        threshold_perc: 95
    metadata_insights: true
```
````

````{tab-item} data-coverage
Class balance, metadata gaps, ontology findings, and embedding blind spots.
See the {doc}`Data Coverage tutorial <../notebooks/data_coverage>` for a full walkthrough.

```yaml
workflows:
  - name: coverage_check
    type: data-coverage
    coverage_method: adaptive
    balance: true
    run_gap_analysis: true
    ontology: config/taxonomy.ttl   # optional; unlocks the label-space findings
    health_thresholds:
      leaf_coverage: 0.9
      dark_branch_count: 0
```
````

````{tab-item} data-prioritization
Rank an abundant or unlabeled pool so the most informative samples come first.
See the {doc}`Prioritization tutorial <../notebooks/data_prioritization>` for a full walkthrough.

```yaml
workflows:
  - name: label_next
    type: data-prioritization
    method: knn                    # knn | kmeans_distance | kmeans_complexity
                                   # | hdbscan_distance | hdbscan_complexity
    order: hard_first              # or easy_first
    policy: difficulty             # difficulty | stratified | class_balanced
```
````

````{tab-item} parameter-sweep
Sweep data-cleaning parameters across a grid and compare the results.
See the {doc}`Parameter Sweep tutorial <../notebooks/parameter_sweep>` for a full walkthrough.

A sweep varies data-cleaning parameters. There is no separate grid block: any
swept field takes a *list* of values and the workflow runs every combination,
while static fields take a single value shared across all runs.

```yaml
workflows:
  - name: threshold_sweep
    type: parameter-sweep
    outlier_flags: [dimension, pixel, visual]   # static — the stat groups to test
    outlier_method: [modzscore, iqr]            # swept — 2 values
    outlier_threshold: [2.5, 3.0, 3.5]          # swept — 3 values, so 6 runs
```
````
`````

### Tasks

Tasks tie everything together. Each task references a workflow, one or more
sources, and an optional extractor:

```yaml
tasks:
  - name: clean_train
    workflow: standard_clean
    sources: train_full
    extractor: bovw_extractor
    enabled: true                  # set false to skip (default: true)

  - name: analyze_all
    workflow: full_analysis
    sources:
      - train_full
      - train_subset
    extractor: resnet_extractor
```

### Complete example

A minimal end-to-end config for data cleaning:

```yaml
# workspace/config/params.yaml

datasets:
  - name: my_dataset
    format: huggingface
    path: my-dataset
    split: train
    task: image_classification

sources:
  - name: my_source
    dataset: my_dataset

extractors:
  - name: bovw
    model: bovw
    vocab_size: 512
    batch_size: 32

workflows:
  - name: clean
    type: data-cleaning
    outlier_method: adaptive
    outlier_flags: [dimension, pixel, visual]

tasks:
  - name: clean_my_data
    workflow: clean
    sources: my_source
    extractor: bovw
```

```{tip}
The repository includes annotated example configs at `config/params.example.yaml`
and `config/params.multi-dataset.example.yaml`. A JSON Schema is available at
`config/params.schema.json` for IDE autocompletion.
```

(run-the-container)=

## 4. Run the container

### CPU

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    --mount type=bind,source="$(pwd)/workspace/cache",target=/cache \
    harbor.jatic.net/aria/dataeval:cpu
```

### GPU

Add `--gpus all` and use a CUDA variant:

```bash
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    --mount type=bind,source="$(pwd)/workspace/cache",target=/cache \
    harbor.jatic.net/aria/dataeval:cu130
```

### Specifying a config file

Point at a specific config file or folder within your data directory:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval:cpu \
    python -m dataeval_flow --config config/params.yaml
```

You can also mount a config directory independently from your data. Use a
separate bind mount and pass the container-side path with `--config`:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/config",target=/config,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval:cpu \
    python -m dataeval_flow --config /config/params.yaml
```

### Verbosity

Pass `-v` flags to increase output detail:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval:cpu \
    python -m dataeval_flow -v
```

| Flag | Level |
| --- | --- |
| `-v` | Show full report output |
| `-vv` | Report + INFO logging |
| `-vvv` | Report + DEBUG logging |

### Running a subset of the tasks

`--task` runs one task by name; repeat it to run several, in the order given. A named
task runs whether or not its config entry sets `enabled: false`, so you can keep a task
defined but dormant and still reach for it when you need it:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval:cpu \
    python -m dataeval_flow --task clean_my_data
```

With no `--task`, every task the config marks `enabled` runs.

### Failing the pipeline on health warnings

By default the run exits `0` whenever every task *ran*, whatever its findings say — a
warning is a prompt to look, not a failure. `--fail-on-warning` makes findings that
breached their health thresholds fatal instead, so a CI job can gate on data quality:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval:cpu \
    python -m dataeval_flow --output /output --fail-on-warning
```

| Exit code | Meaning |
| --- | --- |
| `0` | Every task succeeded; no warnings, or warnings without `--fail-on-warning` |
| `1` | A task failed, or `--fail-on-warning` was set and a task raised warnings |

Results are written either way — the gate decides the exit code, not whether the run's
artifacts survive.

### Discovering the available workflows

The image ships without the TUI extra, so `workflows` is how you ask it what it can run
and what a given workflow type accepts:

```bash
docker run --rm harbor.jatic.net/aria/dataeval:cpu \
    python -m dataeval_flow workflows

docker run --rm harbor.jatic.net/aria/dataeval:cpu \
    python -m dataeval_flow workflows data-cleaning
```

`python -m dataeval_flow --version` reports the build inside the image.

## 5. View results

Results are written to the output mount as one merged file per format, plus a run log:

```bash
find workspace/output -type f
# workspace/output/result.log
# workspace/output/results/result.json
# workspace/output/results/result.txt
# workspace/output/results/encoding.json
```

`result.json` is keyed by task name — each entry holds that task's `metadata`, `health`,
`raw`, and `report` sections, the same data you'd get from `result.to_dict()` in the Python
API. `health` is the roll-up `--fail-on-warning` gates on, so a pipeline can read it
directly rather than parsing the text report:

```bash
jq -r 'to_entries[] | "\(.key)\t\(.value.health.status)\t\(.value.health.warnings)"' \
    workspace/output/results/result.json
```

`result.txt` holds the detailed text reports, the same as `result.report()`.
`encoding.json` is the metadata encoding descriptor the run was computed under, ready
to review and commit — see
{doc}`Configure metadata binning <configure_metadata_binning>`. It is omitted when a
run's tasks encoded their factors differently, since no single descriptor describes it.

```bash
jq -r 'keys[]' workspace/output/results/result.json
jq -r '.clean_my_data.report.findings[] | "\(.severity)\t\(.title)"' \
    workspace/output/results/result.json
```

## Container mount reference

| Mount point | Required | Mode | Purpose |
| --- | --- | --- | --- |
| `/dataeval` | Yes | read-only | Data root — datasets, models, and config files |
| `/output` | Yes | read-write | Reports and results |
| `/cache` | No | read-write | Embedding and stats cache (speeds up re-runs) |

## Troubleshooting

Run the container with `--help` to see full usage:

```bash
docker run harbor.jatic.net/aria/dataeval:cpu python -m dataeval_flow --help
```

Common issues:

- **"Data directory not found or not mounted"** — verify the `--mount source=` path exists on the host
- **"No tasks defined in config"** — ensure `params.yaml` has a `tasks` list
- **"No GPU detected"** — add `--gpus all` to the `docker run` command, or use the `:cpu` image
- **"Output mount not writable"** — pass `--user "$(id -u):$(id -g)"` or `chmod 777` the host directory
- **"Permission denied"** — check host directory permissions with `ls -la`
