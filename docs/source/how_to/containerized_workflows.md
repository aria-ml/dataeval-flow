# Run workflows in containers

Deploy DataEval Flow workflows as a container — pull a pre-built image, write a
config file, and launch with bind-mounted data.

## Used in these tutorials

Every DataEval Flow workflow can be run from a container, so this guide is
referenced from all of the tutorials:

- {doc}`Run a full evaluation pipeline end to end <../notebooks/end_to_end>`
- {doc}`Clean a dataset <../notebooks/data_cleaning>`
- {doc}`Audit a set of splits before training <../notebooks/audit>`
- {doc}`Assess dataset coverage <../notebooks/data_coverage>`
- {doc}`Split a dataset <../notebooks/dataset_splitting>`
- {doc}`Monitor incoming data for drift <../notebooks/drift_monitoring>`
- {doc}`Detect classwise drift <../notebooks/classwise_drift>`
- {doc}`Detect out-of-distribution samples <../notebooks/ood_detection>`
- {doc}`Prioritize unlabeled data for labeling <../notebooks/data_prioritization>`
- {doc}`Tune data cleaning with a matrix <../notebooks/tune_data_cleaning>`

## Prerequisites

- Docker Engine 20.10+ (or Docker Desktop)
- Your dataset saved to disk in a [supported format](#dataset-formats)
- For GPU variants: NVIDIA Container Toolkit (`nvidia-container-toolkit`)

## 1. Pull the Docker image

Pre-built images are published to the JATIC Harbor registry:

```bash
# CPU-only
docker pull harbor.jatic.net/aria/dataeval-flow:latest-cpu

# GPU (CUDA 13.0 — recommended for modern GPUs)
docker pull harbor.jatic.net/aria/dataeval-flow:latest-cu130
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

Nine workflow types are built in. Define named instances in the `workflows`
section, then reference them from tasks.

`````{tab-set}
````{tab-item} data-cleaning
Outlier and duplicate detection with configurable thresholds.
See the {doc}`Data Cleaning tutorial <../notebooks/data_cleaning>` for a full walkthrough.

```yaml
workflows:
  - name: standard_clean
    type: data-cleaning
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: [adaptive, 3.5]   # method | [method, bound]: adaptive | zscore | modzscore | iqr
    duplicates:
      cluster_sensitivity: 0.5
      cluster_algorithm: hdbscan
    checks:
      image-duplicates: {exact: 0.0, near: 5.0}
      image-outliers: {warning: 5.0}
```
````
````{tab-item} audit
A verdict on one or more splits before training, with a record of what was audited. The task's first source is train.
See the {doc}`audit tutorial <../notebooks/audit>` for a full walkthrough.

```yaml
metadata:
  - name: standard
    intrinsic_factors: [visual, pixel]   # measured off the imagery, then binned like any factor

workflows:
  - name: release_audit
    type: audit
    metadata: standard
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: adaptive   # method | [method, bound]: adaptive | zscore | modzscore | iqr
    diversity: {method: simpson}    # simpson | shannon
    divergence: {method: mst}       # mst | fnn (cross-split; needs the task's extractor)
    accepted:
      class-imbalance: "Rare class by design; weighted loss in training."
```
````
````{tab-item} data-splitting
Partition a dataset into train/val/test splits.

```yaml
workflows:
  - name: stratified_split
    type: data-splitting
    folds: 1                # 1 splits once; 2 or more run k-fold
    test_frac: 0.2
    val_frac: 0.1           # with folds: 1 only
    stratify: true
```
````
````{tab-item} drift-monitoring
Detect distribution drift between a reference and each test dataset.
See the {doc}`Drift Monitoring tutorial <../notebooks/drift_monitoring>` for a full walkthrough, and
{doc}`Monitor drift with steps <monitor_drift>` for merging test sources and testing by class.

```yaml
workflows:
  - name: ks_drift
    type: drift-monitoring
    detectors:
      - type: drift-univariate     # drift-univariate | drift-mmd | drift-domain-classifier | drift-kneighbors
        method: ks                 # ks | cvm | mwu | anderson | bws
        p_val: 0.05
        correction: bonferroni
    classwise: {drift-univariate: class}  # also run this detector once per class
```
````
````{tab-item} ood-detection
Identify out-of-distribution images, by each detector and by their agreement.
See the {doc}`OOD Detection tutorial <../notebooks/ood_detection>` for a full walkthrough.

```yaml
workflows:
  - name: ood_knn
    type: ood-detection
    detectors:
      - type: ood-kneighbors       # ood-kneighbors | ood-domain-classifier
        k: 5
        distance_metric: cosine    # cosine | euclidean
        threshold_perc: 95
    factor-deviation: {max_items: 50}   # false leaves it out; `factor-predictors: false` leaves out that step
```
````

````{tab-item} data-coverage
Class balance, metadata gaps, and embedding blind spots; a `label-space` entry on the same source judges the labels
against an ontology.
See the {doc}`Data Coverage tutorial <../notebooks/data_coverage>` for a full walkthrough.

```yaml
workflows:
  - name: coverage_check
    type: data-coverage
    coverage: {method: adaptive}     # adaptive | naive; embeds only when the task names an extractor
    factor-gaps: {mi_threshold: 0.1, min_representation: 5}   # false leaves out the gap analysis
    checks:
      class-imbalance: {warning: 5.0}
      factor-coverage-gaps: {warning: 2}
```
````

````{tab-item} label-space
Judge a dataset's labels against a declared ontology: leaf coverage, conformance, alignment and structure.
See {doc}`Declare an ontology <declare_an_ontology>` for the options.

```yaml
workflows:
  - name: vocab_check
    type: label-space
    ontology: config/taxonomy.ttl    # an ontologies: entry, an RDF file, or an inline hierarchy
    checks:
      leaf-coverage: {coverage: 0.9, empty_branches: 0}
      label-conformance: {warning: 0}
```
````

````{tab-item} metadata-triage
Find the metadata a run failed to read, and the policy stanza that repairs it.
See the {doc}`Metadata Triage tutorial <../notebooks/metadata_triage>` for a full walkthrough.

```yaml
workflows:
  - name: triage
    type: metadata-triage
    metadata: standard             # the policy under triage
    checks:
      metadata-issues: {max_examples: 20}
    verify: true                   # re-read the metadata under the suggestions
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
    select:
      n: 200                       # keep each pool's top 200 as `selected`; omit to keep all
```

To rank clean data, run it as a step after a `data-cleaning` step, as the
[Preset Catalog](../reference/presets.md#data-prioritization) shows.
````

````{tab-item} matrix
Run any workflow once per combination of its settings and compare the runs in
one table. A matrix is not a workflow type: it goes on the task, and the entry
it varies is an ordinary one, valid on its own. See
{doc}`Sweep settings with a matrix <run_a_matrix>` for every way to write one,
and the {doc}`Tune data cleaning with a matrix <../notebooks/tune_data_cleaning>`
tutorial for a worked run.

```yaml
workflows:
  - name: threshold_tuning
    type: data-cleaning
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: modzscore

tasks:
  - name: tune_train
    workflow: threshold_tuning
    sources: train_full
    matrix:
      outlier_threshold: [2.5, 3.0, 3.5]   # 3 runs, compared in one result
```
````
`````

### Tasks

Each task references a workflow, one or more sources, and an optional
extractor:

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
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: adaptive

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
    harbor.jatic.net/aria/dataeval-flow:latest-cpu
```

### GPU

Add `--gpus all` and use a CUDA variant:

```bash
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    --mount type=bind,source="$(pwd)/workspace/cache",target=/cache \
    harbor.jatic.net/aria/dataeval-flow:latest-cu130
```

The image and the GPUs it sees decide the device, not the config: tools compute on a GPU where PyTorch sees one, and
on the CPU otherwise. `--gpus device=1` runs on the second GPU, and `-e CUDA_VISIBLE_DEVICES=` keeps a CUDA image on
the CPU. Each result's `metadata.device` records the device its task ran on.

### Specifying a config file

Point at a specific config file or folder within your data directory:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval-flow:latest-cpu \
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
    harbor.jatic.net/aria/dataeval-flow:latest-cpu \
    python -m dataeval_flow --config /config/params.yaml
```

### Verbosity

Pass `-v` flags to increase output detail:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval-flow:latest-cpu \
    python -m dataeval_flow -v
```

| Flag | Level |
| --- | --- |
| `-v` | Show full report output |
| `-vv` | Report + INFO logging |
| `-vvv` | Report + DEBUG logging |

### Running a subset of the tasks

`--task` runs one task by name. Repeat it to run several, in the order given. A named
task runs whether its config entry sets `enabled: false`, so you can keep a task
defined but dormant.

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval-flow:latest-cpu \
    python -m dataeval_flow --task clean_my_data
```

With no `--task`, every task the config marks `enabled` runs.

### Failing the pipeline on health warnings

By default the run exits `0` whenever every task *ran*, whatever its findings say. A
warning is a prompt to look, not a failure. `--fail-on-warning`, or `fail_on: warning` in the
pipeline's `result:` block, makes findings that passed their checks' thresholds fatal, so a CI
job can gate on data quality:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval-flow:latest-cpu \
    python -m dataeval_flow --output /output --fail-on-warning
```

| Exit code | Meaning |
| --- | --- |
| `0` | Every task succeeded, with no warnings the gate fails on; or `fail_on: never` with no `require` |
| `1` | A task failed, or an export couldn't be written |
| `2` | The command line was mistyped: an unknown flag, or a value it refuses |
| `3` | Every task succeeded, but a task raised warnings and the gate fails on them |
| `4` | A task's verdict is worse than `result: require:` or `--require` |

When more than one applies, `1` comes first, then `4`, then `3`. Exit `4` gates on an `audit`'s
verdict; see [Gate training on an audit](gate_training_on_an_audit.md#read-the-verdict). `fail_on`
in the `result:` block sets the gate: `failure` (the default), `warning`, or `never`, which fails
the job on neither a failed task nor a warning, though a set `require` still exits `4`.
`--fail-on-warning` and `--no-fail-on-warning`, or `DATAEVAL_FAIL_ON_WARNING`, override it. Since a
warning exits `3`, a job can tell a data-quality gate from a crash or a mistyped flag:

```yaml
# .gitlab-ci.yml
data-quality:
  script:
    - python -m dataeval_flow --output output
  allow_failure:
    exit_codes: [3]     # a warning marks the job, a failed task fails the pipeline
  artifacts:
    when: always
    paths: [output/results/]
    reports:
      junit: output/results/result.xml
```

with `fail_on: warning` and `formats: [json, html, junit, markdown]` in the pipeline's `result:`
block. With `per_task: true`, each task writes its own report, so `junit:` takes a glob,
`output/results/result-*.xml`. The JUnit report makes each task a test suite and each finding a
test case, failing where the finding is a warning, so the merge request's test view lists the
checks the data didn't pass. The Markdown summary is each task's findings as a table, ready for
`$GITHUB_STEP_SUMMARY` or a merge-request comment. Both also name any task that failed.

Results are written either way — the gate decides the exit code, not whether the run's
artifacts survive.

### Discovering the available workflows

The image ships without the TUI extra, so `workflows` is how you ask it what it can run
and what a given workflow type accepts:

```bash
docker run --rm harbor.jatic.net/aria/dataeval-flow:latest-cpu \
    python -m dataeval_flow workflows

docker run --rm harbor.jatic.net/aria/dataeval-flow:latest-cpu \
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
# workspace/output/results/result.html
```

`result.json` is keyed by task name — each entry holds that task's `metadata`, `health`,
`raw`, and `report` sections, the same data `result.to_dict()` returns in the Python
API. A `data-cleaning` task's entry, like a custom workflow's, holds `steps` and
`findings` in place of `raw` and `report`: each step's outcome, and the findings its
checks made. `health` is the roll-up `--fail-on-warning` gates on. A pipeline can read it
directly:

```bash
jq -r 'to_entries[] | "\(.key)\t\(.value.health.status)\t\(.value.health.warnings)"' \
    workspace/output/results/result.json
```

`result.txt` holds the detailed text reports, the same as `result.report()`. `result.html`
holds the same reports as one self-contained page, the same as `result.to_html()`: it opens
offline in any browser, and prints or saves to PDF as it shows.
`encoding.json` is the metadata encoding descriptor the run was computed under, ready
to review and commit — see
{doc}`Configure metadata binning <configure_metadata_binning>`. It is written only where
a task records an encoding, as `audit`, `data-coverage` and `ood-detection` do, so
the `data-cleaning` run above writes none. It is also omitted when a run's tasks encoded
their factors differently, since no single descriptor describes it.
`manifests/` holds each `content-digest` step's manifest, one hash per item, under
`manifests/<task>/`. It is written only where a task digests a split, as `audit` does; see
[Gate training on an audit](gate_training_on_an_audit.md#find-what-changed).

A pipeline's `result:` block shapes these files. Every key is optional:

```yaml
result:
  name: release             # release.json, release.txt, release.html (default: result)
  formats: [json, html]     # json, text, html, junit (.xml), markdown (.md) (default: json, text, html)
  detail: summary           # the text and HTML files' detail: full or summary (default: full)
  per_task: true            # one set of files per task: release-<task>.json, … (default: false)
  fail_on: warning          # what fails the job: failure, warning or never (default: failure)
  require: ready-with-accepted-risks  # the worst verdict that passes, else exit 4 (default: none)
  width: 100                # the text report's width, at least 40 (default: 80)
  max_images: 100           # thumbnails per task's result; 0: none, -1: every item named (default: 200)
  max_rows: 1000            # rows a table of items lists; -1: every row (default: 500)
  preview_rows: 20          # rows of it the text report shows; -1: every row (default: 10)
```

`--report-width` and `DATAEVAL_REPORT_WIDTH` override `width`. The console keeps printing the summary, or the full
report with `-v`, whatever `detail` says.

```bash
jq -r 'keys[]' workspace/output/results/result.json
jq -r '.clean_my_data.findings[] | "\(.severity)\t\(.title)"' \
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
docker run harbor.jatic.net/aria/dataeval-flow:latest-cpu python -m dataeval_flow --help
```

Common issues:

- **"Data directory not found or not mounted"** — verify the `--mount source=` path exists on the host
- **"No tasks defined in config"** — ensure `params.yaml` has a `tasks` list
- **"No GPU detected"** — add `--gpus all` to the `docker run` command, or use the `:cpu` image
- **"Output mount not writable"** — pass `--user "$(id -u):$(id -g)"` or `chmod 777` the host directory
- **"Permission denied"** — check host directory permissions with `ls -la`
