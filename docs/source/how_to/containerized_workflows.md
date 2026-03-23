# Run workflows in containers

Deploy DataEval workflows as a container — pull a pre-built image, write a
config file, and launch with bind-mounted data.

## Prerequisites

- Docker Engine 20.10+ (or Docker Desktop)
- Your dataset saved to disk in a [supported format](#dataset-formats)
- For GPU variants: NVIDIA Container Toolkit (`nvidia-container-toolkit`)

## 1. Pull the Docker image

Pre-built images are published to the JATIC Harbor registry:

```bash
# CPU-only
docker pull harbor.jatic.net:443/aria/dataeval:cpu

# GPU (CUDA 12.8 — recommended for modern GPUs)
docker pull harbor.jatic.net:443/aria/dataeval:cu128
```

Available variants:

| Tag | Base | Use case |
| --- | --- | --- |
| `cpu` | Ubuntu 24.04 | Machines without NVIDIA GPU |
| `cu118` | Ubuntu 22.04 | Older GPUs / CUDA 11.8 drivers |
| `cu124` | Ubuntu 22.04 | Mid-range GPUs / CUDA 12.4 drivers |
| `cu128` | Ubuntu 24.04 | Modern GPUs (RTX 40/50 series) / CUDA 12.8 drivers |

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
| `selections` | No | Named selection/filtering pipelines |
| `sources` | Yes | Bundles a dataset with an optional selection |
| `preprocessors` | No | Named preprocessing pipelines (torchvision transforms) |
| `extractors` | No | Model + optional preprocessor + batch size |
| `workflows` | Yes | Named workflow instances (type + parameters) |
| `tasks` | Yes | Lightweight composition — references a workflow, sources, and optional extractor |
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
    classes_file: classes.txt

  # YOLO format
  - name: yolo_train
    format: yolo
    path: yolo-data
    images_dir: images
    labels_dir: labels
    classes_file: classes.txt
```

### Sources, extractors, and selections

Sources bundle a dataset with an optional selection. Extractors bundle a model
with an optional preprocessor. Tasks reference these by name.

```yaml
selections:
  - name: first_5k
    steps:
      - type: Limit
        params:
          size: 5000

sources:
  - name: train_full
    dataset: hf_train

  - name: train_subset
    dataset: hf_train
    selection: first_5k

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

Five workflow types are available. Define named instances in the `workflows`
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
    harbor.jatic.net:443/aria/dataeval:cpu
```

### GPU

Add `--gpus all` and use a CUDA variant:

```bash
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    --mount type=bind,source="$(pwd)/workspace/cache",target=/cache \
    harbor.jatic.net:443/aria/dataeval:cu128
```

### Specifying a config file

Point at a specific config file or folder within your data directory:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net:443/aria/dataeval:cpu \
    python src/container_run.py --config config/params.yaml
```

You can also mount a config directory independently from your data. Use a
separate bind mount and pass the container-side path with `--config`:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/config",target=/config,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net:443/aria/dataeval:cpu \
    python src/container_run.py --config /config/params.yaml
```

### Verbosity

Pass `-v` flags to increase output detail:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net:443/aria/dataeval:cpu \
    python src/container_run.py -v
```

| Flag | Level |
| --- | --- |
| `-v` | Show full report output |
| `-vv` | Report + INFO logging |
| `-vvv` | Report + DEBUG logging |

## 5. View results

Results are written to the output mount under a directory named after each task:

```bash
ls workspace/output/
# clean_my_data/
#   results.json
```

The JSON report contains the workflow results — the same data you'd get from
`result.report(format="json")` in the Python API.

## Container mount reference

| Mount point | Required | Mode | Purpose |
| --- | --- | --- | --- |
| `/dataeval` | Yes | read-only | Data root — datasets, models, and config files |
| `/output` | Yes | read-write | Reports and results |
| `/cache` | No | read-write | Embedding and stats cache (speeds up re-runs) |

## Troubleshooting

Run the container with `--help` to see full usage:

```bash
docker run harbor.jatic.net:443/aria/dataeval:cpu python src/container_run.py --help
```

Common issues:

- **"Data directory not found or not mounted"** — verify the `--mount source=` path exists on the host
- **"No tasks defined in config"** — ensure `params.yaml` has a `tasks` list
- **"No GPU detected"** — add `--gpus all` to the `docker run` command, or use the `:cpu` image
- **"Output mount not writable"** — pass `--user "$(id -u):$(id -g)"` or `chmod 777` the host directory
- **"Permission denied"** — check host directory permissions with `ls -la`
