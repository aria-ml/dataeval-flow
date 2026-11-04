# Write a pipeline config

A pipeline config names each part once and references it by name. To run a workflow, declare its datasets, sources,
extractors, workflow and task in `params.yaml`. [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md)
explains how a workflow type runs, and the [Preset Catalog](../reference/presets.md) lists every setting of each type.
You need your dataset saved to disk in one of the formats below.

## Lay out the config sections

The config follows a define-once, reference-by-name pattern with these sections:

| Section | Required | Purpose |
| --- | --- | --- |
| `datasets` | Yes | Named dataset definitions |
| `views` | No | Named view pipelines (dataset operations) |
| `sources` | Yes | Bundles a dataset with an optional view |
| `preprocessors` | No | Named preprocessing pipelines (torchvision transforms) |
| `extractors` | No | Model + optional preprocessor + batch size |
| `metadata` | No | Named metadata policies (encoding, vocabularies, exclusions), referenced by workflows |
| `workflows` | Yes | Named workflow instances (type + parameters) |
| `tasks` | Yes | Lightweight composition: references a workflow, sources, and optional extractor |
| `seed` | No | Seed for every stochastic component of the run |
| `deterministic` | No | Force PyTorch deterministic algorithms (only meaningful alongside `seed`) |
| `logging` | No | App and library log levels |

(dataset-formats)=

## Declare a dataset

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

Each object-detection format selects a split its own way.

### Select a COCO split

COCO has one annotation file per split, so `annotations_file` (with `images_dir`, when the file names are not relative
to the dataset root) picks one:

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

### Select a YOLO split

YOLO keeps every split under one root, so `split` picks one, in either Ultralytics arrangement (`images/train/` +
`labels/train/`, or `train/images/` + `train/labels/`):

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

Keep `path` on the dataset root rather than pointing it at a split subdirectory: the root is where `data.yaml` lives,
and without it class names fall back to the numeric ids from the label files. Two more optional YOLO fields cover
non-standard layouts: `yaml_file` for a config that is not at the root under a conventional name, and `ann_dir` for
labels kept outside the `labels/` sibling of `images/`. Both are relative to `path`.

## Bundle sources, extractors and views

Sources bundle a dataset with an optional view. Extractors bundle a model with an optional preprocessor. Tasks
reference these by name.

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

## Choose a workflow type

Nine workflow types are built in. Define named instances in the `workflows` section, then reference them from tasks.

`````{tab-set}
````{tab-item} quality
The [`quality`](../reference/presets.md#quality) workflow runs outlier and duplicate detection with configurable
thresholds. See the {doc}`../notebooks/data_cleaning` tutorial for a full walkthrough.

```yaml
workflows:
  - name: standard_clean
    type: quality
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
The [`audit`](../reference/presets.md#audit) workflow gives a verdict on one or more splits before training, with a
record of what was audited. The task's first source is train. See the {doc}`../notebooks/audit` tutorial for a full
walkthrough.

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
````{tab-item} splits
The [`splits`](../reference/presets.md#splits) workflow partitions a dataset into train/val/test splits. See the
{doc}`../notebooks/dataset_splitting` tutorial for a full walkthrough.

```yaml
workflows:
  - name: stratified_split
    type: splits
    folds: 1                # 1 splits once; 2 or more run k-fold
    test_frac: 0.2
    val_frac: 0.1           # with folds: 1 only
    stratify: true
```
````
````{tab-item} shift (drift)
The [`shift`](../reference/presets.md#shift) workflow detects distribution drift between a reference and each test
dataset. See the {doc}`../notebooks/drift_monitoring` tutorial for a full walkthrough, the
{doc}`../notebooks/classwise_drift` tutorial for the `classwise` setting, {doc}`test_merged_sources_for_drift` for merging
test sources, and {doc}`monitor_drift` for testing by class.

```yaml
workflows:
  - name: ks_drift
    type: shift
    detectors:
      - type: drift-univariate     # drift-univariate | drift-mmd | drift-domain-classifier | drift-kneighbors
        method: ks                 # ks | cvm | mwu | anderson | bws
        p_val: 0.05
        correction: bonferroni
    classwise: {drift-univariate: class}  # also run this detector once per class
```
````
````{tab-item} shift (OOD)
The `shift` workflow identifies out-of-distribution images, by each detector and by their agreement. See the
{doc}`../notebooks/ood_detection` tutorial for a full walkthrough.

```yaml
workflows:
  - name: ood_knn
    type: shift
    detectors:
      - type: ood-kneighbors       # ood-kneighbors | ood-domain-classifier
        k: 5
        distance_metric: cosine    # cosine | euclidean
        threshold_perc: 95
    factor-deviation: {max_items: 50}   # false leaves it out; `factor-predictors: false` leaves out that step
```
````

````{tab-item} scope
The [`scope`](../reference/presets.md#scope) workflow finds embedding blind spots and the classes short of their
share. A `bias` entry on the same source judges class balance and metadata gaps, and a `taxonomy` entry judges the
labels against an ontology. See the {doc}`../notebooks/data_coverage` tutorial for a full walkthrough.

```yaml
workflows:
  - name: coverage_check
    type: scope
    coverage: {method: adaptive}     # adaptive | naive; embeds only when the task names an extractor
    checks:
      dimensional-completeness: {warning: 0.5}
```
````

````{tab-item} bias
The [`bias`](../reference/presets.md#bias) workflow judges class balance, and how the metadata factors relate to the
class: shortcut risk, association and under-represented combinations. It needs no extractor.

```yaml
workflows:
  - name: bias_check
    type: bias
    factor-gaps: {mi_threshold: 0.1, min_representation: 5}   # false leaves out the gap analysis
    checks:
      class-imbalance: {warning: 5.0}
      shortcut-risk: {warning: 0.1}
      factor-coverage-gaps: {warning: 2}
```
````

````{tab-item} taxonomy
The [`taxonomy`](../reference/presets.md#taxonomy) workflow judges a dataset's labels against a declared ontology: leaf
coverage, conformance, alignment and structure. See {doc}`declare_an_ontology` for the options.

```yaml
workflows:
  - name: vocab_check
    type: taxonomy
    ontology: config/taxonomy.ttl    # an ontologies: entry, an RDF file, or an inline hierarchy
    checks:
      leaf-coverage: {coverage: 0.9, empty_branches: 0}
      label-conformance: {warning: 0}
```
````

````{tab-item} triage
The [`triage`](../reference/presets.md#triage) workflow finds the metadata a run failed to read, and the policy stanza
that repairs it. See the {doc}`../notebooks/metadata_triage` tutorial for a full walkthrough.

```yaml
workflows:
  - name: triage
    type: triage
    metadata: standard             # the policy under triage
    checks:
      factor-issues: {max_examples: 20}
    verify: true                   # re-read the metadata under the suggestions
```
````

````{tab-item} prioritization
The [`prioritization`](../reference/presets.md#prioritization) workflow ranks an abundant or unlabeled pool so the most
informative samples come first. See the {doc}`../notebooks/data_prioritization` tutorial for a full walkthrough.

```yaml
workflows:
  - name: label_next
    type: prioritization
    prioritization:
      method: knn                  # knn | kmeans_distance | kmeans_complexity
                                   # | hdbscan_distance | hdbscan_complexity
      order: hard_first            # or easy_first
      policy: difficulty           # difficulty | stratified | class_balanced
    select:
      n: 200                       # keep each pool's top 200 as `selected`; omit to keep all
```

To rank clean data, run it as a step after a `quality` step, as the
[Preset Catalog](../reference/presets.md#prioritization) shows.
````

````{tab-item} matrix
Run any workflow once per combination of its settings and compare the runs in one table. A matrix is not a workflow
type: it goes on the task, and the entry it varies is an ordinary one, valid on its own. See {doc}`run_a_matrix` for
every way to write one, and the {doc}`../notebooks/tune_data_cleaning` tutorial for a worked run.

```yaml
workflows:
  - name: threshold_tuning
    type: quality
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

## Compose the tasks

Each task references a workflow, one or more sources, and an optional extractor:

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

## Put the sections together

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
    type: quality
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
`config/params.schema.json` for IDE autocompletion, and the
{ref}`Config Reference <config-reference>` lists every key it takes.
```

## Next steps

- [Run a full evaluation pipeline end to end](../notebooks/end_to_end.py) — Run a pipeline defined in
  `end_to_end.yaml` from the data to the report.
- [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) — Learn how a workflow type runs as a chain of
  steps.
- [Run workflows in containers](containerized_workflows.md) — Launch the pipeline from a container with bind-mounted
  data.
