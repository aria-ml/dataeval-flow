# Tutorials

Learning-oriented guides that walk you through a complete T&E task with DataEval Flow, step by step. Each one states
who it is for, where the task fits in a larger workflow, and ends with links to the how-to guides for its component
steps.

New to DataEval Flow? Run the {doc}`Quickstart <../home/quickstart>` first — it installs the package and gets one
evaluation working end to end.

The tutorials are grouped by where the task falls in the data lifecycle.

## Start here

The whole operating loop once, start to finish, before drilling into any single workflow.

```{toctree}
:hidden:

../notebooks/end_to_end
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`Run a full evaluation pipeline end to end <../notebooks/end_to_end>`
  - Load a dataset, write one config, run three workflows against it, read and export the
    results — then run the identical pipeline in a container.

:::

## Preparing data

Assessing and conditioning a dataset before it is used to train or evaluate.

```{toctree}
:hidden:

../notebooks/data_cleaning
../notebooks/data_analysis
../notebooks/data_coverage
../notebooks/dataset_splitting
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`Clean a dataset <../notebooks/data_cleaning>`
  - Flag outliers, duplicates, and label problems before the data is used downstream.
- - {doc}`Analyze dataset quality across splits <../notebooks/data_analysis>`
  - Profile quality, bias, and cross-split leakage in a single multi-split report.
- - {doc}`Assess dataset coverage <../notebooks/data_coverage>`
  - Find class imbalance, metadata gaps, missing label-space regions, and embedding blind spots.
- - {doc}`Split a dataset <../notebooks/dataset_splitting>`
  - Build defensible, leakage-free train/validation/test splits.

:::

## Curating and tuning

Deciding what to label next, and choosing the parameters an evaluation runs with.

```{toctree}
:hidden:

../notebooks/data_prioritization
../notebooks/parameter_sweep
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`Prioritize unlabeled data for labeling <../notebooks/data_prioritization>`
  - Rank an abundant unlabeled pool so the most informative samples are annotated first.
- - {doc}`Parameter Sweep for Data Cleaning <../notebooks/parameter_sweep>`
  - Run a workflow across a grid of parameters and compare the results side by side.

:::

## Monitoring a deployment

Watching operational data for the shifts that degrade a deployed model.

```{toctree}
:hidden:

../notebooks/drift_monitoring
../notebooks/classwise_drift
../notebooks/ood_detection
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`Monitor incoming data for drift <../notebooks/drift_monitoring>`
  - Detect population-level shift in operational batches against a reference dataset.
- - {doc}`Detect classwise drift <../notebooks/classwise_drift>`
  - Localize a shift to the specific classes responsible for it.
- - {doc}`Detect out-of-distribution samples <../notebooks/ood_detection>`
  - Score individual incoming samples rather than a whole batch.

:::
