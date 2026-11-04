# Tutorials

Each tutorial walks through a complete T&E task with DataEval Flow. It states who it is for and where the task falls in
a larger workflow. It ends with the guides to read next and the settings to change when you run it on your own data.

If you are new to DataEval Flow, start with the {doc}`Quickstart <../home/quickstart>`. It installs the package and gets
one evaluation working end to end.

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

- - {doc}`../notebooks/end_to_end`
  - Load a dataset, write one config, run three workflows against it, read and export the results, then run the
    identical pipeline in a container.

:::

## Preparing data

Assessing and conditioning a dataset before it is used to train or evaluate.

```{toctree}
:hidden:

../notebooks/metadata_triage
../notebooks/data_cleaning
../notebooks/data_coverage
../notebooks/dataset_splitting
../notebooks/audit
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`../notebooks/metadata_triage`
  - Find the metadata columns a run silently failed to read, and get a config change for each.
- - {doc}`../notebooks/data_cleaning`
  - Flag outliers, duplicates, and label problems before the data is used downstream.
- - {doc}`../notebooks/data_coverage`
  - Find class imbalance, metadata gaps, missing label-space regions, and embedding blind spots.
- - {doc}`../notebooks/dataset_splitting`
  - Build defensible, leakage-free train/validation/test splits.
- - {doc}`../notebooks/audit`
  - The gate before training: get a verdict on train and evaluation splits, accept a risk in writing, and keep a
    record of what was audited.

:::

## Curating and tuning

Deciding what to label next, whether the data can support a performance target, and choosing the parameters an
evaluation runs with.

```{toctree}
:hidden:

../notebooks/data_prioritization
../notebooks/measure_data_sufficiency
../notebooks/check_target_feasibility
../notebooks/tune_data_cleaning
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`../notebooks/data_prioritization`
  - Rank an abundant unlabeled pool so the most informative samples are annotated first.
- - {doc}`../notebooks/measure_data_sufficiency`
  - Train a model on growing subsets of a training set, and learn whether more data would reach a target score and how
    much it would take.
- - {doc}`../notebooks/check_target_feasibility`
  - Ask whether your labeled images show that a model can reach an accuracy you need, and read each answer with the
    proof and the assumptions it rests on.
- - {doc}`../notebooks/tune_data_cleaning`
  - Run data cleaning across a grid of thresholds, compare every run's findings in one table, and choose settings.

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

- - {doc}`../notebooks/drift_monitoring`
  - Detect population-level shift in operational batches against a reference dataset.
- - {doc}`../notebooks/classwise_drift`
  - Localize a shift to the specific classes responsible for it.
- - {doc}`../notebooks/ood_detection`
  - Score individual incoming images rather than a whole batch.

:::
