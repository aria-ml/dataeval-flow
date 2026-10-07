# How-to Guides

Task-oriented guides for specific goals with DataEval Flow. Each takes one problem and walks through the solution.
New to DataEval Flow? Start with the {doc}`Quickstart <../home/quickstart>` and the
{doc}`Tutorials <../tutorials/index>`. These guides assume you have run a workflow before.
If you know your question but not the step, see {doc}`Find the Right Step <../reference/index>`.

The guides are grouped by the part of a pipeline they address.

## Configuring the data

Getting the right data in the right representation in front of a workflow.

```{toctree}
:hidden:

build_dataset_views
../notebooks/torchvision_datasets
declare_an_ontology
measure_band_groups
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`Narrow a dataset with views <build_dataset_views>`
  - Limit, filter, shuffle, or index into a dataset before a workflow sees it — and keep it reproducible.
- - {doc}`Use a torchvision dataset <../notebooks/torchvision_datasets>`
  - Feed a `torchvision` classification or detection dataset straight into a workflow.
- - {doc}`Declare an ontology <declare_an_ontology>`
  - Define the sanctioned label space so `taxonomy` can name classes that were never collected.
- - {doc}`Measure band groups <measure_band_groups>`
  - Measure a dataset's channels separately, and its image background, without moving any cleaning result or bias
    number until a policy names them.

:::

## Choosing a representation

Most evaluators measure in embedding space; these guides cover getting there.

```{toctree}
:hidden:

../notebooks/onnx_embeddings
torch_embeddings
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`Use an ONNX model for embeddings <../notebooks/onnx_embeddings>`
  - Configure a pretrained ONNX model with preprocessing transforms for higher-fidelity embeddings.
- - {doc}`Use a PyTorch model for embeddings <torch_embeddings>`
  - Read an intermediate layer of your own `.pt` model — including the model under test.

:::

## Tuning a workflow

```{toctree}
:hidden:

configure_outlier_detection
configure_metadata_binning
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`Configure outlier detection <configure_outlier_detection>`
  - Pick a statistical method, choose which statistics to test, add cluster-based detection, and set the thresholds
    that turn a finding into a warning.
- - {doc}`Configure metadata binning <configure_metadata_binning>`
  - Choose how factors are discretized before bias and coverage analyses read them, declare the range of float
    image data, and read back what the run decided.

:::

## Building a workflow of your own

```{toctree}
:hidden:

write_a_custom_workflow
monitor_drift
reuse_a_workflow
run_a_matrix
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`Chain steps into a workflow of your own <write_a_custom_workflow>`
  - Conform and merge two datasets, remove their duplicates, export the result, and check, split and audit what is
    left, as one chain of steps.
- - {doc}`Monitor drift with steps <monitor_drift>`
  - Read what the drift preset makes for each test source, merge sources to test them as one, compare classes or
    groups, and drift on the crops of detection data.
- - {doc}`Reuse a cleaning chain on new data <reuse_a_workflow>`
  - Keep a chain in your config, in a file each project loads, or saved from Python, and run it on each new dataset.
- - {doc}`Sweep settings with a matrix <run_a_matrix>`
  - Run a task once per combination of settings, sources or extractors, and compare every run's findings in one
    table.

:::

## Running and reading results

```{toctree}
:hidden:

run_a_single_evaluator
evaluator_recipes
read_evaluation_outputs
gate_training_on_an_audit
../notebooks/view_html_reports
export_a_dataset
reuse_results_with_cache
containerized_workflows
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`Run a single evaluator <run_a_single_evaluator>`
  - Run one DataEval evaluator, such as finding duplicates, and read its output with no health status.
- - {doc}`Evaluator recipes <evaluator_recipes>`
  - One worked example per question, each answered by evaluators: bias, representation, coverage, prioritization,
    drift and out-of-distribution.
- - {doc}`Read evaluation outputs <read_evaluation_outputs>`
  - Interpret the report and its severities, export the result envelope, and reach the raw numbers behind a finding.
- - {doc}`Gate training on an audit <gate_training_on_an_audit>`
  - Gate a run on an audit's verdict with `--require`, refuse data whose content digest doesn't match the record, and
    find which items changed.
- - {doc}`View a report as HTML <../notebooks/view_html_reports>`
  - Render a result as one self-contained page with cards, sortable and filterable tables, and each flag's
    measurements on hover.
- - {doc}`Export a dataset <export_a_dataset>`
  - Write a conformed or merged source out as a dataset on disk, with the provenance that produced it.
- - {doc}`Reuse results with the disk cache <reuse_results_with_cache>`
  - Persist embeddings and statistics across runs, and know what invalidates them.
- - {doc}`Run workflows in containers <containerized_workflows>`
  - Pull a pre-built image, write a config, and launch with bind-mounted data.

:::
