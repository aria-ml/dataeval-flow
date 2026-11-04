# How-to guides

Each guide takes one task for DataEval Flow and shows the shortest way to do it. If you are new to DataEval Flow, start
with the {doc}`Quickstart <../home/quickstart>` and the {doc}`Tutorials <../tutorials/index>`. These guides assume you
have run a workflow before. If you know your question but not the step, see {doc}`Find the Right Step <../reference/index>`.

The guides are grouped by the part of a pipeline they address.

## Configuring the data

Getting the right data in the right representation in front of a workflow.

```{toctree}
:hidden:

build_dataset_views
merge_sources
../notebooks/torchvision_datasets
declare_an_ontology
measure_band_groups
declare_a_value_range
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`build_dataset_views`
  - Limit, filter, shuffle or index into a dataset before a workflow sees it, and keep the result reproducible.
- - {doc}`merge_sources`
  - Conform two datasets to a shared vocabulary with `Relabel` and combine them with `merge:`.
- - {doc}`../notebooks/torchvision_datasets`
  - Feed a `torchvision` classification or detection dataset straight into a workflow.
- - {doc}`declare_an_ontology`
  - Define the sanctioned label space so `taxonomy` can name classes that were never collected.
- - {doc}`measure_band_groups`
  - Measure a dataset's channels separately, and its image background, without moving any cleaning result or bias
    number until a policy names them.
- - {doc}`declare_a_value_range`
  - Declare `value_range` on a float dataset so the image statistics that need it stop answering `NaN`.

:::

## Choosing a representation

Most evaluators measure in embedding space. These guides cover getting there.

```{toctree}
:hidden:

../notebooks/onnx_embeddings
torch_embeddings
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`../notebooks/onnx_embeddings`
  - Configure a pretrained ONNX model with preprocessing transforms for higher-fidelity embeddings.
- - {doc}`torch_embeddings`
  - Read an intermediate layer of your own `.pt` model, including the model under test.

:::

## Tuning a workflow

```{toctree}
:hidden:

configure_outlier_detection
configure_metadata_binning
read_metadata_binning
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`configure_outlier_detection`
  - Pick a statistical method, choose which statistics to test, add cluster-based detection, and set the thresholds
    that turn a finding into a warning.
- - {doc}`configure_metadata_binning`
  - Choose how factors are discretized before bias and coverage analyses read them, and drop the factors that are not
    evidence.
- - {doc}`read_metadata_binning`
  - Read the binning record from the report and from Python, and tell whether two results are comparable.

:::

## Running one evaluator

An evaluator reports DataEval's own answer to one question and renders no health status.

```{toctree}
:hidden:

run_a_single_evaluator
look_for_metadata_shortcuts
check_class_representation
find_uncovered_regions
rank_data_to_label
detect_drift_with_an_evaluator
flag_out_of_distribution_items
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`run_a_single_evaluator`
  - Run one DataEval evaluator, such as finding duplicates, and read its output with no health status.
- - {doc}`look_for_metadata_shortcuts`
  - Run `balance` and `parity` to find which metadata factors predict the class label.
- - {doc}`check_class_representation`
  - Run `representation`, with an optional ontology and expected floors.
- - {doc}`find_uncovered_regions`
  - Run `coverage` on embeddings to find the regions the data does not cover.
- - {doc}`rank_data_to_label`
  - Run `prioritization` on an unlabeled pool against a labeled set.
- - {doc}`detect_drift_with_an_evaluator`
  - Run `drift-mmd`, chunked drift and `drift-wasserstein` as evaluators.
- - {doc}`flag_out_of_distribution_items`
  - Run `ood-kneighbors` on a reference and on the data to test.

:::

## Building a workflow of your own

```{toctree}
:hidden:

write_a_custom_workflow
run_a_step_once_per_class
check_a_set_of_splits
judge_a_workflow_like_a_preset
reuse_a_workflow
save_a_workflow_from_python
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`write_a_custom_workflow`
  - Conform and merge two datasets, remove their duplicates, export the result, and check what is left, as one chain of
    steps.
- - {doc}`run_a_step_once_per_class`
  - Use `by:` and `groups:` on an evaluate step and its check.
- - {doc}`check_a_set_of_splits`
  - Collect the evaluation parts of a split and run `audit` as a step.
- - {doc}`judge_a_workflow_like_a_preset`
  - Use `groups:`, `verdict:`, `record:`, `next_steps:`, `preflight:` and `scopes:` to judge a custom workflow the way a
    preset does.
- - {doc}`reuse_a_workflow`
  - Keep a chain in your config, or in a file each project loads, and run it on each new dataset.
- - {doc}`save_a_workflow_from_python`
  - Build a custom workflow in Python, save it as YAML, and run it on new data.

:::

## Monitoring drift with steps

```{toctree}
:hidden:

monitor_drift
test_merged_sources_for_drift
compare_class_groups
detect_drift_on_detection_crops
monitor_model_uncertainty
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`monitor_drift`
  - Read what the drift preset makes for each test source and for each class, and read many verdicts with care.
- - {doc}`test_merged_sources_for_drift`
  - Merge the test sources and run one detector on the pool.
- - {doc}`compare_class_groups`
  - Narrow each side with a `view` step and run a detector on two groups of classes.
- - {doc}`detect_drift_on_detection_crops`
  - Wrap detection data with `DetectionCrops` and run the `shift` preset on the crops.
- - {doc}`monitor_model_uncertainty`
  - Test the `uncertainty` extractor with drift detectors, and flag images with OOD detectors.

:::

## Sweeping settings with a matrix

```{toctree}
:hidden:

run_a_matrix
name_a_matrix_key
gate_a_matrix_run
move_from_parameter_sweep
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`run_a_matrix`
  - Run a task once per combination of settings, sources or extractors, and compare every run's findings in one table.
- - {doc}`name_a_matrix_key`
  - Reach sources, extractors, list items, mapping keys and other entries' settings from a matrix key.
- - {doc}`gate_a_matrix_run`
  - Set the exit code with `fail_on`, export a dataset for each run, and read the files a matrix writes.
- - {doc}`move_from_parameter_sweep`
  - Rewrite a removed `parameter-sweep` entry as a `quality` entry plus a matrix.

:::

## Reading results

```{toctree}
:hidden:

read_evaluation_outputs
read_the_result_envelope
read_report_blocks
get_the_numbers_behind_a_finding
../notebooks/view_html_reports
gate_training_on_an_audit
export_a_dataset
reuse_results_with_cache
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`read_evaluation_outputs`
  - Read the text, chain and HTML reports, their severities and the health line.
- - {doc}`read_the_result_envelope`
  - Export a result and read its envelope, provenance fields, evaluator results and matrix results.
- - {doc}`read_report_blocks`
  - Read a finding's keys and typed blocks, item references, flags and thumbnails.
- - {doc}`get_the_numbers_behind_a_finding`
  - Reach a result's step outputs, verdict, digests, binning record and datasets from Python.
- - {doc}`../notebooks/view_html_reports`
  - Render a result as one self-contained page with cards, sortable and filterable tables, and each flag's measurements
    on hover.
- - {doc}`gate_training_on_an_audit`
  - Gate a run on an audit's verdict with `--require`, refuse data whose content digest does not match the record, and
    find which items changed.
- - {doc}`export_a_dataset`
  - Write a conformed or merged source out as a dataset on disk, with the provenance that produced it.
- - {doc}`reuse_results_with_cache`
  - Persist embeddings and statistics across runs, and know what invalidates them.

:::

## Running Flow

```{toctree}
:hidden:

write_a_pipeline_config
containerized_workflows
fail_a_ci_job_on_warnings
run_flow_as_a_service
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`write_a_pipeline_config`
  - Declare the datasets, sources, extractors, a workflow of each type and the tasks in `params.yaml`.
- - {doc}`containerized_workflows`
  - Pull a pre-built image, mount your data and config, run it and find the results.
- - {doc}`fail_a_ci_job_on_warnings`
  - Turn health warnings into exit code 3, read the exit codes, and publish JUnit and Markdown reports.
- - {doc}`run_flow_as_a_service`
  - Keep Flow running behind an HTTP API that queues pipelines, and follow each run's status, logs and results.

:::
