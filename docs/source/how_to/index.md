# How-to Guides

Task-oriented guides for accomplishing specific goals with DataEval Flow. Each one takes a problem you are likely to
hit once you know the basics and walks through the solution. If you are new to DataEval Flow, start with the
{doc}`Quickstart <../home/quickstart>` and the {doc}`Tutorials <../tutorials/index>` instead — these guides assume you
have run a workflow before.

The guides are grouped by the part of a pipeline they address.

## Configuring the data

Getting the right data, in the right representation, in front of a workflow.

```{toctree}
:hidden:

build_dataset_views
../notebooks/torchvision_datasets
declare_an_ontology
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`Narrow a dataset with views <build_dataset_views>`
  - Limit, filter, shuffle, or index into a dataset before a workflow sees it — and keep it reproducible.
- - {doc}`Use a torchvision dataset <../notebooks/torchvision_datasets>`
  - Feed a `torchvision` classification or detection dataset straight into a workflow.
- - {doc}`Declare an ontology <declare_an_ontology>`
  - Define the sanctioned label space so coverage can name classes that were never collected.

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
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`Configure outlier detection <configure_outlier_detection>`
  - Pick a statistical method, choose which statistics to test, add cluster-based detection, and set the thresholds
    that turn a finding into a warning.

:::

## Running and reading results

```{toctree}
:hidden:

read_evaluation_outputs
reuse_results_with_cache
containerized_workflows
```

:::{list-table}
:widths: 35 65
:header-rows: 0

- - {doc}`Read evaluation outputs <read_evaluation_outputs>`
  - Interpret the report and its severities, export the result envelope, and reach the raw numbers behind a finding.
- - {doc}`Reuse results with the disk cache <reuse_results_with_cache>`
  - Persist embeddings and statistics across runs, and know what invalidates them.
- - {doc}`Run workflows in containers <containerized_workflows>`
  - Pull a pre-built image, write a config, and launch with bind-mounted data.

:::
