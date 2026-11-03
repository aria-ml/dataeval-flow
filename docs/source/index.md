# DataEval Flow

::::{grid}
:reverse:
:gutter: 3 4 4 4
:margin: 1 2 1 2

:::{grid-item}
:columns: 12 4 4 4

:::{image} _static/images/DataEvalFlow_Logo.png
:width: 200px
:class: sd-m-auto
:alt: DataEval Flow

:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-fs-5

DataEval Flow is a containerized workflow engine for DataEval analytics within
the JATIC suite, providing native MAITE interoperability. It allows T&E engineers
to configure multi-stage pipelines in YAML or JSON without writing Python glue
code.

:::
::::

- **Modular & Extensible:** Use turnkey presets for cleaning, metadata triage, label checks, coverage, bias, splitting,
  prioritization, audit, and drift and OOD monitoring, or compose your own workflows from the same steps and add more
  with plugins.
- **Flexible Runtime:** Trigger workflows headlessly or interactively via CLI, or queue jobs over HTTP (preview).
- **Deploy Anywhere:** Runs unchanged on laptops, CUDA containers, clusters, or air-gapped systems, outputting
  human-readable reports and result envelopes that satisfy JATIC interoperability requirements.

## T&E tasks and the workflows that support them

The tasks follow the machine-learning lifecycle, as DataEval's own guides do, with data engineering split into
assessing a dataset and preparing it for training.

### Assess the data

| T&E task | Workflow type | What it produces |
| --- | --- | --- |
| Find and flag dataset quality issues | [`quality`](reference/presets.md#quality) | Outliers and duplicates, removed |
| Find metadata the run could not read | [`triage`](reference/presets.md#triage) | Unreadable and unpinned metadata factors, with suggested corrections |
| Check labels against a declared ontology | [`taxonomy`](reference/presets.md#taxonomy) | Leaf coverage, conformance, alignment and structure findings |
| Find gaps in dataset coverage before training | [`scope`](reference/presets.md#scope) | Embedding blind spots and what to acquire per class |
| Find shortcuts and imbalance in labels and metadata | [`bias`](reference/presets.md#bias) | Class imbalance, metadata factors tied to the class, and under-represented class-factor combinations |

### Prepare data for training

| T&E task | Workflow type | What it produces |
| --- | --- | --- |
| Build stratified or grouped train/val/test splits | [`splits`](reference/presets.md#splits) | Train, val and test splits, stratified or grouped, or k folds, with each part's stratification judged |
| Rank abundant/unlabeled data for labeling | [`prioritization`](reference/presets.md#prioritization) | Ranked sample ordering |
| Audit a set of splits before training | [`audit`](reference/presets.md#audit) | A verdict, a record of what was audited, and findings under five questions; also a step after a split ([how-to](how_to/write_a_custom_workflow.md#11-check-a-set-of-splits)) |

### Monitoring

| T&E task | Workflow type | What it produces |
| --- | --- | --- |
| Monitor operational data for population drift | [`shift`](reference/presets.md#shift) | Per-batch drift flags and p-values |
| Flag anomalous individual samples | [`shift`](reference/presets.md#shift) | Per-sample out-of-distribution scores |

Any task can be tuned across a grid of settings with a task matrix, which gives one table comparing every run's
findings ([how-to](how_to/run_a_matrix.md)).

See [Find the Right Step](reference/index.md) to go from a question to the preset or steps that answer it, the
[Tutorials](tutorials/index.md) for end-to-end walkthroughs and the [Explanations](concepts/index.md) for the concepts
behind each workflow.

## Critical limitations and requirements for use

- **Computer-vision image datasets only**: no NLP or tabular data.
- **MAITE for native interoperability**: non-MAITE sources are consumed through
  the built-in adapters (HuggingFace, COCO, YOLO, TorchVision, ImageFolder).
- **Some workflows need metadata**: bias, parity, and metadata factor analyses
  require per-sample metadata factors.
- **Some workflows need a model or embeddings**: embedding-space drift, OOD
  detection, and prioritization require a feature extractor or precomputed
  embeddings.
- **Drift and OOD need a representative reference dataset.**
- **Batch execution**: the container runs a pipeline to completion and exits. The
  optional `serve` HTTP service (preview) queues pipelines and runs each the same way.

Starting here? The [Quickstart](home/quickstart.md) installs the package and runs a first
evaluation end to end. See the [Installation guide](home/installation.md) for every
supported install path, and the [Container Reference](reference/containers.md) for
hardware, architecture, and network requirements.

<!-- TOC TREE -->

:::{toctree}
:caption: Getting Started
:hidden:

Welcome <self>
home/installation.md
home/quickstart.md
What's New <home/whats_new/index.md>
Change Log <home/changelog.md>
:::

:::{toctree}
:caption: Tutorials
:hidden:

Overview <tutorials/index>
:::

:::{toctree}
:caption: How-to Guides
:hidden:

Overview <how_to/index>
:::

:::{toctree}
:caption: Explanation
:hidden:

Overview <concepts/index>
:::

:::{toctree}
:caption: Reference
:hidden:

Find the Right Step <reference/index>
Container Reference <reference/containers>
Service Reference <reference/service>
JATIC Maturity <reference/maturity>
Evaluator Catalog <reference/evaluators>
Transform Catalog <reference/transforms>
Combine Catalog <reference/combines>
Check Catalog <reference/checks>
Preset Catalog <reference/presets>
Naming Conventions <reference/naming>
Config Reference <reference/config/index>
API Reference <reference/autoapi/dataeval_flow/index>
reference/glossary
:::

## Acknowledgment

### CDAO Funding Acknowledgment

This material is based upon work supported by the Chief Digital and Artificial
Intelligence Office under Contract No. W519TC-23-9-2033. The views and
conclusions contained herein are those of the author(s) and should not be
interpreted as necessarily representing the official policies or endorsements,
either expressed or implied, of the U.S. Government.
