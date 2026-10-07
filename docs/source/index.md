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

DataEval Flow wraps DataEval's analytics in a containerized workflow engine.
Its steps compose into pipelines that run identically locally or in a CUDA-
enabled container. Cleaning, audit, coverage, drift, OOD, splitting, and
prioritization ship as presets, driven by YAML or JSON from a headless or
interactive CLI.

:::
::::

DataEval Flow turns DataEval's analytics into a composable workflow engine, in
two complementary ways:

- **Composable.** Each capability, such as duplicate detection, outlier detection,
  splitting, or export, is a self-contained step with declared inputs and
  outputs. Users compose steps into their own workflows in configuration, and
  the built-in workflows are presets made from the same steps. The set is
  extensible: new steps can be added as plugins.
- **Containerized.** Steps run in a reproducible, CUDA-enabled container (or
  locally), so a pipeline behaves the same on a laptop, a cluster, or an
  air-gapped system, and emits the same machine-readable results.

DataEval Flow lets T&E engineers compose and run multi-stage data evaluation
pipelines without Python glue code. Pipelines are described in YAML or
JSON, executed locally or in a CUDA-enabled container, and produce both
human-readable reports and machine-readable result envelopes that satisfy JATIC
interoperability requirements. It is part of the JATIC suite of tools and builds
directly on the [DataEval](https://dataeval.readthedocs.io/) library; the
underlying evaluators are the same algorithms DataEval exposes, wrapped in a
reproducible orchestration layer with native MAITE interoperability.

## T&E tasks and the workflows that support them

| T&E task | Workflow type | What it produces |
| --- | --- | --- |
| Find and flag dataset quality issues | [`quality`](reference/presets.md#quality) | Outliers and duplicates, removed |
| Audit a set of splits before training | [`audit`](reference/presets.md#audit) | A verdict, a record of what was audited, and findings under five questions; also a step after a split ([how-to](how_to/write_a_custom_workflow.md#11-check-a-set-of-splits)) |
| Check labels against a declared ontology | [`taxonomy`](reference/presets.md#taxonomy) | Leaf coverage, conformance, alignment and structure findings |
| Find gaps in dataset coverage before training | [`scope`](reference/presets.md#scope) | Embedding blind spots and what to acquire per class |
| Find shortcuts and imbalance in labels and metadata | [`bias`](reference/presets.md#bias) | Class imbalance, metadata factors tied to the class, and under-represented class-factor combinations |
| Build stratified or grouped train/val/test splits | [`splits`](reference/presets.md#splits) | Train, val and test splits, stratified or grouped, or k folds, with each part's stratification judged |
| Monitor operational data for population drift | [`shift`](reference/presets.md#shift) | Per-batch drift flags and p-values |
| Flag anomalous individual samples | [`shift`](reference/presets.md#shift) | Per-sample out-of-distribution scores |
| Rank abundant/unlabeled data for labeling | [`prioritization`](reference/presets.md#prioritization) | Ranked sample ordering |
| Find metadata the run could not read | [`triage`](reference/presets.md#triage) | Unreadable and unpinned metadata factors, with suggested corrections |
| Tune workflow parameters across a grid | Any workflow, with a task matrix | One table comparing every run's findings |

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
- **Batch execution**: the container runs a pipeline to completion and exits; it
  is not a long-running service.

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
JATIC Maturity <reference/maturity>
Evaluator Catalog <reference/evaluators>
Transform Catalog <reference/transforms>
Combine Catalog <reference/combines>
Check Catalog <reference/checks>
Preset Catalog <reference/presets>
Naming Conventions <reference/naming>
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
