# DataEval Flow

DataEval Flow provides workflow orchestration for DataEval evaluators, packaging
data cleaning, drift monitoring, OOD detection, analysis, splitting,
prioritization, and parameter-sweep pipelines behind a single declarative
configuration format and both headless and interactive CLIs.

DataEval Flow lets T&E engineers compose and run multi-stage data evaluation
pipelines without writing Python glue code. Pipelines are described in YAML or
JSON, executed locally or in a CUDA-enabled container, and produce both
human-readable reports and machine-readable result envelopes that satisfy JATIC
interoperability requirements. It is part of the JATIC suite of tools and builds
directly on the [DataEval](https://dataeval.readthedocs.io/) library — the
underlying evaluators are the same algorithms DataEval exposes, wrapped in a
reproducible orchestration layer with native MAITE interoperability.

## T&E tasks and the workflows that support them

| T&E task | Workflow | What it produces |
| --- | --- | --- |
| Find and flag dataset quality issues | Data Cleaning | Outliers, duplicates, and label issues |
| Profile dataset quality across splits | Data Analysis | Statistical summaries and quality metrics |
| Build leakage-free train/val/test splits | Dataset Splitting | Stratified or random splits |
| Monitor operational data for population drift | Drift Detection | Per-batch drift flags and p-values |
| Track drift on a per-class basis | Classwise Drift | Per-class drift signals |
| Flag anomalous individual samples | OOD Detection | Per-sample out-of-distribution scores |
| Rank abundant/unlabeled data for labeling | Prioritization | Ranked sample ordering |
| Tune workflow parameters across a grid | Parameter Sweep | Per-configuration result comparison |

See the [Tutorials](tutorials/index) for end-to-end walkthroughs and the
[Explanations](concepts/index) for the concepts behind each workflow.

## Critical limitations and requirements for use

- **Computer-vision image datasets only** — no NLP or tabular data.
- **MAITE for native interoperability** — non-MAITE sources are consumed through
  the built-in adapters (HuggingFace, COCO, YOLO, TorchVision, ImageFolder).
- **Some workflows need metadata** — bias, parity, and metadata-insight analyses
  require per-sample metadata factors.
- **Some workflows need a model or embeddings** — embedding-space drift, OOD
  detection, and prioritization require a feature extractor or precomputed
  embeddings.
- **Drift and OOD need a representative reference dataset.**
- **Batch execution** — the container runs a pipeline to completion and exits; it
  is not a long-running service.

See the [Container Reference](reference/containers) for hardware, architecture,
and network requirements, and the [Installation guide](home/installation) to get
started.

<!-- TOC TREE -->

:::{toctree}
:caption: Getting Started
:hidden:

Welcome <self>
home/installation.md
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

Container Reference <reference/containers>
API Reference <reference/autoapi/dataeval_flow/index>
reference/glossary
:::

## Acknowledgement

### CDAO Funding Acknowledgement

This material is based upon work supported by the Chief Digital and Artificial
Intelligence Office under Contract No. W519TC-23-9-2033. The views and
conclusions contained herein are those of the author(s) and should not be
interpreted as necessarily representing the official policies or endorsements,
either expressed or implied, of the U.S. Government.
