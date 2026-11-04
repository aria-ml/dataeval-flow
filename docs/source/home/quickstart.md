# Quickstart

Run your first DataEval Flow evaluation. You will install the package, describe a pipeline in YAML, and read the
result.

## Check your machine

DataEval Flow is developed and tested on Linux (Ubuntu 22.04 and 24.04, including WSL2) on linux/amd64 (x86-64),
against Python 3.11–3.14 (CI runs all four). macOS and Windows are supported only through Docker or WSL2 and are not
part of the CI test matrix. If you hit a problem on those hosts, the OS or hardware may be the root cause. arm64 /
Apple Silicon is not built or tested.

A GPU is never required. It only accelerates model-based embedding extraction. Plan on 2 CPU cores and 8 GB of memory
as a floor. Datasets and embeddings are held in memory during a run, so memory is the primary limit on dataset size.

The {doc}`Container Reference <../reference/containers>` has the full hardware, architecture, and network requirements
for both the container and the Python-library forms.

## Install DataEval Flow

```bash
pip install dataeval-flow --extra-index-url https://download.pytorch.org/whl/cpu
```

PyTorch lives on a separate wheel index, and that flag selects the CPU build. Without it, pip resolves the much larger
CUDA-bundled build from PyPI.

For a CUDA build, install `torch` from its index first, then DataEval Flow. `--extra-index-url` is not reliable for
the CUDA variants. The `cpu` / `cu126` / `cu130` extras do not select a PyTorch variant under pip. They exist for uv
source installs. See {doc}`installation` for both points in full, plus the uv, conda, source, and Docker paths.

## Describe a pipeline

A pipeline is a YAML (or JSON) file that follows a define-once, reference-by-name pattern. You name your `datasets`,
optionally narrow them with `views`, bundle them into `sources`, choose an `extractor` if the workflow needs
embeddings, configure a `workflow`, and compose it all in a `task`.

Save this as `params.yaml` next to your data:

```yaml
datasets:
  - name: my_dataset
    format: image_folder
    path: data/my-images       # relative to the data root
    infer_labels: true

sources:
  - name: my_source
    dataset: my_dataset

extractors:
  - name: bovw_ext
    model: bovw                # model-free — no model file needed
    vocab_size: 512
    batch_size: 32

workflows:
  - name: quality_check
    type: quality
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: adaptive

tasks:
  - name: check_my_data
    workflow: quality_check
    sources: my_source
    extractor: bovw_ext
```

## Run the pipeline

From the command line:

```bash
dataeval-flow --config params.yaml --data . --output ./results
```

Or run the same config from Python through the public API:

```python
from pathlib import Path

from dataeval_flow import load_config, run_tasks

config = load_config(Path("params.yaml"))  # a file, or a folder of config files
results = run_tasks(config, data_dir=Path("."))  # keyed by task name

result = results["check_my_data"]
print(result.report())  # human-readable text report
result.export("./results")  # machine-readable result envelope (JSON)
```

`report()` prints the findings, their severities, and an overall health line. `export()` writes the JSON **result
envelope**. The envelope holds the findings plus provenance metadata (timestamp, tool version, dataset and model
identifiers, fully resolved configuration) that makes each finding auditable and interoperable with other JATIC tools.

To build a config without writing YAML by hand, `dataeval-flow config` walks you through it on the command line and
`dataeval-flow app` opens an interactive TUI dashboard (needs the `app` extra).

## Where to go next

- {doc}`Tutorials <../tutorials/index>` — Follow an end-to-end walkthrough of each T&E task, from cleaning a dataset to
  monitoring a deployed model for drift.
- {doc}`How-to guides <../how_to/index>` — Find focused answers to specific problems, such as tuning outlier detection,
  reading evaluation outputs, using your own model for embeddings and running in a container.
- {doc}`Explanations <../concepts/index>` — Read the concepts behind the workflows, and how reproducibility and
  provenance are built into every result.
- {doc}`Glossary <../reference/glossary>` — Look up the vocabulary used throughout the documentation.
