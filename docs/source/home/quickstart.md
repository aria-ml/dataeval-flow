# Quickstart

Run your first DataEval Flow evaluation — install the package, describe a pipeline in YAML, and read the result.

## Step 1: Check your machine

DataEval Flow is developed and tested on **Linux** (Ubuntu 22.04 and 24.04, including WSL2) on **linux/amd64
(x86-64)**, against **Python 3.10–3.14** (CI runs all five). macOS and Windows are supported only through
Docker or WSL2 and are not part of the CI test matrix — if you hit a problem on those hosts, the OS or hardware may be
the root cause. arm64 / Apple Silicon is not built or tested.

A GPU is never required; it only accelerates model-based embedding extraction. Plan on **2 CPU cores and 8 GB of
memory** as a floor, and remember that datasets and embeddings are held in memory during a run, so **memory is the
primary limit on dataset size**.

The {doc}`Container Reference <../reference/containers>` has the full hardware, architecture, and network requirements
for both the container and the Python-library forms.

## Step 2: Install

```bash
pip install dataeval-flow --extra-index-url https://download.pytorch.org/whl/cpu
```

PyTorch lives on a separate wheel index, and that flag is what gets you the CPU build — without it you get the much
larger CUDA-bundled build from PyPI.

For a **CUDA** build, install `torch` from its index first and then DataEval Flow; `--extra-index-url` is not reliable
for the CUDA variants. And note that the `cpu` / `cu118` / `cu128` extras do *not* select a PyTorch variant under pip —
they exist for uv source installs. See {doc}`installation` for both points in full, plus the uv, Poetry, conda, source,
and Docker paths.

## Step 3: Describe a pipeline

A pipeline is a YAML (or JSON) file using a define-once, reference-by-name pattern: name your `datasets`, optionally
narrow them with `views`, bundle them into `sources`, choose an `extractor` if the workflow needs embeddings,
configure a `workflow`, and compose it all in a `task`.

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
    type: data-cleaning
    mode: advisory             # report only; do not modify the dataset
    outlier_method: adaptive
    outlier_flags: [dimension, pixel, visual]

tasks:
  - name: check_my_data
    workflow: quality_check
    sources: my_source
    extractor: bovw_ext
```

## Step 4: Run it

From the command line:

```bash
dataeval-flow --config params.yaml --data . --output ./results
```

Or from Python — the same config, executed through the public API:

```python
from pathlib import Path

from dataeval_flow import load_config, run_tasks

config = load_config(Path("params.yaml"))
results = run_tasks(config, data_dir=Path("."))

result = results[0]
print(result.report())  # human-readable text report
result.export("./results")  # machine-readable result envelope (JSON)
```

`report()` prints the findings, their severities, and an overall health line. `export()` writes the JSON **result
envelope** — the findings plus the provenance metadata (timestamp, tool version, dataset and model identifiers, and the
fully resolved configuration) that makes the finding auditable and interoperable with other JATIC tools.

To build a config without writing YAML by hand, `dataeval-flow config` walks you through it on the command line and
`dataeval-flow app` opens an interactive TUI dashboard (needs the `app` extra).

## Where to go next

- {doc}`Tutorials <../tutorials/index>` — end-to-end walkthroughs of each T&E task, from cleaning a dataset to
  monitoring a deployed model for drift.
- {doc}`How-to Guides <../how_to/index>` — targeted answers to specific problems: tuning outlier detection, reading
  evaluation outputs, using your own model for embeddings, running in a container.
- {doc}`Explanations <../concepts/index>` — the concepts behind the workflows, and why reproducibility and provenance
  are built into every result.
- {doc}`Glossary <../reference/glossary>` — the vocabulary used throughout the documentation.
