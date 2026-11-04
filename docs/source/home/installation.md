# Installation

DataEval Flow supports Python 3.11–3.14. It is developed and tested on Linux (Ubuntu 22.04 / 24.04, including WSL2) on
linux/amd64. See {doc}`quickstart` for the supported-platform summary and {doc}`../reference/containers` for the full
hardware, architecture, and network requirements.

## Install PyTorch and torchvision

DataEval Flow requires PyTorch, which it picks up through its [DataEval](https://dataeval.readthedocs.io/) dependency.
By default, `pip install dataeval-flow` pulls PyTorch from PyPI, which bundles CUDA support on Linux and is a much
larger download than the CPU build.

To choose a specific PyTorch variant, install `torch` from that variant's wheel index first, then install DataEval
Flow. It accepts the build already in the environment:

```bash
# 1. Pick your PyTorch build (cpu / cu126 / cu130)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. Install DataEval Flow
pip install dataeval-flow
```

See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for all available PyTorch installation
options.

:::{warning}
Do not use `--extra-index-url` to select a CUDA variant. It adds an index rather than replacing PyPI, and pip then
picks the highest version across both. The CUDA indexes lag the latest PyTorch release, so PyPI usually wins and you
get the default CUDA-bundled build instead of the one you asked for. The install succeeds and nothing warns you.

`--index-url` (as above) replaces the index outright, which is why it is reliable.
:::

A CPU-only install has a one-line shortcut because the CPU index tracks the latest release:

```bash
pip install dataeval-flow --extra-index-url https://download.pytorch.org/whl/cpu
```

`torchvision` is not installed by default. It is imported lazily and is only needed if you use preprocessing
pipelines, the torchvision dataset adapter, or the interactive TUI's transform discovery. When you need it, install it
in the same step as `torch` so both come from the same index:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install dataeval-flow
```

:::{important}
The `cpu`, `cu126`, and `cu130` extras are not a way to select a PyTorch variant with pip. All three declare the same
requirements (`torch` and `torchvision`). What distinguishes them is `[tool.uv.sources]` in the project's
`pyproject.toml`, which routes those two packages to the right wheel index. That routing is project metadata. uv
applies it when resolving from a source checkout, and it is not part of the published wheel. Installing
`dataeval-flow[cpu]` from PyPI therefore adds `torchvision` but still resolves PyTorch from whichever index pip is
pointed at.

Select the variant with `--index-url` under pip, `--torch-backend` under `uv pip`, and use the extras only when
installing from source.
:::

## Install feature extras

These extras behave normally under every installer, including pip:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `onnx` | `onnx`, `onnxruntime` | ONNX feature extractors (CPU inference) |
| `onnx-cu126` | `onnx`, `onnxruntime-gpu` (CUDA 12.x build) | ONNX feature extractors on a `cu126` install |
| `onnx-cu130` | `onnx`, `onnxruntime-gpu` (CUDA 13.x build) | ONNX feature extractors on a `cu130` install |
| `opencv` | `opencv-python-headless` (capped below 4.13) | Bag-of-Visual-Words (SIFT) extraction |
| `app` | `textual` | The interactive TUI (`dataeval-flow app`) |
| `service` | `fastapi`, `uvicorn` | The long-running HTTP service (`dataeval-flow serve`); see [Run Flow as a service](../how_to/run_flow_as_a_service.md) |
| `ontology` | `dataeval[ontology]` | Loading an ontology from an RDF file |

```bash
pip install "dataeval-flow[onnx,opencv,app]"
```

## Install from PyPI with uv

When you install from PyPI, let uv select the PyTorch index with `--torch-backend`:

```bash
uv pip install dataeval-flow --torch-backend cpu   # or cu126 / cu130 / auto
```

## Install from source

Clone once, then use whichever toolchain you prefer. `pyproject.toml` is the source of truth, and requirements are
resolved against their respective committed requirement files.

```bash
git clone https://github.com/aria-ml/dataeval-flow.git
cd dataeval-flow
```

### Install with uv

```bash
uv sync --extra cpu                                        # torch + torchvision, CPU build
uv sync --extra cpu --extra onnx --extra opencv --extra app  # plus the feature extras
```

Substitute `cu126` / `cu130` for the CUDA variants, and the matching `onnx-cu126` / `onnx-cu130` for `onnx` alongside
them. An `onnxruntime-gpu` wheel links against one specific CUDA major, so the ONNX extra has to track the PyTorch one.
The PyTorch variant extras are mutually exclusive, as are the three `onnx` extras. uv enforces both.

### Install with conda or mamba

Conda manages the environment from the committed `environment.yml`. The package itself is then installed into that
environment. PyTorch is installed from PyPI / `download.pytorch.org` (it is no longer maintained on conda-forge), so
the conda path installs the CPU build of PyTorch.

```bash
conda env create -f environment.yml
conda activate dataeval-flow
pip install -e .
```

## Pull a Docker image

Pre-built images are available from the JATIC Harbor registry, each bundling its matching PyTorch build. See
{doc}`../how_to/containerized_workflows` for the full containerized workflow guide.

### Pull the CPU image

```bash
docker pull harbor.jatic.net/aria/dataeval-flow:latest-cpu
docker run --rm harbor.jatic.net/aria/dataeval-flow:latest-cpu
```

### Pull a CUDA image

```bash
# CUDA 13.0 — recommended for modern GPUs
docker pull harbor.jatic.net/aria/dataeval-flow:latest-cu130
docker run --rm --gpus all harbor.jatic.net/aria/dataeval-flow:latest-cu130
```

| Tag | Base | Use case |
| --- | --- | --- |
| `cpu` | Ubuntu 24.04 | Machines without NVIDIA GPU |
| `cu126` | Ubuntu 24.04 | Older GPUs / CUDA 12.6 drivers |
| `cu130` | Ubuntu 24.04 | Modern GPUs (RTX 50 series) / CUDA 13.0 drivers |
