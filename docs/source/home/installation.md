# Installation

DataEval Flow supports **Python 3.10–3.13**. It is developed and tested on Linux
(Ubuntu 22.04 / 24.04, including WSL2) on linux/amd64. See {doc}`quickstart` for the
supported-platform summary and {doc}`../reference/containers` for the full hardware,
architecture, and network requirements.

## PyTorch and torchvision

DataEval Flow requires PyTorch, which it picks up through its
[DataEval](https://dataeval.readthedocs.io/) dependency. By default,
`pip install dataeval-flow` pulls PyTorch from PyPI, which bundles CUDA support on
Linux and is a much larger download than the CPU build.

To choose a specific PyTorch variant, install `torch` from that variant's wheel index
**first**, then install DataEval Flow — it accepts the build already in the
environment:

```bash
# 1. Pick your PyTorch build (cpu / cu118 / cu128)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install DataEval Flow
pip install dataeval-flow
```

See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for all
available PyTorch installation options.

:::{warning}
Do **not** reach for `--extra-index-url` to select a CUDA variant. It *adds* an index
rather than replacing PyPI, and pip then picks the highest version across both. The
CUDA indexes lag the latest PyTorch release, so PyPI usually wins and you silently get
the default CUDA-bundled build instead of the one you asked for — the install succeeds
and nothing warns you.

`--index-url` (as above) replaces the index outright, which is why it is reliable.
:::

For a CPU-only install there is a one-line shortcut, because the CPU index does track
the latest release:

```bash
pip install dataeval-flow --extra-index-url https://download.pytorch.org/whl/cpu
```

`torchvision` is **not** installed by default. It is imported lazily and is only needed
if you use preprocessing pipelines, the torchvision dataset adapter, or the interactive
TUI's transform discovery. When you need it, install it in the same step as `torch` so
both come from the same index:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install dataeval-flow
```

:::{important}
The `cpu`, `cu118`, and `cu128` extras are **not** a way to select a PyTorch variant
with pip. All three declare exactly the same requirements (`torch` and `torchvision`);
what distinguishes them is `[tool.uv.sources]` in the project's `pyproject.toml`, which
routes those two packages to the right wheel index. That routing is project metadata —
uv applies it when resolving **from a source checkout**, and it is not part of the
published wheel. Installing `dataeval-flow[cpu]` from PyPI therefore adds `torchvision`
but still resolves PyTorch from whichever index pip is pointed at.

Select the variant with `--index-url` under pip, `--torch-backend` under `uv pip`, and
use the extras only when installing from source.
:::

## Feature extras

These extras behave normally under every installer, including pip:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `onnx` | `onnx`, `onnxruntime` | ONNX feature extractors (CPU inference) |
| `onnx-gpu` | `onnx`, `onnxruntime-gpu` | ONNX feature extractors (GPU inference) |
| `opencv` | `opencv-python-headless` | Bag-of-Visual-Words (SIFT) extraction |
| `app` | `textual` | The interactive TUI (`dataeval-flow app`) |
| `ontology` | `dataeval[ontology]` | Loading an ontology from an RDF file |

```bash
pip install "dataeval-flow[onnx,opencv,app]"
```

## Using uv

Installing from PyPI, let uv select the PyTorch index with `--torch-backend`:

```bash
uv pip install dataeval-flow --torch-backend cpu   # or cu118 / cu128 / auto
```

## From source

Clone once, then use whichever toolchain you prefer — uv and Poetry both read
`pyproject.toml` as the source of truth and resolve against their respective committed
lockfiles (`uv.lock` / `poetry.lock`). This is where the PyTorch variant extras apply.

```bash
git clone https://github.com/aria-ml/dataeval-flow.git
cd dataeval-flow
```

### With uv

```bash
uv sync --extra cpu                                        # torch + torchvision, CPU build
uv sync --extra cpu --extra onnx --extra opencv --extra app  # plus the feature extras
```

Substitute `cu118` / `cu128` for the CUDA variants, and `onnx-gpu` for `onnx` alongside
them. The PyTorch variant extras are mutually exclusive, as are `onnx` and `onnx-gpu`;
uv enforces both.

### With Poetry

```bash
poetry install --extras "cpu onnx opencv app"
```

Poetry resolves `torch` and `torchvision` from the CPU wheel index, so the Poetry path
installs the CPU build regardless of which extra you name.

### With conda / mamba

Conda manages the environment from the committed `environment.yml`; the package itself
is then installed into that environment. PyTorch is installed from PyPI /
`download.pytorch.org` (it is no longer maintained on conda-forge), so the conda path
installs the CPU build of PyTorch.

```bash
conda env create -f environment.yml
conda activate dataeval-flow
pip install -e .
```

## Docker

Pre-built images are available from the JATIC Harbor registry and bundle the matching
PyTorch build already. See {doc}`../how_to/containerized_workflows` for the full
containerized workflow guide.

### CPU only

```bash
docker pull harbor.jatic.net/aria/dataeval:cpu
docker run --rm harbor.jatic.net/aria/dataeval:cpu
```

### GPU (CUDA)

```bash
# CUDA 12.8 — recommended for modern GPUs
docker pull harbor.jatic.net/aria/dataeval:cu128
docker run --rm --gpus all harbor.jatic.net/aria/dataeval:cu128
```

| Tag | Base | Use case |
| --- | --- | --- |
| `cpu` | Ubuntu 24.04 | Machines without NVIDIA GPU |
| `cu118` | Ubuntu 22.04 | Older GPUs / CUDA 11.8 drivers |
| `cu128` | Ubuntu 24.04 | Modern GPUs (RTX 40/50 series) / CUDA 12.8 drivers |
