# Installation

## Using pip

PyTorch is hosted on a separate wheel index, so pass `--extra-index-url`
matching the variant you want. Omit it and pip pulls the CUDA-bundled
manylinux build of `torch` from PyPI, which is much larger and may not
match your hardware.

```bash
# CPU-only PyTorch
pip install "dataeval-flow[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 PyTorch
pip install "dataeval-flow[cu118]" --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA 12.8 PyTorch
pip install "dataeval-flow[cu128]" --extra-index-url https://download.pytorch.org/whl/cu128
```

## Using uv

```bash
uv pip install "dataeval-flow[cpu]" --torch-backend cpu   # or cu118 / cu128 for CUDA variants
```

## From source

Clone once, then use whichever toolchain you prefer — uv and Poetry both
read `pyproject.toml` as the source of truth and resolve
against their respective committed lockfiles (`uv.lock` / `poetry.lock`).

```bash
git clone https://gitlab.jatic.net/jatic/aria/dataeval-flow.git
cd dataeval-flow
```

### With uv

```bash
uv sync --extra all-cpu  # or all-cu118 / all-cu128 for CUDA variants
```

### With Poetry

```bash
poetry install --extras all-cpu
```

## Docker

Pre-built images are available from the JATIC Harbor registry. See
{doc}`../how_to/containerized_workflows` for the full containerized workflow
guide.

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
