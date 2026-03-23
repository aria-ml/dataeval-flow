# Installation

## Using pip

```bash
pip install dataeval-flow
```

## Using uv

```bash
uv add dataeval-flow
```

## From source

```bash
git clone https://gitlab.jatic.net/jatic/aria/dataeval-flow.git
cd dataeval-flow
uv sync
```

## Docker

Pre-built images are available from the JATIC Harbor registry. See
{doc}`../how_to/containerized_workflows` for the full containerized workflow
guide.

### CPU only

```bash
docker pull harbor.jatic.net:443/aria/dataeval:cpu
docker run --rm harbor.jatic.net:443/aria/dataeval:cpu
```

### GPU (CUDA)

```bash
# CUDA 12.8 — recommended for modern GPUs
docker pull harbor.jatic.net:443/aria/dataeval:cu128
docker run --rm --gpus all harbor.jatic.net:443/aria/dataeval:cu128
```

| Tag | Base | Use case |
| --- | --- | --- |
| `cpu` | Ubuntu 24.04 | Machines without NVIDIA GPU |
| `cu118` | Ubuntu 22.04 | Older GPUs / CUDA 11.8 drivers |
| `cu124` | Ubuntu 22.04 | Mid-range GPUs / CUDA 12.4 drivers |
| `cu128` | Ubuntu 24.04 | Modern GPUs (RTX 40/50 series) / CUDA 12.8 drivers |

To build from source instead, see {doc}`../how_to/build_from_source`.
