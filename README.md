# DataEval Workflows

Workflow orchestration for DataEval with GPU support.

## Quick Start

```bash
# 1. Build CUDA 11.8 container
docker build -f Dockerfile.cu118 -t dataeval:cu118 .

# 2. Show help
docker run dataeval:cu118

# 3. Run with config and dataset
docker run --gpus all \
  --mount type=bind,source=/path/to/config,target=/data/config,readonly \
  --mount type=bind,source=/path/to/dataset,target=/data/dataset,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval:cu118
```

## Wrapper Scripts

For easier usage (GPU is default):

```bash
# Linux/Mac
./run.sh --config /path/to/config --dataset /path/to/dataset
./run.sh --config /path/to/config --dataset /path/to/dataset --cpu

# Windows PowerShell
.\run.ps1 -Config /path/to/config -Dataset /path/to/dataset
.\run.ps1 -Config /path/to/config -Dataset /path/to/dataset -CPU
```

## Requirements

| Requirement | Version |
|-------------|---------|
| Docker | >= 20.10 |
| NVIDIA GPU | Any (for GPU mode) |
| NVIDIA Driver | >= 520 (for GPU mode) |
| CUDA | 11.8.0 (for GPU mode) |

### Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Volume Mounts

| Path | Mode | Purpose |
|------|------|---------|
| `/data/config` | ro | Config files (required) |
| `/data/dataset` | ro | Dataset (required) |
| `/data/model` | ro | Model files (optional) |
| `/output` | rw | Results (required) |

## Dataset Formats

Currently supported dataset structures:

| Format | Structure | Example |
|--------|-----------|---------|
| **Dataset** | Single split, used directly | `cifar10_test/` |
| **DatasetDict** | Multiple splits (dict), configured via config YAML | `cifar10_full/` |

## CPU Fallback

For machines without NVIDIA GPU:

```bash
docker build -f Dockerfile.cpu -t dataeval:cpu .
docker run dataeval:cpu  # Shows help
docker run \
  --mount type=bind,source=/path/to/config,target=/data/config,readonly \
  --mount type=bind,source=/path/to/dataset,target=/data/dataset,readonly \
  dataeval:cpu
```

## Dependencies

- `dataeval>=0.95.0` - Core evaluation library
- `datasets>=4.0.0` - Dataset loading library
- `maite-datasets>=0.0.9` - MAITE protocol adapter

## Troubleshooting

### Build appears stuck at `uv sync`

The Docker build may appear frozen during the `uv sync` step:

```
=> [builder 7/7] RUN uv sync --frozen --no-dev --no-install-project    1139.3s
```

**This is normal.** The step downloads ~2GB of dependencies (PyTorch, scipy, etc.) with no progress indicator.

| Network Speed | Expected Build Time |
|---------------|---------------------|
| 100 Mbps | ~10 minutes |
| 30 Mbps | ~20 minutes |
| 10 Mbps | ~45 minutes |

**Tip:** First build is slow; subsequent builds use Docker cache and complete in seconds.

## Running Without Container

The `dataeval_flow` package can be used standalone without Docker.

**Installation:**
```bash
git clone https://gitlab.jatic.net/jatic/aria/dataeval-flow.git
cd dataeval-flow
uv sync
```

**CLI Usage:**
```bash
python -m dataeval_flow --config /path/to/config --output /path/to/output
```

**Python API Usage:**
```python
from pathlib import Path
from dataeval_flow import load_dataset

dataset = load_dataset(Path("/path/to/dataset"))
```

**Development:**
```bash
uv sync --group dev
nox
```

## License

MIT
