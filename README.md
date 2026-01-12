# DataEval Application

Minimal container for DataEval data evaluation with GPU support.

## Quick Start

```bash
# 1. Build GPU container
docker build -t dataeval:gpu .

# 2. Show help
docker run dataeval:gpu

# 3. Run with dataset
docker run --gpus all \
  --mount type=bind,source=/path/to/dataset,target=/data/dataset,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval:gpu
```

## Wrapper Scripts

For easier usage (GPU is default):

```bash
# Linux/Mac
./run.sh --dataset /path/to/dataset --output /path/to/output
./run.sh --dataset /path/to/dataset --output /path/to/output -cpu

# Windows PowerShell
.\run.ps1 -Dataset /path/to/dataset -Output /path/to/output
.\run.ps1 -Dataset /path/to/dataset -Output /path/to/output -CPU
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
| `/data/dataset` | ro | Dataset (required) |
| `/data/model` | ro | Model files (optional) |
| `/data/incoming` | ro | Raw images (optional) |
| `/output` | rw | Results |

## Dataset Formats

Currently supported dataset structures:

| Format | Structure | Example |
|--------|-----------|---------|
| **Dataset** | Single split, used directly | `cifar10_test/` |
| **DatasetDict** | Multiple splits (dict), requires `DATASET_SPLIT` | `cifar10_full/` with `train/` and `test/` |

### Using DatasetDict (multi-split datasets)

For datasets with multiple splits, set `DATASET_SPLIT` env var:

```bash
# List available splits (will error and show options)
docker run --gpus all \
  -v /path/to/cifar10_full:/data/dataset:ro \
  dataeval:gpu

# Output:
# DatasetDict detected. Available splits: ['train', 'test']
# ERROR: Multiple splits found. Set DATASET_SPLIT env var.
#   Example: -e DATASET_SPLIT=train
#   Available: ['train', 'test']

# Specify split
docker run --gpus all \
  -e DATASET_SPLIT=test \
  -v /path/to/cifar10_full:/data/dataset:ro \
  dataeval:gpu
```

## CPU Fallback

For machines without NVIDIA GPU:

```bash
docker build -f Dockerfile.cpu -t dataeval:cpu .
docker run dataeval:cpu  # Shows help
docker run \
  --mount type=bind,source=/path/to/dataset,target=/data/dataset,readonly \
  dataeval:cpu
```

## Docker Compose

```bash
# 1. Copy and edit config
cp .env.example .env
# Edit .env with your paths

# 2. Run
docker compose up
```

## Dependencies

- `dataeval>=0.93.1` - Core evaluation library
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

The `dataeval_app` package can be used standalone without Docker.

**Installation:**
```bash
git clone https://gitlab.jatic.net/jatic/aria/dataeval-app.git
cd dataeval-app
uv sync
```

**CLI Usage:**
```bash
python -m dataeval_app --dataset-path /path/to/your/dataset
python -m dataeval_app --dataset-path /path/to/dataset --split train
```

**Python API Usage:**
```python
from pathlib import Path
from dataeval_app import load_dataset, inspect_dataset

dataset = load_dataset(Path("/path/to/dataset"), split="train")
inspect_dataset(Path("/path/to/dataset"))
```

**Development:**
```bash
uv sync --group dev
nox
```

## License

MIT
