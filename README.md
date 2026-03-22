# DataEval Workflows

Workflow orchestration for DataEval with GPU support.

## Quick Start

```bash
# 1. Build CUDA 11.8 container
docker build -f Dockerfile.cu118 -t dataeval:cu118 .

# 2. Show help
docker run dataeval:cu118

# 3. Run with data and output
docker run --gpus all \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval:cu118
```

## Wrapper Scripts

For easier usage (GPU is default):

```bash
# Linux/Mac
./run.sh --data /path/to/data --output /path/to/output
./run.sh --data /path/to/data --output /path/to/output --cpu

# Windows PowerShell
.\run.ps1 -Data /path/to/data -Output /path/to/output
.\run.ps1 -Data /path/to/data -Output /path/to/output -CPU
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
| `/dataeval` | ro | Data directory — datasets, models, configs (required) |
| `/output` | rw | Results (required) |
| `/cache` | rw | Computation cache (optional) |

### File Permissions

The container runs as a non-root user (`dataeval`, UID 1000). Mounted directories for `/output` and `/cache` must be writable by the container process. There are two approaches:

#### Option 1: Pass your host UID (recommended)

Use `--user` to run the container as your host user, so mounted directories are naturally writable:

```bash
docker run --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval:cu118
```

#### Option 2: Open directory permissions

Make the output and cache directories world-writable on the host:

```bash
chmod 777 /path/to/output /path/to/cache
```

Then run without `--user`. This is simpler but less secure.

### Custom Data Root

The data root path can be overridden via the `DATAEVAL_DATA` environment variable:

```bash
docker run --gpus all \
  -e DATAEVAL_DATA=/data \
  --mount type=bind,source=/path/to/data,target=/data,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval:cu118
```

## Configuration

Config files (YAML or JSON) can be placed anywhere in your data directory. By default, all YAML/JSON files at the root of the data mount are auto-discovered and merged.

To specify a config path explicitly:

```bash
# Config folder within data directory
./run.sh --data /path/to/data --output /path/to/output --config config/

# Single config file
./run.sh --data /path/to/data --output /path/to/output --config params.yaml
```

Dataset and model paths in config files are resolved relative to the data root (`/dataeval` by default).

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
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
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
python -m dataeval_flow --data-root /path/to/data --output /path/to/output
```

**Python API Usage:**
```python
from pathlib import Path
from dataeval_flow import load_config, run_tasks

config = load_config(Path("/path/to/data/config.yaml"))
results = run_tasks(config, data_dir=Path("/path/to/data"))
print(results[0].report())
```

**Development:**
```bash
uv sync --group dev
nox
```

## License

MIT
