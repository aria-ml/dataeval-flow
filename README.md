# DataEval Workflows

Workflow orchestration for DataEval with GPU support.

## Quick Start

```bash
# 1. Build CUDA 11.8 container
docker build -f docker/Dockerfile.cu118 -t dataeval:cu118 .

# 2. Show help
docker run dataeval:cu118

# 3. Run with data and output
docker run --gpus all \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval:cu118
```

## Pulling pre-built images

Pre-built, cosign-signed images are published to Harbor for every merge to `main` and every release tag. Pull one of these instead of building from source if you don't need to modify the code.

**Rolling channel** — tracks the latest commit on `main`. The tag is overwritten on every merge.

```bash
docker pull harbor.jatic.net/aria/dataeval:cu118   # cpu / cu118 / cu124 / cu128
```

**Pinned release channel** — immutable, version-tagged images cut from `v*` git tags. Use these for reproducible workloads.

```bash
docker pull harbor.jatic.net/aria/dataeval:0.1.0-cu118
```

**Verifying the signature** — every published image is signed with [cosign](https://docs.sigstore.dev/cosign/). The public key is committed at [docker/cosign.pub](docker/cosign.pub).

```bash
cosign verify --key docker/cosign.pub harbor.jatic.net/aria/dataeval:cu118
```

Then drop the `dataeval:cu118` reference in the Quick Start `docker run` commands above with the fully-qualified `harbor.jatic.net/aria/dataeval:cu118` (or pinned version) and skip step 1.

> **Note on feature branches.** Containers are only built and published from `main` and release tags — no image is produced for MRs or topic branches. If you want to run a feature branch as a container, check it out and follow the Quick Start to build locally; the resulting image will pick up the branch's version via `git describe`.

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
docker run --gpus all \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval:cu118 --config config/

# Single config file
docker run --gpus all \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval:cu118 --config params.yaml
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
docker build -f docker/Dockerfile.cpu -t dataeval:cpu .
docker run dataeval:cpu  # Shows help
docker run \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval:cpu
```

## CLI Modes

DataEval Flow has three modes:

| Command                | Purpose                                                          |
| ---------------------- | ---------------------------------------------------------------- |
| `dataeval-flow [opts]` | Headless execution — for automation and CI/CD pipelines          |
| `dataeval-flow app`    | Interactive TUI dashboard — configure, execute, and view results |
| `dataeval-flow config` | Simple CLI config builder — create/edit configs without the TUI  |

### Interactive TUI (`app`)

**Installation:**

```bash
uv sync --extra app          # or: pip install dataeval-flow[app]
```

**Usage:**

```bash
# Launch with a blank config
python -m dataeval_flow app

# Load an existing config for editing
python -m dataeval_flow app --config /path/to/params.yaml
```

The TUI provides a three-pane dashboard for config editing, task execution, and result viewing. It auto-discovers available torchvision transforms, dataeval selection classes, and workflow types, generating dynamic parameter forms from their schemas.

### Simple CLI Config Builder (`config`)

For environments without the TUI dependency:

```bash
python -m dataeval_flow config
python -m dataeval_flow config --config /path/to/params.yaml
```

Configs can be saved as YAML or JSON.

## Dependencies

- `dataeval` - Core evaluation library
- `datasets` - Huggingface library
- `maite-datasets` - MAITE protocol adapter
- `maite` - MAITE protocol library
- `pydantic` - Structural typing and schema validation

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
python -m dataeval_flow --data /path/to/data --output /path/to/output
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

## Versioning

The package version is **derived from git tags** — there is no hardcoded version anywhere in the source tree. `hatch-vcs` reads `git describe --tags` at build/install time and writes the resolved version to a generated `src/dataeval_flow/_version.py` (gitignored), which `dataeval_flow.__init__` imports at runtime.

**Release flow for container images:**

1. Push a semver tag (e.g. `v0.2.0`) — this is the single source of truth for the release version.
2. The `push:docker` CI job runs `git describe --tags --always --dirty | sed 's/^v//'` to resolve `${VERSION}`, then passes `--build-arg DATAEVAL_FLOW_VERSION="${VERSION}"` to `docker buildx build` for both the `test` and `prod` stages.
3. The `prod` stage in [docker/Dockerfile.j2](docker/Dockerfile.j2) redeclares the ARG and:
   - Bakes the resolved version into `/app/src/dataeval_flow/_version.py` so `dataeval_flow.__version__` matches the wheel version at runtime.
   - Stamps the OCI `org.opencontainers.image.version` label with the same value.
4. The image is pushed to Harbor and cosign-signed.

The `ARG DATAEVAL_FLOW_VERSION="…"` default rendered into each committed `docker/Dockerfile.<variant>` by `docker/generate.py` is only used for **local** `docker build` invocations that don't pass `--build-arg`. Release builds always override it, so the committed default is allowed to drift from the latest tag and does not need to be regenerated at release time.

## License

MIT
