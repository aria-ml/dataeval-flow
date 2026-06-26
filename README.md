# DataEval Flow

DataEval Flow provides workflow orchestration for DataEval evaluators, packaging
data cleaning, drift monitoring, OOD detection, analysis, splitting,
prioritization, and parameter-sweep pipelines behind a single declarative
configuration format and both headless and interactive CLIs.

📖 **Documentation:** <https://dataeval-flow.readthedocs.io/>

## What DataEval Flow is for

<!-- start needs -->

DataEval Flow lets T&E engineers compose and run multi-stage data evaluation
pipelines without writing Python glue code. Pipelines are described in YAML or
JSON, executed locally or in a CUDA-enabled container, and produce both
human-readable reports and machine-readable result envelopes that satisfy JATIC
interoperability requirements. It builds directly on the
[DataEval](https://dataeval.readthedocs.io/) library, so the underlying
evaluators — outlier and duplicate detection, drift and OOD monitoring, dataset
splitting, prioritization, and statistical analysis — are the same algorithms
DataEval exposes, wrapped in a reproducible orchestration layer.

<!-- end needs -->

## Target Audience

<!-- start JATIC interop -->

DataEval Flow is intended for data scientists, ML engineers, and T&E engineers
who want to run automated data-evaluation pipelines against their image datasets
and models. It is part of the JATIC suite of tools: DataEval Flow has native
interoperability when using MAITE-compliant datasets and models, so its outputs
compose with other MAITE-conforming JATIC tools.

<!-- end JATIC interop -->

## Limitations and requirements for use

- **Computer-vision image datasets only.** DataEval Flow operates on image
  classification and object-detection datasets; it does not handle NLP or
  tabular data.
- **MAITE for native interoperability.** Non-MAITE sources are consumed through
  the built-in adapters (HuggingFace, COCO, YOLO, TorchVision, ImageFolder);
  native JATIC interoperability requires MAITE-compliant datasets/models.
- **Some workflows need metadata.** Bias, parity, and metadata-insight analyses
  require per-sample metadata factors to be present in the dataset.
- **Some workflows need a model or embeddings.** Embedding-space drift, OOD
  detection, and prioritization require a feature extractor (ONNX/PyTorch) or
  precomputed embeddings.
- **Drift and OOD need a representative reference.** Detection baselines are only
  as good as the reference dataset they are fit on.
- **Batch container, not a service.** The container runs a pipeline to
  completion and exits; it is not a long-running web service (no health-check
  endpoint).

## System Requirements

The guidance below applies to both the container and the Python-library forms.

### Tested platforms

DataEval Flow is developed and tested on Linux (Ubuntu 22.04 and 24.04, including
WSL2). The Python package supports **Python 3.10–3.13**; the CI test matrix runs
3.10, 3.11, and 3.12. The container images are built on Ubuntu 22.04 (cu118) and
Ubuntu 24.04 (cpu, cu128). macOS and Windows are supported only through Docker or
WSL2 and are not part of the CI test matrix — if you hit an issue on those hosts,
the OS/hardware may be the root cause.

### Architecture

All container images and the dependency stack target **linux/amd64 (x86-64)**.
arm64 / Apple Silicon is not built or tested; on those hosts run the CPU image
under emulation or install the library from source. The `dataeval_flow` package
ships no compiled extensions of its own, so the library form runs anywhere its
dependencies (PyTorch, NumPy, SciPy) provide x86-64 wheels.

### Recommended minimum hardware

| Resource | Minimum         | Recommended         | Notes                                                                                                                                                                                                                                                      |
| -------- | --------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU      | 2 cores         | 4+ cores            | Dataset loading and statistical analysis are CPU-bound.                                                                                                                                                                                                    |
| Memory   | 8 GB            | 16+ GB              | Datasets and embeddings are held in memory during a run; peak scales with dataset and batch size. **Memory is the primary limit on dataset size.**                                                                                                         |
| Disk     | 10 GB           | 20+ GB              | Several GB for the container image / dependencies, plus dataset and `/cache` storage.                                                                                                                                                                      |
| GPU      | none (optional) | NVIDIA, ≥ 4 GB VRAM | Optional — used only to accelerate model-based embedding extraction (ONNX / PyTorch). Every workflow runs CPU-only via the `cpu` image / `[cpu]` extra; a GPU mainly speeds up embedding-heavy workflows (drift / OOD / prioritization) on large datasets. |

A GPU is never required. When deploying the container under Kubernetes, request
at least the minimum CPU/memory above; size memory to your largest dataset.

### Internet access

- **Installation** needs network access to PyPI and the PyTorch wheel index (or,
  for the container, to the base image and the Harbor registry).
- **First run** downloads any datasets referenced from the HuggingFace Hub (and,
  in the tutorials, sample datasets such as MNIST / CPPE-5). Model weights
  referenced by URL are likewise fetched on first use.
- **Offline / air-gapped operation** is supported once the image, datasets, and
  models are staged locally: point the config at on-disk dataset/model paths and
  set `HF_HUB_OFFLINE=1` (and `HF_DATASETS_OFFLINE=1`). With local inputs the
  batch container makes no outbound network calls of its own at run time.

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

Pre-built, cosign-signed images are published to Harbor for every merge to
`main` and every release tag. Pull one of these instead of building from source
if you don't need to modify the code.

**Rolling channel** — tracks the latest commit on `main`. The tag is overwritten on every merge.

```bash
docker pull harbor.jatic.net/aria/dataeval:cu118   # cpu / cu118 / cu128
```

**Pinned release channel** — immutable, version-tagged images cut from `v*` git tags. Use these for reproducible workloads.

```bash
docker pull harbor.jatic.net/aria/dataeval:0.1.0-cu118
```

**Verifying the signature** — every published image is signed with
[cosign](https://docs.sigstore.dev/cosign/). The public key is committed at
[docker/cosign.pub](docker/cosign.pub).

```bash
cosign verify --key docker/cosign.pub harbor.jatic.net/aria/dataeval:cu118
```

Then drop the `dataeval:cu118` reference in the Quick Start `docker run`
commands above with the fully-qualified `harbor.jatic.net/aria/dataeval:cu118`
(or pinned version) and skip step 1.

> **Note on feature branches.** Containers are only built and published from
> `main` and release tags — no image is produced for MRs or topic branches. If
> you want to run a feature branch as a container, check it out and follow the
> Quick Start to build locally; the resulting image will pick up the branch's
> version via `git describe`.

## Requirements

| Requirement   | Version               |
| ------------- | --------------------- |
| Docker        | >= 20.10              |
| NVIDIA GPU    | Any (for GPU mode)    |
| NVIDIA Driver | >= 520 (for GPU mode) |
| CUDA          | 11.8.0 (for GPU mode) |

### Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Volume Mounts

| Path        | Mode | Purpose                                               |
| ----------- | ---- | ----------------------------------------------------- |
| `/dataeval` | ro   | Data directory — datasets, models, configs (required) |
| `/output`   | rw   | Results (required)                                    |
| `/cache`    | rw   | Computation cache (optional)                          |

### File Permissions

The container runs as a non-root user (`dataeval`, UID 1000). Mounted
directories for `/output` and `/cache` must be writable by the container
process. There are two approaches:

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

## Environment Variables

All runtime environment variables are optional; command-line options take
precedence over them (see [Input Precedence](#input-precedence) below).

| Variable          | Purpose                                         | Default                                                   |
| ----------------- | ----------------------------------------------- | --------------------------------------------------------- |
| `DATAEVAL_DATA`   | Input data root — datasets, models, and configs | `/dataeval` in the container; current directory otherwise |
| `DATAEVAL_OUTPUT` | Directory for results and reports               | `/output` in the container                                |
| `DATAEVAL_CACHE`  | Disk-backed computation cache (optional)        | `/cache` when that mount is present                       |

No secret mounts or credentials are required — DataEval Flow uses no API keys,
tokens, or passwords. (`DATAEVAL_FLOW_VERSION` and `DATAEVAL_NOX_UV_EXTRAS_OVERRIDE`
are build-time only and are not read at run time.)

## Input Precedence

For any input, the resolution order is:

1. **Command-line option** — `--config`, `--data`, `--output`, `--cache`
2. **Environment variable** — `DATAEVAL_DATA`, `DATAEVAL_OUTPUT`, `DATAEVAL_CACHE`
3. **Built-in default** — the container mount paths above (or the current
   directory outside the container)

Dataset and model paths inside a config file are resolved relative to the data
root; a relative path not found directly is also looked up under the conventional
`data/` (datasets) and `models/` (models) subfolders of the data root.

## Interface Documentation

The container prints its full interface — mounts, environment variables, CLI
options, precedence, and examples — via its help command, which is also the
default when the container runs with no pipeline arguments:

```bash
docker run dataeval:cu118 --help
```

The library form exposes the same options via `python -m dataeval_flow --help`.
The published [container reference](https://dataeval-flow.readthedocs.io/en/latest/reference/containers.html)
documents every input, default, and configuration dependency.

## Configuration

Config files (YAML or JSON) can be placed anywhere in your data directory. By
default, all YAML/JSON files at the root of the data mount are auto-discovered
and merged.

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

| Format          | Structure                                          | Example         |
| --------------- | -------------------------------------------------- | --------------- |
| **Dataset**     | Single split, used directly                        | `cifar10_test/` |
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

The TUI provides a three-pane dashboard for config editing, task execution, and
result viewing. It auto-discovers available torchvision transforms, dataeval
selection classes, and workflow types, generating dynamic parameter forms from
their schemas.

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

```text
=> [builder 7/7] RUN uv sync --frozen --no-dev --no-install-project    1139.3s
```

**This is normal.** The step downloads ~2GB of dependencies (PyTorch, scipy, etc.) with no progress indicator.

| Network Speed | Expected Build Time |
| ------------- | ------------------- |
| 100 Mbps      | ~10 minutes         |
| 30 Mbps       | ~20 minutes         |
| 10 Mbps       | ~45 minutes         |

**Tip:** First build is slow; subsequent builds use Docker cache and complete in seconds.

## Running Without Container

The `dataeval_flow` package can be used standalone without Docker.

**Installation:**

Three installer toolchains are supported. Choose whichever fits your environment;
all three install the same dependencies pinned in their respective lockfiles.

`uv` (default toolchain):

```bash
git clone https://github.com/aria-ml/dataeval-flow.git
cd dataeval-flow
uv sync --extra cpu      # or cu118 / cu128 for CUDA variants
```

`pip` from PyPI (no source checkout). PyTorch is hosted on a separate
wheel index, so pass `--extra-index-url` matching the variant you want
(omit it and you'll get the CUDA-bundled manylinux build of torch from
PyPI, which is much larger):

```bash
# CPU-only PyTorch
pip install "dataeval-flow[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 PyTorch
pip install "dataeval-flow[cu118]" --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA 12.8 PyTorch
pip install "dataeval-flow[cu128]" --extra-index-url https://download.pytorch.org/whl/cu128
```

`poetry` (source checkout; uses committed `poetry.lock`):

```bash
git clone https://github.com/aria-ml/dataeval-flow.git
cd dataeval-flow
poetry install
```

`conda` / `mamba` (source checkout; uses committed `environment.yml`):

```bash
git clone https://github.com/aria-ml/dataeval-flow.git
cd dataeval-flow
conda env create -f environment.yml
conda activate dataeval-flow
pip install -e .         # install the package itself; conda manages deps
```

Notes:

- PyTorch is installed from PyPI/`download.pytorch.org` in every path
  (it is no longer maintained on conda-forge).
- GPU variants (`cu118`, `cu128`) are only wired through `uv` and
  `pip` today; the Poetry/conda paths install the CPU build of PyTorch.

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

The package version is **derived from git tags** — there is no hardcoded version
anywhere in the source tree. `hatch-vcs` reads `git describe --tags` at
build/install time and writes the resolved version to a generated
`src/dataeval_flow/_version.py` (gitignored), which `dataeval_flow.__init__`
imports at runtime.

**Release flow for container images:**

1. Push a semver tag (e.g. `v0.2.0`) — this is the single source of truth for the release version.
2. The `push:docker` CI job runs `git describe --tags --always --dirty | sed
   's/^v//'` to resolve `${VERSION}`, then passes `--build-arg
   DATAEVAL_FLOW_VERSION="${VERSION}"` to `docker buildx build` for both the
   `test` and `prod` stages.
3. The `prod` stage in [docker/Dockerfile.j2](docker/Dockerfile.j2) redeclares the ARG and:
   - Bakes the resolved version into `/app/src/dataeval_flow/_version.py` so
     `dataeval_flow.__version__` matches the wheel version at runtime.
   - Stamps the OCI `org.opencontainers.image.version` label with the same value.
4. The image is pushed to Harbor and cosign-signed.

The `ARG DATAEVAL_FLOW_VERSION="…"` default rendered into each committed
`docker/Dockerfile.<variant>` by `docker/generate.py` is only used for **local**
`docker build` invocations that don't pass `--build-arg`. Release builds always
override it, so the committed default is allowed to drift from the latest tag and
does not need to be regenerated at release time.

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for bug reports and contribution
guidelines, and [BRANCHING.md](BRANCHING.md) for the branching and release
strategy.

## Contact

For questions or feedback, reach out to <dataeval-flow@ariacoustics.com>.

## Acknowledgement

### CDAO Funding Acknowledgement

<!-- start acknowledgement -->

This material is based upon work supported by the Chief Digital and Artificial
Intelligence Office under Contract No. W519TC-23-9-2033. The views and
conclusions contained herein are those of the author(s) and should not be
interpreted as necessarily representing the official policies or endorsements,
either expressed or implied, of the U.S. Government.

<!-- end acknowledgement -->
