<!-- markdownlint-disable MD041 -->
![DataEval Flow](https://raw.githubusercontent.com/aria-ml/dataeval-flow/main/docs/source/_static/images/DataEvalFlow_Logo.png)

# DataEval Flow

DataEval Flow wraps DataEval's analytics in a containerized workflow engine. Its steps compose into pipelines that run
identically locally or in a CUDA-enabled container. Cleaning, audit, coverage, drift, OOD, splitting, and prioritization
ship as presets, driven by YAML or JSON from a headless or interactive CLI.

Read the documentation at <https://dataeval-flow.readthedocs.io/>.

## What DataEval Flow is for

<!-- start needs -->

DataEval Flow turns DataEval's analytics into a composable workflow engine, in two complementary ways:

- It is composable. Each capability, such as duplicate detection, outlier detection, splitting, or export, is a
  self-contained step with declared inputs and outputs. You compose steps into your own workflows in configuration, and
  the built-in workflows are presets made from the same steps. The set is extensible, and new steps can be added as
  plugins.
- It is containerized. Steps run in a reproducible, CUDA-enabled container (or locally), so a pipeline behaves the same
  on a laptop, a cluster, or an air-gapped system, and emits the same machine-readable results.

DataEval Flow lets T&E engineers compose and run multi-stage data evaluation pipelines without writing Python glue code.
Pipelines are described in YAML or JSON, executed locally or in a CUDA-enabled container, and produce both
human-readable reports and machine-readable result envelopes that satisfy JATIC interoperability requirements. It builds
directly on the [DataEval](https://dataeval.readthedocs.io/) library. The underlying evaluators, namely outlier and
duplicate detection, coverage and gap analysis, drift and OOD monitoring, dataset splitting, prioritization, and
statistical analysis, are the same algorithms DataEval exposes, wrapped in a reproducible orchestration layer. Each
DataEval evaluator can also run on its own, as an **evaluator**. It reports what DataEval determined, with no health
status. For example, you can run duplicate detection without the rest of data cleaning.

<!-- end needs -->

## Target audience

<!-- start JATIC interop -->

DataEval Flow is for data scientists, ML engineers, and T&E engineers who want to run automated data-evaluation
pipelines against their image datasets and models. It is part of the JATIC suite of tools. DataEval Flow has native
interoperability when you use MAITE-compliant datasets and models, so its outputs compose with other MAITE-conforming
JATIC tools.

<!-- end JATIC interop -->

## Limitations and requirements for use

- DataEval Flow supports computer-vision image datasets only. It operates on image classification and object-detection
  datasets and does not handle NLP or tabular data.
- Non-MAITE sources are consumed through the built-in adapters (HuggingFace, COCO, YOLO, TorchVision, ImageFolder).
  Native JATIC interoperability requires MAITE-compliant datasets and models.
- Some workflows need metadata. Bias, parity, and metadata-insight analyses require per-sample metadata factors to be
  present in the dataset.
- Some workflows need a model or embeddings. Embedding-space drift, OOD detection, and prioritization require a feature
  extractor (ONNX/PyTorch) or precomputed embeddings.
- Drift and OOD need a representative reference. Detection baselines are only as good as the reference dataset they are
  fit on.
- The service is a preview. The container runs a pipeline to completion and exits by default. `dataeval-flow serve`
  keeps it running behind an HTTP API whose endpoints may change in any release without a deprecation cycle.

## System requirements

The guidance below applies to both the container and the Python-library forms.

### Tested platforms

DataEval Flow is developed and tested on Linux (Ubuntu 22.04 and 24.04, including WSL2). The Python package supports
Python 3.11–3.14, and the CI test matrix runs all four. The container images are all built on Ubuntu 24.04 (cpu, cu126,
cu130). macOS and Windows are supported only through Docker or WSL2 and are not part of the CI test matrix. If you hit
an issue on those hosts, the OS/hardware may be the root cause.

### Architecture

All container images and the dependency stack target linux/amd64 (x86-64). arm64 / Apple Silicon is not built or tested.
On those hosts, run the CPU image under emulation or install the library from source. The `dataeval_flow` package ships
no compiled extensions of its own, so the library form runs anywhere its dependencies (PyTorch, NumPy, SciPy) provide
x86-64 wheels.

### Recommended minimum hardware

| Resource | Minimum         | Recommended         | Notes                                                                                                                                                                                                                                                         |
| -------- | --------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU      | 2 cores         | 4+ cores            | Dataset loading and statistical analysis are CPU-bound.                                                                                                                                                                                                       |
| Memory   | 8 GB            | 16+ GB              | Datasets and embeddings are held in memory during a run; peak scales with dataset and batch size. Memory is the primary limit on dataset size.                                                                                                                |
| Disk     | 10 GB           | 20+ GB              | Several GB for the container image / dependencies, plus dataset and `/cache` storage.                                                                                                                                                                         |
| GPU      | none (optional) | NVIDIA, ≥ 4 GB VRAM | Optional. It is used only to accelerate model-based embedding extraction (ONNX / PyTorch). Every workflow runs CPU-only via the `cpu` image or a CPU torch build; a GPU speeds up embedding-heavy workflows (drift / OOD / prioritization) on large datasets. |

A GPU is never required. When deploying the container under Kubernetes, request at least the minimum CPU/memory above.
Size memory to your largest dataset.

### Internet access

- Installation needs network access to PyPI and the PyTorch wheel index (or, for the container, to the base image and
  the Harbor registry).
- The first run downloads any datasets referenced from the HuggingFace Hub (and, in the tutorials, sample datasets such
  as MNIST / CPPE-5). Model weights referenced by URL are likewise fetched on first use.
- Offline / air-gapped operation is supported once the image, datasets, and models are staged locally. Point the config
  at on-disk dataset/model paths and set `HF_HUB_OFFLINE=1` (and `HF_DATASETS_OFFLINE=1`). With local inputs the batch
  container makes no outbound network calls of its own at run time.

## Quick start

```bash
# 1. Build CUDA 12.6 container
docker build -f docker/Dockerfile.cu126 -t dataeval-flow:cu126 .

# 2. Show help
docker run dataeval-flow:cu126

# 3. Run with data and output
docker run --gpus all \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval-flow:cu126
```

## Pulling pre-built images

Pre-built, cosign-signed images are published to Harbor for every merge to `main` and every release tag. Pull one of
these if you do not need to modify the code.

The release channel, `latest-<variant>`, points at the newest stable release. It is a retag of a version tag, not a
separate build, so it carries that release's signature, scan report and SBOM.

```bash
docker pull harbor.jatic.net/aria/dataeval-flow:latest-cu126   # cpu / cu126 / cu130
```

The pinned release channel holds immutable, version-tagged images cut from `v*` git tags. Use these for reproducible
workloads.

```bash
docker pull harbor.jatic.net/aria/dataeval-flow:0.3.0-cu126
```

The rolling channel, `main-<variant>`, tracks the latest commit on `main` and is overwritten on every merge. It is
unstable and is not a release.

```bash
docker pull harbor.jatic.net/aria/dataeval-flow:main-cu126
```

Every published image is signed with [cosign](https://docs.sigstore.dev/cosign/). The public key is committed at
[docker/cosign.pub](https://github.com/aria-ml/dataeval-flow/blob/main/docker/cosign.pub). Verify the signature with:

```bash
cosign verify --key docker/cosign.pub harbor.jatic.net/aria/dataeval-flow:latest-cu126
```

Then replace the `dataeval-flow:cu126` reference in the Quick start `docker run` commands above with the fully qualified
`harbor.jatic.net/aria/dataeval-flow:latest-cu126` (or a pinned version) and skip step 1.

Containers are only built and published from `main` and release tags. No image is produced for MRs or topic branches. To
run a feature branch as a container, check it out and follow the Quick start to build locally. The resulting image picks
up the branch's version via `git describe`.

## Requirements

| Requirement   | Version                                                     |
| ------------- | ----------------------------------------------------------- |
| Docker        | >= 20.10                                                    |
| NVIDIA GPU    | Any (for GPU mode)                                          |
| NVIDIA Driver | >= 525 for the `cu126` image; >= 580 for `cu130` (GPU mode) |
| CUDA          | 12.6 (`cu126`) or 13.0 (`cu130`), bundled in the image      |

The CUDA runtime libraries ship inside the image via PyTorch and `onnxruntime-gpu`, so the host needs only the NVIDIA
driver and the NVIDIA Container Toolkit, with no host CUDA install. The driver floors above are the minimums for the
CUDA major versions. Any driver from the 525 series up runs a CUDA 12.x image under minor-version compatibility, and
CUDA 13.0 requires a 580-series or newer driver.

### Verify GPU access

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

## Volume mounts

| Path        | Mode | Purpose                                                      |
| ----------- | ---- | ------------------------------------------------------------ |
| `/dataeval` | ro   | Data directory with datasets, models, and configs (required) |
| `/output`   | rw   | Results (required)                                           |
| `/cache`    | rw   | Computation cache (optional)                                 |

### File permissions

The container runs as a non-root user (`dataeval`, UID 1000). Mounted directories for `/output` and `/cache` must be
writable by the container process. There are two approaches.

#### Pass your host UID (recommended)

Use `--user` to run the container as your host user, so mounted directories are writable:

```bash
docker run --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval-flow:cu126
```

#### Open directory permissions

Make the output and cache directories world-writable on the host:

```bash
chmod 777 /path/to/output /path/to/cache
```

Then run without `--user`. This is simpler but less secure.

### Custom data root

Override the data root path with the `DATAEVAL_DATA` environment variable:

```bash
docker run --gpus all \
  -e DATAEVAL_DATA=/data \
  --mount type=bind,source=/path/to/data,target=/data,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval-flow:cu126
```

## Environment variables

All runtime environment variables are optional. Command-line options take precedence over them (see [Input
precedence](#input-precedence) below).

| Variable                | Purpose                                           | Default                                                   |
| ----------------------- | ------------------------------------------------- | --------------------------------------------------------- |
| `DATAEVAL_DATA`         | Input data root for datasets, models, and configs | `/dataeval` in the container; current directory otherwise |
| `DATAEVAL_OUTPUT`       | Directory for results and reports                 | `/output` in the container                                |
| `DATAEVAL_CACHE`        | Disk-backed computation cache (optional)          | `/cache` when that mount is present and writable          |
| `DATAEVAL_REPORT_WIDTH` | Characters per line of the text report            | `80`; at least `40`                                       |

No secret mounts or credentials are required. DataEval Flow uses no API keys, tokens, or passwords.

`DATAEVAL_FLOW_VERSION` and `DATAEVAL_NOX_UV_EXTRAS_OVERRIDE` are build-time only and are not read at run time.
`UV_EXTRAS_OVERRIDE` and `CONTAINER_MODE` are baked into the image by the build and read at run time by the entrypoint,
only to label the help text and decide whether to run the GPU check. Neither is meant to be set by the caller.

## Input precedence

For any input, the resolution order is:

1. Command-line option: `--config`, `--data`, `--output`, `--cache`
2. Environment variable: `DATAEVAL_DATA`, `DATAEVAL_OUTPUT`, `DATAEVAL_CACHE`
3. Built-in default: the container mount paths above (or the current directory outside the container)

Dataset and model paths inside a config file are resolved relative to the data root. A relative path not found directly
is also looked up under the conventional `data/` (datasets) and `models/` (models) subfolders of the data root.

## Interface documentation

The container prints its full interface (mounts, environment variables, CLI options, precedence, and examples) via its
help command. Help is also the default when the container runs with no pipeline arguments:

```bash
docker run dataeval-flow:cu126 --help
```

The library form exposes the same options via `python -m dataeval_flow --help`. The published [container
reference](https://dataeval-flow.readthedocs.io/en/latest/reference/containers.html) documents every input, default, and
configuration dependency.

## Configuration

Place config files (YAML or JSON) anywhere in your data directory. By default, all YAML/JSON files at the root of the
data mount are auto-discovered and merged.

To specify a config path explicitly:

```bash
# Config folder within data directory
docker run --gpus all \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval-flow:cu126 --config config/

# Single config file
docker run --gpus all \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval-flow:cu126 --config params.yaml
```

Dataset and model paths in config files are resolved relative to the data root (`/dataeval` by default).

## Dataset formats

A `datasets` entry selects its loader with `format`:

| `format`       | Structure                                                  | Split selection                     |
| -------------- | ---------------------------------------------------------- | ----------------------------------- |
| `huggingface`  | HuggingFace arrow dataset or dataset dict                  | `split:` (omit for a single split)  |
| `image_folder` | Directory of images, optionally one subdirectory per class | n/a                                 |
| `coco`         | COCO images plus a JSON annotation file                    | `annotations_file:` + `images_dir:` |
| `yolo`         | Ultralytics root with `data.yaml` plus image/label trees   | `split:` (omit to load every split) |

Both single-split datasets and multi-split dataset dicts are supported. In-memory MAITE and TorchVision datasets are
also accepted programmatically via `DatasetProtocolConfig`, which is not serializable and so cannot be named in a config
file. See the [Run workflows in
containers](https://dataeval-flow.readthedocs.io/en/latest/how_to/containerized_workflows.html) guide for the full field
list of each format.

## CPU fallback

For machines without an NVIDIA GPU:

```bash
docker build -f docker/Dockerfile.cpu -t dataeval-flow:cpu .
docker run dataeval-flow:cpu  # Shows help
docker run \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  dataeval-flow:cpu
```

## CLI modes

DataEval Flow has nine modes:

| Command                    | Purpose                                                              |
| -------------------------- | -------------------------------------------------------------------- |
| `dataeval-flow [opts]`     | Headless execution for automation and CI/CD pipelines                |
| `dataeval-flow serve`      | Preview: long-running HTTP service that queues pipelines (`service`) |
| `dataeval-flow app`        | Interactive TUI dashboard to configure, execute, and view results    |
| `dataeval-flow config`     | Simple CLI config builder to create/edit configs without the TUI     |
| `dataeval-flow encoding`   | Write the metadata encoding descriptor a result was computed under   |
| `dataeval-flow verify`     | Check that a source still holds the items a run's manifest records   |
| `dataeval-flow workflows`  | List the available workflow types, or print one's parameter schema   |
| `dataeval-flow evaluators` | List the available evaluator types, or print one's parameter schema  |
| `dataeval-flow steps`      | List every step a custom workflow can chain, or describe one         |

Use `dataeval-flow --version` to report the installed build, which identifies the image that produced a result.

### Headless execution

```bash
# Every enabled task in the config
dataeval-flow --config params.yaml --data . --output ./results

# One task by name, whether or not the config marks it enabled
dataeval-flow --config params.yaml --task clean_my_data

# Several, in the order given
dataeval-flow --config params.yaml --task clean_my_data --task analyze_my_data

# Stop the pipeline when a finding breaches its health threshold
dataeval-flow --config params.yaml --output ./results --fail-on-warning
```

The exit codes are `0` when every task succeeded, `1` when any task failed, and `4` when an audit's verdict is worse
than `--require`, or `require` in the pipeline's `result:` block. Health warnings are reported but are not fatal unless
you pass `--fail-on-warning`, or set `fail_on: warning` in the pipeline's `result:` block, which turns them into an exit
code of `3`. A mistyped flag exits with `2`. When more than one applies, `1` comes first, then `4`, then `3`. Every
run's `result.json` carries a `health` block per task (`status`, `warnings`, `findings`) so a pipeline can gate on
findings without parsing the text report.

### Service (`serve`)

```bash
pip install "dataeval-flow[service]"
dataeval-flow serve --data /path/to/data --output /path/to/service-output
```

The service listens on `127.0.0.1:8001` (`--host`/`--port`, or `DATAEVAL_SERVICE_HOST`/`DATAEVAL_SERVICE_PORT`).
`POST /v1/runs` queues a pipeline. Each run executes as the headless command would, one at a time, and keeps its
snapshot, logs and result files under `<output>/runs/<id>/`. `/healthz`, `/livez` and `/readyz` report its health, and
`/openapi.json` describes the API. It has no authentication, so expose it on a trusted network only. The service is a
preview. Its HTTP API may change or be removed in any release without a deprecation cycle. See [Run Flow as a
service](https://dataeval-flow.readthedocs.io/en/latest/how_to/run_flow_as_a_service.html) and the [Service
Reference](https://dataeval-flow.readthedocs.io/en/latest/reference/service.html).

### Workflow discovery (`workflows`)

For images that ship without the TUI extra:

```bash
# What can this build run?
dataeval-flow workflows
dataeval-flow workflows --json

# What does a given workflow type accept?
dataeval-flow workflows quality
```

`dataeval-flow steps` lists every step a custom workflow can chain. `--json` prints the catalog Studio reads.

### Interactive TUI (`app`)

Install the extra:

```bash
uv sync --extra app          # or: pip install dataeval-flow[app]
```

Run the TUI:

```bash
# Launch with a blank config
python -m dataeval_flow app

# Load an existing config for editing
python -m dataeval_flow app --config /path/to/params.yaml
```

The TUI provides a three-pane dashboard for config editing, task execution, and result viewing. It auto-discovers
available torchvision transforms, dataeval selection classes, and workflow types, generating dynamic parameter forms
from their schemas.

### Simple CLI config builder (`config`)

For environments without the TUI dependency:

```bash
python -m dataeval_flow config
python -m dataeval_flow config --config /path/to/params.yaml
```

Configs can be saved as YAML or JSON.

### Encoding descriptor (`encoding`)

Extract the metadata encoding descriptor from an archived `result.json` and write it where it can be reviewed and
committed, so a later dataset is cut the same way:

```bash
python -m dataeval_flow encoding output/results/result.json -o policy/factor_bins.json
python -m dataeval_flow encoding output/results/result.json --task analyze_my_data
```

Reference the committed descriptor from a metadata policy's `encoding` field. A run with `-o` already writes
`results/encoding.json` beside its results. This command recovers one from a result archived earlier. See [Configure
metadata binning](https://dataeval-flow.readthedocs.io/en/latest/how_to/configure_metadata_binning.html).

## Dependencies

Declared runtime dependencies:

- `dataeval` - Core evaluation library
- `datamaite` - MAITE protocol dataset loaders (which brings in `maite` itself)
- `pydantic` - Structural typing and schema validation
- `click` - Interactive prompts for the simple CLI config builder
- `pyyaml` - YAML config parsing
- `numpy` - Array handling

PyTorch arrives transitively through `dataeval`. Optional extras (`onnx`, `opencv`, `app`, `ontology`, and the CUDA
variants) are listed under [Running without container](#running-without-container).

## Troubleshooting

### Build appears stuck at `uv sync`

The Docker build may appear frozen during the `uv sync` step:

```text
=> [builder 7/7] RUN uv sync --frozen --no-dev --no-install-project    1139.3s
```

This is normal. The step downloads ~2GB of dependencies (PyTorch, scipy, etc.) with no progress indicator.

| Network speed | Expected build time |
| ------------- | ------------------- |
| 100 Mbps      | ~10 minutes         |
| 30 Mbps       | ~20 minutes         |
| 10 Mbps       | ~45 minutes         |

The first build is slow. Later builds use the Docker cache and complete in seconds.

## Running without container

You can use the `dataeval_flow` package standalone, without Docker.

### Installation

See the [Installation guide](https://dataeval-flow.readthedocs.io/en/latest/home/installation.html) for the full matrix.
In short:

`pip` from PyPI (no source checkout). PyTorch arrives transitively via `dataeval`. To choose a variant, install `torch`
from that variant's index first, then install DataEval Flow, which accepts the build already present. If you omit step
1, you get the CUDA-bundled manylinux build of torch from PyPI, which is much larger:

```bash
# 1. Pick your PyTorch build (cpu / cu126 / cu130)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. Install DataEval Flow
pip install dataeval-flow
```

> Use `--index-url`, not `--extra-index-url`, to pick a CUDA build. `--extra-index-url` adds an index instead of
> replacing PyPI, and pip then takes the highest version across both. The CUDA indexes lag the latest PyTorch release,
> so PyPI usually wins and you get the default CUDA-bundled build. The install succeeds with no warning. `--index-url`
> replaces the index, so it is reliable. (For CPU only,
> `pip install dataeval-flow --extra-index-url https://download.pytorch.org/whl/cpu` does work, because the CPU index
> tracks the latest release.)

`torchvision` is not installed by default. It is imported lazily and is only needed for preprocessing pipelines, the
torchvision dataset adapter, and the TUI's transform discovery. Install it together with `torch` in step 1 so both come
from the same index. Feature extras (`onnx`, `onnx-cu126`, `onnx-cu130`, `opencv`, `app`, `ontology`) work normally
under pip and are independent of the PyTorch variant:

```bash
pip install "dataeval-flow[onnx,opencv,app]"
```

> The `cpu` / `cu126` / `cu130` extras do not select a PyTorch variant under pip. All three declare the same
> requirements (`torch`, `torchvision`). What distinguishes them is `[tool.uv.sources]`, which routes those packages to
> the right wheel index. That is project metadata applied by uv when resolving from source, not part of the published
> wheel. Under pip their only effect is pulling in `torchvision`. Select the variant with `--index-url` under pip,
> `--torch-backend` under `uv pip`, and use the extras only for source installs.

`uv` from PyPI:

```bash
uv pip install dataeval-flow --torch-backend cpu     # or cu126 / cu130 / auto
```

`uv` from source (default toolchain; uses committed `uv.lock`); extras apply here:

```bash
git clone https://github.com/aria-ml/dataeval-flow.git
cd dataeval-flow
uv sync --extra cpu      # or cu126 / cu130; add --extra onnx --extra opencv --extra app as needed
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

- PyTorch is installed from PyPI/`download.pytorch.org` in every path (it is no longer maintained on conda-forge).
- GPU variants (`cu126`, `cu130`) are only wired through `uv` and `pip` today. The conda path installs the CPU build of
  PyTorch.

### CLI usage

```bash
python -m dataeval_flow --config /path/to/config --output /path/to/output
python -m dataeval_flow --data /path/to/data --output /path/to/output
```

### Python API usage

```python
from pathlib import Path
from dataeval_flow import load_config, run_tasks

config = load_config(Path("/path/to/data/config.yaml"))  # a file, or a folder of config files
results = run_tasks(config, data_dir=Path("/path/to/data"))  # keyed by task name
for name, result in results.items():
    print(result.report())
```

### Development

```bash
uvx --with nox-uv nox -s dev     # build .venv with the dev toolchain
source .venv/bin/activate
nox                              # run the default sessions
```

`nox -s dev` prompts for the Python version and device variant, records the choice in `.cuda-version` so the other
sessions match it, and installs the `onnx` and `app` extras alongside. See [Development
setup](https://github.com/aria-ml/dataeval-flow/blob/main/CONTRIBUTING.md#development-setup) for the arguments it takes.

## Versioning

The package version is derived from git tags, with no hardcoded version anywhere in the source tree. `hatch-vcs` reads
`git describe --tags` at build/install time and writes the resolved version to a generated
`src/dataeval_flow/_version.py` (gitignored), which `dataeval_flow.__init__` imports at runtime.

The release flow for container images is:

1. Push a semver tag (for example `v0.3.0`), the single source of truth for the release version.
2. The `push:docker` CI job runs `git describe --tags --always --dirty | sed 's/^v//'` to resolve `${VERSION}`, then
   passes `--build-arg DATAEVAL_FLOW_VERSION="${VERSION}"` to `docker buildx build` for both the `test` and `prod`
   stages.
3. The `prod` stage in
   [docker/Dockerfile.j2](https://github.com/aria-ml/dataeval-flow/blob/main/docker/Dockerfile.j2) redeclares the ARG
   and:
   - Bakes the resolved version into `/app/src/dataeval_flow/_version.py` so `dataeval_flow.__version__` matches the
     wheel version at runtime.
   - Stamps the OCI `org.opencontainers.image.version` label with the same value.
4. The image is pushed to Harbor and cosign-signed.

The `ARG DATAEVAL_FLOW_VERSION="…"` default rendered into each committed `docker/Dockerfile.<variant>` by
`docker/generate.py` is only used for local `docker build` invocations that do not pass `--build-arg`. Release builds
always override it, so the committed default is allowed to drift from the latest tag and does not need to be regenerated
at release time.

## License

MIT. See [LICENSE](https://github.com/aria-ml/dataeval-flow/blob/main/LICENSE).

## Contributing

See
[CONTRIBUTING.md](https://github.com/aria-ml/dataeval-flow/blob/main/CONTRIBUTING.md)
for bug reports and contribution guidelines, and
[BRANCHING.md](https://github.com/aria-ml/dataeval-flow/blob/main/BRANCHING.md)
for the branching and release strategy.

## Contact

For questions or feedback, reach out to <dataeval-flow@ariacoustics.com>.

## Acknowledgment

### CDAO funding acknowledgment

<!-- start acknowledgment -->

This material is based upon work supported by the Chief Digital and Artificial
Intelligence Office under Contract No. W519TC-23-9-2033. The views and
conclusions contained herein are those of the author(s) and should not be
interpreted as necessarily representing the official policies or endorsements,
either expressed or implied, of the U.S. Government.

<!-- end acknowledgment -->
