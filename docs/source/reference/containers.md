# Container Reference

Reference for running **DataEval Flow** as a container: every input the
container accepts, their defaults and precedence, volume mounts, environment
variables, dependencies between configuration parameters, and the hardware,
architecture, and network requirements for both the container and the
Python-library forms.

By default the container is a **batch** application: it runs a configured
pipeline to completion, writes its artifacts, and exits. Run with `serve`, it is
a **long-running service** instead. The service takes pipelines over HTTP, runs
each one as the batch command, and has health-check endpoints (see
[Health checks](#health-checks) and {doc}`../how_to/run_flow_as_a_service`).

## Image tags

Every image is published to `harbor.jatic.net/aria/dataeval-flow` in three variants,
`cpu`, `cu126` and `cu130`, under three kinds of tag:

| Tag                   | Example      | Points at                                  |
| --------------------- | ------------ | ------------------------------------------ |
| `<version>-<variant>` | `0.3.0-cpu`  | One release build; never reassigned        |
| `latest-<variant>`    | `latest-cpu` | The highest stable release                 |
| `main-<variant>`      | `main-cpu`   | The newest build of `main`; not a release  |

**Pin to `<version>-<variant>` for anything reproducible.** It is the only tag
that never moves, and it is the tag the cosign signature and SBOM attestation are
bound to.

`latest-<variant>` moves only to a higher stable version, so a patch cut on an
older release line, such as v0.2.3 after v0.3.0, never moves it backwards. A
release line's patches are published under their version tags only.

Prereleases (`0.4.0-rc0-cpu`) are published under their exact version tag only.
`latest-<variant>` never points at one.

See [BRANCHING.md](https://github.com/aria-ml/dataeval-flow/blob/main/BRANCHING.md)
for how release lines are cut and maintained.

## Obtaining the interface documentation

The container ships its own interface description. Print it with the help
command, which is also the default action when the container runs with no
pipeline arguments:

```bash
docker run harbor.jatic.net/aria/dataeval-flow:latest-cu130 --help
```

The library form prints the same options with `python -m dataeval_flow --help`, and
`python -m dataeval_flow serve --help` describes the service. A running service also
describes its HTTP API at `/openapi.json`, and renders it at `/docs`, whose page assets
load from a CDN.
The sections below mirror that in-container help; if the two ever disagree, the
in-container help for your specific image tag is authoritative.

## Volume mounts

| Path        | Mode       | Purpose                                              | Required |
| ----------- | ---------- | ---------------------------------------------------- | -------- |
| `/dataeval` | read-only  | Input data root — datasets, models, and config files | Yes      |
| `/output`   | read-write | Results and human-readable reports                   | Yes      |
| `/cache`    | read-write | Disk-backed computation cache                        | Optional |

Mount your host directories onto these paths, e.g.:

```bash
docker run --gpus all \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  --mount type=bind,source=/path/to/cache,target=/cache \
  harbor.jatic.net/aria/dataeval-flow:latest-cu130
```

The data root can be relocated with `DATAEVAL_DATA` / `--data` (see below).

## Secrets

DataEval Flow uses **no API keys, tokens, or passwords**, so **no secret mounts
or secret-management mechanism are required**.

## Environment variables

All runtime environment variables are optional.

| Variable                 | Purpose                                     | Default                                                                  |
| ------------------------ | ------------------------------------------- | ------------------------------------------------------------------------ |
| `DATAEVAL_DATA`          | Input data root (datasets, models, configs) | `/dataeval` in the container; current working directory otherwise        |
| `DATAEVAL_OUTPUT`        | Output directory for results and reports    | `/output` in the container                                               |
| `DATAEVAL_CACHE`         | Disk-backed computation cache directory     | Auto-set to `/cache` when that mount is present and writable (see below) |
| `DATAEVAL_REPORT_WIDTH`  | Characters per line of the text report      | The config's `result: width`, else `80`; at least `40`                   |
| `DATAEVAL_REPORT_IMAGES` | Thumbnails of the items reports name        | On; `0`, `false` or `no` turns them off                                  |
| `DATAEVAL_REQUIRE`       | The worst verdict that passes, else exit 4  | The config's `result: require`, else none                                |
| `DATAEVAL_SERVICE_HOST`  | Address `serve` listens on                  | `0.0.0.0` in the container; `127.0.0.1` otherwise                        |
| `DATAEVAL_SERVICE_PORT`  | Port `serve` listens on                     | `8001`                                                                   |

`DATAEVAL_DATA` and `DATAEVAL_OUTPUT` are baked into the image as `/dataeval` and
`/output`. `DATAEVAL_CACHE` is **not**. The entrypoint sets it to `/cache`
only when you have not set it and `/cache` exists, is writable, and is a real
mount, not the image's unmounted placeholder. If any check fails, the variable
stays unset and caching falls back to in-memory only: the run succeeds but
nothing persists between runs. Pass `--cache` explicitly to be sure.

`HF_HUB_OFFLINE` / `HF_DATASETS_OFFLINE` are standard HuggingFace variables you
may set to force fully offline operation (see [Internet access](#internet-access)).

The following are **build-time only** and are not read at run time:
`DATAEVAL_FLOW_VERSION` (stamps the wheel/image version) and
`DATAEVAL_NOX_UV_EXTRAS_OVERRIDE` (selects extras during the image build).

Two more are set by the build and read at run time, but only by the entrypoint
and not by the application: `UV_EXTRAS_OVERRIDE` (names the variant in
the help text) and `CONTAINER_MODE` (decides whether the GPU check runs).
Overriding either changes only what the container prints and whether it insists on
a GPU; neither is intended as a caller-facing knob.

## Command-line options

| Option                 | Purpose                                | Default                                            |
| ---------------------- | -------------------------------------- | -------------------------------------------------- |
| `-c`, `--config PATH`  | Config file or folder                  | Auto-discover YAML/JSON at the data root           |
| `-d`, `--data PATH`    | Input data root                        | `$DATAEVAL_DATA`, else the container default / CWD |
| `-o`, `--output PATH`  | Output directory for artifacts         | `$DATAEVAL_OUTPUT`, else `/output`                 |
| `-k`, `--cache PATH`   | Disk-backed computation cache          | `$DATAEVAL_CACHE`, else `/cache` if mounted        |
| `-v`, `--verbose`      | Increase verbosity (repeatable)        | Off — see below                                    |
| `--report-width N`     | Characters per line of the text report | `$DATAEVAL_REPORT_WIDTH`, else `result: width`     |
| `--[no-]report-images` | Thumbnails of the items reports name   | `$DATAEVAL_REPORT_IMAGES`, else on                 |
| `--require LEVEL`      | Exit 4 when a verdict is worse         | `$DATAEVAL_REQUIRE`, else `result: require`        |
| `-h`, `--help`         | Print the interface help and exit      | —                                                  |

`--verbose` is a counting flag. Without it, reports print to stdout in their short form;
`-v` prints them in full, `-vv` adds `INFO` logs, and `-vvv` adds `DEBUG` logs.
Artifacts are written to the output directory regardless of verbosity.

Optional sub-commands (default is the headless pipeline):

| Command    | Purpose                                                              |
| ---------- | -------------------------------------------------------------------- |
| `app`      | Interactive TUI dashboard (requires the `app` extra)                 |
| `config`   | Simple CLI config builder                                            |
| `encoding` | Write the metadata encoding descriptor a result was computed under   |
| `verify`   | Check that a source still holds the items a run's manifest records   |
| `serve`    | Long-running HTTP service that queues pipelines (`service` extra)    |

`encoding` takes the path to a `result.json` written by a run, plus an optional
`-o`/`--output` for where to write the descriptor (default: print it) and
`--task` to pick one task's encoding when a result holds several that differ.

`verify` takes the path to a manifest a run wrote under `results/manifests/`, plus
`--config` and `--source` for the source to check and `--data` for the data root,
which must be the run's. It exits `0` when the source holds the recorded items and `1`
otherwise, including when the manifest can't be read or the source can't be loaded.

`serve` takes `-d`/`--data`, `-o`/`--output` and `-k`/`--cache` as the batch command does,
plus `--host` and `--port`. It keeps each run under `<output>/runs/<id>/` and shares
`--cache`, else `<output>/cache`, between runs. Publish its port (`-p 8001:8001`) and run
with `--init`, which reaps the processes finished runs leave behind. The service has no
authentication: expose it on a trusted network only. The
[Service Reference](service.md) documents its API, run states and files.

## Input precedence

For every input the resolution order is:

1. **Command-line option** (`--config`, `--data`, `--output`, `--cache`)
2. **Environment variable** (`DATAEVAL_DATA`, `DATAEVAL_OUTPUT`, `DATAEVAL_CACHE`)
3. **Built-in default** (the container mount paths above, or the current
   directory outside the container)

`serve` resolves `--host` and `--port` the same way, over `DATAEVAL_SERVICE_HOST` and
`DATAEVAL_SERVICE_PORT`. Its options follow `serve`; the top-level `--log-format`
precedes it. A run the service starts takes none of the service's own `DATAEVAL_*`
variables: its pipeline snapshot alone defines it.

Dataset and model paths inside a config file are resolved **relative to the data
root**. A relative path not found directly under the data root is also looked up
under the conventional `data/` (datasets) and `models/` (models) subfolders.
Absolute paths in a config are used as-is.

## Supported inputs and formats

- **Configuration files:** YAML or JSON. When `--config` points at a folder (or
  is omitted), all YAML/JSON files at the data root are auto-discovered and
  merged into a single pipeline configuration.
- **Datasets:** HuggingFace Vision, COCO, YOLO, TorchVision, ImageFolder, and
  raw MAITE-compatible dataset objects. Both single-split datasets and
  multi-split dataset dicts are supported.
- **Models / extractors:** ONNX, PyTorch, Bag-of-Visual-Words (SIFT), Flatten,
  and Uncertainty extractors.

## Configuration defaults

Defaults for the top-level inputs are listed in the tables above. Within a
config, notable defaults include: `--config` auto-discovers and merges root-level
YAML/JSON; ONNX extractors default `flatten: true`; BoVW defaults
`vocab_size: 2048`. Each evaluator workflow carries its own defaults; see the
{ref}`Config Reference <config-reference>` for every key's type and default, or the
generated [API Reference](autoapi/dataeval_flow/index) for the config models.

## Dependencies between configuration parameters

Some configuration fields are only meaningful — or only valid — in combination
with others:

- **Extractor model type drives required fields.** An extractor's `model`
  selector determines which fields are required:
  - `model: onnx` **requires** `model_path`; `output_name` is optional;
    `image_height` and `image_width` **must be set together** (setting only one
    is rejected) and, when both are set, override the model's native input size.
  - `model: torch` **requires** `model_path`; `layer_name` and `use_output` are optional.
  - `model: uncertainty` **requires** `model_path`, `metadata_path` (DataEval's model metadata) and `preds_type`
    (`logits`, `probs` or `sigmoid`), and `confidence` for a detector;
    `image_height` and `image_width` are set together, and needed when the model's metadata leaves its input size
    open; only drift and OOD evaluators read it
  - `model: bovw` and `model: flatten` need **no** `model_path`.
- **Preprocessor references must resolve.** An extractor's `preprocessor` field,
  if set, must name a preprocessor defined in the same configuration.
- **GPU execution requires a CUDA image and runtime.** Tools compute on a GPU
  where PyTorch sees one: a CUDA image variant (`cu126` / `cu130`) run with
  `--gpus all`. The `cpu` image computes on the CPU, and a config names no
  device. `--gpus device=1` picks a GPU; `-e CUDA_VISIBLE_DEVICES=` keeps a CUDA
  image on the CPU. Each result's `metadata.device` records the device used.
- **Metadata-dependent analyses.** Bias, parity, and metadata factor outputs
  require per-sample metadata factors to be present in the dataset; without them
  those analyses are skipped.
- **The `app` sub-command requires the `app` extra** to be installed in the image.
- **`serve` requires the `service` extra**, which every image ships, and an output
  directory: `--output`, or `DATAEVAL_OUTPUT`, which images set to `/output`. Without
  `--cache` or `DATAEVAL_CACHE`, its runs share `<output>/cache`.
- **`DATAEVAL_SERVICE_HOST` and `DATAEVAL_SERVICE_PORT` apply to `serve` only.** The
  batch command, `app` and `config` ignore them.
- **A pipeline submitted to `serve` must keep `json` in `result: formats`**: the service
  reads each run's results from its JSON file, and refuses such a pipeline with `422`.

## Recommended minimum hardware

The same rough-order-of-magnitude guidance applies to the container and the
Python-library forms.

| Resource | Minimum         | Recommended         | Notes                                                                                                                                              |
| -------- | --------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU      | 2 cores         | 4+ cores            | Dataset loading and statistical analysis are CPU-bound.                                                                                            |
| Memory   | 8 GB            | 16+ GB              | Datasets and embeddings are held in memory during a run; peak scales with dataset and batch size. **Memory is the primary limit on dataset size.** |
| Disk     | 10 GB           | 20+ GB              | Several GB for the image / dependencies, plus dataset and `/cache` storage.                                                                        |
| GPU      | none (optional) | NVIDIA, ≥ 4 GB VRAM | Optional — accelerates model-based embedding extraction only. Every workflow runs CPU-only.                                                        |

When scheduling the container on Kubernetes, request at least the minimum CPU and
memory above and size memory to your largest dataset. A GPU is never required.

Under `serve`, the service process itself uses about 1 GB, since it loads Flow and
PyTorch, on top of the memory of the run it is executing. Size memory for both. Each
run's directory under `<output>/runs/` keeps its result files and logs until you
remove it.

## Supported architectures

All images and the dependency stack target **linux/amd64 (x86-64)**. arm64 /
Apple Silicon is not built or tested; on those hosts run the CPU image under
emulation or install the library from source. The `dataeval_flow` package ships
no compiled extensions of its own, so the library form runs anywhere its
dependencies (PyTorch, NumPy, SciPy) provide x86-64 wheels.

## Internet access

- **Build / install** requires network access to the base image, the Harbor
  registry, PyPI, and the PyTorch wheel index.
- **First run** downloads any datasets referenced from the HuggingFace Hub, and
  any model weights referenced by URL, on first use.
- **Offline / air-gapped operation** is supported once the image, datasets, and
  models are staged locally: reference them by on-disk path and set
  `HF_HUB_OFFLINE=1` (and `HF_DATASETS_OFFLINE=1`). With local inputs the batch
  container makes **no outbound network calls of its own** at run time.
- **`serve`** listens for inbound connections on its port and makes no outbound
  calls of its own; its runs need what the batch command needs. The `/docs` page
  loads its assets from a CDN in the viewer's browser; `/openapi.json` needs no
  network.

## Health checks

A batch run exposes **no health-check endpoint**. It reports success or failure through
the process exit code and the logs and reports written to the output directory: `0` for
success, `1` for a failed task or export, `2` for a mistyped command line, `3` for
health warnings under `fail_on: warning`, and `4` for a verdict worse than `--require`.
When more than one applies, `1` comes first, then `4`, then `3`.

`serve` exposes three, on its port (IR-2.3-H-2, IR-2.3-S-1):

| Endpoint   | `200` when                           | `503` when                                          |
| ---------- | ------------------------------------ | --------------------------------------------------- |
| `/healthz` | The service takes and runs work      | Its queue has stopped, or its run store is unusable |
| `/readyz`  | Same as `/healthz`                   | Same as `/healthz`                                  |
| `/livez`   | The process can run work             | Its queue has stopped and it needs a restart        |

A `503` body names its reasons: `queue-stopped` or `store-unavailable`. The service
describes its API at `/openapi.json` (IR-2.4-S-1).
