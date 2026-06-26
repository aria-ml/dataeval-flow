# Container Reference

This page is the reference for running **DataEval Flow** as a container. It
documents every input the container accepts, their defaults and precedence, the
volume mounts and environment variables, dependencies between configuration
parameters, and the hardware, architecture, and network requirements for both
the container and the Python-library forms.

The container is a **batch** application: it runs a configured pipeline to
completion, writes its artifacts, and exits. It is not a long-running web
service, so there is no health-check endpoint.

## Obtaining the interface documentation

The container ships its own authoritative interface description. Print it with
the help command — which is also the default action when the container is run
with no pipeline arguments:

```bash
docker run harbor.jatic.net/aria/dataeval:cu128 --help
```

The library form prints the same options with `python -m dataeval_flow --help`.
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
  harbor.jatic.net/aria/dataeval:cu128
```

The data root can be relocated with `DATAEVAL_DATA` / `--data` (see below).

## Secrets

DataEval Flow uses **no API keys, tokens, or passwords**, so **no secret mounts
or secret-management mechanism are required**. Nothing needs to be injected as a
secret to run any workflow.

## Environment variables

All runtime environment variables are optional.

| Variable          | Purpose                                     | Default                                                                   |
| ----------------- | ------------------------------------------- | ------------------------------------------------------------------------- |
| `DATAEVAL_DATA`   | Input data root (datasets, models, configs) | `/dataeval` in the container; current working directory otherwise         |
| `DATAEVAL_OUTPUT` | Output directory for results and reports    | `/output` in the container                                                |
| `DATAEVAL_CACHE`  | Disk-backed computation cache directory     | `/cache` when that mount is present (otherwise caching is in-memory only) |

`HF_HUB_OFFLINE` / `HF_DATASETS_OFFLINE` are standard HuggingFace variables you
may set to force fully offline operation (see [Internet access](#internet-access)).

The following are **build-time only** and are not read at run time:
`DATAEVAL_FLOW_VERSION` (stamps the wheel/image version) and
`DATAEVAL_NOX_UV_EXTRAS_OVERRIDE` (selects extras during the image build).

## Command-line options

| Option                | Purpose                           | Default                                            |
| --------------------- | --------------------------------- | -------------------------------------------------- |
| `-c`, `--config PATH` | Config file or folder             | Auto-discover YAML/JSON at the data root           |
| `-d`, `--data PATH`   | Input data root                   | `$DATAEVAL_DATA`, else the container default / CWD |
| `-o`, `--output PATH` | Output directory for artifacts    | `$DATAEVAL_OUTPUT`, else `/output`                 |
| `-k`, `--cache PATH`  | Disk-backed computation cache     | `$DATAEVAL_CACHE`, else `/cache` if mounted        |
| `-h`, `--help`        | Print the interface help and exit | —                                                  |

Optional sub-commands (default is the headless pipeline):

| Command  | Purpose                                              |
| -------- | ---------------------------------------------------- |
| `app`    | Interactive TUI dashboard (requires the `app` extra) |
| `config` | Simple CLI config builder                            |

## Input precedence

For every input the resolution order is:

1. **Command-line option** (`--config`, `--data`, `--output`, `--cache`)
2. **Environment variable** (`DATAEVAL_DATA`, `DATAEVAL_OUTPUT`, `DATAEVAL_CACHE`)
3. **Built-in default** (the container mount paths above, or the current
   directory outside the container)

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
generated [API Reference](autoapi/dataeval_flow/index) for the full field-level
defaults of every config model.

## Dependencies between configuration parameters

Some configuration fields are only meaningful — or only valid — in combination
with others:

- **Extractor model type drives required fields.** An extractor's `model`
  selector determines which fields are required:
  - `model: onnx` **requires** `model_path`; `output_name` is optional;
    `image_height` and `image_width` **must be set together** (setting only one
    is rejected) and, when both are set, override the model's native input size.
  - `model: torch` **requires** `model_path`; `layer_name`, `use_output`, and
    `device` are optional.
  - `model: uncertainty` **requires** `model_path`; `preds_type`
    (`probs`/`logits`) and `device` are optional.
  - `model: bovw` and `model: flatten` need **no** `model_path`.
- **Preprocessor references must resolve.** An extractor's `preprocessor` field,
  if set, must name a preprocessor defined in the same configuration.
- **GPU execution requires a CUDA image and runtime.** Setting an extractor
  `device: cuda:0` requires running a CUDA image variant (`cu118` / `cu128`)
  with `--gpus all`; on the `cpu` image, models run on CPU regardless.
- **Metadata-dependent analyses.** Bias, parity, and metadata-insight outputs
  require per-sample metadata factors to be present in the dataset; without them
  those analyses are skipped.
- **The `app` sub-command requires the `app` extra** to be installed in the image.

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

## Health checks

DataEval Flow is a batch container, not a long-running service, so it exposes
**no health-check endpoint** (IR-2.3 monitoring requirements are not applicable).
Success or failure is reported through the process exit code and the logs/reports
written to the output directory.
