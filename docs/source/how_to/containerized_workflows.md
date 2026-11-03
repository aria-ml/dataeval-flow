# Run workflows in containers

To run a DataEval Flow workflow without installing the library, pull a pre-built image, write a config file, and
launch the container with bind-mounted data. Every workflow can run from a container. The container is a batch
application: it runs the configured pipeline to completion, writes its artifacts, and exits. The
[Container Reference](../reference/containers.md) lists every input, variable, mount and command-line option, and
[Execution Modes: Library, Command and Service](../concepts/ExecutionModes.md) explains when to run it as a command and
when as a service.

You need Docker Engine 20.10+ (or Docker Desktop) and your dataset saved to disk in a supported format (see
{doc}`write_a_pipeline_config`). For GPU variants, you also need the NVIDIA Container Toolkit
(`nvidia-container-toolkit`).

## Pull the image

Pre-built images are published to the JATIC Harbor registry:

```bash
# CPU-only
docker pull harbor.jatic.net/aria/dataeval-flow:latest-cpu

# GPU (CUDA 13.0 — recommended for modern GPUs)
docker pull harbor.jatic.net/aria/dataeval-flow:latest-cu130
```

Available variants:

| Tag | Base | Use case |
| --- | --- | --- |
| `cpu` | Ubuntu 24.04 | Machines without NVIDIA GPU |
| `cu126` | Ubuntu 24.04 | Older GPUs / CUDA 12.6 drivers |
| `cu130` | Ubuntu 24.04 | Modern GPUs (RTX 50 series) / CUDA 13.0 drivers |

All GPU variants bundle their own CUDA runtime libraries via PyTorch, so the host only needs the NVIDIA driver and
Container Toolkit.

## Prepare the host directories

Create the directory layout the container expects:

```bash
mkdir -p workspace/{config,output,cache}
```

The container reads and writes three mounts:

| Mount point | Required | Mode | Purpose |
| --- | --- | --- | --- |
| `/dataeval` | Yes | read-only | Data root: datasets, models, and config files |
| `/output` | Yes | read-write | Reports and results |
| `/cache` | No | read-write | Embedding and stats cache (speeds up re-runs) |

By default the container looks for config files inside the data mount (`/dataeval`). You can also mount a config
directory independently. See the "Select a config file" section for both approaches.

### Make the directories writable

The container runs as a non-root user (`dataeval`). Mounted directories for `/output` and `/cache` must be writable by
the container process. You have two options.

#### Pass your host user ID (recommended)

Use `--user` to run the container as your host user, so mounted directories are naturally writable:

```bash
docker run --user "$(id -u):$(id -g)" ...
```

#### Open the directory permissions

Make the output and cache directories world-writable on the host:

```bash
chmod 777 workspace/output workspace/cache
```

## Write the config file

Create `params.yaml` in your config directory (for example `workspace/config/params.yaml`) or inside your data
directory. {doc}`write_a_pipeline_config` shows each section and a complete example for data cleaning.

(run-the-container)=

## Run the container

### Run on the CPU

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    --mount type=bind,source="$(pwd)/workspace/cache",target=/cache \
    harbor.jatic.net/aria/dataeval-flow:latest-cpu
```

### Run on a GPU

Add `--gpus all` and use a CUDA variant:

```bash
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    --mount type=bind,source="$(pwd)/workspace/cache",target=/cache \
    harbor.jatic.net/aria/dataeval-flow:latest-cu130
```

The image and the GPUs it sees decide the device, not the config. DataEval Flow computes on a GPU where PyTorch sees
one, and on the CPU otherwise. `--gpus device=1` runs on the second GPU, and `-e CUDA_VISIBLE_DEVICES=` keeps a CUDA
image on the CPU. Each result's `metadata.device` records the device its task ran on.

### Select a config file

Point at a specific config file or folder within your data directory:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval-flow:latest-cpu \
    python -m dataeval_flow --config config/params.yaml
```

You can also mount a config directory independently from your data. Use a separate bind mount and pass the
container-side path with `--config`:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/config",target=/config,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval-flow:latest-cpu \
    python -m dataeval_flow --config /config/params.yaml
```

### Increase the verbosity

Pass `-v` flags to increase output detail:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval-flow:latest-cpu \
    python -m dataeval_flow -v
```

| Flag | Level |
| --- | --- |
| `-v` | Show full report output |
| `-vv` | Report + INFO logging |
| `-vvv` | Report + DEBUG logging |

### Run a subset of the tasks

`--task` runs one task by name. Repeat it to run several, in the order given. A named task runs whether its config
entry sets `enabled: false`, so you can keep a task defined but dormant.

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval-flow:latest-cpu \
    python -m dataeval_flow --task clean_my_data
```

With no `--task`, every task the config marks `enabled` runs.

### Discover the available workflows

The image ships without the TUI extra, so `workflows` is how you ask it what it can run and what a given workflow type
accepts:

```bash
docker run --rm harbor.jatic.net/aria/dataeval-flow:latest-cpu \
    python -m dataeval_flow workflows

docker run --rm harbor.jatic.net/aria/dataeval-flow:latest-cpu \
    python -m dataeval_flow workflows quality
```

`python -m dataeval_flow --version` reports the build inside the image.

## Find the results

Results are written to the output mount as one merged file per format, plus a run log:

```bash
find workspace/output -type f
# workspace/output/result.log
# workspace/output/results/result.json
# workspace/output/results/result.txt
# workspace/output/results/result.html
```

| File | Holds |
| --- | --- |
| `result.json` | Each task's result envelope, keyed by task name. It is the same data `result.to_dict()` returns in the Python API. |
| `result.txt` | The detailed text reports, the same as `result.report()`. |
| `result.html` | The same reports as one self-contained page, the same as `result.to_html()`. It opens offline in any browser, and prints or saves to PDF as it shows. |
| `encoding.json` | The metadata encoding descriptor the run was computed under, ready to review and commit. |
| `manifests/` | Each `content-digest` step's manifest, one hash per item, under `manifests/<task>/`. |

A run writes `encoding.json` only where a task records an encoding, as `audit`, `bias` and `shift` do, so the `quality`
run above writes none. It also omits the file when a run's tasks encoded their factors differently, since no single
descriptor describes it. {doc}`configure_metadata_binning` shows how to review the encoding. A run writes `manifests/`
only where a task digests a split, as `audit` does. See
[Gate training on an audit](gate_training_on_an_audit.md#find-what-changed).

{doc}`read_the_result_envelope` describes the sections of each task's entry in `result.json`. `health` is the roll-up
that `--fail-on-warning` gates on, and a pipeline can read it directly:

```bash
jq -r 'to_entries[] | "\(.key)\t\(.value.health.status)\t\(.value.health.warnings)"' \
    workspace/output/results/result.json
```

List the task names, or the findings of one task:

```bash
jq -r 'keys[]' workspace/output/results/result.json
jq -r '.clean_my_data.findings[] | "\(.severity)\t\(.title)"' \
    workspace/output/results/result.json
```

## Choose the result files

A pipeline's `result:` block shapes these files. Every key is optional:

```yaml
result:
  name: release             # release.json, release.txt, release.html (default: result)
  formats: [json, html]     # json, text, html, junit (.xml), markdown (.md) (default: json, text, html)
  detail: summary           # the text and HTML files' detail: full or summary (default: full)
  per_task: true            # one set of files per task: release-<task>.json, … (default: false)
  fail_on: warning          # what fails the job: failure, warning or never (default: failure)
  require: ready-with-accepted-risks  # the worst verdict that passes, else exit 4 (default: none)
  width: 100                # the text report's width, at least 40 (default: 80)
  max_images: 100           # thumbnails per task's result; 0: none, -1: every item named (default: 200)
  max_rows: 1000            # rows a table of items lists; -1: every row (default: 500)
  preview_rows: 20          # rows of it the text report shows; -1: every row (default: 10)
```

`--report-width` and `DATAEVAL_REPORT_WIDTH` override `width`. The console keeps printing the summary, or the full
report with `-v`, whatever `detail` says.

## Troubleshoot a run

Run the container with `--help` to see full usage:

```bash
docker run harbor.jatic.net/aria/dataeval-flow:latest-cpu python -m dataeval_flow --help
```

| Message | Fix |
| --- | --- |
| "Data directory not found or not mounted" | Verify that the `--mount source=` path exists on the host. |
| "No tasks defined in config" | Ensure `params.yaml` has a `tasks` list. |
| "No GPU detected" | Add `--gpus all` to the `docker run` command, or use the `:cpu` image. |
| "Output mount not writable" | Pass `--user "$(id -u):$(id -g)"` or `chmod 777` the host directory. |
| "Permission denied" | Check host directory permissions with `ls -la`. |

## Next steps

- [Write a pipeline config](write_a_pipeline_config.md) — Declare the datasets, sources, extractors, workflows and
  tasks the container runs.
- [Fail a CI job on warnings](fail_a_ci_job_on_warnings.md) — Turn health warnings into a failing exit code.
- [Run Flow as a service](run_flow_as_a_service.md) — Keep the container running behind an HTTP API.
