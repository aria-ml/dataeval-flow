# Build and run containers from source

Build DataEval workflow containers locally from the repository source code.
This is useful for development, testing local changes, or environments without
access to the Harbor registry.

For the full containerized workflow guide — writing configs, mounting data, and
viewing results — see {doc}`containerized_workflows`.

## Prerequisites

- Docker Engine 20.10+ (or Docker Desktop)
- Git
- For GPU variants: NVIDIA Container Toolkit (`nvidia-container-toolkit`)

## Clone and build

```bash
git clone https://gitlab.jatic.net/jatic/aria/dataeval-flow.git
cd dataeval-flow
```

Build the image for your target variant:

```bash
# CPU
docker build -f Dockerfile.cpu --target prod -t dataeval:cpu .

# GPU (pick a CUDA variant)
docker build -f Dockerfile.cu128 --target prod -t dataeval:cu128 .
```

````{tip}
Other CUDA Dockerfiles: `Dockerfile.cu118`, `Dockerfile.cu124`.
````

## Run with `run.sh`

The repository includes a helper script that wraps `docker run` with
convenient flags:

```bash
./run.sh \
    -d "$(pwd)/data" \
    -o "$(pwd)/output" \
    -c config \
    --cpu
```

| Flag | Description |
| --- | --- |
| `-d, --data PATH` | Data directory (required, mounted read-only) |
| `-o, --output PATH` | Output directory (required, mounted read-write) |
| `-c, --config PATH` | Config file or folder relative to data dir (optional — auto-discovers if omitted) |
| `-k, --cache PATH` | Cache directory for embeddings (optional, mounted read-write) |
| `-v, --verbose` | Increase verbosity: `-v` report, `-vv` +INFO, `-vvv` +DEBUG |
| `--cpu` | Use CPU container (default: GPU) |
| `--cuda VERSION` | CUDA variant: `cu118`, `cu124`, `cu128` (default: `cu124`) |

## Run with `docker run`

You can also run the locally built image directly. See the
{ref}`docker run examples <run-the-container>` in the main containerized
workflows guide.

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/output",target=/output \
    dataeval:cpu
```
