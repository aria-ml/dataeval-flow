# Installation

```{admonition} Coming Soon
Installation guide is a place-holder.  Official installation notes will be updated.
```

## Using pip

```bash
pip install dataeval-flow
```

## Using uv

```bash
uv add dataeval-flow
```

## From source

```bash
git clone https://gitlab.jatic.net/jatic/aria/dataeval-flow.git
cd dataeval-flow
uv sync
```

## Docker

### CUDA 11.8 (GPU)

```bash
docker build -f Dockerfile.cu118 -t dataeval-flow:cu118 .
docker run --gpus all dataeval-flow:cu118
```

### CPU only

```bash
docker build -f Dockerfile.cpu -t dataeval-flow:cpu .
docker run dataeval-flow:cpu
```

See the project [README](https://gitlab.jatic.net/jatic/aria/dataeval-flow) for full
Docker configuration options and environment variables.
