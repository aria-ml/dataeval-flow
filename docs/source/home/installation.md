# Installation

```{admonition} Coming Soon
Installation guide is a place-holder.  Official installation notes will be updated.
```

## Using pip

```bash
pip install dataeval-app
```

## Using uv

```bash
uv add dataeval-app
```

## From source

```bash
git clone https://gitlab.jatic.net/jatic/aria/dataeval-app.git
cd dataeval-app
uv sync
```

## Docker

### CUDA 11.8 (GPU)

```bash
docker build -f Dockerfile.cu118 -t dataeval-app:cu118 .
docker run --gpus all dataeval-app:cu118
```

### CPU only

```bash
docker build -f Dockerfile.cpu -t dataeval-app:cpu .
docker run dataeval-app:cpu
```

See the project [README](https://gitlab.jatic.net/jatic/aria/dataeval-app) for full
Docker configuration options and environment variables.
