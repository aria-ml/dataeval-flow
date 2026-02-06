# Installation

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

### GPU (CUDA)

```bash
docker compose up
```

### CPU only

```bash
docker build -f Dockerfile.cpu -t dataeval-app:cpu .
docker run dataeval-app:cpu
```

See the project [README](https://gitlab.jatic.net/jatic/aria/dataeval-app) for full
Docker configuration options and environment variables.
