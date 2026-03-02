#!/bin/bash
# DataEval Application Container Entrypoint
# Validates environment and provides helpful usage info

set -e

# ============== HELP ==============
show_help() {
    if [[ "$CONTAINER_MODE" == "cpu" ]]; then
        cat << 'EOF'
DataEval Application Container - CPU Version

USAGE:
    docker run [OPTIONS] dataeval:cpu [COMMAND]
EOF
    else
        cat << 'EOF'
DataEval Application Container - GPU Version

USAGE:
    docker run --gpus all [OPTIONS] dataeval:gpu [COMMAND]
EOF
    fi

    cat << 'EOF'

================================================================================
VOLUME MOUNTS
================================================================================

REQUIRED:
  /data/config     YAML config files (read-only)
  /data/dataset    Reference dataset (read-only)
  /output          Results and output files (read-write)

OPTIONAL:
  /data/model      Model files (read-only)
  /cache           Computation cache (read-write)

--------------------------------------------------------------------------------
MOUNT SYNTAX
--------------------------------------------------------------------------------

    --mount type=bind,source=HOST_PATH,target=CONTAINER_PATH[,readonly]

    source=    Your local directory path
    target=    Container mount point (use paths above)
    readonly   Optional: prevents container from modifying files

--------------------------------------------------------------------------------
EXAMPLES
--------------------------------------------------------------------------------

Minimal (config + dataset + output):
    docker run --gpus all \
        --mount type=bind,source=/home/user/config,target=/data/config,readonly \
        --mount type=bind,source=/home/user/cifar10,target=/data/dataset,readonly \
        --mount type=bind,source=/home/user/results,target=/output \
        dataeval:gpu

All mounts (with model):
    docker run --gpus all \
        --mount type=bind,source=/home/user/config,target=/data/config,readonly \
        --mount type=bind,source=/home/user/cifar10,target=/data/dataset,readonly \
        --mount type=bind,source=/home/user/models,target=/data/model,readonly \
        --mount type=bind,source=/home/user/results,target=/output \
        --mount type=bind,source=/home/user/cache,target=/cache \
        dataeval:gpu

Windows PowerShell:
    docker run --gpus all `
        --mount type=bind,source=C:\data\config,target=/data/config,readonly `
        --mount type=bind,source=C:\data\cifar10,target=/data/dataset,readonly `
        --mount type=bind,source=C:\output,target=/output `
        dataeval:gpu

--------------------------------------------------------------------------------
OTHER OPTIONS
--------------------------------------------------------------------------------

Interactive shell:
    docker run -it --gpus all \
        --mount type=bind,source=/home/user/config,target=/data/config,readonly \
        --mount type=bind,source=/home/user/cifar10,target=/data/dataset,readonly \
        --entrypoint /bin/bash \
        dataeval:gpu

Debug (bypass GPU check):
    docker run \
        --mount type=bind,source=/home/user/config,target=/data/config,readonly \
        --mount type=bind,source=/home/user/cifar10,target=/data/dataset,readonly \
        --entrypoint python \
        dataeval:gpu src/container_run.py

--------------------------------------------------------------------------------
CPU-ONLY MACHINES
--------------------------------------------------------------------------------

    docker run \
        --mount type=bind,source=/home/user/config,target=/data/config,readonly \
        --mount type=bind,source=/home/user/cifar10,target=/data/dataset,readonly \
        --mount type=bind,source=/home/user/results,target=/output \
        dataeval:cpu

--------------------------------------------------------------------------------
TROUBLESHOOTING
--------------------------------------------------------------------------------

"No GPU detected"      -> Add --gpus all to command
"No config mounted"    -> Add --mount for /data/config
"No dataset mounted"   -> Add --mount for /data/dataset
"Output not mounted"   -> Add --mount for /output
"invalid mount config" -> Check source path exists on host
"Permission denied"    -> Check host directory permissions

================================================================================
EOF
    exit 0
}

# ============== PARSE ARGS ==============
if [[ "$1" == "--help" || "$1" == "-h" || "$1" == "-help" ]]; then
    show_help
fi

# ============== VALIDATE DATASET MOUNT ==============
# Marker file exists = no mount attempted = show help
if [[ -f "/data/dataset/.not_mounted" ]]; then
    show_help
fi

# No marker but empty = mount attempted with bad path = show error
if [[ -z "$(ls -A /data/dataset 2>/dev/null)" ]]; then
    echo ""
    echo "ERROR: Dataset mount is empty at /data/dataset"
    echo ""
    echo "Check that your source path exists on the host:"
    echo "  -v /path/to/dataset:/data/dataset:ro"
    echo "       ↑ verify this path exists"
    echo ""
    echo "Run with --help for usage information."
    exit 1
fi

# ============== VALIDATE CONFIG MOUNT (REQUIRED) ==============
# Marker file exists = no mount attempted = show help
if [[ -f "/data/config/.not_mounted" ]]; then
    show_help
fi

# No marker but empty = mount attempted with bad path = show error
if [[ -z "$(ls -A /data/config 2>/dev/null)" ]]; then
    echo ""
    echo "ERROR: Config mount is empty at /data/config"
    echo ""
    echo "Check that your source path exists on the host:"
    echo "  -v /path/to/config:/data/config:ro"
    echo "       ↑ verify this path exists"
    echo ""
    echo "Run with --help for usage information."
    exit 1
fi

# ============== VALIDATE OUTPUT MOUNT (REQUIRED) ==============
# Marker file exists = no mount attempted = show error
if [[ -f "/output/.not_mounted" ]]; then
    echo ""
    echo "ERROR: Output directory not mounted at /output"
    echo ""
    echo "Mount an output directory:"
    echo "  --mount type=bind,source=/path/to/output,target=/output"
    echo ""
    echo "Run with --help for usage information."
    exit 1
fi

# ============== VALIDATE OPTIONAL MOUNTS ==============
# Check if mount was attempted (no marker) but directory is empty (bad path)

# /data/model - Model files (optional)
if [[ ! -f "/data/model/.not_mounted" ]] && [[ -z "$(ls -A /data/model 2>/dev/null)" ]]; then
    echo ""
    echo "ERROR: Model mount is empty at /data/model"
    echo ""
    echo "Check that your source path exists on the host:"
    echo "  -v /path/to/models:/data/model:ro"
    echo "       ↑ verify this path exists"
    exit 1
fi

# /output - Results (required, check writability)
if [[ ! -w "/output" ]]; then
    echo ""
    echo "ERROR: Output mount at /output is not writable"
    echo ""
    echo "Check directory permissions on host."
    exit 1
fi

# /cache - Computation cache (optional, check writability if mounted)
if [[ -d "/cache" ]] && [[ ! -w "/cache" ]]; then
    echo ""
    echo "ERROR: Cache mount at /cache is not writable"
    echo ""
    echo "Check directory permissions on host."
    exit 1
fi

# ============== VALIDATE GPU (skip for CPU mode) ==============
if [[ "$CONTAINER_MODE" == "cpu" ]]; then
    echo "Running in CPU mode (GPU check skipped)"
else
    echo "Checking GPU access..."

    if ! command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "ERROR: nvidia-smi not found."
        echo ""
        echo "Did you forget --gpus all?"
        echo ""
        echo "    docker run --gpus all \\"
        echo "        --mount type=bind,source=/path/to/config,target=/data/config,readonly \\"
        echo "        --mount type=bind,source=/path/to/dataset,target=/data/dataset,readonly \\"
        echo "        dataeval:gpu"
        echo ""
        echo "For CPU-only machines, use: dataeval:cpu"
        echo "Run with --help for full usage."
        exit 1
    fi

    if ! nvidia-smi &> /dev/null; then
        echo ""
        echo "ERROR: GPU not accessible."
        echo ""
        echo "Ensure nvidia-container-toolkit is installed and run with --gpus all"
        echo ""
        echo "    docker run --gpus all \\"
        echo "        --mount type=bind,source=/path/to/config,target=/data/config,readonly \\"
        echo "        --mount type=bind,source=/path/to/dataset,target=/data/dataset,readonly \\"
        echo "        dataeval:gpu"
        echo ""
        echo "Run with --help for full usage."
        exit 1
    fi
fi

# ============== SUCCESS ==============
echo ""
DATASET_COUNT=$(ls /data/dataset 2>/dev/null | wc -l)
if [[ "$CONTAINER_MODE" == "cpu" ]]; then
    echo "Mode: CPU"
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "GPU detected: $GPU_NAME"
fi
echo "Dataset mounted: $DATASET_COUNT items in /data/dataset"
echo ""

exec "$@"
