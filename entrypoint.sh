#!/bin/bash
# DataEval Workflows Container Entrypoint
# Validates environment and provides helpful usage info

set -e

# Resolve data root (matches Python-side DATAEVAL_DATA env var)
DATA_DIR="${DATAEVAL_DATA:-/dataeval}"

# Image tag for help text (e.g. dataeval:cpu, dataeval:cu124)
IMAGE_TAG="dataeval:${UV_EXTRAS_OVERRIDE:-cpu}"

# GPU flag for docker run examples
if [[ "$CONTAINER_MODE" == "cpu" ]]; then
    GPU_FLAG=""
else
    GPU_FLAG=" --gpus all"
fi

# ============== HELP ==============
show_help() {
    if [[ "$CONTAINER_MODE" == "cpu" ]]; then
        cat << EOF
DataEval Workflows Container - CPU Version

USAGE:
    docker run [OPTIONS] $IMAGE_TAG [COMMAND]
EOF
    else
        cat << EOF
DataEval Workflows Container - GPU Version ($UV_EXTRAS_OVERRIDE)

USAGE:
    docker run --gpus all [OPTIONS] $IMAGE_TAG [COMMAND]
EOF
    fi

    # Double backslashes (\\) produce literal backslashes in the output for
    # displaying line-continuation examples to the user.
    cat << _EOF_

================================================================================
VOLUME MOUNTS
================================================================================

REQUIRED:
  $DATA_DIR          Data directory — datasets, models, configs (read-only)

OPTIONAL:
  /output            Results and output files (read-write)
  /cache             Computation cache (read-write)

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

Minimal (data + output):
    docker run${GPU_FLAG} \\
        --mount type=bind,source=/home/user/myproject,target=$DATA_DIR,readonly \\
        --mount type=bind,source=/home/user/results,target=/output \\
        $IMAGE_TAG

With cache:
    docker run${GPU_FLAG} \\
        --mount type=bind,source=/home/user/myproject,target=$DATA_DIR,readonly \\
        --mount type=bind,source=/home/user/results,target=/output \\
        --mount type=bind,source=/home/user/cache,target=/cache \\
        $IMAGE_TAG

With config path override (config in a subdirectory):
    docker run${GPU_FLAG} \\
        --mount type=bind,source=/home/user/myproject,target=$DATA_DIR,readonly \\
        --mount type=bind,source=/home/user/results,target=/output \\
        $IMAGE_TAG python -m dataeval_flow --config config/

Verbose output (-v report, -vv +INFO, -vvv +DEBUG):
    docker run${GPU_FLAG} \\
        --mount type=bind,source=/home/user/myproject,target=$DATA_DIR,readonly \\
        --mount type=bind,source=/home/user/results,target=/output \\
        $IMAGE_TAG python -m dataeval_flow -v

Windows PowerShell:
    docker run${GPU_FLAG} \`
        --mount type=bind,source=C:\\data\\myproject,target=$DATA_DIR,readonly \`
        --mount type=bind,source=C:\\output,target=/output \`
        $IMAGE_TAG

--------------------------------------------------------------------------------
OTHER OPTIONS
--------------------------------------------------------------------------------

Interactive shell:
    docker run -it${GPU_FLAG} \\
        --mount type=bind,source=/home/user/myproject,target=$DATA_DIR,readonly \\
        --entrypoint /bin/bash \\
        $IMAGE_TAG

Custom data root (override DATAEVAL_DATA):
    docker run${GPU_FLAG} \\
        -e DATAEVAL_DATA=/data \\
        --mount type=bind,source=/home/user/myproject,target=/data,readonly \\
        --mount type=bind,source=/home/user/results,target=/output \\
        $IMAGE_TAG

--------------------------------------------------------------------------------
TROUBLESHOOTING
--------------------------------------------------------------------------------

"No GPU detected"      -> Add --gpus all to command
"No data mounted"      -> Add --mount for $DATA_DIR
"Output not mounted"   -> Add --mount for /output
"invalid mount config" -> Check source path exists on host
"Permission denied"    -> Check host directory permissions

================================================================================
_EOF_
    exit 0
}

# ============== PARSE ARGS ==============
if [[ "$1" == "--help" || "$1" == "-h" || "$1" == "-help" ]]; then
    show_help
fi

# ============== VALIDATE DATA MOUNT ==============
# Marker file exists = no mount attempted = show help
if [[ -f "$DATA_DIR/.not_mounted" ]]; then
    show_help
fi

# No marker but empty = mount attempted with bad path = show error
if [[ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]]; then
    echo ""
    echo "ERROR: Data mount is empty at $DATA_DIR"
    echo ""
    echo "Check that your source path exists on the host:"
    echo "  --mount type=bind,source=/path/to/data,target=$DATA_DIR,readonly"
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
        echo "        --mount type=bind,source=/path/to/data,target=$DATA_DIR,readonly \\"
        echo "        --mount type=bind,source=/path/to/output,target=/output \\"
        echo "        $IMAGE_TAG"
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
        echo "        --mount type=bind,source=/path/to/data,target=$DATA_DIR,readonly \\"
        echo "        --mount type=bind,source=/path/to/output,target=/output \\"
        echo "        $IMAGE_TAG"
        echo ""
        echo "Run with --help for full usage."
        exit 1
    fi
fi

# ============== AUTO-DETECT CACHE ==============
# Set DATAEVAL_CACHE only if /cache is mounted and writable (not a marker stub)
if [[ -z "${DATAEVAL_CACHE:-}" ]] && [[ -d "/cache" ]] && [[ -w "/cache" ]] && [[ ! -f "/cache/.not_mounted" ]]; then
    export DATAEVAL_CACHE="/cache"
fi

# ============== SUCCESS ==============
echo ""
DATA_COUNT=$(ls "$DATA_DIR" 2>/dev/null | wc -l)
if [[ "$CONTAINER_MODE" == "cpu" ]]; then
    echo "Mode: CPU"
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "GPU detected: $GPU_NAME"
fi
echo "Data mounted: $DATA_COUNT items in $DATA_DIR"
echo ""

exec "$@"
