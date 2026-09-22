#!/bin/bash
# DataEval Workflows Container Entrypoint
# Validates environment and provides helpful usage info

set -e

# Resolve data root (matches Python-side DATAEVAL_DATA env var)
DATA_DIR="${DATAEVAL_DATA:-/dataeval}"

# Resolve output and cache roots (matches DATAEVAL_OUTPUT and DATAEVAL_CACHE env vars)
OUTPUT_DIR="${DATAEVAL_OUTPUT:-/output}"
CACHE_DIR="${DATAEVAL_CACHE:-/cache}"

# Image tag for help text (e.g. dataeval-flow:cpu, dataeval-flow:cu130)
IMAGE_TAG="dataeval-flow:${UV_EXTRAS_OVERRIDE:-cpu}"

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
  $OUTPUT_DIR        Results and output files (read-write)
  $CACHE_DIR         Computation cache (read-write)

  No secret mounts are required: this tool uses no API keys, tokens, or
  passwords. None are read from the environment or baked into the image.

================================================================================
ENVIRONMENT VARIABLES
================================================================================

  DATAEVAL_DATA      Input data root inside the container (default: $DATA_DIR).
                     Datasets, models, and configs resolve relative to it.
  DATAEVAL_OUTPUT    Output directory for results/reports (default: $OUTPUT_DIR).
  DATAEVAL_CACHE     Computation cache directory (default: $CACHE_DIR when that
                     volume is mounted and writable).
  DATAEVAL_CONFIG    Config file or folder (default: auto-discover at the data root).
  DATAEVAL_VERBOSITY Verbosity 0-3, as -v/-vv/-vvv (default: 0).
  DATAEVAL_TASKS     Comma-separated task names to run (default: every enabled task).
  DATAEVAL_FAIL_ON_WARNING
                     Exit non-zero on health warnings: true/false (default: false).
  DATAEVAL_LOG_FORMAT
                     Console format: structured or plain (default: structured).
                     'structured' prefixes each record with an ISO-8601 UTC
                     timestamp and level.

  All are optional. Command-line options below take precedence over them.

================================================================================
COMMAND-LINE OPTIONS
================================================================================

  Usage: docker run [MOUNTS] $IMAGE_TAG python -m dataeval_flow [OPTIONS] [COMMAND]

  -c, --config PATH   Config file or folder (default: auto-discover YAML/JSON
                      at the data root).
  -d, --data PATH     Input data root (default: \$DATAEVAL_DATA or CWD).
  -o, --output PATH   Output directory for artifacts (default: \$DATAEVAL_OUTPUT).
  -k, --cache PATH    Disk-backed computation cache (default: \$DATAEVAL_CACHE).
  -t, --task NAME     Run only this task. Repeat to run several, in order.
      --log-format F  Console format: structured (default) or plain.
      --fail-on-warning / --no-fail-on-warning
                      Exit non-zero when a task reports health warnings.
  -v, --verbose       Increase verbosity (-v report, -vv +INFO, -vvv +DEBUG).
  -h, --help          Show this help and exit.

  COMMANDS (optional; default runs the headless pipeline):
    app               Launch the interactive TUI (requires the 'app' extra).
    config            Create or edit pipeline config files.
    encoding          Write the metadata encoding descriptor a result was
                      computed under, ready to review and commit. Takes a
                      result.json path, with -o PATH and --task NAME.

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
        --mount type=bind,source=/home/user/results,target=$OUTPUT_DIR \\
        $IMAGE_TAG

With cache:
    docker run${GPU_FLAG} \\
        --mount type=bind,source=/home/user/myproject,target=$DATA_DIR,readonly \\
        --mount type=bind,source=/home/user/results,target=$OUTPUT_DIR \\
        --mount type=bind,source=/home/user/mycache,target=$CACHE_DIR \\
        $IMAGE_TAG

With config path override (config in a subdirectory):
    docker run${GPU_FLAG} \\
        --mount type=bind,source=/home/user/myproject,target=$DATA_DIR,readonly \\
        --mount type=bind,source=/home/user/results,target=$OUTPUT_DIR \\
        $IMAGE_TAG python -m dataeval_flow --config config/

Verbose output (-v report, -vv +INFO, -vvv +DEBUG):
    docker run${GPU_FLAG} \\
        --mount type=bind,source=/home/user/myproject,target=$DATA_DIR,readonly \\
        --mount type=bind,source=/home/user/results,target=$OUTPUT_DIR \\
        $IMAGE_TAG python -m dataeval_flow -v

Windows PowerShell:
    docker run${GPU_FLAG} \`
        --mount type=bind,source=C:\\data\\myproject,target=$DATA_DIR,readonly \`
        --mount type=bind,source=C:\\output,target=$OUTPUT_DIR \`
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
        --mount type=bind,source=/home/user/results,target=$OUTPUT_DIR \\
        $IMAGE_TAG

--------------------------------------------------------------------------------
TROUBLESHOOTING
--------------------------------------------------------------------------------

"No GPU detected"      -> Add --gpus all to command
"No data mounted"      -> Add --mount for $DATA_DIR
"Output not mounted"   -> Add --mount for $OUTPUT_DIR
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
# Validate output directory is mounted
if [[ -f "$OUTPUT_DIR/.not_mounted" ]] || [[ ! -d "$OUTPUT_DIR" ]]; then
    echo ""
    echo "ERROR: Output directory not mounted at $OUTPUT_DIR"
    echo ""
    echo "Mount an output directory:"
    echo "  --mount type=bind,source=/path/to/results,target=$OUTPUT_DIR"
    echo ""
    echo "Run with --help for usage information."
    exit 1
fi

if [[ ! -w "$OUTPUT_DIR" ]]; then
    echo ""
    echo "ERROR: Output mount at $OUTPUT_DIR is not writable"
    echo ""
    echo "Check directory permissions on host."
    exit 1
fi

if [[ -d "$CACHE_DIR" ]] && [[ ! -w "$CACHE_DIR" ]]; then
    echo ""
    echo "ERROR: Cache mount at $CACHE_DIR is not writable"
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
        echo "        --mount type=bind,source=/path/to/results,target=$OUTPUT_DIR \\"
        echo "        $IMAGE_TAG"
        echo ""
        echo "For CPU-only machines, use: dataeval-flow:cpu"
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
        echo "        --mount type=bind,source=/path/to/results,target=$OUTPUT_DIR \\"
        echo "        $IMAGE_TAG"
        echo ""
        echo "Run with --help for full usage."
        exit 1
    fi
fi

# ============== AUTO-DETECT CACHE ==============
# Set DATAEVAL_CACHE if cache mount is present and writable
if [[ -z "${DATAEVAL_CACHE:-}" ]] && [[ -d "$CACHE_DIR" ]] && [[ -w "$CACHE_DIR" ]] \
   && [[ ! -f "$CACHE_DIR/.not_mounted" ]]; then
    export DATAEVAL_CACHE="$CACHE_DIR"
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
