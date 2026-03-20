#!/bin/bash
# DataEval Workflows Runner
# Usage: ./run.sh -d PATH -o PATH [-c PATH] [-k PATH] [-v] [--cpu] [--cuda VERSION]

set -e

# Initialize variables
DATA_PATH=""
OUTPUT_PATH=""
CONFIG_ARG=""
CACHE_PATH=""
VERBOSE_FLAGS=""
USE_CPU=false
CUDA_VERSION="cu124"
SHOW_HELP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data)
            DATA_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_ARG="$2"
            shift 2
            ;;
        -k|--cache)
            CACHE_PATH="$2"
            shift 2
            ;;
        -v|--verbose)
            # Stack -v flags (supports -v, -vv, -vvv or repeated -v -v)
            if [[ "$1" =~ ^-v+$ ]]; then
                VERBOSE_FLAGS="${VERBOSE_FLAGS}${1}"
            else
                VERBOSE_FLAGS="${VERBOSE_FLAGS}-v"
            fi
            shift
            ;;
        --cpu)
            USE_CPU=true
            shift
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested or missing required args
if [[ "$SHOW_HELP" == true ]] || [[ -z "$DATA_PATH" ]] || [[ -z "$OUTPUT_PATH" ]]; then
    echo "DataEval Workflows Runner"
    echo ""
    echo "Usage:"
    echo "  ./run.sh -d PATH -o PATH [-c PATH] [-k PATH] [-v] [--cpu] [--cuda VERSION]"
    echo ""
    echo "Options:"
    echo "  -d, --data PATH      Path to data directory (required, mounted read-only)"
    echo "  -o, --output PATH    Path for output files (required, mounted read-write)"
    echo "  -c, --config PATH    Config file or folder relative to data dir (optional)"
    echo "  -k, --cache PATH     Path for computation cache (optional, mounted read-write)"
    echo "  -v, --verbose        Increase verbosity (-v report, -vv +INFO, -vvv +DEBUG)"
    echo "      --cpu            Use CPU container (default: GPU)"
    echo "      --cuda VERSION   CUDA variant: cu118, cu124, cu128 (default: cu124)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh -d /mnt/c/data/myproject -o /mnt/c/data/output"
    echo "  ./run.sh -d /mnt/c/data/myproject -o /mnt/c/data/output -c config/"
    echo "  ./run.sh -d /mnt/c/data/myproject -o /mnt/c/data/output --cuda cu118"
    echo "  ./run.sh -d /mnt/c/data/myproject -o /mnt/c/data/output -k /mnt/c/data/cache -v --cpu"
    echo ""
    exit 0
fi

# Build mount arguments as array (handles paths with spaces)
MOUNTS=(
    -v "$DATA_PATH:/dataeval:ro"
    -v "$OUTPUT_PATH:/output"
)

if [[ -n "$CACHE_PATH" ]]; then
    MOUNTS+=(-v "$CACHE_PATH:/cache")
fi

# Build container command — forward flags to container_run.py
CMD=("python" "src/container_run.py")
if [[ -n "$CONFIG_ARG" ]]; then
    CMD+=("--config" "$CONFIG_ARG")
fi
if [[ -n "$VERBOSE_FLAGS" ]]; then
    CMD+=("$VERBOSE_FLAGS")
fi

# Select image and GPU flag
if [[ "$USE_CPU" == true ]]; then
    echo "Running with CPU..."
    docker run --rm "${MOUNTS[@]}" "dataeval:cpu" "${CMD[@]}"
else
    echo "Running with GPU (dataeval:${CUDA_VERSION})..."
    docker run --rm --gpus all "${MOUNTS[@]}" "dataeval:${CUDA_VERSION}" "${CMD[@]}"
fi
