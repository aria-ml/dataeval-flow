#!/bin/bash
# DataEval Application Runner
# Usage: ./run.sh -f PATH -d PATH -o PATH [-m PATH] [-k PATH] [--cpu]

set -e

# Initialize variables
CONFIG_PATH=""
DATASET_PATH=""
OUTPUT_PATH=""
MODEL_PATH=""
CACHE_PATH=""
USE_CPU=false
SHOW_HELP=false

# Parse arguments (Linux convention: -s for short, --long for long)
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -k|--cache)
            CACHE_PATH="$2"
            shift 2
            ;;
        -c|--cpu)
            USE_CPU=true
            shift
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
if [[ "$SHOW_HELP" == true ]] || [[ -z "$DATASET_PATH" ]] || [[ -z "$CONFIG_PATH" ]] || [[ -z "$OUTPUT_PATH" ]]; then
    echo "DataEval Application Runner"
    echo ""
    echo "Usage:"
    echo "  ./run.sh -f PATH -d PATH -o PATH [-m PATH] [-k PATH] [--cpu]"
    echo ""
    echo "Options:"
    echo "  -f, --config PATH    Path to config folder (required)"
    echo "  -d, --dataset PATH   Path to dataset (required)"
    echo "  -o, --output PATH    Path for output files (required)"
    echo "  -m, --model PATH     Path to model files (optional)"
    echo "  -k, --cache PATH     Path for computation cache (optional)"
    echo "  -c, --cpu            Use CPU container (default: GPU)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh -f /mnt/c/data/config -d /mnt/c/data/cifar10_test -o /mnt/c/data/output"
    echo "  ./run.sh -f /mnt/c/data/config -d /mnt/c/data/cifar10_test -o /mnt/c/data/output -m /mnt/c/data/model -k /mnt/c/data/cache --cpu"
    echo ""
    exit 0
fi

# Build mount arguments as array (handles paths with spaces)
MOUNTS=(
    -v "$CONFIG_PATH:/data/config:ro"
    -v "$DATASET_PATH:/data/dataset:ro"
    -v "$OUTPUT_PATH:/output"
)

if [[ -n "$MODEL_PATH" ]]; then
    MOUNTS+=(-v "$MODEL_PATH:/data/model:ro")
fi

if [[ -n "$CACHE_PATH" ]]; then
    MOUNTS+=(-v "$CACHE_PATH:/cache")
fi

# Select image and GPU flag
if [[ "$USE_CPU" == true ]]; then
    echo "Running with CPU..."
    docker run --rm "${MOUNTS[@]}" dataeval-app:cpu
else
    echo "Running with GPU..."
    docker run --rm --gpus all "${MOUNTS[@]}" dataeval-app:gpu
fi
