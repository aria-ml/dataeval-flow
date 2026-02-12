#!/bin/bash
# DataEval Application Runner
# Usage: ./run.sh -f PATH -d PATH [-o PATH] [--cpu]

set -e

# Initialize variables
CONFIG_PATH=""
DATASET_PATH=""
OUTPUT_PATH=""
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
if [[ "$SHOW_HELP" == true ]] || [[ -z "$DATASET_PATH" ]] || [[ -z "$CONFIG_PATH" ]]; then
    echo "DataEval Application Runner"
    echo ""
    echo "Usage:"
    echo "  ./run.sh -f PATH -d PATH [-o PATH] [--cpu]"
    echo ""
    echo "Options:"
    echo "  -f, --config PATH    Path to config folder (required)"
    echo "  -d, --dataset PATH   Path to dataset (required)"
    echo "  -o, --output PATH    Path for output files (optional)"
    echo "  -c, --cpu            Use CPU container (default: GPU)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh --config /mnt/c/data/config --dataset /mnt/c/data/cifar10_test"
    echo "  ./run.sh -f /mnt/c/data/config -d /mnt/c/data/cifar10_test -o /mnt/c/data/output"
    echo "  ./run.sh -f /mnt/c/data/config -d /mnt/c/data/cifar10_test --cpu"
    echo ""
    exit 0
fi

# Build mount arguments (config and dataset are required)
MOUNTS="--mount type=bind,source=$CONFIG_PATH,target=/data/config,readonly"
MOUNTS="$MOUNTS --mount type=bind,source=$DATASET_PATH,target=/data/dataset,readonly"

if [[ -n "$OUTPUT_PATH" ]]; then
    MOUNTS="$MOUNTS --mount type=bind,source=$OUTPUT_PATH,target=/output"
fi

# Select image and GPU flag
if [[ "$USE_CPU" == true ]]; then
    echo "Running with CPU..."
    docker run $MOUNTS dataeval:cpu
else
    echo "Running with GPU..."
    docker run --gpus all $MOUNTS dataeval:gpu
fi
