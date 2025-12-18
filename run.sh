#!/bin/bash
# DataEval Application Runner
# Usage: ./run.sh -d PATH [-o PATH] [-s NAME] [--cpu]

set -e

# Initialize variables
DATASET_PATH=""
OUTPUT_PATH=""
SPLIT=""
USE_CPU=false
SHOW_HELP=false

# Parse arguments (Linux convention: -s for short, --long for long)
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -s|--split)
            SPLIT="$2"
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

# Show help if requested or no dataset provided
if [[ "$SHOW_HELP" == true ]] || [[ -z "$DATASET_PATH" ]]; then
    echo "DataEval Application Runner"
    echo ""
    echo "Usage:"
    echo "  ./run.sh -d PATH [-o PATH] [-s NAME] [--cpu]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset PATH   Path to dataset (required)"
    echo "  -o, --output PATH    Path for output files (optional)"
    echo "  -s, --split NAME     Dataset split name for DatasetDict (optional, e.g., train, test)"
    echo "  -c, --cpu            Use CPU container (default: GPU)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh --dataset /mnt/c/data/cifar10_test"
    echo "  ./run.sh -d /mnt/c/data/cifar10_test -o /mnt/c/data/output"
    echo "  ./run.sh -d /mnt/c/data/cifar10_full --split test"
    echo "  ./run.sh -d /mnt/c/data/cifar10_test --cpu"
    echo ""
    exit 0
fi

# Build mount arguments
MOUNTS="--mount type=bind,source=$DATASET_PATH,target=/data/dataset,readonly"

if [[ -n "$OUTPUT_PATH" ]]; then
    MOUNTS="$MOUNTS --mount type=bind,source=$OUTPUT_PATH,target=/output"
fi

# Build env arguments
ENV_ARGS=""
if [[ -n "$SPLIT" ]]; then
    ENV_ARGS="-e DATASET_SPLIT=$SPLIT"
fi

# Select image and GPU flag
if [[ "$USE_CPU" == true ]]; then
    echo "Running with CPU..."
    docker run $ENV_ARGS $MOUNTS dataeval:cpu
else
    echo "Running with GPU..."
    docker run --gpus all $ENV_ARGS $MOUNTS dataeval:gpu
fi
