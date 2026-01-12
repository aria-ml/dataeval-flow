#!/usr/bin/env python3
"""Container entry point - maps container mounts to library calls.

This script is container-specific and NOT part of the dataeval_app package.
It knows about container mount points and maps them to library parameters.

The container expects these mount points:
- /data/dataset: Input dataset (required)
- /data/model: Model files (optional)
- /data/incoming: Raw images (optional)
- /output: Results directory (optional)
"""

import os
import sys
from pathlib import Path

# Container mount points (container-specific knowledge)
CONTAINER_MOUNTS = {
    "dataset": Path("/data/dataset"),
    "model": Path("/data/model"),
    "incoming": Path("/data/incoming"),
    "output": Path("/output"),
}


def get_dataset_path() -> Path:
    """Resolve dataset path from container mount or environment override.

    Returns
    -------
    Path
        Dataset path from DATASET_PATH env var or default container mount.
    """
    if env_path := os.environ.get("DATASET_PATH"):
        return Path(env_path)
    return CONTAINER_MOUNTS["dataset"]


def get_output_path() -> Path:
    """Resolve output path from container mount or environment override.

    Returns
    -------
    Path
        Output path from OUTPUT_PATH env var or default container mount.
    """
    if env_path := os.environ.get("OUTPUT_PATH"):
        return Path(env_path)
    return CONTAINER_MOUNTS["output"]


def main() -> int:
    """Container entry point - resolve paths and call library.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for error.
    """
    dataset_path = get_dataset_path()
    split = os.environ.get("DATASET_SPLIT")

    # Container-specific validation (before importing heavy dependencies)
    if not dataset_path.exists():
        print(f"ERROR: Dataset path not found: {dataset_path}")
        print("Troubleshooting:")
        print("  1. Ensure DATASET_PATH is set in docker-compose")
        print("  2. Check that the path contains a valid dataset")
        return 1

    # Import after validation to give helpful error messages first
    try:
        from dataeval_app import inspect_dataset
    except ImportError as e:
        print(f"ERROR: Failed to import dataeval_app: {e}")
        print("Ensure dependencies are installed correctly.")
        return 1

    try:
        return inspect_dataset(dataset_path, split)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
