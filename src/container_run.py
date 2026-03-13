#!/usr/bin/env python3
"""Container entry point - maps container mounts to library calls.

This script is container-specific and NOT part of the dataeval_flow package.
It knows about container mount points and maps them to library parameters.

The container expects these mount points:
- /data/config: Config files (required)
- /data/dataset: Input dataset (required)
- /output: Results directory (required)
- /data/model: Model files (optional)
- /cache: Computation cache (optional)
"""

import sys
from pathlib import Path

# Container mount points (container-specific knowledge)
CONTAINER_MOUNTS = {
    "config": Path("/data/config"),
    "dataset": Path("/data/dataset"),
    "model": Path("/data/model"),
    "output": Path("/output"),
    "cache": Path("/cache"),
}


def main() -> int:
    """Container entry point - resolve paths and run tasks.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for error.
    """
    config_path = CONTAINER_MOUNTS["config"]
    if not config_path.exists() or (config_path / ".not_mounted").exists():
        print("ERROR: Config path not found or not mounted: /data/config")
        print("Troubleshooting:")
        print("  1. Ensure the config directory is mounted correctly")
        print("  2. Check that the path contains valid YAML config files")
        return 1

    try:
        from dataeval_flow.runner import run_all_tasks
    except ImportError as e:
        print(f"ERROR: Failed to import dataeval_flow: {e}")
        print("Ensure dependencies are installed correctly.")
        return 1

    return run_all_tasks(config_path, CONTAINER_MOUNTS["output"])


if __name__ == "__main__":
    sys.exit(main())
