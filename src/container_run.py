#!/usr/bin/env python3
"""Container entry point - maps container mounts to library calls.

This script is container-specific and NOT part of the dataeval_app package.
It knows about container mount points and maps them to library parameters.

The container expects these mount points:
- /data/config: Config files (required)
- /data/dataset: Input dataset (required)
- /data/model: Model files (optional)
- /output: Results directory (optional)
"""

import os
import sys
from pathlib import Path

# Container mount points (container-specific knowledge)
CONTAINER_MOUNTS = {
    "config": Path("/data/config"),
    "dataset": Path("/data/dataset"),
    "model": Path("/data/model"),
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
        from dataeval_app.config import load_config_folder
        from dataeval_app.workflow import run_task
    except ImportError as e:
        print(f"ERROR: Failed to import dataeval_app: {e}")
        print("Ensure dependencies are installed correctly.")
        return 1

    config = load_config_folder(config_path)
    if not config.tasks:
        print("No tasks defined in config.")
        return 0

    output_path = get_output_path()
    print(f"Running {len(config.tasks)} task(s)...")
    exit_code = 0
    for task in config.tasks:
        run_result = run_task(task, config)
        status = "OK" if run_result.success else "FAILED"
        print(f"  {task.name}: {status}")
        if not run_result.success:
            exit_code = 1
            continue

        task_dir = output_path / task.name
        out = run_result.report(path=task_dir)
        if isinstance(out, str):
            print(out)
        else:
            print(f"  Wrote {out}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
