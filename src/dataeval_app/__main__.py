#!/usr/bin/env python3
"""CLI entry point for standalone usage: python -m dataeval_app."""

import argparse
import sys
from pathlib import Path
from typing import NoReturn


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="dataeval_app",
        description="DataEval Application - Data evaluation tools",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config folder (default: /data/config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output directory (default: /output)",
    )
    return parser.parse_args()


def main() -> NoReturn:
    """CLI entry point."""
    args = parse_args()
    try:
        _run_tasks(args.config, args.output)
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def _run_tasks(config_path: Path | None, output_dir: Path | None = None) -> NoReturn:
    """Load config and run all tasks."""
    from dataeval_app.config import load_config_folder
    from dataeval_app.workflow import run_task

    config = load_config_folder(config_path)

    if not config.tasks:
        print("No tasks defined in config.")
        sys.exit(0)

    if output_dir is None:
        output_dir = Path("/output")

    print(f"Running {len(config.tasks)} task(s)...")
    failures = 0
    for task in config.tasks:
        print(f"\n--- Task: {task.name} (workflow: {task.workflow}) ---")
        result = run_task(task, config)
        if not result.success:
            print(f"  FAILED: {result.errors}")
            failures += 1
            continue

        # Determine output path for file-based formats
        path = output_dir / task.name if output_dir.exists() else None
        out = result.report(path=path)
        if isinstance(out, str):
            print(out)
        else:
            print(f"  OK: wrote {out}")

    print(f"\nDone. {len(config.tasks) - failures}/{len(config.tasks)} succeeded.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
