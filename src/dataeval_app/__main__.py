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
        required=True,
        help="Path to output directory for results and log.txt",
    )
    return parser.parse_args()


def main() -> NoReturn:
    """CLI entry point."""
    args = parse_args()
    try:
        from dataeval_app.runner import run_all_tasks

        sys.exit(run_all_tasks(args.config, args.output))
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
