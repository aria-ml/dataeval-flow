#!/usr/bin/env python3
"""CLI entry point for standalone usage: python -m dataeval_app.

This module provides the command-line interface for the DataEval Application.
It can be invoked directly via: python -m dataeval_app --dataset-path /path/to/data
"""

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from dataeval_app import inspect_dataset


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing dataset_path.
    """
    parser = argparse.ArgumentParser(
        prog="dataeval_app",
        description="DataEval Application - Data evaluation and inspection tools",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the dataset directory",
    )
    return parser.parse_args()


def main() -> NoReturn:
    """CLI entry point.

    Parses command line arguments and runs dataset inspection.
    Exits with code 0 on success, 1 on error.
    """
    args = parse_args()
    try:
        result = inspect_dataset(args.dataset_path)
        sys.exit(result)
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
