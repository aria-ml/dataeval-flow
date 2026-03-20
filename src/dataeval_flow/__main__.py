#!/usr/bin/env python3
"""CLI entry point for standalone usage: python -m dataeval_flow."""

import argparse
import sys
from pathlib import Path
from typing import NoReturn


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="dataeval_flow",
        description="DataEval Workflows - Data evaluation tools",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity: -v text report, -vv +INFO logs, -vvv +DEBUG logs.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to config file or folder. If omitted, auto-discovers YAML/JSON at the data root.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        default=None,
        help="Root directory for data files (default: $DATAEVAL_DATA or current directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output directory for artifacts. If omitted, results print to console only.",
    )
    parser.add_argument(
        "-k",
        "--cache",
        type=Path,
        default=None,
        help="Directory for disk-backed computation cache (embeddings, metadata, stats).",
    )
    return parser.parse_args()


def main() -> NoReturn:
    """CLI entry point."""
    args = parse_args()
    try:
        from dataeval_flow.runner import run

        sys.exit(run(args.config, args.output, data_dir=args.data, verbosity=args.verbose, cache_dir=args.cache))
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
