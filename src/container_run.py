#!/usr/bin/env python3
"""Container entry point - maps container mounts to library calls.

This script is container-specific and NOT part of the dataeval_flow package.
It knows about container mount points and maps them to library parameters.

The container expects these mount points:
- /dataeval: Data directory (read-only) — datasets, models, and optionally config files
- /output: Results directory (read-write)
- /cache: Computation cache (read-write, optional)

The data root can be overridden via the DATAEVAL_DATA environment variable.

Flags mirror ``python -m dataeval_flow`` but with container-aware defaults.
"""

import argparse
import os
import sys
from pathlib import Path

_FALLBACK_DATA = "/dataeval"
_DEFAULT_OUTPUT = Path("/output")
_DEFAULT_CACHE = Path("/cache")


def parse_args() -> argparse.Namespace:
    """Parse container entry-point arguments."""
    parser = argparse.ArgumentParser(
        prog="container_run",
        description="DataEval Workflows — container entry point",
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
        help="Config file or folder (relative to data root). Auto-discovers if omitted.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        default=Path(os.environ.get("DATAEVAL_DATA", _FALLBACK_DATA)),
        help=f"Data root directory (default: $DATAEVAL_DATA or {_FALLBACK_DATA}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output directory for artifacts (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "-k",
        "--cache",
        type=Path,
        default=None,
        help="Cache directory for embeddings/metadata. Not used unless specified.",
    )
    return parser.parse_args()


def main() -> int:
    """Container entry point - resolve paths and run tasks.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for error.
    """
    args = parse_args()

    data_dir: Path = args.data
    if not data_dir.exists() or (data_dir / ".not_mounted").exists():
        print(f"ERROR: Data directory not found or not mounted: {data_dir}")
        print("Troubleshooting:")
        print("  1. Ensure the data directory is mounted correctly")
        print(f"  2. Mount your data to {data_dir}:")
        print(f"     --mount type=bind,source=/path/to/data,target={data_dir},readonly")
        return 1

    # Default to /cache when it is mounted (writable) and no explicit --cache given
    cache_dir = args.cache
    if cache_dir is None and _DEFAULT_CACHE.is_dir() and os.access(_DEFAULT_CACHE, os.W_OK):
        cache_dir = _DEFAULT_CACHE

    try:
        from dataeval_flow.runner import run
    except ImportError as e:
        print(f"ERROR: Failed to import dataeval_flow: {e}")
        print("Ensure dependencies are installed correctly.")
        return 1

    return run(
        args.config,
        args.output,
        data_dir=data_dir,
        verbosity=args.verbose,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
