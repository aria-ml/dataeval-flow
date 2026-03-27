#!/usr/bin/env python3
"""CLI entry point for standalone usage: python -m dataeval_flow."""

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="dataeval_flow",
        description="DataEval Flow - Data evaluation and monitoring pipelines",
    )

    # Headless execution flags (top-level, no subcommand needed)
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

    _data_default = os.environ.get("DATAEVAL_DATA")
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        default=Path(_data_default) if _data_default else None,
        help="Root directory for data files (default: $DATAEVAL_DATA or current directory)",
    )

    _output_default = os.environ.get("DATAEVAL_OUTPUT")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(_output_default) if _output_default else None,
        help="Path to output directory for artifacts (default: $DATAEVAL_OUTPUT or None).",
    )

    _cache_default = os.environ.get("DATAEVAL_CACHE")
    parser.add_argument(
        "-k",
        "--cache",
        type=Path,
        default=Path(_cache_default) if _cache_default else None,
        help="Directory for disk-backed computation cache (default: $DATAEVAL_CACHE or None).",
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- app (interactive TUI) ---
    app_parser = subparsers.add_parser(
        "app",
        help="Launch interactive TUI dashboard",
        description="Launch the interactive TUI dashboard. Requires: pip install dataeval-flow[app]",
    )
    app_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to an existing config file or folder to load on startup",
    )
    app_parser.add_argument(
        "-d",
        "--data",
        type=Path,
        default=None,
        help="Root directory for data files (default: $DATAEVAL_DATA or current directory)",
    )
    app_parser.add_argument(
        "-k",
        "--cache",
        type=Path,
        default=None,
        help="Directory for disk-backed computation cache (embeddings, metadata, stats).",
    )

    # --- config (simple CLI builder) ---
    config_parser = subparsers.add_parser(
        "config",
        help="Create or edit config files (simple CLI)",
        description="Interactive CLI config builder. Create and edit pipeline config files.",
    )
    config_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to an existing config file or folder to load on startup",
    )

    return parser


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = _build_parser()
    return parser.parse_args()


def main() -> NoReturn:
    """CLI entry point."""
    args = parse_args()

    if args.command == "app":
        try:
            from dataeval_flow._app.app import run_builder
        except ImportError:
            print("ERROR: The interactive TUI requires the 'app' extra.")
            print("")
            print("Install with:")
            print("  pip install dataeval-flow[app]")
            print("")
            print("For the simple CLI config editor, use:")
            print("  dataeval-flow config")
            sys.exit(1)

        run_builder(config_path=args.config, data_dir=args.data, cache_dir=args.cache)
        sys.exit(0)

    if args.command == "config":
        from dataeval_flow._app.cli import run_cli_builder

        run_cli_builder(config_path=args.config)
        sys.exit(0)

    # Headless execution (no subcommand)
    try:
        from dataeval_flow.runner import run

        sys.exit(run(args.config, args.output, data_dir=args.data, verbosity=args.verbose, cache_dir=args.cache))
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
