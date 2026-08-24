#!/usr/bin/env python3
"""CLI entry point for standalone usage: python -m dataeval_flow."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import NoReturn

_logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    from dataeval_flow import __version__

    parser = argparse.ArgumentParser(
        prog="dataeval_flow",
        description="DataEval Flow - Data evaluation and monitoring pipelines",
    )

    # Long form only: -v is --verbose, and a container asking which build it is running
    # is a different question from how loudly it should report.
    parser.add_argument(
        "--version",
        action="version",
        version=f"dataeval-flow {__version__}",
        help="Show the installed dataeval-flow version and exit.",
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
    parser.add_argument(
        "-t",
        "--task",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Run only this task, by name. Repeat to run several, in the order given. "
            "Naming a task runs it whether or not the config marks it enabled. "
            "Default: every enabled task."
        ),
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help=(
            "Exit non-zero when a task succeeds but reports findings that breached their "
            "health thresholds. Off by default, so a warning stays a prompt to look."
        ),
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- workflows (discovery) ---
    workflows_parser = subparsers.add_parser(
        "workflows",
        help="List available workflow types, or show one's parameter schema",
        description=(
            "List the workflow types this build provides. Naming one prints the JSON "
            "Schema for its parameters — the fields a `workflows:` entry of that type accepts."
        ),
    )
    workflows_parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Workflow type to describe (e.g. data-cleaning). Omit to list them all.",
    )
    workflows_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the listing as JSON rather than a table.",
    )

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

    # --- encoding (extract a committable descriptor from a result) ---
    encoding_parser = subparsers.add_parser(
        "encoding",
        help="Write the encoding descriptor a result was computed under",
        description=(
            "Extract the encoding descriptor from an archived result.json and write it "
            "where it can be reviewed and committed. Reference it from a metadata policy's "
            "`encoding` to cut a later dataset the same way."
        ),
    )
    encoding_parser.add_argument(
        "result",
        type=Path,
        help="Path to a result.json written by a run",
    )
    encoding_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to write the descriptor (default: print it)",
    )
    encoding_parser.add_argument(
        "--task",
        default=None,
        help="Which task's encoding to extract, when the result holds several that differ",
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


def _list_workflows(name: str | None, *, as_json: bool) -> int:
    """Print the available workflow types, or one workflow's parameter schema.

    Discovery without a TUI: the container image ships no browser and no Python REPL
    worth the name, so the question "what can this build run, and what does it take?"
    needs an answer from the command line.
    """
    import json

    from dataeval_flow.workflow import get_workflow, list_workflows

    if name is not None:
        try:
            workflow = get_workflow(name)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        schema = workflow.params_schema
        if schema is None:
            print(f"{workflow.name} takes no parameters.")
            return 0
        print(json.dumps(schema.model_json_schema(), indent=2))
        return 0

    entries = sorted(list_workflows(), key=lambda w: w["name"])
    if as_json:
        print(json.dumps(entries, indent=2))
        return 0

    width = max(len(w["name"]) for w in entries)
    for entry in entries:
        print(f"  {entry['name']:<{width}}  {entry['description']}")
    return 0


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

    if args.command == "encoding":
        from dataeval_flow._encoding_cli import write_encoding
        from dataeval_flow._logging import setup_logging

        # Console logging first: the package attaches a NullHandler to its own logger, so
        # without this every message this command emits — the error explaining why it
        # refused, and the "commit it" hand-off when it succeeds — is dropped and the user
        # is left with a bare exit code. At INFO because this command's whole output is
        # one artifact and one sentence saying where it went.
        setup_logging(verbosity=max(args.verbose, 2))
        sys.exit(write_encoding(args.result, args.output, args.task))

    if args.command == "workflows":
        sys.exit(_list_workflows(args.name, as_json=args.json))

    if args.command == "config":
        from dataeval_flow._app.cli import run_cli_builder

        run_cli_builder(config_path=args.config)
        sys.exit(0)

    # Headless execution (no subcommand)
    # Enable clean console logging up front so failures during import or config
    # resolution are reported even before the runner configures the file log.
    from dataeval_flow._logging import setup_logging

    setup_logging(verbosity=args.verbose)
    try:
        from dataeval_flow.runner import run

        sys.exit(
            run(
                args.config,
                args.output,
                data_dir=args.data,
                verbosity=args.verbose,
                cache_dir=args.cache,
                tasks=args.task,
                fail_on_warning=args.fail_on_warning,
            )
        )
    except (FileNotFoundError, ValueError, ImportError) as e:
        _logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
