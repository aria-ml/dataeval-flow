#!/usr/bin/env python3
"""CLI entry point for standalone usage: python -m dataeval_flow."""

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from dataeval_flow._env import env_bool, env_choice, env_int, env_list, env_path

if TYPE_CHECKING:
    from dataeval_flow.evaluators._evaluator import Evaluator

_logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    from dataeval_flow import __version__

    parser = argparse.ArgumentParser(
        prog="dataeval_flow",
        description="DataEval Flow - Data evaluation and monitoring pipelines",
    )

    # Long form only: -v is --verbose.
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
        default=env_path("DATAEVAL_CONFIG"),
        help=(
            "Path to config file or folder (default: $DATAEVAL_CONFIG). If omitted, "
            "auto-discovers YAML/JSON at the data root."
        ),
    )
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        default=env_path("DATAEVAL_DATA"),
        help="Root directory for data files (default: $DATAEVAL_DATA or current directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=env_path("DATAEVAL_OUTPUT"),
        help="Path to output directory for artifacts (default: $DATAEVAL_OUTPUT or None).",
    )
    parser.add_argument(
        "-k",
        "--cache",
        type=Path,
        default=env_path("DATAEVAL_CACHE"),
        help="Directory for disk-backed computation cache (default: $DATAEVAL_CACHE or None).",
    )
    parser.add_argument(
        "--log-format",
        choices=("structured", "plain"),
        default=env_choice("DATAEVAL_LOG_FORMAT", ("structured", "plain")) or "structured",
        help=(
            "Console log format (default: $DATAEVAL_LOG_FORMAT, else structured). "
            "'structured' prefixes each record with an ISO-8601 UTC timestamp and level; "
            "'plain' prints bare messages."
        ),
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
        action=argparse.BooleanOptionalAction,
        default=env_bool("DATAEVAL_FAIL_ON_WARNING") or False,
        help=(
            "Exit non-zero when a task succeeds but reports findings that breached their "
            "health thresholds (default: $DATAEVAL_FAIL_ON_WARNING, else off). "
            "Use --no-fail-on-warning to override the environment."
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

    # --- evaluators (discovery) ---
    evaluators_parser = subparsers.add_parser(
        "evaluators",
        help="List available evaluator types, or show one's parameter schema",
        description=(
            "List the evaluator types this build provides: single DataEval evaluators that report their "
            "determinations, with no health status. Naming one prints the JSON Schema for its parameters — the "
            "fields an `evaluators:` entry of that type accepts."
        ),
    )
    evaluators_parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Evaluator type to describe (e.g. quality.duplicates). Omit to list them all.",
    )
    evaluators_parser.add_argument(
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
        default=env_path("DATAEVAL_DATA"),
        help="Root directory for data files (default: $DATAEVAL_DATA or current directory)",
    )
    app_parser.add_argument(
        "-k",
        "--cache",
        type=Path,
        default=env_path("DATAEVAL_CACHE"),
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
    # Do not read DATAEVAL_OUTPUT here: encoding defaults to stdout unless -o is explicitly specified.
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

    Container images ship no browser or REPL, so type discovery is available from
    the command line.
    """
    import json

    from dataeval_flow.workflows._registry import get_workflow, list_workflows

    if name is not None:
        try:
            workflow = get_workflow(name)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(json.dumps(workflow.config_type.model_json_schema(), indent=2))
        return 0

    entries = [{"name": cls.name, "description": cls.description} for cls in list_workflows()]
    if as_json:
        print(json.dumps(entries, indent=2))
        return 0

    width = max(len(w["name"]) for w in entries)
    for entry in entries:
        print(f"  {entry['name']:<{width}}  {entry['description']}")
    return 0


def _list_evaluators(name: str | None, *, as_json: bool) -> int:
    """Print the available evaluator types, or one evaluator's parameter schema.

    The evaluator counterpart of :func:`_list_workflows`. Each entry also states the
    consumed inputs and the source count, which decide what a task must provide.
    """
    import json

    from dataeval_flow.evaluators._registry import get_evaluator, list_evaluators

    if name is not None:
        try:
            evaluator = get_evaluator(name)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(json.dumps(evaluator.config_type.model_json_schema(), indent=2))
        return 0

    entries = [_evaluator_entry(cls) for cls in list_evaluators()]
    if as_json:
        print(json.dumps(entries, indent=2))
        return 0

    width = max(len(e["name"]) for e in entries)
    for entry in entries:
        print(f"  {entry['name']:<{width}}  {entry['description']}")
        print(f"  {'':<{width}}  consumes: {entry['consumes']}; sources: {entry['sources']}")
    return 0


def _evaluator_entry(cls: "type[Evaluator[Any, Any]]") -> dict[str, str]:
    """An evaluator's listing: its name and description, what it consumes, and how many sources it reads."""
    spec = cls.config_type.inputs
    consumes = [*sorted(spec.required), *(f"{kind} (optional)" for kind in sorted(spec.optional))]
    return {
        "name": cls.name,
        "description": cls.description,
        "consumes": ", ".join(consumes),
        "sources": spec.sources.value,
    }


def apply_env_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply environment variable defaults that cannot be handled by argparse defaults.

    For options with ``count`` or ``append`` actions (such as ``--verbose`` and
    ``--task``), setting defaults in argparse causes command-line arguments to
    increment or append to the default instead of overriding it. Applying these
    environment variables post-parsing ensures CLI arguments take precedence.

    ``DATAEVAL_TASKS`` applies only to headless execution, not subcommands.
    """
    if getattr(args, "verbose", 0) == 0:
        args.verbose = env_int("DATAEVAL_VERBOSITY") or 0
    if args.command is None and getattr(args, "task", None) is None:
        args.task = env_list("DATAEVAL_TASKS")
    return args


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and apply environment defaults."""
    parser = _build_parser()
    return apply_env_defaults(parser.parse_args())


def main() -> NoReturn:
    """CLI entry point."""
    try:
        args = parse_args()
    except ValueError as e:
        # Report invalid environment variable values without a traceback.
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

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
        # without this this command's messages are dropped and the caller gets a bare exit
        # code. At INFO: this command's output is one artifact and one line saying where
        # it went.
        setup_logging(verbosity=max(args.verbose, 2), log_format=args.log_format)
        sys.exit(write_encoding(args.result, args.output, args.task))

    if args.command == "workflows":
        sys.exit(_list_workflows(args.name, as_json=args.json))

    if args.command == "evaluators":
        sys.exit(_list_evaluators(args.name, as_json=args.json))

    if args.command == "config":
        from dataeval_flow._app.cli import run_cli_builder

        run_cli_builder(config_path=args.config)
        sys.exit(0)

    # Headless execution (no subcommand)
    # Enable clean console logging up front so failures during import or config
    # resolution are reported even before the runner configures the file log.
    from dataeval_flow._logging import setup_logging

    setup_logging(verbosity=args.verbose, log_format=args.log_format)
    try:
        from dataeval_flow._runner import run

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
