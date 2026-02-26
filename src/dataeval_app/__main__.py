#!/usr/bin/env python3
"""CLI entry point for standalone usage: python -m dataeval_app."""

import argparse
import logging
import sys
from pathlib import Path
from typing import NoReturn

logger: logging.Logger = logging.getLogger(__name__)


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
        _run_tasks(args.config, args.output)
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def _run_tasks(config_path: Path | None, output_dir: Path) -> NoReturn:
    """Load config and run all tasks."""
    from dataeval_app._logging import configure_log_levels, flush_logs, setup_logging
    from dataeval_app.config import load_config_folder
    from dataeval_app.workflow import run_task

    setup_logging(output_dir)

    config = load_config_folder(config_path)

    if config.logging:
        configure_log_levels(config.logging.app_level, config.logging.lib_level)

    if not config.tasks:
        logger.info("No tasks defined in config.")
        sys.exit(0)

    logger.info("Running %d task(s)...", len(config.tasks))
    failures = 0
    for task in config.tasks:
        logger.info("--- Task: %s (workflow: %s) ---", task.name, task.workflow)
        result = run_task(task, config)
        if not result.success:
            logger.error("  FAILED: %s", task.name)
            for error in result.errors:
                logger.error("    %s", error)
            failures += 1
            flush_logs()
            continue

        path = output_dir / task.name
        out = result.report(path=path)
        if isinstance(out, str):
            logger.info(out)
        else:
            logger.info("  OK: wrote %s", out)
        flush_logs()

    logger.info("Done. %d/%d succeeded.", len(config.tasks) - failures, len(config.tasks))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
