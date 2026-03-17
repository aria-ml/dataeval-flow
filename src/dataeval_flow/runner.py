"""Shared task runner used by both CLI and container entry points."""

import logging
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


def run_all_tasks(config_path: Path | None, output_dir: Path) -> int:
    """Load config and run all tasks, returning an exit code.

    Parameters
    ----------
    config_path : Path | None
        Path to config folder, or None for default.
    output_dir : Path
        Directory for results, logs, and reports.

    Returns
    -------
    int
        0 if all tasks succeed, 1 if any fail.
    """
    from dataeval_flow._logging import configure_log_levels, flush_logs, setup_logging
    from dataeval_flow.config import load_config_folder
    from dataeval_flow.workflow import run_tasks

    setup_logging(output_dir)

    config = load_config_folder(config_path)

    if config.logging:
        configure_log_levels(config.logging.app_level, config.logging.lib_level)

    if not config.tasks:
        logger.info("No tasks defined in config.")
        return 0

    results = run_tasks(config)

    failures = 0
    for task, result in zip(config.tasks, results, strict=True):
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

        # Always write the text report alongside the primary output
        text_report = result.report(format="text")
        path.mkdir(parents=True, exist_ok=True)
        (path / "report.txt").write_text(text_report)

        flush_logs()

    logger.info("Done. %d/%d succeeded.", len(config.tasks) - failures, len(config.tasks))
    return 1 if failures else 0
