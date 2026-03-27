"""Shared CLI/container runner — loads config, runs tasks, writes reports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig

logger: logging.Logger = logging.getLogger(__name__)


def _resolve_config(config_arg: Path | str | None, data_dir: Path) -> PipelineConfig:
    """Resolve and load config from an explicit path or auto-discover from data root."""
    from dataeval_flow.config._loader import load_config, load_config_folder

    if config_arg is not None:
        config_path = Path(config_arg)
        if not config_path.is_absolute():
            config_path = data_dir / config_path
    else:
        config_path = data_dir

    if config_path.is_file():
        return load_config(config_path)
    if config_path.is_dir():
        return load_config_folder(config_path)

    msg = f"Config path not found: {config_path}"
    raise FileNotFoundError(msg)


def run(
    config_arg: Path | str | None,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    verbosity: int = 0,
    cache_dir: Path | None = None,
) -> int:
    """Load config, execute all tasks, and write reports.

    This is the shared entry point for CLI (``__main__.py``) and container
    (container) usage.  For programmatic use, prefer
    :func:`~dataeval_flow.load_config` + :func:`~dataeval_flow.run_tasks`.

    Parameters
    ----------
    config_arg : Path | str | None
        Path to config file or folder, or None for auto-discovery at data root.
    output_dir : Path | None
        Directory for results, logs, and reports.  When ``None``, results
        are printed to the console only — no file artifacts are created.
    data_dir : Path | None
        Root directory for data files. Defaults to ``$DATAEVAL_DATA`` or current directory.
    verbosity : int
        Console verbosity (0=quiet, 1=text report, 2=+INFO, 3=+DEBUG).
    cache_dir : Path | None
        Directory for disk-backed computation cache (embeddings, metadata, stats).

    Returns
    -------
    int
        0 if all tasks succeed, 1 if any fail.
    """
    from dataeval_flow._logging import configure_log_levels, flush_logs, setup_logging
    from dataeval_flow.config._loader import get_data_dir
    from dataeval_flow.workflow import run_tasks

    setup_logging(output_dir, verbosity)

    resolved_data = get_data_dir(data_dir)
    config = _resolve_config(config_arg, resolved_data)

    if config.logging:
        configure_log_levels(config.logging.app_level, config.logging.lib_level)

    if not config.tasks:
        logger.info("No tasks defined in config.")
        return 0

    results = run_tasks(config, data_dir=resolved_data, cache_dir=cache_dir)

    failures = 0
    merged: dict[str, dict] = {}
    text_parts: list[str] = []

    for task, result in zip(config.tasks, results, strict=True):
        if not result.success:
            logger.error("  FAILED: %s", task.name)
            for error in result.errors:
                logger.error("    %s", error)
            failures += 1
            flush_logs()
            continue

        # --- Text report: summary (no flag) or full detail (-v) ---
        print(result.report(detailed=verbosity >= 1))

        # --- Collect for file output ---
        merged[task.name] = result.to_dict()
        text_parts.append(result.report(detailed=True))

        logger.info("  OK: %s", task.name)
        flush_logs()

    # --- Write file artifacts (only when output_dir is set) ---
    if output_dir is not None and merged:
        import json as json_mod

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(json_mod.dumps(merged, indent=2), encoding="utf-8")
        (output_dir / "result.txt").write_text("\n".join(text_parts), encoding="utf-8")
        logger.info("  Wrote result.json and result.txt to %s", output_dir)

    logger.info("Done. %d/%d succeeded.", len(config.tasks) - failures, len(config.tasks))
    return 1 if failures else 0
