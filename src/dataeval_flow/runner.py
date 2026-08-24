"""Shared CLI/container runner — loads config, runs tasks, writes reports."""

from __future__ import annotations

import json as json_mod
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.config.schemas import TaskConfig
    from dataeval_flow.workflow import WorkflowResult

_logger: logging.Logger = logging.getLogger(__name__)


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


@dataclass
class _Collected:
    """What one pass over a run's results leaves behind, ready to write and to gate on."""

    failures: int = 0
    warned: list[str] = field(default_factory=list)
    merged: dict[str, dict] = field(default_factory=dict)
    text_parts: list[str] = field(default_factory=list)
    binning: dict[str, dict] = field(default_factory=dict)


def _collect_results(
    selected: Sequence[TaskConfig],
    results: Sequence[WorkflowResult[Any, Any]],
    *,
    verbosity: int,
) -> _Collected:
    """Print each task's report and gather what the run has to write and answer for.

    ``selected`` and ``results`` are the same length by construction — both come from
    the one selection :func:`~dataeval_flow.workflow.select_tasks` resolved — so a task
    is always paired with the result it produced.
    """
    from dataeval_flow._logging import flush_logs

    collected = _Collected()

    for task, result in zip(selected, results, strict=True):
        if not result.success:
            _logger.error("  FAILED: %s", task.name)
            for error in result.errors:
                _logger.error("    %s", error)
            collected.failures += 1
            flush_logs()
            continue

        # --- Text report: summary (no flag) or full detail (-v) ---
        print(result.report(detailed=verbosity >= 1))

        # --- Collect for file output ---
        collected.merged[task.name] = result.to_dict()
        collected.text_parts.append(result.report(detailed=True))
        if record := getattr(result.metadata, "metadata_binning", None):
            collected.binning[task.name] = record

        if result.warning_count:
            collected.warned.append(task.name)

        _logger.info("  OK: %s", task.name)
        flush_logs()

    return collected


def run(
    config_arg: Path | str | None,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    verbosity: int = 0,
    cache_dir: Path | None = None,
    tasks: str | Sequence[str] | None = None,
    fail_on_warning: bool = False,
) -> int:
    """Load config, execute the selected tasks, and write reports.

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
    tasks : str | Sequence[str] | None
        Which tasks to run.  ``None`` (the default) runs every enabled task;
        naming tasks runs those, in the order given, whether or not they are enabled.
    fail_on_warning : bool
        Return a non-zero exit code when a task that otherwise succeeded reports
        findings at ``severity="warning"``.  Off by default: a warning is a prompt
        to look, and only the caller knows whether their pipeline should stop for one.

    Returns
    -------
    int
        0 if every task succeeded (and, under ``fail_on_warning``, raised no
        warnings); 1 otherwise.
    """
    from dataeval_flow._logging import configure_log_levels, flush_logs, setup_logging
    from dataeval_flow.config._loader import get_data_dir
    from dataeval_flow.workflow import run_tasks, select_tasks

    setup_logging(output_dir, verbosity)

    resolved_data = get_data_dir(data_dir)
    config = _resolve_config(config_arg, resolved_data)

    if config.logging:
        configure_log_levels(config.logging.app_level, config.logging.lib_level)

    if not config.tasks:
        _logger.info("No tasks defined in config.")
        return 0

    # Resolve the selection here as well as inside run_tasks, so the results pair back
    # to the tasks that produced them.  run_tasks returns one result per *executed*
    # task, and a result names its workflow type rather than its task, so pairing
    # against config.tasks drops a task's results the moment one is disabled.
    selected = select_tasks(config, tasks)
    results = run_tasks(config, tasks, data_dir=resolved_data, cache_dir=cache_dir)

    collected = _collect_results(selected, results, verbosity=verbosity)

    # --- Write file artifacts (only when output_dir is set) ---
    if output_dir is not None and collected.merged:
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "result.json").write_text(json_mod.dumps(collected.merged, indent=2), encoding="utf-8")
        (results_dir / "result.txt").write_text("\n".join(collected.text_parts), encoding="utf-8")
        _logger.info("  Wrote result.json and result.txt to %s", results_dir)
        _write_encoding_descriptor(collected.binning, results_dir)

    failures = collected.failures
    warned = collected.warned
    _logger.info("Done. %d/%d succeeded.", len(selected) - failures, len(selected))

    if failures:
        return 1

    if warned:
        # Reported either way: a run whose findings breached their thresholds is worth
        # saying out loud even where the caller did not ask for it to be fatal.
        _logger.warning("  Health warnings raised by: %s", ", ".join(warned))
        if fail_on_warning:
            _logger.error("  Failing on health warnings (--fail-on-warning).")
            flush_logs()
            return 1

    return 0


def _write_encoding_descriptor(binning: dict[str, dict], results_dir: Path) -> None:
    """Write the run's encoding descriptor beside its results, when there is one to write.

    The artifact stage six of the lifecycle asks for: lock the encoding in, commit it, and
    hand it back through a policy's ``encoding`` so the next dataset is cut the same way.
    Writing it here makes that a copy rather than a transcription from a report.

    Only where the tasks agree.  A run whose workflows encoded a dataset differently has no
    single descriptor, and writing one of them would hand somebody a policy nobody chose —
    ``dataeval-flow encoding <result.json> --task <name>`` extracts a specific one instead.

    Never fatal: the descriptor is a convenience, and ``result.json`` already carries every
    record it is built from.
    """
    from dataeval_flow.binning import descriptor_from_record

    def _descriptor(name: str, record: dict) -> dict | None:
        """One task's descriptor, or None where it has none to give."""
        try:
            return descriptor_from_record(record)
        except ValueError as exc:  # nothing to write, or splits that disagree
            _logger.debug("  No encoding descriptor for task '%s': %s", name, exc)
            return None

    descriptors = {
        name: descriptor for name, record in binning.items() if (descriptor := _descriptor(name, record)) is not None
    }

    if not descriptors:
        return

    distinct = {json_mod.dumps(d, sort_keys=True) for d in descriptors.values()}
    if len(distinct) > 1:
        _logger.warning(
            "  Tasks %s encoded their factors differently, so no single encoding.json was "
            "written. Extract one with `dataeval-flow encoding <result.json> --task <name>`.",
            sorted(descriptors),
        )
        return

    path = results_dir / "encoding.json"
    path.write_text(json_mod.dumps(next(iter(descriptors.values())), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _logger.info("  Wrote encoding.json to %s — commit it and reference it as a policy `encoding`", results_dir)
