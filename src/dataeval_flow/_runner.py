"""Shared CLI/container runner — loads config, runs tasks, writes reports."""

from __future__ import annotations

import json as json_mod
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataeval_flow._blocks._text import DEFAULT_WIDTH, MIN_WIDTH

if TYPE_CHECKING:
    from dataeval_flow._result import Result
    from dataeval_flow.config._models import PipelineConfig, ResultConfig

_logger: logging.Logger = logging.getLogger(__name__)


def _resolve_config(config_arg: Path | str | None, data_dir: Path) -> PipelineConfig:
    """Resolve and load config from an explicit path or auto-discover from data root."""
    from dataeval_flow.config._loader import load_config

    if config_arg is not None:
        config_path = Path(config_arg)
        if not config_path.is_absolute():
            config_path = data_dir / config_path
    else:
        config_path = data_dir

    if not config_path.exists():
        msg = f"Config path not found: {config_path}"
        raise FileNotFoundError(msg)
    return load_config(config_path)


@dataclass
class _Collected:
    """What one pass over a run's results leaves behind, ready to write and to gate on."""

    failures: int = 0
    warned: list[str] = field(default_factory=list)
    merged: dict[str, dict] = field(default_factory=dict)
    reported: dict[str, Result[Any, Any]] = field(default_factory=dict)
    # Every task's result, failed or not, in the order the tasks ran: what the CI reports name.
    everything: dict[str, Result[Any, Any]] = field(default_factory=dict)
    # Each task's text report as printed, and whether in full, so a file wanting the same needn't draw it again.
    printed: dict[str, tuple[bool, str]] = field(default_factory=dict)
    binning: dict[str, dict] = field(default_factory=dict)


def _collect_results(
    results: Mapping[str, Result[Any, Any]], *, verbosity: int, report_width: int = DEFAULT_WIDTH
) -> _Collected:
    """Print each task's report; collect its failures, warnings, and the payloads to write.

    ``results`` is keyed by the task that produced each result, as ``run_tasks`` returns it.
    """
    from dataeval_flow._logging import flush_logs
    from dataeval_flow.workflows._result import WorkflowResult

    collected = _Collected()

    for name, result in results.items():
        collected.everything[name] = result
        if not result.success:
            _logger.error("  FAILED: %s", name)
            for error in result.errors:
                _logger.error("    %s", error)
            collected.failures += 1
            flush_logs()
            continue

        # --- Text report: summary (no flag) or full detail (-v) ---
        text = result.report(detailed=verbosity >= 1, width=report_width)
        print(text)
        collected.printed[name] = (verbosity >= 1, text)

        # --- Collect for file output ---
        collected.merged[name] = result.to_dict()
        collected.reported[name] = result
        if record := getattr(result.metadata, "metadata_binning", None):
            collected.binning[name] = record

        # Only a workflow judges health; an evaluator makes determinations, never a verdict.
        if isinstance(result, WorkflowResult) and result.warning_count:
            collected.warned.append(name)

        _logger.info("  OK: %s", name)
        flush_logs()

    return collected


def _write_results(collected: _Collected, results_dir: Path, settings: ResultConfig, width: int) -> list[str]:
    """Write the results in each configured format, one file for the run or one per task; return the names written.

    The JSON, text and HTML files hold the tasks that succeeded, and none is written where no task did. The
    JUnit and Markdown files, which a CI job reads, name the failed tasks too.
    """
    tasks = list(collected.everything)
    groups = [(f"{settings.name}-{task}", [task]) for task in tasks] if settings.per_task else [(settings.name, tasks)]
    written: list[str] = []
    for stem, names in groups:
        for kind in settings.formats:
            text = _file_text(kind, names, collected, settings.detail == "full", width)
            if text is not None:
                results_dir.mkdir(parents=True, exist_ok=True)
                path = results_dir / f"{stem}.{_EXTENSIONS[kind]}"
                path.write_text(text, encoding="utf-8")
                written.append(path.name)
    if not collected.reported and (unwritten := [kind for kind in settings.formats if kind in _SUCCEEDED_ONLY]):
        _logger.warning("  No task succeeded, so no file was written for: %s.", ", ".join(unwritten))
    return written


def _file_text(kind: str, names: Sequence[str], collected: _Collected, detailed: bool, width: int) -> str | None:
    """The named tasks' results as one file of *kind*, or ``None`` where it would hold nothing."""
    from dataeval_flow._ci_reports import junit_report, markdown_summary
    from dataeval_flow._result import results_html

    if kind in ("junit", "markdown"):
        results = {name: collected.everything[name] for name in names}
        return junit_report(results) if kind == "junit" else markdown_summary(results)
    succeeded = [name for name in names if name in collected.reported]  # json, text and html
    if not succeeded:
        return None
    if kind == "json":
        return json_mod.dumps({name: collected.merged[name] for name in succeeded}, indent=2)
    if kind == "text":
        return "\n".join(_text(name, collected, detailed, width) for name in succeeded)
    return results_html([collected.reported[name] for name in succeeded], detailed=detailed)


def _text(name: str, collected: _Collected, detailed: bool, width: int) -> str:
    """A task's text report, reusing the one printed where it holds the same detail."""
    printed_detailed, printed = collected.printed[name]
    return printed if printed_detailed == detailed else collected.reported[name].report(detailed=detailed, width=width)


def _gate(fail_on: str, fail_on_warning: bool | None) -> str:
    """What fails the run: the config's ``fail_on``, unless ``--fail-on-warning`` or its variable says otherwise."""
    if fail_on_warning is None:
        return fail_on
    if fail_on_warning:
        return "warning"
    return "failure" if fail_on == "warning" else fail_on


# The extension of each format's file.
_EXTENSIONS = {"json": "json", "text": "txt", "html": "html", "junit": "xml", "markdown": "md"}
# The formats that hold only the tasks that succeeded.
_SUCCEEDED_ONLY = ("json", "text", "html")


def run(
    config_arg: Path | str | None,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    verbosity: int = 0,
    cache_dir: Path | None = None,
    tasks: str | Sequence[str] | None = None,
    fail_on_warning: bool | None = None,
    report_width: int | None = None,
    report_images: bool = True,
) -> int:
    """Load config, execute the selected tasks, and write reports.

    This is the shared entry point for CLI (``__main__.py``) and container
    usage.  For programmatic use, prefer
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
        A task named twice runs once.
    fail_on_warning : bool | None
        ``True`` returns 3 when a task that otherwise succeeded reports findings at ``severity="warning"``, and
        ``False`` doesn't. ``None`` (the default) leaves it to the config's ``result: fail_on``.
    report_width : int | None
        Characters per line of the text report, on the console and in ``result.txt``; at least 40. ``None`` (the
        default) takes the config's ``result: width``.
    report_images : bool
        Whether results keep thumbnails of the items their reports name, which ``result.html`` shows and
        ``result.json`` holds. ``False`` reads no item and keeps none.

    Returns
    -------
    int
        0 if nothing the gate fails on happened, or the gate is ``fail_on: never``; 1 if a task failed or
        an export couldn't be written; 3 if the gate fails on warnings and a task raised one.

    Raises
    ------
    ValueError
        If ``report_width`` is below 40, before any task runs.
    """
    from dataeval_flow._logging import configure_log_levels, flush_logs, setup_logging
    from dataeval_flow._orchestrator import run_tasks
    from dataeval_flow.config._loader import get_data_dir

    # Checked before the tasks run: a width the report refuses would otherwise surface only
    # after every task had finished, and before any of their results were written.
    if report_width is not None and report_width < MIN_WIDTH:
        raise ValueError(f"report_width must be at least {MIN_WIDTH}, got {report_width}")

    setup_logging(output_dir, verbosity)

    resolved_data = get_data_dir(data_dir)
    config = _resolve_config(config_arg, resolved_data)

    if config.logging:
        configure_log_levels(config.logging.app_level, config.logging.lib_level)

    if not config.tasks:
        # An export names a source, not a task, so a config that runs nothing still has a
        # corpus to write.
        _logger.info("No tasks defined in config.")
        export_failures = _write_declared_exports(config, output_dir, resolved_data)
        return 1 if export_failures and _gate(config.result.fail_on, fail_on_warning) != "never" else 0

    # Keyed by the executed tasks' names, so a disabled task cannot misalign a result
    # with the task that produced it.
    results = run_tasks(config, tasks, data_dir=resolved_data, cache_dir=cache_dir, report_images=report_images)

    width = config.result.width if report_width is None else report_width
    collected = _collect_results(results, verbosity=verbosity, report_width=width)

    # --- Write file artifacts (only when output_dir is set) ---
    if output_dir is not None:
        results_dir = output_dir / "results"
        if written := _write_results(collected, results_dir, config.result, width):
            _logger.info("  Wrote %s to %s", ", ".join(written), results_dir)
        if collected.merged:
            _write_encoding_descriptor(collected.binning, results_dir)

    export_failures = _write_declared_exports(config, output_dir, resolved_data)

    failures = collected.failures
    warned = collected.warned
    _logger.info("Done. %d/%d succeeded.", len(results) - failures, len(results))

    gate = _gate(config.result.fail_on, fail_on_warning)
    if (failures or export_failures) and gate != "never":
        return 1

    if warned:
        # Printed either way: a warning that breached a threshold is worth stating even
        # when it is not fatal.
        _logger.warning("  Health warnings raised by: %s", ", ".join(warned))
        if gate == "warning":
            _logger.error("  Failing on health warnings (fail_on: warning, or --fail-on-warning).")
            flush_logs()
            return 3

    return 0


def _write_declared_exports(config: PipelineConfig, output_dir: Path | None, data_dir: Path) -> int:
    """Write the run's declared exports, and report how many failed.

    An export the config asked for and did not get is a failure the caller must see:
    succeeding quietly would hand somebody an empty output directory. Writing needs an
    output directory, so a run that produces no file artifacts produces no exports either.
    """
    if output_dir is None or not config.exports:
        return 0

    from dataeval_flow._export import write_exports

    export_failures = write_exports(config, output_dir, data_dir=data_dir)
    if export_failures:
        _logger.error("  %d export(s) failed to write.", export_failures)
    return export_failures


def _write_encoding_descriptor(binning: dict[str, dict], results_dir: Path) -> None:
    """Write the run's encoding descriptor beside its results, when there is one to write.

    The artifact stage six of the lifecycle asks for: lock the encoding in, commit it, and
    hand it back through a policy's ``encoding`` so the next dataset is cut the same way.
    Writing it here makes that a copy rather than a transcription from a report.

    Only where the tasks agree.  A run whose workflows encoded a dataset differently has no
    single descriptor, and writing one of them picks a policy nobody chose —
    ``dataeval-flow encoding <result.json> --task <name>`` extracts a specific one instead.

    Never fatal: the descriptor is a convenience, and ``result.json`` already carries every
    record it is built from.
    """
    from dataeval_flow._binning import descriptor_from_record

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
            "written. Extract one from the JSON results with `dataeval-flow encoding <results.json> --task <name>`.",
            sorted(descriptors),
        )
        return

    path = results_dir / "encoding.json"
    path.write_text(json_mod.dumps(next(iter(descriptors.values())), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _logger.info("  Wrote encoding.json to %s — commit it and reference it as a policy `encoding`", results_dir)
