"""A task's matrix run: each run as the task would run alone, over one draw of each source, into one result
(task-matrix spec §5)."""

__all__ = ["run_matrix"]

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataeval_flow._cache import DatasetCache
    from dataeval_flow._chain._run import ExtractorSetup
    from dataeval_flow._matrix._build import MatrixRunPlan
    from dataeval_flow._matrix._result import MatrixResult
    from dataeval_flow._sources import ResolvedSource
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.config._schemas._task import TaskConfig
    from dataeval_flow.workflows._context import DatasetContext

_logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Drawn:
    """A source the matrix reads, resolved once: its resolution, its cache, and the one draw of its view."""

    resolved: "ResolvedSource"
    cache: "DatasetCache | None"
    dataset: Any


def _draw(
    names: list[str], config: "PipelineConfig", data_dir: Path | None, cache_dir: Path | None
) -> dict[str, _Drawn]:
    """Resolve and draw each source once, as a lone task resolves its sources. Raises on a source that does not
    resolve, declares conflicting value ranges or channel groups, or has a view that can't be built: every run reads
    them, so the error is the matrix's, not a run's."""
    from dataeval_flow._orchestrator import _source_contexts
    from dataeval_flow._view import build_view

    contexts, resolved = _source_contexts(names, config, None, data_dir, cache_dir)
    drawn: dict[str, _Drawn] = {}
    for name, source in zip(names, resolved, strict=True):
        context = contexts[name]
        ops = context.view_operations
        drawn[name] = _Drawn(source, context.cache, build_view(context.dataset, list(ops)) if ops else context.dataset)
    return drawn


def _contexts(
    drawn: dict[str, _Drawn], names: list[str], setup: "ExtractorSetup | None"
) -> "dict[str, DatasetContext]":
    """Each source's context for one run: its one draw, read with the run's extractor."""
    from dataeval_flow._orchestrator import _dataset_context

    return {
        name: _dataset_context(name, drawn[name].resolved, setup, drawn[name].cache, drawn=drawn[name].dataset)
        for name in names
    }


def _varies_extractor(plans: "list[MatrixRunPlan]") -> bool:
    return any(key == "extractor" or key.startswith("extractors.") for plan in plans for key in plan.values)


def run_matrix(
    task: "TaskConfig",
    config: "PipelineConfig",
    *,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    report_images: bool = True,
    output_dir: Path | None = None,
) -> "MatrixResult":
    """Run every run of `task`'s matrix and return them as one result.

    What no run varies is resolved once and raises: the sources and their draws, and the task's extractor where the
    matrix leaves it alone. What a run varies is resolved in that run, and an error there fails only that run.
    """
    from dataeval_flow._embeddings import new_extractor_scope, shared_extractor_scope
    from dataeval_flow._matrix._build import expand_matrix
    from dataeval_flow._matrix._result import MatrixResult, MatrixRun
    from dataeval_flow._orchestrator import (
        _apply_device,
        _apply_seed,
        _extractor_setup,
        _refused,
        _resolve_evaluator,
        _resolve_workflow,
        _run_resolved,
        _target_of,
    )
    from dataeval_flow._result import failure_message

    # A task run straight through `run_task` may name an entry the config lacks: say which, as a lone task does.
    if task.kind == "evaluator":
        _resolve_evaluator(task.workflow, config)
    else:
        _resolve_workflow(task.workflow, config)
    plans = expand_matrix(task, config)
    _logger.info("Task '%s': a matrix of %d runs (%s)", task.name, len(plans), _target_of(task))
    start = time.monotonic()
    _apply_seed(config)
    _apply_device()
    drawn = _draw(
        list(dict.fromkeys(name for plan in plans for name in plan.task.source_names)), config, data_dir, cache_dir
    )
    varied = _varies_extractor(plans)
    fixed = None if varied else _extractor_setup(task.extractor, config, data_dir)
    scopes: dict[tuple[str, ...], Any] = {}
    runs: list[MatrixRun] = []
    for plan in plans:
        _logger.info("Task '%s': run %d/%d (%s)", task.name, plan.number, len(plans), plan.label)
        _apply_seed(config)
        _apply_device()
        names = plan.task.source_names
        if tuple(names) not in scopes:
            scopes[tuple(names)] = new_extractor_scope()
        scope = scopes[tuple(names)]
        setup = fixed  # a run that varies the extractor resolves its own below
        resolved = [drawn[name].resolved for name in names]
        run_start = time.monotonic()
        try:
            with shared_extractor_scope(scope):
                if varied:
                    setup = _extractor_setup(plan.task.extractor, plan.pipeline, data_dir)
                result = _run_resolved(
                    plan.task,
                    plan.pipeline,
                    setup,
                    _contexts(drawn, names, setup),
                    resolved,
                    data_dir=data_dir,
                    cache_dir=cache_dir,
                    report_images=report_images,
                    output_dir=output_dir,
                )
        except Exception as error:  # a run that raised is a failed run; the others go on
            _logger.exception("Task '%s': run %d raised", task.name, plan.number)
            message = failure_message(error)
            contexts = _contexts(drawn, names, setup)
            elapsed = time.monotonic() - run_start
            result = _refused(
                plan.task, plan.pipeline, message, setup, contexts, resolved, data_dir=data_dir, elapsed=elapsed
            )
        runs.append(MatrixRun(number=plan.number, label=plan.label, values=plan.values, result=result))
    keys = list(dict.fromkeys(key for plan in plans for key in plan.values))
    matrix = MatrixResult(type=runs[0].result.type, keys=keys, runs=runs)
    _envelope(matrix, task, config, drawn, fixed, time.monotonic() - start)
    _logger.info("Task '%s': matrix finished (success=%s)", task.name, matrix.success)
    return matrix


def _envelope(
    result: "MatrixResult",
    task: "TaskConfig",
    config: "PipelineConfig",
    drawn: dict[str, _Drawn],
    setup: "ExtractorSetup | None",
    elapsed: float,
) -> None:
    """Fill the matrix's envelope: the task as written, the sources its runs read, the extractor where all share
    one (`setup`, resolved once where no run varies it), the device, and the matrix's total time (task-matrix spec
    §6). Each run's envelope is its own."""
    from dataeval.config import get_device

    from dataeval_flow import __version__
    from dataeval_flow._orchestrator import _device_name, _source_description

    metadata = result.metadata
    datasets = list(
        dict.fromkeys(
            operand.source.dataset
            for source in drawn.values()
            for operand in source.resolved.operands
            if operand.source.dataset
        )
    )
    metadata.dataset_id = datasets[0] if len(datasets) == 1 else ",".join(datasets)
    metadata.source_descriptions = [_source_description(source.resolved) for source in drawn.values()]
    metadata.tool_version = __version__
    metadata.execution_time_s = round(elapsed, 3)
    metadata.device = _device_name(get_device())
    if setup is not None:
        metadata.model_id = f"{setup.config.name} ({setup.config.model})"
    metadata.resolved_config = {"task": task.model_dump(), "sources": list(drawn), "seed": config.seed}
    result._entry = task.workflow  # noqa: SLF001 - the banner names the entry its runs ran
    # Every null in the matrix is a value the user wrote: its report keeps them, where it drops unset defaults.
    written = _nulls(metadata.resolved_config["task"].get("matrix"), ("task", "matrix"))
    result._set_nulls = frozenset(written)  # noqa: SLF001 - as `_entry`


def _nulls(value: Any, path: tuple[str | int, ...]) -> set[tuple[str | int, ...]]:
    """The key and index paths of every ``None`` in `value`, through mappings and lists."""
    if value is None:
        return {path}
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    return {found for key, item in items for found in _nulls(item, (*path, key))}
