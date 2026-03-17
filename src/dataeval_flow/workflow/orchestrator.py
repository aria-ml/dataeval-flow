"""Task orchestration — config → execution bridge."""

__all__ = ["run_tasks"]

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, runtime_checkable

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig, SourceConfig
    from dataeval_flow.config.schemas import DataCleaningWorkflowConfig, DriftMonitoringWorkflowConfig
    from dataeval_flow.config.schemas._task import TaskConfig
    from dataeval_flow.workflow import DatasetContext, WorkflowResult


@runtime_checkable
class _Named(Protocol):
    """Protocol for config objects with a name attribute."""

    name: str


T = TypeVar("T", bound=_Named)


def _resolve_by_name(items: Sequence[T] | None, name: str, kind: str) -> T:
    """Find a config object by name.

    Parameters
    ----------
    items : list[T] | None
        List of config objects with a ``name`` attribute.
    name : str
        Name to look up.
    kind : str
        Human-readable kind for error messages (e.g. "dataset").

    Raises
    ------
    ValueError
        If *items* is ``None`` or *name* is not found.
    """
    if items is None:
        raise ValueError(f"No {kind} configs defined, cannot resolve '{name}'")
    for item in items:
        if item.name == name:
            return item
    available = [item.name for item in items]
    raise ValueError(f"Unknown {kind}: '{name}'. Available: {available}")


def _resolve_workflow(
    workflow_name: str,
    config: "PipelineConfig",
) -> "DataCleaningWorkflowConfig | DriftMonitoringWorkflowConfig":
    """Resolve a workflow by name from ``config.workflows``."""
    return _resolve_by_name(config.workflows, workflow_name, "workflow")


def _run_single_task(task: "TaskConfig", config: "PipelineConfig") -> "WorkflowResult[Any, Any]":
    """Run a single resolved task against a pipeline config.

    This is the internal workhorse — resolves all references (sources,
    extractor) against ``PipelineConfig``, builds contexts, and executes
    the workflow.
    """
    from dataeval_flow.cache import DatasetCache
    from dataeval_flow.config._models import SourceConfig
    from dataeval_flow.config.schemas import ExtractorConfig, PreprocessorConfig, SelectionConfig
    from dataeval_flow.dataset import resolve_dataset
    from dataeval_flow.preprocessing import build_preprocessing
    from dataeval_flow.workflow import DatasetContext, WorkflowContext, get_workflow

    logger.info("Task '%s': starting (workflow_instance=%s)", task.name, task.workflow)

    # 1. Normalize sources to list
    source_names: list[str] = [task.sources] if isinstance(task.sources, str) else list(task.sources)

    # 2. Resolve extractor config (optional — single per task)
    extractor_cfg: ExtractorConfig | None = None
    transforms = None
    batch_size: int | None = None

    if task.extractor is not None:
        extractor_cfg = _resolve_by_name(config.extractors, task.extractor, "extractor")
        batch_size = extractor_cfg.batch_size

        # Resolve preprocessor from extractor (optional)
        if extractor_cfg.preprocessor is not None:
            pre_config: PreprocessorConfig = _resolve_by_name(
                config.preprocessors, extractor_cfg.preprocessor, "preprocessor"
            )
            transforms = build_preprocessing(pre_config.steps)

    # 3. Build a DatasetContext per source
    dataset_contexts: dict[str, DatasetContext] = {}
    dataset_names: list[str] = []
    resolved_sources: list[SourceConfig] = []

    for src_name in source_names:
        source: SourceConfig = _resolve_by_name(config.sources, src_name, "source")
        resolved_sources.append(source)
        ds_config = _resolve_by_name(config.datasets, source.dataset, "dataset")
        resolved = resolve_dataset(ds_config)
        dataset_names.append(source.dataset)

        # Resolve selection from source (optional)
        selection_steps = None
        if source.selection is not None:
            sel_config: SelectionConfig = _resolve_by_name(config.selections, source.selection, "selection")
            selection_steps = sel_config.steps

        # Build per-dataset cache
        cache_path = Path(task.cache_dir) if task.cache_dir else None
        ds_cache = DatasetCache.get_or_create(
            cache_dir=cache_path,
            name=resolved.name,
            cache_key=resolved.cache_key,
        )

        dataset_contexts[src_name] = DatasetContext(
            name=src_name,
            dataset=resolved.dataset,
            extractor=extractor_cfg,
            transforms=transforms,
            selection_steps=selection_steps,
            batch_size=batch_size,
            label_source=resolved.label_source,
            cache=ds_cache,
        )

    if task.cache_dir:
        logger.info("Cache enabled: %s", Path(task.cache_dir))

    # 4. Build WorkflowContext
    context = WorkflowContext(
        dataset_contexts=dataset_contexts,
        batch_size=batch_size,
    )

    logger.debug("Task '%s': resolved %d source(s): %s", task.name, len(source_names), source_names)

    # 5. Resolve workflow → type + params
    instance = _resolve_workflow(task.workflow, config)
    workflow = get_workflow(instance.type)

    # 6. Run workflow with timing
    logger.debug("Task '%s': executing workflow", task.name)
    start = time.monotonic()
    result = workflow.execute(context, instance)
    elapsed = time.monotonic() - start
    logger.info("Task '%s': finished in %.1fs (success=%s)", task.name, elapsed, result.success)

    # 7. Populate metadata envelope
    _populate_result_metadata(
        result, dataset_names, dataset_contexts, resolved_sources, extractor_cfg, task.output_format, elapsed
    )

    return result


def _populate_result_metadata(
    result: "WorkflowResult[Any, Any]",
    dataset_names: Sequence[str],
    dataset_contexts: "Mapping[str, DatasetContext]",
    sources: "Sequence[SourceConfig]",
    extractor_cfg: Any,
    output_format: "Literal['text', 'json', 'yaml']",
    elapsed: float,
) -> None:
    """Fill in the JATIC metadata envelope from resolved source/extractor context."""
    from dataeval_flow import __version__

    result.metadata.dataset_id = dataset_names[0] if len(dataset_names) == 1 else ",".join(dataset_names)
    result.metadata.tool_version = __version__
    result.metadata.execution_time_s = round(elapsed, 3)
    result.format = output_format

    # Source context — selection info
    selection_names = [s.selection for s in sources if s.selection is not None]
    if selection_names:
        result.metadata.selection_id = selection_names[0] if len(selection_names) == 1 else ",".join(selection_names)

    # Extractor context — model + preprocessor info
    if extractor_cfg is not None:
        result.metadata.model_id = f"{extractor_cfg.name} ({extractor_cfg.model})"
        if extractor_cfg.preprocessor is not None:
            result.metadata.preprocessor_id = extractor_cfg.preprocessor

    # Annotate dataset source when label provenance is known
    dc = next(iter(dataset_contexts.values()))
    if dc.label_source:
        result.metadata.label_source = dc.label_source


def run_tasks(
    config: "PipelineConfig",
    tasks: str | Sequence[str] | None = None,
) -> "list[WorkflowResult[Any, Any]]":
    """Run tasks from a pipeline configuration.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration containing datasets, sources, extractors,
        workflows, and tasks.
    tasks : str | list[str] | None
        Which tasks to run:

        - ``None`` (default) — run all enabled tasks
        - ``str`` — run a single task by name
        - ``list[str]`` — run specific tasks by name, in the given order

    Returns
    -------
    list[WorkflowResult]
        One result per task executed, in execution order.

    Raises
    ------
    ValueError
        If no tasks are defined, all are disabled, or a named task is
        not found.
    """
    if not config.tasks:
        raise ValueError("No tasks defined in pipeline config")

    if tasks is None:
        # Run all enabled tasks
        to_run = [t for t in config.tasks if t.enabled]
        skipped = len(config.tasks) - len(to_run)
        if skipped:
            logger.info("Skipping %d disabled task(s)", skipped)
        if not to_run:
            raise ValueError("All tasks are disabled — nothing to run")
    elif isinstance(tasks, str):
        to_run = [_resolve_by_name(config.tasks, tasks, "task")]
    else:
        to_run = [_resolve_by_name(config.tasks, name, "task") for name in tasks]

    logger.info("Running %d task(s)", len(to_run))
    results: list[WorkflowResult[Any, Any]] = []
    for task in to_run:
        logger.info("--- Task: %s (workflow: %s) ---", task.name, task.workflow)
        results.append(_run_single_task(task, config))
    return results
