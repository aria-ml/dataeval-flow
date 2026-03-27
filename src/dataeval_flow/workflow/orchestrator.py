"""Task orchestration — config → execution bridge."""

__all__ = ["run_task", "run_tasks"]

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload, runtime_checkable

from pydantic import BaseModel

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dataeval_flow.config import PipelineConfig, SourceConfig, TaskConfig
    from dataeval_flow.config.schemas import WorkflowConfig
    from dataeval_flow.config.schemas._task import (
        DataAnalysisTaskConfig,
        DataCleaningTaskConfig,
        DataPrioritizationTaskConfig,
        DriftMonitoringTaskConfig,
        OODDetectionTaskConfig,
    )
    from dataeval_flow.workflow import DatasetContext, WorkflowResult
    from dataeval_flow.workflows.analysis.outputs import DataAnalysisMetadata, DataAnalysisOutputs
    from dataeval_flow.workflows.cleaning.outputs import DataCleaningMetadata, DataCleaningOutputs
    from dataeval_flow.workflows.drift.outputs import DriftMonitoringMetadata, DriftMonitoringOutputs
    from dataeval_flow.workflows.ood.outputs import OODDetectionMetadata, OODDetectionOutputs
    from dataeval_flow.workflows.prioritization.outputs import DataPrioritizationMetadata, DataPrioritizationOutputs


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
) -> "WorkflowConfig":
    """Resolve a workflow by name from ``config.workflows``."""
    return _resolve_by_name(config.workflows, workflow_name, "workflow")


E = TypeVar("E", bound=BaseModel)


def _resolve_extractor_paths(extractor_cfg: E, data_dir: Path | None) -> E:
    """Resolve relative ``model_path`` on extractor configs against *data_dir*."""
    model_path: str | None = getattr(extractor_cfg, "model_path", None)

    if model_path is not None:
        from dataeval_flow.config._loader import resolve_path

        resolved = str(resolve_path(model_path, data_dir))
        if resolved != model_path:
            return extractor_cfg.model_copy(update={"model_path": resolved})

    return extractor_cfg


def _run_single_task(
    task: "TaskConfig",
    config: "PipelineConfig",
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> "WorkflowResult[Any, Any]":
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
        extractor_cfg = _resolve_extractor_paths(extractor_cfg, data_dir)
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
        resolved = resolve_dataset(ds_config, data_dir=data_dir)
        dataset_names.append(source.dataset)

        # Resolve selection from source (optional)
        selection_steps = None
        if source.selection is not None:
            sel_config: SelectionConfig = _resolve_by_name(config.selections, source.selection, "selection")
            selection_steps = sel_config.steps

        # Build per-dataset cache
        ds_cache = DatasetCache.get_or_create(
            cache_dir=cache_dir,
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

    if cache_dir:
        logger.info("Cache enabled: %s", cache_dir)

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
        result,
        dataset_names,
        dataset_contexts,
        resolved_sources,
        extractor_cfg,
        elapsed,
        instance,
        config,
        data_dir=data_dir,
    )

    return result


def _populate_result_metadata(
    result: "WorkflowResult[Any, Any]",
    dataset_names: Sequence[str],
    dataset_contexts: "Mapping[str, DatasetContext]",
    sources: "Sequence[SourceConfig]",
    extractor_cfg: Any,
    elapsed: float,
    workflow_instance: "WorkflowConfig | None" = None,
    pipeline_config: "PipelineConfig | None" = None,
    data_dir: Path | None = None,
) -> None:
    """Fill in the JATIC metadata envelope from resolved source/extractor context."""
    from dataeval_flow import __version__

    result.metadata.dataset_id = dataset_names[0] if len(dataset_names) == 1 else ",".join(dataset_names)
    result.metadata.tool_version = __version__
    result.metadata.execution_time_s = round(elapsed, 3)

    # Source context — selection info
    selection_names = [s.selection for s in sources if s.selection is not None]
    if selection_names:
        result.metadata.selection_id = selection_names[0] if len(selection_names) == 1 else ",".join(selection_names)

    # Build human-readable source descriptions: "src_name (dataset[selection])"
    source_descs: list[str] = []
    for src in sources:
        if src.selection is not None:
            source_descs.append(f"{src.name} ({src.dataset}[{src.selection}])")
        else:
            source_descs.append(f"{src.name} ({src.dataset})")
    result.metadata.source_descriptions = source_descs

    # Extractor context — model + preprocessor info
    if extractor_cfg is not None:
        result.metadata.model_id = f"{extractor_cfg.name} ({extractor_cfg.model})"
        if extractor_cfg.preprocessor is not None:
            result.metadata.preprocessor_id = extractor_cfg.preprocessor

    # Annotate dataset source when label provenance is known
    dc = next(iter(dataset_contexts.values()))
    if dc.label_source:
        result.metadata.label_source = dc.label_source

    # Build fully resolved config snapshot for report traceability
    result.metadata.resolved_config = _build_resolved_config(
        sources, workflow_instance, extractor_cfg, pipeline_config, data_dir=data_dir
    )


def _build_resolved_config(
    sources: "Sequence[SourceConfig]",
    workflow_instance: "WorkflowConfig | None",
    extractor_cfg: Any,
    pipeline_config: "PipelineConfig | None",
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a fully resolved config dict for report traceability."""
    cfg: dict[str, Any] = {}

    # Sources — expand dataset and selection configs inline
    source_entries: list[dict[str, Any]] = []
    for src in sources:
        entry: dict[str, Any] = {"name": src.name, "dataset": src.dataset}
        if pipeline_config is not None:
            ds = _resolve_by_name(pipeline_config.datasets, src.dataset, "dataset")
            if getattr(ds, "serializable", True):
                entry["dataset_config"] = ds.model_dump(mode="json")
            else:
                dumped = ds.model_dump(mode="json", exclude={"dataset"})
                runtime_obj = getattr(ds, "dataset", None)
                dumped["dataset"] = {
                    "type": "protocol",
                    "class": type(runtime_obj).__qualname__ if runtime_obj is not None else "unknown",
                    "id": getattr(runtime_obj, "metadata", {}).get("id", "unknown"),
                }
                entry["dataset_config"] = dumped
        if src.selection is not None:
            entry["selection"] = src.selection
            if pipeline_config is not None:
                sel = _resolve_by_name(pipeline_config.selections, src.selection, "selection")
                entry["selection_config"] = sel.model_dump(mode="json")
        source_entries.append(entry)
    cfg["sources"] = source_entries

    # Workflow params
    if workflow_instance is not None:
        cfg["workflow"] = workflow_instance.model_dump(mode="json")

    # Extractor
    if extractor_cfg is not None:
        cfg["extractor"] = extractor_cfg.model_dump(mode="json")

    return _relativize_paths(cfg, root=data_dir)


def _relativize_paths(obj: Any, root: Path | None = None) -> Any:
    """Recursively convert absolute path strings to relative paths.

    Only strings that resolve to a path under *root* are relativized.
    If *root* is ``None``, the object is returned unchanged.
    """
    if root is None:
        return obj
    root = root.resolve()
    if isinstance(obj, dict):
        return {k: _relativize_paths(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relativize_paths(v, root) for v in obj]
    if isinstance(obj, str) and obj.startswith("/"):
        p = Path(obj)
        try:
            return str(p.relative_to(root))
        except ValueError:
            return obj
    return obj


def run_tasks(
    config: "PipelineConfig",
    tasks: str | Sequence[str] | None = None,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
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
    data_dir : Path | None
        Root directory for resolving relative paths in configs.
    cache_dir : Path | None
        Directory for disk-backed computation cache.

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
        results.append(_run_single_task(task, config, data_dir=data_dir, cache_dir=cache_dir))
    return results


@overload
def run_task(
    task: "DataAnalysisTaskConfig",
    config: "PipelineConfig",
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> "WorkflowResult[DataAnalysisMetadata, DataAnalysisOutputs]": ...
@overload
def run_task(
    task: "DataCleaningTaskConfig",
    config: "PipelineConfig",
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> "WorkflowResult[DataCleaningMetadata, DataCleaningOutputs]": ...
@overload
def run_task(
    task: "DriftMonitoringTaskConfig",
    config: "PipelineConfig",
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> "WorkflowResult[DriftMonitoringMetadata, DriftMonitoringOutputs]": ...
@overload
def run_task(
    task: "OODDetectionTaskConfig",
    config: "PipelineConfig",
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> "WorkflowResult[OODDetectionMetadata, OODDetectionOutputs]": ...
@overload
def run_task(
    task: "DataPrioritizationTaskConfig",
    config: "PipelineConfig",
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> "WorkflowResult[DataPrioritizationMetadata, DataPrioritizationOutputs]": ...
@overload
def run_task(
    task: "TaskConfig", config: "PipelineConfig", data_dir: Path | None = None, cache_dir: Path | None = None
) -> "WorkflowResult[Any, Any]": ...
def run_task(
    task: "TaskConfig", config: "PipelineConfig", data_dir: Path | None = None, cache_dir: Path | None = None
) -> "WorkflowResult[Any, Any]":
    """Run a single task, returning a narrowly typed result based on the task type.

    Unlike :func:`run_tasks`, this function accepts the task config object
    directly rather than looking it up by name, which allows type checkers to
    narrow the return type to the appropriate workflow result type.

    Parameters
    ----------
    task : TaskConfig
        The task configuration to execute.
    config : PipelineConfig
        Pipeline configuration supplying datasets, sources, extractors, and
        workflow definitions.  The task does **not** need to appear in
        ``config.tasks``.
    data_dir : Path | None
        Root directory for resolving relative paths in configs.
    cache_dir : Path | None
        Directory for disk-backed computation cache.

    Returns
    -------
    WorkflowResult
        A result typed to the specific workflow — e.g.
        ``WorkflowResult[OODDetectionMetadata, OODDetectionOutputs]`` when
        *task* is an :class:`~dataeval_flow.config.OODDetectionTaskConfig`.
    """
    logger.info("--- Task: %s (workflow: %s) ---", task.name, task.workflow)
    return _run_single_task(task, config, data_dir=data_dir, cache_dir=cache_dir)
