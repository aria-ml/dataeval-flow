"""Task orchestration — config → execution bridge."""

__all__ = ["run_task"]

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, overload, runtime_checkable

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from dataeval_flow.config.models import WorkflowConfig
    from dataeval_flow.config.schemas.metadata import ResultMetadata
    from dataeval_flow.config.schemas.task import DataCleaningTaskConfig, DriftMonitoringTaskConfig, TaskConfig
    from dataeval_flow.workflow import DatasetContext, WorkflowResult
    from dataeval_flow.workflows.cleaning.outputs import DataCleaningResult
    from dataeval_flow.workflows.drift.outputs import DriftMonitoringResult


@runtime_checkable
class _Named(Protocol):
    """Protocol for config objects with a name attribute."""

    name: str


T = TypeVar("T", bound=_Named)


def _resolve_by_name(items: list[T] | None, name: str, kind: str) -> T:
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


def _resolve_optional_mapping(
    mapping_or_name: str | Mapping[str, str] | None,
    dataset_name: str,
    items: list[T] | None,
    kind: str,
) -> T | None:
    """Resolve a config reference that may be shared or per-dataset.

    Parameters
    ----------
    mapping_or_name : str | Mapping[str, str] | None
        - ``None`` → return ``None`` (not configured)
        - ``str`` → shared across all datasets, resolve by name
        - ``Mapping`` → per-dataset; look up *dataset_name* key
    dataset_name : str
        Current dataset name (used as key into mapping).
    items : list[T] | None
        Config list to resolve names against.
    kind : str
        Human-readable kind for error messages.

    Returns
    -------
    T | None
        Resolved config object, or ``None`` if not applicable.
    """
    if mapping_or_name is None:
        return None
    if isinstance(mapping_or_name, str):
        return _resolve_by_name(items, mapping_or_name, kind)
    # Mapping path — missing key means this dataset has no config
    name = mapping_or_name.get(dataset_name)
    if name is None:
        return None
    return _resolve_by_name(items, name, kind)


def _validate_mapping_keys(
    mapping: Mapping[str, str],
    dataset_names: list[str],
    kind: str,
) -> None:
    """Raise ValueError if mapping references unknown dataset names.

    Parameters
    ----------
    mapping : Mapping[str, str]
        Per-dataset config mapping to validate.
    dataset_names : list[str]
        Valid dataset names.
    kind : str
        Human-readable kind for error messages.
    """
    extra = set(mapping.keys()) - set(dataset_names)
    if extra:
        raise ValueError(f"{kind} mapping references unknown datasets: {sorted(extra)}")


@overload
def run_task(  # pyright: ignore[reportOverlappingOverload]
    task: "DriftMonitoringTaskConfig", config: "WorkflowConfig"
) -> "DriftMonitoringResult": ...
@overload
def run_task(  # pyright: ignore[reportOverlappingOverload]
    task: "DataCleaningTaskConfig", config: "WorkflowConfig"
) -> "DataCleaningResult": ...
@overload
def run_task(task: "TaskConfig", config: "WorkflowConfig") -> "WorkflowResult[ResultMetadata, BaseModel]": ...


def run_task(task: "TaskConfig", config: "WorkflowConfig") -> "WorkflowResult[Any, Any]":
    """Run a single task using config-driven resolution.

    This is the primary entry point for config-driven execution.
    Resolves all ``TaskConfig`` references (datasets, model, preprocessor,
    selection) against ``WorkflowConfig`` lists, builds a
    ``WorkflowContext``, validates params, and runs the workflow.

    Supports both single-dataset (``datasets`` as ``str``) and
    multi-dataset (``datasets`` as ``list[str]``) modes. In
    multi-dataset mode, model/preprocessor/selection may be a
    ``Mapping[str, str]`` for per-dataset resolution.

    Parameters
    ----------
    task : TaskConfig
        Task to execute. References are resolved against *config*.
    config : WorkflowConfig
        Full workflow configuration containing datasets, models, etc.

    Returns
    -------
    WorkflowResult
        Result from the workflow execution.

    Raises
    ------
    ValueError
        If a referenced config (dataset, model, etc.) is not found,
        or if a mapping references unknown dataset names.
    """
    from dataeval_flow.cache import DatasetCache
    from dataeval_flow.config.models import ModelConfig
    from dataeval_flow.config.schemas import (
        PreprocessorConfig,
        SelectionConfig,
    )
    from dataeval_flow.dataset import resolve_dataset
    from dataeval_flow.preprocessing import build_preprocessing
    from dataeval_flow.workflow import DatasetContext, WorkflowContext, get_workflow

    logger.info("Task '%s': starting (workflow=%s)", task.name, task.workflow)

    # 1. Normalize datasets to list
    dataset_names: list[str] = [task.datasets] if isinstance(task.datasets, str) else list(task.datasets)

    # 2. Validate mapping keys (if any config is a mapping)
    for attr, kind in [("models", "models"), ("preprocessors", "preprocessors"), ("selections", "selections")]:
        value = getattr(task, attr)
        if isinstance(value, Mapping):
            _validate_mapping_keys(value, dataset_names, kind)

    # 3. Build a DatasetContext per dataset (cache is per-dataset so that
    #    different workflows sharing the same dataset can reuse cached results)
    dataset_contexts: dict[str, DatasetContext] = {}
    for ds_name in dataset_names:
        ds_config = _resolve_by_name(config.datasets, ds_name, "datasets")
        resolved = resolve_dataset(ds_config)

        # Resolve model → extractor (optional)
        extractor_config = None
        model_config_obj: ModelConfig | None = _resolve_optional_mapping(task.models, ds_name, config.models, "models")
        if model_config_obj is not None:
            extractor_config = model_config_obj.extractor

        # Resolve preprocessor → transforms (optional)
        transforms = None
        pre_config: PreprocessorConfig | None = _resolve_optional_mapping(
            task.preprocessors, ds_name, config.preprocessors, "preprocessor"
        )
        if pre_config is not None:
            transforms = build_preprocessing(pre_config.steps)

        # Resolve selection → steps (optional)
        selection_steps = None
        sel_config: SelectionConfig | None = _resolve_optional_mapping(
            task.selections, ds_name, config.selections, "selection"
        )
        if sel_config is not None:
            selection_steps = sel_config.steps

        # Build per-dataset cache (keyed to this dataset's config only).
        # Always instantiated — disk-backed when cache_dir is set,
        # memory-only otherwise (singleton via get_or_create).
        cache_path = Path(task.cache_dir) if task.cache_dir else None
        ds_cache = DatasetCache.get_or_create(
            cache_dir=cache_path,
            name=resolved.name,
            cache_key=resolved.cache_key,
        )

        dataset_contexts[ds_name] = DatasetContext(
            name=ds_name,
            dataset=resolved.dataset,
            extractor=extractor_config,
            transforms=transforms,
            selection_steps=selection_steps,
            batch_size=task.batch_size,
            label_source=resolved.label_source,
            cache=ds_cache,
        )

    if task.cache_dir:
        logger.info("Cache enabled: %s", Path(task.cache_dir))

    # 4. Build WorkflowContext (metadata config is task-wide, not per-dataset)
    context = WorkflowContext(
        dataset_contexts=dataset_contexts,
        metadata_auto_bin_method=task.metadata_auto_bin_method,
        metadata_exclude=task.metadata_exclude or [],
        metadata_continuous_factor_bins=task.metadata_continuous_factor_bins,
        batch_size=task.batch_size,
    )

    logger.debug("Task '%s': resolved %d dataset(s): %s", task.name, len(dataset_names), dataset_names)

    # 5. Validate params against workflow's params_schema
    workflow = get_workflow(task.workflow)
    params = None
    if workflow.params_schema is not None:
        params = workflow.params_schema.model_validate(task.params)

    # 6. Run workflow with timing
    import time

    logger.debug("Task '%s': executing workflow", task.name)
    start = time.monotonic()
    result = workflow.execute(context, params)
    elapsed = time.monotonic() - start
    logger.info("Task '%s': finished in %.1fs (success=%s)", task.name, elapsed, result.success)

    # 7. Populate metadata envelope (JATIC fields + timing)
    _populate_result_metadata(result, dataset_names, dataset_contexts, task, task.output_format, elapsed)

    return result


def _populate_result_metadata(
    result: "WorkflowResult[Any, Any]",
    dataset_names: list[str],
    dataset_contexts: "dict[str, DatasetContext]",
    task: "TaskConfig",
    output_format: "Literal['text', 'json', 'yaml']",
    elapsed: float,
) -> None:
    """Fill in the JATIC metadata envelope and dataset source annotation."""
    from dataeval_flow import __version__

    result.metadata.dataset_id = dataset_names[0] if len(dataset_names) == 1 else ",".join(dataset_names)
    result.metadata.tool_version = __version__
    result.metadata.execution_time_s = round(elapsed, 3)
    result.format = output_format

    # Config reference names
    if task.models is not None:
        result.metadata.model_id = task.models if isinstance(task.models, str) else ", ".join(task.models.values())
    if task.preprocessors is not None:
        result.metadata.preprocessor_id = (
            task.preprocessors if isinstance(task.preprocessors, str) else ", ".join(task.preprocessors.values())
        )
    if task.selections is not None:
        result.metadata.selection_id = (
            task.selections if isinstance(task.selections, str) else ", ".join(task.selections.values())
        )

    # Annotate dataset source when label provenance is known
    dc = next(iter(dataset_contexts.values()))
    if dc.label_source:
        result.metadata.label_source = dc.label_source
