"""Task orchestration — config → execution bridge."""

__all__ = ["run_task", "run_tasks", "select_tasks"]

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload, runtime_checkable

from pydantic import BaseModel

from dataeval_flow._logging import capture_diagnostics

_logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dataeval_flow.config import PipelineConfig, TaskConfig
    from dataeval_flow.config.schemas import WorkflowConfig
    from dataeval_flow.config.schemas._task import (
        DataAnalysisTaskConfig,
        DataCleaningTaskConfig,
        DataCoverageTaskConfig,
        DataPrioritizationTaskConfig,
        DriftMonitoringTaskConfig,
        OODDetectionTaskConfig,
        ParameterSweepTaskConfig,
    )
    from dataeval_flow.policy import ResolvedPolicy
    from dataeval_flow.sources import ResolvedSource, SourceOperand
    from dataeval_flow.workflow import DatasetContext, ResolvedOntology, WorkflowResult
    from dataeval_flow.workflows.analysis.outputs import DataAnalysisMetadata, DataAnalysisOutputs
    from dataeval_flow.workflows.cleaning.outputs import DataCleaningMetadata, DataCleaningOutputs
    from dataeval_flow.workflows.coverage.outputs import DataCoverageMetadata, DataCoverageOutputs
    from dataeval_flow.workflows.drift.outputs import DriftMonitoringMetadata, DriftMonitoringOutputs
    from dataeval_flow.workflows.ood.outputs import OODDetectionMetadata, OODDetectionOutputs
    from dataeval_flow.workflows.parameter_sweep.outputs import (
        ParameterSweepMetadata,
        ParameterSweepOutputs,
    )
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


def _resolve_metadata_policy(
    instance: "WorkflowConfig",
    config: "PipelineConfig",
    data_dir: Path | None,
) -> "ResolvedPolicy | None":
    """Resolve the metadata policy for one workflow, or None where it reads no metadata.

    Kept here rather than inside the workflows because resolving it needs the pipeline the
    policy pool lives on and the data root a descriptor path is relative to, and because
    every check it runs is worth running before the dataset is walked.
    """
    from dataeval_flow.policy import resolve_policy
    from dataeval_flow.workflow.base import MetadataConfigMixin

    if not isinstance(instance, MetadataConfigMixin):
        return None
    return resolve_policy(instance, config, data_dir)


def _apply_dataset_value_range(
    policy: "ResolvedPolicy | None",
    ranges: "Sequence[tuple[float, float] | None]",
    workflow_name: str,
) -> "ResolvedPolicy | None":
    """Stamp the datasets' declared value range onto the resolved policy.

    Authored on the dataset because it describes the imagery, carried on the policy because
    it changes the injected values and therefore the codes.  This is also what puts it in
    ``policy_key``, without which two runs over different ranges would share one metadata
    archive holding different numbers.

    Raises
    ------
    ValueError
        When two datasets in one workflow declare different ranges.  Statistics compared
        across incompatible pixel scales are not comparable, and refusing here costs a
        config error rather than an hour of walking images.
    """
    from dataclasses import replace

    declared = sorted({value for value in ranges if value is not None})
    if len(declared) > 1:
        raise ValueError(
            f"Workflow {workflow_name!r} reads datasets declaring different `value_range`s "
            f"({declared[0]} and {declared[1]}). Statistics measured on different pixel "
            "scales are not comparable, so there is no right answer to pick — give the "
            "datasets one range, or run them as separate workflows.",
        )
    if not declared or policy is None:
        return policy
    return replace(policy, value_range=declared[0])


def _value_range_of(resolved: "ResolvedSource") -> "tuple[float, float] | None":
    """Return the value range *resolved* declares, or None where no operand declares one.

    Raises
    ------
    ValueError
        When two operands of one source declare different ranges.  Refuse here rather
        than dropping to None: the range is what the statistics are measured against and
        what keys their cache, so an undeclared range would answer NaN and share an
        archive with the other scale.
    """
    ranges = [getattr(operand.dataset_config, "value_range", None) for operand in resolved.operands]
    declared = sorted({value for value in ranges if value is not None})
    if len(declared) > 1:
        raise ValueError(
            f"Source {resolved.name!r} merges datasets declaring different `value_range`s "
            f"({declared[0]} and {declared[1]}). Statistics measured on different pixel "
            "scales are not comparable, so there is no right answer to pick — give the "
            "datasets one range, or do not merge them.",
        )
    return declared[0] if declared else None


def _label_source_of(label_sources: "Sequence[str | None]") -> "str | Sequence[str] | None":
    """Return where a corpus's labels came from, given each operand's provenance.

    Report None where no operand knows its provenance, and the shared value where every
    operand reports the same one.  Otherwise report one entry per operand in merge order,
    writing an unknown provenance as "unknown": a corpus read from two provenances has two
    answers, and reporting one of them hides the other.
    """
    distinct = set(label_sources)
    if not distinct or distinct == {None}:
        return None
    if len(distinct) == 1:
        return next(iter(distinct))
    return [source or "unknown" for source in label_sources]


def _resolve_ontology(
    instance: Any,
    config: "PipelineConfig | None",
    data_dir: Path | None,
) -> "ResolvedOntology | None":
    """Resolve the task's ontology up front. Return any failure rather than raising it.

    Resolve here for the same reason as the metadata policy: a name needs the pipeline's
    pool and a path needs the data root. Return the failure instead of raising it, because
    ``data-coverage`` degrades on an ontology problem by contract and moving the work
    earlier must not change that.
    """
    from dataeval_flow.workflow import ResolvedOntology
    from dataeval_flow.workflows._ontology import OntologyLoadError, resolve_ontology

    spec = getattr(instance, "ontology", None)
    if spec is None:
        return None

    pool = getattr(config, "ontologies", None) if config is not None else None
    try:
        ontology, source = resolve_ontology(spec, pool, data_dir=data_dir)
    except OntologyLoadError as exc:
        _logger.warning("Task ontology could not be resolved: %s", exc)
        return ResolvedOntology(ontology=None, source=str(spec), error=str(exc))
    return ResolvedOntology(ontology=ontology, source=source)


E = TypeVar("E", bound=BaseModel)


def _resolve_extractor_paths(extractor_cfg: E, data_dir: Path | None) -> E:
    """Resolve relative ``model_path`` on extractor configs against *data_dir*."""
    model_path: str | None = getattr(extractor_cfg, "model_path", None)

    if model_path is not None:
        from dataeval_flow.config._loader import resolve_path

        # Models default to the `models` folder of the input mount.
        resolved = str(resolve_path(model_path, data_dir, default_subdir="models"))
        if resolved != model_path:
            return extractor_cfg.model_copy(update={"model_path": resolved})

    return extractor_cfg


def _apply_seed(config: "PipelineConfig") -> None:
    """Apply the pipeline's seed through DataEval's seed configuration [CR-7-S-1].

    Delegates to :func:`dataeval.config.set_seed` so DataEval's evaluators and the
    NumPy/PyTorch global generators are all pinned by the same value — a partial
    seeding would leave clustering or sampling free to vary between runs.

    A ``seed`` of ``None`` is a no-op: it leaves whatever randomness state the
    process already has rather than actively reseeding it.
    """
    if config.seed is None:
        return

    from dataeval.config import set_seed

    set_seed(config.seed, all_generators=True, deterministic=config.deterministic)
    _logger.info(
        "Seeded run with seed=%d (deterministic=%s)",
        config.seed,
        config.deterministic,
    )


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
    from dataeval_flow.config.schemas import ExtractorConfig, PreprocessorConfig
    from dataeval_flow.preprocessing import build_preprocessing
    from dataeval_flow.sources import resolve_source
    from dataeval_flow.workflow import DatasetContext, WorkflowContext, get_workflow

    _logger.info("Task '%s': starting (workflow_instance=%s)", task.name, task.workflow)

    # 0. Seed every stochastic component [CR-7-S-1]. Applied per task rather than
    #    once per pipeline so a task's result does not depend on what ran before it.
    _apply_seed(config)

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
    resolved_sources: list[ResolvedSource] = []

    for src_name in source_names:
        resolved = resolve_source(src_name, config, data_dir=data_dir)
        resolved_sources.append(resolved)

        ds_cache = DatasetCache.get_or_create(
            cache_dir=cache_dir,
            name=resolved.cache_name,
            cache_key=resolved.cache_key,
        )

        dataset_contexts[src_name] = DatasetContext(
            name=src_name,
            dataset=resolved.dataset,
            extractor=extractor_cfg,
            transforms=transforms,
            view_operations=resolved.view_config.operations if resolved.view_config else None,
            batch_size=batch_size,
            label_source=_label_source_of(resolved.label_sources),
            value_range=_value_range_of(resolved),
            cache=ds_cache,
        )

    if cache_dir:
        _logger.info("Cache enabled: %s", cache_dir)

    _logger.debug("Task '%s': resolved %d source(s): %s", task.name, len(source_names), source_names)

    # 4. Resolve workflow → type + params
    instance = _resolve_workflow(task.workflow, config)
    workflow = get_workflow(instance.type)

    # 5. Resolve the metadata policy before anything reads the dataset, so a misspelled
    #    factor or a descriptor that does not exist costs a config error rather than an
    #    hour of walking images.  Workflows that read no metadata simply carry None.
    policy = _resolve_metadata_policy(instance, config, data_dir)

    # Before anything reads the dataset: the range is what the injected statistics are
    # measured against, and a disagreement between datasets has no sound resolution.
    policy = _apply_dataset_value_range(
        policy,
        [ctx.value_range for ctx in dataset_contexts.values()],
        instance.name,
    )

    ontology = _resolve_ontology(instance, config, data_dir)

    # 6. Build WorkflowContext
    context = WorkflowContext(
        dataset_contexts=dataset_contexts,
        batch_size=batch_size,
        metadata_policy=policy,
        ontology=ontology,
    )

    # 7. Run workflow with timing
    _logger.debug("Task '%s': executing workflow", task.name)
    start = time.monotonic()
    # Library diagnostics are captured here rather than left to the log file:
    # they name the binning and value_range decisions this run made, and the
    # envelope has to be able to answer for them on its own.
    with capture_diagnostics() as diagnostics:
        result = workflow.execute(context, instance)
    elapsed = time.monotonic() - start
    if diagnostics:
        result.metadata.diagnostics = list(diagnostics)
    _logger.info("Task '%s': finished in %.1fs (success=%s)", task.name, elapsed, result.success)

    # 8. Backfill the resolved dataset(s) when the workflow left them unset —
    # notably on failure paths, where callers still need the inputs to debug.
    _ensure_result_datasets(result, dataset_contexts)

    # 9. Populate metadata envelope
    _populate_result_metadata(
        result,
        resolved_sources,
        extractor_cfg,
        elapsed,
        instance,
        config,
        data_dir=data_dir,
        ontology=ontology,
    )

    return result


def _ensure_result_datasets(
    result: "WorkflowResult[Any, Any]",
    dataset_contexts: "Mapping[str, DatasetContext]",
) -> None:
    """Fill in ``result.dataset`` / ``result.sources`` when the workflow did not.

    Workflows attach the resolved, post-selection dataset to successful
    results only — an early return or an exception leaves the fields unset,
    which strands callers that want to inspect the inputs that produced the
    failure.  Rebuild the selection here for those cases; already-populated
    fields are left untouched, so the success path is unaffected.
    """
    if not dataset_contexts:
        return

    single = len(dataset_contexts) == 1
    needs_dataset = single and result.dataset is None
    needs_sources = not single and result.sources is None
    if not (needs_dataset or needs_sources):
        return

    from dataeval_flow.view import build_view

    resolved: dict[str, Any] = {}
    for name, dc in dataset_contexts.items():
        dataset = dc.dataset
        if dc.view_operations:
            try:
                dataset = build_view(dataset, list(dc.view_operations))
            except Exception:
                _logger.debug("Could not rebuild selection for source '%s'", name, exc_info=True)
        resolved[name] = dataset

    if needs_dataset:
        result.dataset = next(iter(resolved.values()))
    else:
        result.sources = resolved


def _populate_result_metadata(
    result: "WorkflowResult[Any, Any]",
    resolved_sources: "Sequence[ResolvedSource]",
    extractor_cfg: Any,
    elapsed: float,
    workflow_instance: "WorkflowConfig | None" = None,
    pipeline_config: "PipelineConfig | None" = None,
    data_dir: Path | None = None,
    ontology: "ResolvedOntology | None" = None,
) -> None:
    """Fill in the JATIC metadata envelope from resolved source/extractor context."""
    from dataeval_flow import __version__
    from dataeval_flow.sources import label_space_records

    dataset_names = [
        operand.source.dataset for rs in resolved_sources for operand in rs.operands if operand.source.dataset
    ]
    result.metadata.dataset_id = dataset_names[0] if len(dataset_names) == 1 else ",".join(dataset_names)
    result.metadata.tool_version = __version__
    result.metadata.execution_time_s = round(elapsed, 3)

    # Every view a source reads through, operands first. A merge's conform views define
    # the label space, so leaving them out would drop what the result was read under.
    view_names: list[str] = []
    for rs in resolved_sources:
        view_names.extend(operand.view_config.name for operand in rs.operands if operand.view_config is not None)
        if rs.is_merged and rs.view_config is not None:
            view_names.append(rs.view_config.name)
    if view_names:
        result.metadata.selection_id = view_names[0] if len(view_names) == 1 else ",".join(view_names)

    result.metadata.source_descriptions = [_source_description(rs) for rs in resolved_sources]

    if extractor_cfg is not None:
        result.metadata.model_id = f"{extractor_cfg.name} ({extractor_cfg.model})"
        if extractor_cfg.preprocessor is not None:
            result.metadata.preprocessor_id = extractor_cfg.preprocessor

    label_source = _label_source_of([ls for rs in resolved_sources for ls in rs.label_sources])
    if label_source is not None:
        result.metadata.label_source = label_source

    records = label_space_records(resolved_sources, ontology)
    if records:
        result.metadata.label_space = records
        digests = {record.digest for record in records}
        # Set the scalar only where the run read one vocabulary. A workflow that stamped
        # its own — the coverage audit does — keeps it.
        if len(digests) == 1 and not result.metadata.label_space_digest:
            result.metadata.label_space_digest = records[0].digest

    result.metadata.resolved_config = _build_resolved_config(
        resolved_sources, workflow_instance, extractor_cfg, pipeline_config, data_dir=data_dir
    )


def _operand_description(operand: "SourceOperand") -> str:
    """Render one operand as `dataset[view]`, or `dataset` where it reads no view."""
    if operand.view_config is None:
        return str(operand.source.dataset)
    return f"{operand.source.dataset}[{operand.view_config.name}]"


def _source_description(resolved: "ResolvedSource") -> str:
    """Render one source for the report's Source line.

    A merged source spells out its operands, because the datasets it reads and the views
    that conformed them are what a reader needs and the source name alone hides both.
    """
    if not resolved.is_merged:
        return f"{resolved.name} ({_operand_description(resolved.operands[0])})"
    parts = " + ".join(_operand_description(operand) for operand in resolved.operands)
    own_view = f"[{resolved.view_config.name}]" if resolved.view_config is not None else ""
    return f"{resolved.name} (merge: {parts}){own_view}"


def _build_resolved_config(
    resolved_sources: "Sequence[ResolvedSource]",
    workflow_instance: "WorkflowConfig | None",
    extractor_cfg: Any,
    pipeline_config: "PipelineConfig | None",
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a fully resolved config dict for report traceability."""
    cfg: dict[str, Any] = {"sources": [_source_entry(rs) for rs in resolved_sources]}

    if workflow_instance is not None:
        cfg["workflow"] = workflow_instance.model_dump(mode="json")

    if extractor_cfg is not None:
        cfg["extractor"] = extractor_cfg.model_dump(mode="json")

    # Reproducibility — recorded so the envelope alone is enough to repeat the run.
    if pipeline_config is not None and pipeline_config.seed is not None:
        cfg["seed"] = pipeline_config.seed
        cfg["deterministic"] = pipeline_config.deterministic

    return _relativize_paths(cfg, root=data_dir)


def _source_entry(resolved: "ResolvedSource") -> dict[str, Any]:
    """Expand one source, recursing into a merge's operands.

    Recurse so a merged run is replayable. `view_config` carries each operand's `Relabel`
    verbatim, so the conformed vocabulary lands here once the walk reaches it.
    """
    if not resolved.is_merged:
        return {"name": resolved.name, **_operand_entry(resolved.operands[0])}

    entry: dict[str, Any] = {
        "name": resolved.name,
        "merge": [_operand_entry(operand) for operand in resolved.operands],
    }
    if resolved.view_config is not None:
        entry["view"] = resolved.view_config.name
        entry["view_config"] = resolved.view_config.model_dump(mode="json")
    return entry


def _operand_entry(operand: "SourceOperand") -> dict[str, Any]:
    """Expand one leaf source's dataset and view configs inline."""
    entry: dict[str, Any] = {"dataset": operand.source.dataset}
    ds = operand.dataset_config
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
    if operand.view_config is not None:
        entry["view"] = operand.view_config.name
        entry["view_config"] = operand.view_config.model_dump(mode="json")
    return entry


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


def select_tasks(
    config: "PipelineConfig",
    tasks: str | Sequence[str] | None = None,
) -> "list[TaskConfig]":
    """Resolve which of a config's tasks a run executes, in execution order.

    Callers that pair results back to the tasks that produced them go through this
    rather than re-deriving the selection.  ``run_tasks`` returns one result per
    *executed* task, and a result carries its workflow type rather than its task
    name, so anything zipping results against ``config.tasks`` misaligns the moment
    a task is disabled.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration holding the task list.
    tasks : str | Sequence[str] | None
        Which tasks to select:

        - ``None`` (default) — every enabled task, in config order
        - ``str`` — a single task by name
        - ``Sequence[str]`` — specific tasks by name, in the given order

        Naming a task selects it whether or not it is enabled: an explicit
        request outranks the config's default.

    Returns
    -------
    list[TaskConfig]
        The tasks to run, in execution order.

    Raises
    ------
    ValueError
        If no tasks are defined, all are disabled, or a named task is not found.
    """
    if not config.tasks:
        raise ValueError("No tasks defined in pipeline config")

    if tasks is None:
        to_run = [t for t in config.tasks if t.enabled]
        skipped = len(config.tasks) - len(to_run)
        if skipped:
            _logger.info("Skipping %d disabled task(s)", skipped)
        if not to_run:
            raise ValueError("All tasks are disabled — nothing to run")
        return to_run

    if isinstance(tasks, str):
        return [_resolve_by_name(config.tasks, tasks, "task")]
    return [_resolve_by_name(config.tasks, name, "task") for name in tasks]


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
    to_run = select_tasks(config, tasks)

    _logger.info("Running %d task(s)", len(to_run))
    results: list[WorkflowResult[Any, Any]] = []
    for task in to_run:
        _logger.info("--- Task: %s (workflow: %s) ---", task.name, task.workflow)
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
    task: "ParameterSweepTaskConfig",
    config: "PipelineConfig",
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> "WorkflowResult[ParameterSweepMetadata, ParameterSweepOutputs]": ...
@overload
def run_task(
    task: "DataCoverageTaskConfig",
    config: "PipelineConfig",
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> "WorkflowResult[DataCoverageMetadata, DataCoverageOutputs]": ...
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
    _logger.info("--- Task: %s (workflow: %s) ---", task.name, task.workflow)
    return _run_single_task(task, config, data_dir=data_dir, cache_dir=cache_dir)
