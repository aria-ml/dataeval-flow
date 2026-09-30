"""Task orchestration — config → execution bridge."""

__all__ = ["run_task", "run_tasks", "select_tasks"]

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from dataeval_flow._embeddings import shared_extractor_scope
from dataeval_flow._logging import capture_diagnostics
from dataeval_flow.steps._workflow import CustomWorkflowConfig

_logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dataeval_flow._chain._run import ChainRun, ExtractorSetup
    from dataeval_flow._policy import ResolvedPolicy
    from dataeval_flow._result import Result
    from dataeval_flow._sources import ResolvedSource, SourceOperand
    from dataeval_flow._stats import ResolvedStatsPolicy
    from dataeval_flow._tables import TableLimits
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.config._schemas._task import TaskConfig
    from dataeval_flow.evaluators._base import EvaluatorConfig
    from dataeval_flow.evaluators._evaluator import Evaluator
    from dataeval_flow.steps._result import ChainResult
    from dataeval_flow.workflows._base import Workflow, WorkflowConfig
    from dataeval_flow.workflows._context import DatasetContext, ResolvedOntology, WorkflowContext


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
) -> "WorkflowConfig[Any] | CustomWorkflowConfig":
    """Resolve a workflow by name from ``config.workflows``: a workflow type, or a custom workflow's steps."""
    return _resolve_by_name(config.workflows, workflow_name, "workflow")


def _resolve_evaluator(
    evaluator_name: str,
    config: "PipelineConfig",
) -> "EvaluatorConfig[Any]":
    """Resolve an evaluator by name from ``config.evaluators``."""
    return _resolve_by_name(config.evaluators, evaluator_name, "evaluator")


def _implementation(config: "WorkflowConfig[Any] | EvaluatorConfig[Any]") -> "Workflow[Any, Any] | Evaluator[Any, Any]":
    """A fresh instance of the workflow or evaluator registered under ``config.type``."""
    from dataeval_flow.evaluators._base import EvaluatorConfig
    from dataeval_flow.evaluators._registry import get_evaluator
    from dataeval_flow.workflows._registry import get_workflow

    if isinstance(config, EvaluatorConfig):
        return get_evaluator(config.type)()
    return get_workflow(config.type)()


def _target_of(task: "TaskConfig") -> str:
    """What a task runs, as log lines name it: the way its config file does."""
    return f"{task.kind}: {task.workflow}"


def _resolve_metadata_policy(
    instance: "BaseModel",
    config: "PipelineConfig",
    data_dir: Path | None,
) -> "ResolvedPolicy | None":
    """Resolve the metadata policy for one workflow, or None where it reads no metadata.

    Resolving needs the pipeline the policy pool lives on and the data root a descriptor
    path is relative to, so it stays here, outside the workflows. Its checks run before
    the dataset is walked.
    """
    from dataeval_flow._policy import resolve_policy
    from dataeval_flow.config._schemas._mixins import MetadataConfigMixin

    if not isinstance(instance, MetadataConfigMixin):
        return None
    return resolve_policy(instance, config, data_dir)


def _apply_dataset_value_range(
    policy: "ResolvedPolicy | None",
    ranges: "Sequence[tuple[float, float] | None]",
    target_name: str,
    target_kind: str = "workflow",
) -> "ResolvedPolicy | None":
    """Stamp the datasets' declared value range onto the resolved policy.

    Authored on the dataset because it describes the imagery, carried on the policy because
    it changes the injected values and therefore the codes.  It also goes into
    ``policy_key``, so two runs over different ranges do not share one metadata archive
    holding different numbers.

    Parameters
    ----------
    target_kind : str
        What *target_name* names — ``"workflow"`` or ``"evaluator"`` — so the error names the
        task's actual target instead of always calling it a workflow.

    Raises
    ------
    ValueError
        When two datasets in one task declare different ranges.  Statistics across
        incompatible pixel scales are not comparable; failing here costs a config error,
        not an hour of walking images.
    """
    declared = sorted({value for value in ranges if value is not None})
    if len(declared) > 1:
        raise ValueError(
            f"{target_kind.capitalize()} {target_name!r} reads datasets declaring different `value_range`s "
            f"({declared[0]} and {declared[1]}). Statistics measured on different pixel "
            "scales are not comparable, so there is no right answer to pick — give the "
            "datasets one range, or run them as separate tasks.",
        )
    if not declared or policy is None:
        return policy
    return replace(policy, value_range=declared[0])


def _value_range_of(resolved: "ResolvedSource") -> "tuple[float, float] | None":
    """Return the value range *resolved* declares, or None where no operand declares one.

    Raises
    ------
    ValueError
        When two operands of one source declare different ranges.  The range is what the
        statistics are measured against and what keys their cache; an undeclared range
        would answer NaN and share an archive with the other scale.
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


def _merge_channel_groups(
    declarations: "Iterable[Mapping[str, Any] | None]",
    subject: str,
) -> "Mapping[str, tuple[int, ...]] | None":
    """Union declared band groups, refusing two definitions of one name.

    *subject* names what is being merged, for the error.

    Raises
    ------
    ValueError
        When one group name is given different bands. `ir_mean` measured over different
        bands is not one statistic, and the merged column would hold both.
    """
    merged: dict[str, tuple[int, ...]] = {}
    for declared in declarations:
        for name, bands in (declared or {}).items():
            indices = (bands,) if isinstance(bands, int) else tuple(bands)
            existing = merged.get(name)
            if existing is not None and existing != indices:
                raise ValueError(
                    f"{subject} declares different bands for channel group {name!r} "
                    f"({list(existing)} and {list(indices)}). One column name means one "
                    "measurement, so there is no right answer to pick — give them one "
                    "definition, or rename one group.",
                )
            merged[name] = indices
    return merged or None


def _channel_groups_of(resolved: "ResolvedSource") -> "Mapping[str, tuple[int, ...]] | None":
    """Return the band groups *resolved* declares, or None where no operand declares any."""
    return _merge_channel_groups(
        (getattr(operand.dataset_config, "channel_groups", None) for operand in resolved.operands),
        f"Source {resolved.name!r}",
    )


def _channel_groups_for(
    dataset_contexts: "Mapping[str, DatasetContext]",
) -> "Mapping[str, tuple[int, ...]] | None":
    """Return the band groups every dataset this workflow reads declares."""
    return _merge_channel_groups(
        (ctx.channel_groups for ctx in dataset_contexts.values()),
        "This workflow's datasets",
    )


def _resolve_stats_policy(
    instance: BaseModel,
    config: "PipelineConfig",
    dataset_contexts: "Mapping[str, DatasetContext]",
) -> "ResolvedStatsPolicy | None":
    """Resolve the stats policy for one workflow, or None where it computes no statistics.

    Kept here rather than inside the workflows because resolving it needs the pipeline the
    pool lives on and the datasets that declare the bands, and because every check it runs
    is worth running before the dataset is walked.
    """
    from dataeval_flow._stats import resolve_stats_policy
    from dataeval_flow.config._schemas._mixins import StatsConfigMixin

    if not isinstance(instance, StatsConfigMixin):
        return None
    return resolve_stats_policy(instance, config, _channel_groups_for(dataset_contexts))


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
    instance: "BaseModel",
    config: "PipelineConfig | None",
    data_dir: Path | None,
) -> "ResolvedOntology | None":
    """Resolve the task's ontology up front. Return any failure rather than raising it.

    Resolve here for the same reason as the metadata policy: a name needs the pipeline's
    pool and a path needs the data root. Return the failure instead of raising it, because
    ``data-coverage`` degrades on an ontology problem by contract and moving the work
    earlier must not change that.
    """
    from dataeval_flow.workflows._context import ResolvedOntology
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
    *,
    report_images: bool = True,
    output_dir: Path | None = None,
) -> "Result[Any, Any]":
    """Run a single resolved task against a pipeline config.

    This is the internal workhorse — resolves all references (sources,
    extractor) against ``PipelineConfig``, builds contexts, and executes
    the workflow or evaluator through the step engine: as a one-step graph, or as a custom workflow's chain.

    Resolving raises on a config error: a reference that names nothing, a dataset that cannot load, a
    preprocessing step whose transform cannot be built, a policy that does not resolve. A task that does not meet
    its target's inputs, and any failure of the run itself, come back as a failed result instead.

    `output_dir` is where export steps write, under ``<output_dir>/datasets/``. ``None`` writes nothing, and export
    steps are skipped with a reason.
    """
    from dataeval_flow._kind import input_problem, result_type_of
    from dataeval_flow._tables import TableLimits
    from dataeval_flow.evaluators._result import EvaluatorResult
    from dataeval_flow.workflows._base import WorkflowConfig
    from dataeval_flow.workflows._preset import Preset, expand_preset
    from dataeval_flow.workflows._result import WorkflowResult

    _logger.info("Task '%s': starting (%s)", task.name, _target_of(task))

    # 0. Seed every stochastic component [CR-7-S-1]. Applied per task rather than
    #    once per pipeline so a task's result does not depend on what ran before it.
    _apply_seed(config)

    # 1. Normalize sources to list
    source_names: list[str] = [task.sources] if isinstance(task.sources, str) else list(task.sources)

    # 2. Resolve extractor config (optional — single per task)
    setup = _extractor_setup(task.extractor, config, data_dir)
    extractor_cfg = setup.config if setup is not None else None

    # 3. Build a DatasetContext per source
    dataset_contexts, resolved_sources = _source_contexts(source_names, config, setup, data_dir, cache_dir)
    _logger.debug("Task '%s': resolved %d source(s): %s", task.name, len(source_names), source_names)

    # 4. Resolve the target → type + params. A task runs a workflow, an evaluator, or a custom workflow's chain;
    #    the context, policies, timing and envelope below serve the first two.
    instance: WorkflowConfig[Any] | EvaluatorConfig[Any] | CustomWorkflowConfig
    if task.kind == "evaluator":
        instance = _resolve_evaluator(task.workflow, config)
    else:
        instance = _resolve_workflow(task.workflow, config)
    # The pipeline's `result:` limits on tables of items hold for this run's report builders alone.
    settings = config.result
    limits = TableLimits(rows=_unless_all(settings.max_rows), preview=_unless_all(settings.preview_rows))
    if isinstance(instance, CustomWorkflowConfig):
        return _run_custom_task(
            task,
            instance,
            config,
            dataset_contexts,
            resolved_sources,
            setup,
            data_dir=data_dir,
            cache_dir=cache_dir,
            output_dir=output_dir,
            report_images=report_images,
            limits=limits,
        )
    runner = _implementation(instance)

    # `PipelineConfig` only checks the tasks it holds; a task run directly (not out of
    # `config.tasks`) would skip this and fail later, deep inside the run. Check here for
    # both kinds, with the config-load check's message, as a failed result so a caller
    # sees the same envelope either way.
    problem = input_problem(instance, source_count=len(source_names), has_extractor=task.extractor is not None)
    if problem is not None:
        default = EvaluatorResult if task.kind == "evaluator" else WorkflowResult
        result_type = result_type_of(runner, default)
        message = f"Task '{task.name}' runs {task.kind} '{instance.name}' ({instance.type}), which {problem}"
        refused = result_type.failed(type=runner.name, errors=[message])
        from dataeval_flow.steps._result import ChainResult

        if isinstance(refused, ChainResult):
            refused._preset = isinstance(runner, Preset)  # noqa: SLF001 - a preset's refusal names its type
        # The envelope any other failed result carries. Nothing ran, so no time was spent running.
        _ensure_result_datasets(refused, dataset_contexts)
        _populate_result_metadata(refused, resolved_sources, extractor_cfg, 0.0, instance, config, data_dir=data_dir)
        return refused

    # A preset's settings expand to a chain of steps, which runs as a custom workflow's, under the preset's type id.
    if isinstance(runner, Preset) and isinstance(instance, WorkflowConfig):
        chain, evaluators = expand_preset(instance, type(runner))
        return _run_custom_task(
            task,
            chain,
            config,
            dataset_contexts,
            resolved_sources,
            setup,
            data_dir=data_dir,
            cache_dir=cache_dir,
            output_dir=output_dir,
            report_images=report_images,
            limits=limits,
            evaluators=evaluators,
            entry=instance,
        )

    # 5-7. Resolve the step's policies and ontology, then run it as a one-step graph.
    result, elapsed, ontology = _run_one_step(
        task,
        instance,
        runner,
        config,
        dataset_contexts,
        resolved_sources,
        setup,
        data_dir=data_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        limits=limits,
    )
    _logger.info("Task '%s': finished in %.1fs (success=%s)", task.name, elapsed, result.success)

    # 8. Backfill the resolved dataset(s) when the workflow left them unset —
    # notably on failure paths, where callers still need the inputs to debug.
    _ensure_result_datasets(result, dataset_contexts)

    # 9. Thumbnails of the items the report names, read while the run's datasets are at hand.
    if report_images and result.success:
        _capture_assets(result, dataset_contexts, _unless_all(config.result.max_images))

    # 10. Populate metadata envelope
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


def _extractor_setup(name: str | None, config: "PipelineConfig", data_dir: Path | None) -> "ExtractorSetup | None":
    """The extractor `name` names, with its paths resolved and its preprocessing built; ``None`` for no name."""
    from dataeval_flow._chain._run import ExtractorSetup
    from dataeval_flow._preprocessing import build_preprocessing

    if name is None:
        return None
    extractor_cfg = _resolve_extractor_paths(_resolve_by_name(config.extractors, name, "extractor"), data_dir)
    transforms = None
    if extractor_cfg.preprocessor is not None:
        pre_config = _resolve_by_name(config.preprocessors, extractor_cfg.preprocessor, "preprocessor")
        transforms = build_preprocessing(pre_config.steps)
    return ExtractorSetup(extractor_cfg, transforms, extractor_cfg.batch_size)


def _source_contexts(
    source_names: Sequence[str],
    config: "PipelineConfig",
    setup: "ExtractorSetup | None",
    data_dir: Path | None,
    cache_dir: Path | None,
) -> "tuple[dict[str, DatasetContext], list[ResolvedSource]]":
    """Resolve each source a task reads, and the context its run reads it through, keyed by source name."""
    from dataeval_flow._cache import DatasetCache
    from dataeval_flow._sources import resolve_source
    from dataeval_flow.workflows._context import DatasetContext

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
            extractor=setup.config if setup is not None else None,
            transforms=setup.transforms if setup is not None else None,
            view_operations=resolved.view_config.operations if resolved.view_config else None,
            batch_size=setup.batch_size if setup is not None else None,
            label_source=_label_source_of(resolved.label_sources),
            value_range=_value_range_of(resolved),
            channel_groups=_channel_groups_of(resolved),
            cache=ds_cache,
        )

    if cache_dir:
        _logger.info("Cache enabled: %s", cache_dir)

    return dataset_contexts, resolved_sources


def _run_one_step(
    task: "TaskConfig",
    instance: "WorkflowConfig[Any] | EvaluatorConfig[Any]",
    runner: "Workflow[Any, Any] | Evaluator[Any, Any]",
    config: "PipelineConfig",
    dataset_contexts: "dict[str, DatasetContext]",
    resolved_sources: "list[ResolvedSource]",
    setup: "ExtractorSetup | None",
    *,
    data_dir: Path | None,
    cache_dir: Path | None,
    output_dir: Path | None,
    limits: "TableLimits",
) -> "tuple[Result[Any, Any], float, ResolvedOntology | None]":
    """Run an ``evaluator:`` or workflow-type task as a one-step graph: its result, unwrapped, the seconds it took,
    and the ontology its step resolved.

    The step's metadata policy, value range, stats policy and ontology resolve before anything reads the dataset,
    so a misspelled factor or a missing descriptor costs a config error, not an hour of walking images.
    """
    from dataeval_flow._chain._graph import one_step_graph
    from dataeval_flow._chain._preflight import step_contexts
    from dataeval_flow._chain._run import RunSettings, bind_inputs, run_chain
    from dataeval_flow._kind import result_type_of
    from dataeval_flow._tables import limited_tables
    from dataeval_flow.evaluators._result import EvaluatorResult
    from dataeval_flow.workflows._result import WorkflowResult

    source_names = list(dataset_contexts)
    graph = one_step_graph(task, instance, source_names)
    contexts = step_contexts(graph, config, data_dir, {name: [dataset_contexts[name]] for name in source_names})
    inputs = bind_inputs(graph, source_names, dataset_contexts, {source.name: source for source in resolved_sources})
    run_settings = RunSettings(
        task=task.name,
        pipeline=config,
        data_dir=data_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        extractors={None: setup},
        step_contexts=contexts,
        runners={task.name: runner},
    )
    _logger.debug("Task '%s': executing", task.name)
    start = time.monotonic()
    # Library diagnostics are captured here, not left to the log file: they name the
    # binning and value_range decisions this run made, and the envelope must record them
    # on its own.
    # One extractor scope per task, whichever entry point runs it (`run_task`, `run_tasks`,
    # `run`, the TUI), so every source the task compares is described by the same stateful
    # extractor, fitted once on the first source to ask.
    with capture_diagnostics() as diagnostics, shared_extractor_scope(), limited_tables(limits):
        chain = run_chain(graph, inputs, run_settings)
    elapsed = time.monotonic() - start
    record = chain.steps[task.name]
    result = record.result
    if result is None:  # the step failed before its evaluator or workflow could build a result
        default = EvaluatorResult if task.kind == "evaluator" else WorkflowResult
        result = result_type_of(runner, default).failed(type=runner.name, errors=record.errors)
    if diagnostics:
        result.metadata.diagnostics = list(diagnostics)
    return result, elapsed, contexts[task.name].ontology


def _run_custom_task(
    task: "TaskConfig",
    workflow: CustomWorkflowConfig,
    config: "PipelineConfig",
    dataset_contexts: "dict[str, DatasetContext]",
    resolved_sources: "list[ResolvedSource]",
    setup: "ExtractorSetup | None",
    *,
    data_dir: Path | None,
    cache_dir: Path | None,
    output_dir: Path | None,
    report_images: bool,
    limits: "TableLimits",
    evaluators: "Sequence[EvaluatorConfig[Any]]" = (),
    entry: "WorkflowConfig[Any] | None" = None,
) -> "ChainResult":
    """Run a custom workflow's chain for `task`. Config errors raise; step failures become the result's.

    `entry` is the preset entry `workflow` was expanded from: the result carries its type id, and the envelope records
    its settings rather than the chain's, and each conformed source's label space under its ontology. `evaluators`
    are the entries the preset's steps name.
    """
    from dataeval_flow._chain._graph import binding_problems, build_graph
    from dataeval_flow._chain._preflight import check_kinds, step_contexts
    from dataeval_flow._chain._run import RunSettings, bind_inputs, run_chain
    from dataeval_flow._sources import label_space_records
    from dataeval_flow._tables import limited_tables
    from dataeval_flow.steps._result import ChainMetadata, ChainResult

    names = task.source_names
    extractor_cfg = setup.config if setup is not None else None
    type_id = entry.type if entry is not None else workflow.name
    described = entry if entry is not None else workflow
    # `PipelineConfig` checks only the tasks it holds; a task run directly gets the same checks here, as a failed
    # result, like a workflow-type task's.
    problem = workflow.binding_problem(len(names))
    problems = (
        [f"Task '{task.name}' runs workflow '{workflow.name}', which {problem}"]
        if problem is not None
        else binding_problems(task, workflow, config, evaluators)
    )
    if problems:
        refused = ChainResult.failed(type=type_id, errors=problems)
        refused.metadata = ChainMetadata(workflow=workflow.name)
        refused._preset = entry is not None  # noqa: SLF001 - the banner names a preset, not a custom workflow
        _populate_result_metadata(refused, resolved_sources, extractor_cfg, 0.0, described, config, data_dir=data_dir)
        return refused
    # A preset entry's ontology resolves up front, as a workflow-type task's does, for the envelope to record.
    ontology = _resolve_ontology(entry, config, data_dir) if entry is not None else None
    graph = build_graph(workflow, config, evaluators=evaluators)
    inputs = bind_inputs(graph, names, dataset_contexts, {source.name: source for source in resolved_sources})
    slot_contexts: dict[str, list[DatasetContext]] = {}
    for index, slot in enumerate(graph.slots):
        bound = names[index:] if slot.is_list else [names[index]]
        slot_contexts[slot.name] = [dataset_contexts[name] for name in bound]
    contexts = step_contexts(graph, config, data_dir, slot_contexts)
    check_kinds(graph, inputs)
    # Each step naming its own extractor embeds with it; every other step, with the task's.
    named = {spec.extractor for spec in graph.steps if spec.extractor}
    extractors = {None: setup} | {name: _extractor_setup(name, config, data_dir) for name in named}
    run_settings = RunSettings(
        task=task.name,
        pipeline=config,
        data_dir=data_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        extractors=extractors,
        step_contexts=contexts,
    )
    _logger.debug("Task '%s': executing", task.name)
    start = time.monotonic()
    with capture_diagnostics() as diagnostics, shared_extractor_scope(), limited_tables(limits):
        chain = run_chain(graph, inputs, run_settings)
    elapsed = time.monotonic() - start
    result = ChainResult.from_run(workflow.name, chain, type_id=type_id, preset=entry is not None)
    if diagnostics:
        result.metadata.diagnostics = list(diagnostics)
    _logger.info("Task '%s': finished in %.1fs (success=%s)", task.name, elapsed, result.success)
    result.sources = {source.name: source.realized() for source in resolved_sources}
    if report_images:
        _capture_chain_assets(result, chain, _unless_all(config.result.max_images))
    _populate_result_metadata(
        result, resolved_sources, extractor_cfg, elapsed, described, config, data_dir=data_dir, ontology=ontology
    )
    if chain.label_space:
        # The sources' records, where there are any, replaced the chain's own: keep both, the sources' first.
        result.metadata.label_space = [*label_space_records(resolved_sources, ontology), *chain.label_space]
        # One vocabulary names the run's labels only where every record agrees, the sources' and the chain's.
        digests = {record.digest for record in result.metadata.label_space}
        result.metadata.label_space_digest = next(iter(digests)) if len(digests) == 1 else None
    return result


def _capture_chain_assets(result: "ChainResult", chain: "ChainRun", limit: int | None) -> None:
    """Thumbnails of the items a chain's report names: each `ItemRef` names a Dataset node's address.

    A failure here costs the report its thumbnails, never the result: the run has already finished.
    """
    from dataeval_flow._capture import capture
    from dataeval_flow._chain._nodes import Node, NodeList
    from dataeval_flow.steps._port import DataType

    nodes: list[Node] = []
    for value in chain.nodes.values():
        items = value.present.values() if isinstance(value, NodeList) else [value]
        nodes.extend(item for item in items if isinstance(item, Node) and item.type is DataType.DATASET)
    try:
        datasets = {node.address: node.value for node in nodes}
        ranges = {node.address: node.context.value_range if node.context else None for node in nodes}
        result.assets = capture(result._document(detailed=True).blocks, datasets, ranges, limit=limit)  # noqa: SLF001
    except Exception:
        _logger.warning("Could not capture thumbnails for task '%s'", result.type, exc_info=True)


def _run_target(
    target: "Workflow[Any, Any] | Evaluator[Any, Any]",
    config: Any,
    context: "WorkflowContext",
    *,
    stats_unions: "Mapping[str, ResolvedStatsPolicy] | None" = None,
) -> "Result[Any, Any]":
    """Run a workflow or an evaluator on a resolved context. Never raises: a failure becomes a failed result.

    The failed result is of the config's result class. Only the run itself is covered: resolving the task before it,
    in :func:`_run_single_task`, raises on a config error, a transform's constructor included. `stats_unions` is what
    a chain planned for an evaluator step's sources, handed to :func:`~dataeval_flow.evaluators._execute.execute`.
    """
    from dataeval_flow._kind import result_type_of
    from dataeval_flow._result import failure_message
    from dataeval_flow.evaluators._evaluator import Evaluator
    from dataeval_flow.evaluators._execute import execute
    from dataeval_flow.workflows._result import WorkflowResult

    if isinstance(target, Evaluator):
        return execute(target, context, config, stats_unions=stats_unions)
    result_type: type[WorkflowResult[Any, Any]] = result_type_of(target, WorkflowResult)
    if not isinstance(config, target.config_type):
        return result_type.failed(
            type=target.name, errors=[f"Expected {target.config_type.__name__}, got {type(config).__name__}"]
        )
    try:
        result = target.run(config, context)
    except Exception as error:  # every failure is reported as a failed result
        _logger.exception("Workflow '%s' failed", target.name)
        return result_type.failed(type=target.name, errors=[failure_message(error)])
    if not isinstance(result, result_type):  # `run()` is typed by this class, so another one would lie
        _logger.error("Workflow '%s' returned %s, not a %s", target.name, type(result).__name__, result_type.__name__)
        return result_type.failed(
            type=target.name, errors=[f"{target.name} returned {type(result).__name__}, not a {result_type.__name__}"]
        )
    return result


def _ensure_result_datasets(
    result: "Result[Any, Any]",
    dataset_contexts: "Mapping[str, DatasetContext]",
) -> None:
    """Fill in ``result.dataset`` / ``result.sources`` when the workflow did not.

    Workflows attach the resolved, post-selection dataset to successful
    results only — an early return or an exception leaves the fields unset,
    which prevents callers from inspecting the inputs that produced the failure.
    Rebuild the selection here for those cases; already-populated fields are
    left untouched, so the success path is unaffected.
    """
    if not dataset_contexts:
        return

    single = len(dataset_contexts) == 1
    needs_dataset = single and result.dataset is None
    needs_sources = not single and result.sources is None
    if not (needs_dataset or needs_sources):
        return

    from dataeval_flow._view import build_view

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


def _unless_all(limit: int) -> int | None:
    """A `result:` limit as the code takes it: ``None`` for -1, which lifts it."""
    return None if limit < 0 else limit


def _capture_assets(
    result: "Result[Any, Any]", dataset_contexts: "Mapping[str, DatasetContext]", limit: int | None
) -> None:
    """Keep a thumbnail of each item *result*'s report names, at most *limit*, read from the datasets the run read.

    A failure here costs the report its thumbnails, never the result: the run has already finished.
    """
    from dataeval_flow._capture import capture

    try:
        datasets = result.sources if result.sources is not None else {next(iter(dataset_contexts)): result.dataset}
        ranges = {name: context.value_range for name, context in dataset_contexts.items()}
        result.assets = capture(result._document(detailed=True).blocks, datasets, ranges, limit=limit)  # noqa: SLF001
    except Exception:
        _logger.warning("Could not capture the report's thumbnails, so it names its items instead.", exc_info=True)


def _populate_result_metadata(
    result: "Result[Any, Any]",
    resolved_sources: "Sequence[ResolvedSource]",
    extractor_cfg: Any,
    elapsed: float,
    workflow_instance: "WorkflowConfig[Any] | EvaluatorConfig[Any] | CustomWorkflowConfig | None" = None,
    pipeline_config: "PipelineConfig | None" = None,
    data_dir: Path | None = None,
    ontology: "ResolvedOntology | None" = None,
) -> None:
    """Fill in the JATIC metadata envelope from resolved source/extractor context."""
    from dataeval_flow import __version__
    from dataeval_flow._sources import label_space_records

    dataset_names = [
        operand.source.dataset for rs in resolved_sources for operand in rs.operands if operand.source.dataset
    ]
    if workflow_instance is not None:
        result._entry = workflow_instance.name  # noqa: SLF001 - the envelope's filler names the entry
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
    result._set_nulls = _set_nulls(resolved_sources, workflow_instance, extractor_cfg)  # noqa: SLF001 - as `_entry`


def _nulls_of(value: Any, path: tuple[str | int, ...]) -> set[tuple[str | int, ...]]:
    """The paths, in ``value``'s dump, of the fields a model was given as ``None``, through nested models,
    lists and mappings of them."""
    from pydantic import BaseModel

    found: set[tuple[str | int, ...]] = set()
    if isinstance(value, BaseModel):
        for name in value.model_fields_set:
            item = getattr(value, name, None)
            if item is None:
                found.add((*path, name))
            else:
                found |= _nulls_of(item, (*path, name))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found |= _nulls_of(item, (*path, index))
    elif isinstance(value, dict):
        for key, item in value.items():
            found |= _nulls_of(item, (*path, key))
    return found


def _set_nulls(
    resolved_sources: "Sequence[ResolvedSource]",
    workflow_instance: "WorkflowConfig[Any] | EvaluatorConfig[Any] | CustomWorkflowConfig | None",
    extractor_cfg: Any,
) -> frozenset[tuple[str | int, ...]]:
    """Where ``resolved_config`` holds a ``None`` the user wrote, so its report keeps it beside the unset defaults it
    drops. The sources' dataset and view configs, the workflow or evaluator and the extractor are models, so tracked;
    the rest of ``resolved_config`` (names, the seed) is plain values, where a ``None`` is always dropped."""
    found: set[tuple[str | int, ...]] = set()
    for index, resolved in enumerate(resolved_sources):
        base: tuple[str | int, ...] = ("sources", index)
        if resolved.is_merged:
            for leaf, operand in enumerate(resolved.operands):
                found |= _nulls_of(operand.view_config, (*base, "merge", leaf, "view_config"))
                found |= _nulls_of(operand.dataset_config, (*base, "merge", leaf, "dataset_config"))
            found |= _nulls_of(resolved.view_config, (*base, "view_config"))
        else:
            operand = resolved.operands[0]
            found |= _nulls_of(operand.dataset_config, (*base, "dataset_config"))
            found |= _nulls_of(operand.view_config, (*base, "view_config"))
    if workflow_instance is not None:
        from dataeval_flow.evaluators._base import EvaluatorConfig

        key = "evaluator" if isinstance(workflow_instance, EvaluatorConfig) else "workflow"
        found |= _nulls_of(workflow_instance, (key,))
    found |= _nulls_of(extractor_cfg, ("extractor",))
    return frozenset(found)


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
    workflow_instance: "WorkflowConfig[Any] | EvaluatorConfig[Any] | CustomWorkflowConfig | None",
    extractor_cfg: Any,
    pipeline_config: "PipelineConfig | None",
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a fully resolved config dict for report traceability."""
    cfg: dict[str, Any] = {"sources": [_source_entry(rs) for rs in resolved_sources]}

    if workflow_instance is not None:
        from dataeval_flow.evaluators._base import EvaluatorConfig

        key = "evaluator" if isinstance(workflow_instance, EvaluatorConfig) else "workflow"
        cfg[key] = workflow_instance.model_dump(mode="json")

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

    :func:`~dataeval_flow.run_tasks` runs exactly this selection, and keys its results by
    the selected tasks' names.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration holding the task list.
    tasks : str | Sequence[str] | None
        Which tasks to select:

        - ``None`` (default) — every enabled task, in config order
        - ``str`` — a single task by name
        - ``Sequence[str]`` — specific tasks by name, in the given order; a name
          given twice is selected once, where it first appears

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
    # Results are keyed by task name; a name given twice runs once.
    return [_resolve_by_name(config.tasks, name, "task") for name in dict.fromkeys(tasks)]


def run_tasks(
    config: "PipelineConfig",
    tasks: str | Sequence[str] | None = None,
    *,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    report_images: bool = True,
    output_dir: Path | None = None,
) -> "dict[str, Result[Any, Any]]":
    """Run tasks from a pipeline configuration.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration containing datasets, sources, extractors,
        workflows, and tasks.
    tasks : str | Sequence[str] | None
        Which tasks to run:

        - ``None`` (default) — run all enabled tasks, in config order
        - ``str`` — run a single task by name
        - ``Sequence[str]`` — run specific tasks by name, in the given order; a name
          given twice runs once, where it first appears

        Naming a task runs it whether or not it is enabled.
    data_dir : Path | None, keyword-only
        Root directory for resolving relative paths in configs.
    cache_dir : Path | None, keyword-only
        Directory for disk-backed computation cache.
    report_images : bool, keyword-only
        Whether each result keeps thumbnails of the items its report names, for the HTML report. ``False``
        reads no item and keeps none.
    output_dir : Path | None, keyword-only
        Where export steps write, under ``<output_dir>/datasets/``. ``None`` writes nothing, and export steps are
        skipped with a reason.

    Returns
    -------
    dict[str, Result]
        Each executed task's result, keyed by task name, in execution order. A task
        whose run raised has a failed result, so check ``result.success`` before
        reading ``result.output``.

    Raises
    ------
    ValueError
        If no tasks are defined, all are disabled, or a named task is
        not found.

    Examples
    --------
    >>> from pathlib import Path
    >>> from dataeval_flow import load_config, run_tasks
    >>> results = run_tasks(load_config("params.yaml"), data_dir=Path("."))  # doctest: +SKIP
    >>> print(results["find_dupes"].report())  # doctest: +SKIP
    """
    to_run = select_tasks(config, tasks)

    _logger.info("Running %d task(s)", len(to_run))
    results: dict[str, Result[Any, Any]] = {}
    for task in to_run:
        _logger.info("--- Task: %s (%s) ---", task.name, _target_of(task))
        results[task.name] = _run_single_task(
            task, config, data_dir=data_dir, cache_dir=cache_dir, report_images=report_images, output_dir=output_dir
        )
    return results


def run_task(
    task: "TaskConfig",
    config: "PipelineConfig",
    *,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    report_images: bool = True,
    output_dir: Path | None = None,
) -> "Result[Any, Any]":
    """Run a single task.

    Unlike :func:`~dataeval_flow.run_tasks`, this function accepts the task config object
    directly rather than looking it up by name.

    Parameters
    ----------
    task : TaskConfig
        The task configuration to execute.
    config : PipelineConfig
        Pipeline configuration supplying datasets, sources, extractors,
        workflow, and evaluator definitions.  The task does **not** need to
        appear in ``config.tasks``.
    data_dir : Path | None, keyword-only
        Root directory for resolving relative paths in configs.
    cache_dir : Path | None, keyword-only
        Directory for disk-backed computation cache.
    report_images : bool, keyword-only
        Whether the result keeps thumbnails of the items its report names, for the HTML report. ``False``
        reads no item and keeps none.
    output_dir : Path | None, keyword-only
        Where export steps write, under ``<output_dir>/datasets/``. ``None`` writes nothing, and export steps are
        skipped with a reason.

    Returns
    -------
    Result
        The result of the workflow or evaluator the task runs, as that type's own result class —
        ``isinstance(result, DriftMonitoringResult)`` narrows it. A run that raised returns a failed
        result of the same class. A custom workflow's, or a preset's such as data-cleaning's, is a
        :class:`~dataeval_flow.steps.ChainResult`, holding every step's outcome whether or not one failed.
    """
    _logger.info("--- Task: %s (%s) ---", task.name, _target_of(task))
    return _run_single_task(
        task, config, data_dir=data_dir, cache_dir=cache_dir, report_images=report_images, output_dir=output_dir
    )
