"""The one ``execute`` every evaluator shares: views, cache, producers, the call, and the envelope."""

__all__ = ["execute"]

import contextlib
import logging
from typing import TYPE_CHECKING, Any

from dataeval_flow._input_spec import InputKind
from dataeval_flow._result import failure_message
from dataeval_flow.evaluators._base import EvaluatorConfig
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators._per_class import require_one_label_per_item, run_per_class, serialize_per_class
from dataeval_flow.evaluators._producers import PRODUCERS, ProducerContext
from dataeval_flow.evaluators._result import DataEvalExecution, EvaluatorMetadata, EvaluatorResult
from dataeval_flow.evaluators._serialize import serialize_output

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dataeval.protocols import AnnotatedDataset

    from dataeval_flow._stats import ResolvedStatsPolicy
    from dataeval_flow.evaluators._evaluator import Evaluator
    from dataeval_flow.steps._by import ByConfig
    from dataeval_flow.workflows._context import ResolvedOntology, WorkflowContext

_logger: logging.Logger = logging.getLogger(__name__)


def execute(
    evaluator: "Evaluator[Any, Any]",
    context: "WorkflowContext",
    config: Any,
    *,
    stats_unions: "Mapping[str, ResolvedStatsPolicy] | None" = None,
    by: "ByConfig | None" = None,
) -> "EvaluatorResult[Any]":
    """Run *evaluator* over every source in *context*. Never raises: a failure is a failed result.

    Parameters
    ----------
    evaluator : Evaluator
        The evaluator to run.
    context : WorkflowContext
        The resolved sources, cache and policies, as the orchestrator builds them.
    config : EvaluatorConfig
        The evaluator's configuration entry; must be an instance of its ``config_type``.
    stats_unions : Mapping[str, ResolvedStatsPolicy] or None, optional
        By source, the statistics a chain planned to compute there, each source's
        :attr:`~dataeval_flow.evaluators._producers.ProducerContext.stats_union`. ``None`` outside a chain.
    by : ByConfig or None, optional
        A chain step's ``by:``: run *evaluator* once per class or class group, on each source's embeddings and labels
        sliced by key, into one :class:`~dataeval_flow.evaluators.PerClassOutput`. ``None`` runs it once.

    Returns
    -------
    EvaluatorResult
        DataEval's output in its envelope, or a failed result carrying the error, of the config's result class.
    """
    from dataeval_flow._kind import result_type_of

    result_type: type[EvaluatorResult[Any]] = result_type_of(evaluator, EvaluatorResult)
    if not isinstance(config, evaluator.config_type):
        return result_type.failed(
            type=evaluator.name,
            errors=[f"Expected {evaluator.config_type.__name__}, got {type(config).__name__}"],
        )
    try:
        inputs, datasets = _prepare(
            context, config, stats_unions, extra=frozenset({InputKind.LABELS}) if by is not None else frozenset()
        )
        if by is None:
            output = evaluator.run(config, inputs)
            serialized = serialize_output(output, extras=evaluator.output_extras)
        else:
            require_one_label_per_item(datasets, inputs)
            output = run_per_class(evaluator, config, inputs, by)
            serialized = serialize_per_class(output, extras=evaluator.output_extras)
        # Recording the output reads its `meta()`, which an output that is not DataEval's may lack or break.
        metadata = EvaluatorMetadata(evaluator=evaluator.name, dataeval=DataEvalExecution.from_meta(output.meta()))
        single = len(datasets) == 1
        return result_type(
            type=evaluator.name,
            success=True,
            output=output,
            serialized=serialized,
            source_names=tuple(datasets),
            metadata=metadata,
            dataset=next(iter(datasets.values())) if single else None,
            sources=None if single else datasets,
        )
    except Exception as e:
        _logger.exception("Evaluator '%s' failed", evaluator.name)
        return result_type.failed(type=evaluator.name, errors=[failure_message(e)])


def _prepare(
    context: "WorkflowContext",
    config: "EvaluatorConfig[Any]",
    stats_unions: "Mapping[str, ResolvedStatsPolicy] | None" = None,
    *,
    extra: frozenset[InputKind] = frozenset(),
) -> "tuple[list[EvaluatorInputs], dict[str, AnnotatedDataset[Any]]]":
    """Apply each source's view, then run every wanted producer under that source's cache.

    Every source's inputs carry the task's ontology, resolved once before any source is read, and each source's
    producers the stats union a chain planned for it, if any. `extra` adds kinds the config does not want, such as the
    labels a run with ``by:`` keys items by.
    """
    from dataeval_flow._cache import active_cache, selection_repr
    from dataeval_flow._view import build_view

    wanted = config.wanted_kinds() | extra
    missing = sorted(kind for kind in wanted if kind not in PRODUCERS)
    if missing:
        raise ValueError(f"No producer for {', '.join(missing)} in this build")

    resolved = _task_ontology(context, config)
    ontology = resolved.ontology if resolved is not None else None
    ontology_source = resolved.source if resolved is not None else None

    inputs: list[EvaluatorInputs] = []
    datasets: dict[str, AnnotatedDataset[Any]] = {}
    for name, dc in context.dataset_contexts.items():
        dataset = build_view(dc.dataset, list(dc.view_operations)) if dc.view_operations else dc.dataset
        datasets[name] = dataset
        pc = ProducerContext(
            source=name,
            dataset=dataset,
            dataset_context=dc,
            workflow_context=context,
            config=config,
            stats_union=(stats_unions or {}).get(name),
        )
        produced: dict[str, Any] = {}
        with contextlib.ExitStack() as stack:
            if dc.cache is not None:
                stack.enter_context(active_cache(dc.cache, selection_repr(dataset)))
            for kind in InputKind:
                if kind in wanted:
                    produced.update(PRODUCERS[kind](pc))
        inputs.append(
            EvaluatorInputs(
                source=name,
                ontology=ontology,
                ontology_source=ontology_source,
                label_source=dc.label_source,
                **produced,
            )
        )
    return inputs, datasets


def _task_ontology(context: "WorkflowContext", config: "EvaluatorConfig[Any]") -> "ResolvedOntology | None":
    """The ontology the task names, resolved, with how it was named; ``None`` where it names none.

    One that fails to resolve fails the run.

    The orchestrator resolves it onto the context. A context built by hand carries none, so the config's own value is
    resolved here, as ``policy_for`` resolves a metadata policy, rather than silently dropped.
    """
    resolved = context.ontology
    if resolved is None and getattr(config, "ontology", None) is not None:
        from dataeval_flow._orchestrator import _resolve_ontology

        resolved = _resolve_ontology(config, None, None)
    if resolved is None:
        return None
    if resolved.error is not None:
        raise ValueError(f"The task's ontology could not be resolved: {resolved.error}")
    return resolved
