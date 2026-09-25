"""The one ``execute`` every evaluator shares: views, cache, producers, the call, and the envelope."""

__all__ = ["execute"]

import contextlib
import logging
from typing import TYPE_CHECKING, Any

from dataeval_flow._input_spec import InputKind
from dataeval_flow._result import failure_message
from dataeval_flow.evaluators._base import EvaluatorConfig
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators._producers import PRODUCERS, ProducerContext
from dataeval_flow.evaluators._result import DataEvalExecution, EvaluatorMetadata, EvaluatorResult
from dataeval_flow.evaluators._serialize import serialize_output

if TYPE_CHECKING:
    from dataeval.protocols import AnnotatedDataset

    from dataeval_flow.evaluators._evaluator import Evaluator
    from dataeval_flow.workflows._context import WorkflowContext

_logger: logging.Logger = logging.getLogger(__name__)


def execute(evaluator: "Evaluator[Any, Any]", context: "WorkflowContext", config: Any) -> "EvaluatorResult[Any]":
    """Run *evaluator* over every source in *context*. Never raises: a failure is a failed result.

    Parameters
    ----------
    evaluator : Evaluator
        The evaluator to run.
    context : WorkflowContext
        The resolved sources, cache and policies, as the orchestrator builds them.
    config : EvaluatorConfig
        The evaluator's configuration entry; must be an instance of its ``config_type``.

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
        inputs, datasets = _prepare(context, config)
        output = evaluator.run(config, inputs)
        serialized = serialize_output(output)
        # Recording the output reads its `meta()`, which an output that is not DataEval's may lack or break.
        metadata = EvaluatorMetadata(evaluator=evaluator.name, dataeval=DataEvalExecution.from_meta(output.meta()))
        single = len(datasets) == 1
        return result_type(
            type=evaluator.name,
            success=True,
            output=output,
            serialized=serialized,
            metadata=metadata,
            dataset=next(iter(datasets.values())) if single else None,
            sources=None if single else datasets,
        )
    except Exception as e:
        _logger.exception("Evaluator '%s' failed", evaluator.name)
        return result_type.failed(type=evaluator.name, errors=[failure_message(e)])


def _prepare(
    context: "WorkflowContext", config: "EvaluatorConfig[Any]"
) -> "tuple[list[EvaluatorInputs], dict[str, AnnotatedDataset[Any]]]":
    """Apply each source's view, then run every wanted producer under that source's cache."""
    from dataeval_flow._cache import active_cache, selection_repr
    from dataeval_flow._view import build_view

    wanted = config.wanted_kinds()
    missing = sorted(kind for kind in wanted if kind not in PRODUCERS)
    if missing:
        raise ValueError(f"No producer for {', '.join(missing)} in this build")

    inputs: list[EvaluatorInputs] = []
    datasets: dict[str, AnnotatedDataset[Any]] = {}
    for name, dc in context.dataset_contexts.items():
        dataset = build_view(dc.dataset, list(dc.view_operations)) if dc.view_operations else dc.dataset
        datasets[name] = dataset
        pc = ProducerContext(dataset=dataset, dataset_context=dc, workflow_context=context, config=config)
        produced: dict[str, Any] = {}
        with contextlib.ExitStack() as stack:
            if dc.cache is not None:
                stack.enter_context(active_cache(dc.cache, selection_repr(dataset)))
            for kind in InputKind:
                if kind in wanted:
                    produced.update(PRODUCERS[kind](pc))
        inputs.append(EvaluatorInputs(source=name, **produced))
    return inputs, datasets
