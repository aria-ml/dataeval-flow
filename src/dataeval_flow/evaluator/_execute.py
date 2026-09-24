"""The one ``execute`` every evaluator shares: views, cache, producers, the call, and the envelope."""

__all__ = ["execute"]

import contextlib
import logging
from typing import TYPE_CHECKING, Any

from dataeval_flow.evaluator._producers import PRODUCERS, ProducerContext
from dataeval_flow.evaluator._serialize import serialize_output
from dataeval_flow.evaluator.base import EvaluatorParametersBase, InputKind
from dataeval_flow.evaluator.inputs import Inputs
from dataeval_flow.evaluator.result import DataEvalExecution, EvaluatorMetadata, EvaluatorResult

if TYPE_CHECKING:
    from dataeval.protocols import AnnotatedDataset
    from pydantic import BaseModel

    from dataeval_flow.evaluator.protocol import EvaluatorBase
    from dataeval_flow.workflow import WorkflowContext

_logger: logging.Logger = logging.getLogger(__name__)


def execute(evaluator: "EvaluatorBase", context: "WorkflowContext", params: "BaseModel | None") -> EvaluatorResult:
    """Run *evaluator* over every source in *context*. Never raises: a failure is a failed result.

    Parameters
    ----------
    evaluator : EvaluatorBase
        The evaluator to run.
    context : WorkflowContext
        The resolved sources, cache and policies, as the orchestrator builds them.
    params : BaseModel | None
        The evaluator's configuration entry; must be an instance of its ``params_schema``.

    Returns
    -------
    EvaluatorResult
        The serialized output in its envelope, or a failed result carrying the error.
    """
    if not isinstance(params, evaluator.params_schema):
        return _failed(evaluator.name, f"Expected {evaluator.params_schema.__name__}, got {type(params).__name__}")
    try:
        inputs, datasets = _prepare(context, params)
        output = evaluator.run(params, inputs)
        serialized = serialize_output(output)
    except Exception as e:
        _logger.exception("Evaluator '%s' failed", evaluator.name)
        return _failed(evaluator.name, f"Evaluator execution failed: {e}")
    single = len(datasets) == 1
    return EvaluatorResult(
        name=evaluator.name,
        success=True,
        output=serialized,
        metadata=EvaluatorMetadata(evaluator=evaluator.name, dataeval=DataEvalExecution.from_meta(output.meta())),
        raw=output,
        dataset=next(iter(datasets.values())) if single else None,
        sources=None if single else datasets,
    )


def _prepare(
    context: "WorkflowContext", params: EvaluatorParametersBase
) -> "tuple[list[Inputs], dict[str, AnnotatedDataset[Any]]]":
    """Apply each source's view, then run every wanted producer under that source's cache."""
    from dataeval_flow.cache import active_cache, selection_repr
    from dataeval_flow.view import build_view

    wanted = params.wanted_kinds()
    missing = sorted(kind for kind in wanted if kind not in PRODUCERS)
    if missing:
        raise ValueError(f"No producer for {', '.join(missing)} in this build")

    inputs: list[Inputs] = []
    datasets: dict[str, AnnotatedDataset[Any]] = {}
    for name, dc in context.dataset_contexts.items():
        dataset = build_view(dc.dataset, list(dc.view_operations)) if dc.view_operations else dc.dataset
        datasets[name] = dataset
        pc = ProducerContext(dataset=dataset, dataset_context=dc, workflow_context=context, params=params)
        produced: dict[str, Any] = {}
        with contextlib.ExitStack() as stack:
            if dc.cache is not None:
                stack.enter_context(active_cache(dc.cache, selection_repr(dataset)))
            for kind in InputKind:
                if kind in wanted:
                    produced.update(PRODUCERS[kind](pc))
        inputs.append(Inputs(source=name, **produced))
    return inputs, datasets


def _failed(name: str, error: str) -> EvaluatorResult:
    return EvaluatorResult(
        name=name, success=False, output={}, metadata=EvaluatorMetadata(evaluator=name), errors=[error]
    )
