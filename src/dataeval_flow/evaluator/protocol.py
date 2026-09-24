"""The evaluator base class: its parameters, the DataEval class it wraps, and the one call."""

__all__ = ["EvaluatorBase"]

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from dataeval_flow.evaluator.base import EvaluatorParametersBase, InputKind

if TYPE_CHECKING:
    from dataeval.types import Output
    from pydantic import BaseModel

    from dataeval_flow.evaluator.inputs import Inputs
    from dataeval_flow.evaluator.result import EvaluatorResult
    from dataeval_flow.workflow import WorkflowContext


class EvaluatorBase(ABC):
    """One DataEval evaluator, run by Flow.

    A concrete evaluator declares its parameters (which carry its ``InputSpec``), the
    DataEval class it wraps, the entry point it calls for each input kind, and ``run``.
    Views, the cache, the producers, serialization and the envelope are :meth:`execute`'s,
    and shared by every evaluator.

    ``entry_points`` and ``run`` are the only places an evaluator names a DataEval method.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    params_schema: ClassVar[type[EvaluatorParametersBase]]
    dataeval_class: ClassVar[type]
    entry_points: ClassVar[Mapping[InputKind, str]]

    @abstractmethod
    def run(self, params: Any, inputs: "Sequence[Inputs]") -> "Output[Any]":
        """Call DataEval on the prepared inputs, one per source, and return its output."""

    def execute(self, context: "WorkflowContext", params: "BaseModel | None" = None) -> "EvaluatorResult":
        """Prepare the inputs, run, and wrap the output. Never raises: a failure is a failed result."""
        from dataeval_flow.evaluator._execute import execute

        return execute(self, context, params)
