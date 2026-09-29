"""What a chain's steps produced: one result per step, and, once Task 9 lands, the chain's result."""

__all__ = ["StepResult"]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from dataeval_flow.steps._step import StepKind

if TYPE_CHECKING:
    from dataeval_flow._result import Result

StepStatus = Literal["ok", "failed", "skipped"]


@dataclass
class StepResult:
    """One step's outcome in a chain: its status, what it read, and what it made.

    ``output`` is the live object: a Dataset (a DataEval ``View`` you can go on to use), DataEval's output, a
    workflow's result, an export record, or a mapping of them for a step with several outputs. A step that ran once
    per element of a list has ``elements`` instead, one :class:`StepResult` per key.
    """

    name: str
    kind: StepKind
    type: str
    inputs: list[str]
    status: StepStatus
    output: Any = None
    reason: str | None = None
    errors: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    elements: "dict[str, StepResult] | None" = None
    result: "Result[Any, Any] | None" = None
    optional: bool = False
