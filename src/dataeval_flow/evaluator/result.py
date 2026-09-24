"""Evaluator results: DataEval's output as JSON, in the JATIC envelope, with no health status."""

__all__ = ["DataEvalExecution", "EvaluatorMetadata", "EvaluatorResult"]

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

from dataeval_flow.config.schemas import ResultMetadata, TaskKind
from dataeval_flow.result import Result, _metadata_block

if TYPE_CHECKING:
    from dataeval.types import ExecutionMetadata, Output


class DataEvalExecution(BaseModel):
    """What DataEval recorded about the call: which entry point, which version, when, and for how long."""

    name: str = ""
    version: str = ""
    execution_time: datetime | None = None
    execution_duration: float | None = None

    @classmethod
    def from_meta(cls, meta: "ExecutionMetadata | None") -> "DataEvalExecution":
        """Copy DataEval's execution record, keeping only the version where there is none.

        An output Flow built itself, such as hash and cluster groups merged into one table,
        carries an empty record. The installed version still names what produced it.
        """
        import dataeval

        if meta is None or not meta.name:
            version = meta.version if meta is not None and meta.version else dataeval.__version__
            return cls(version=version)
        return cls(
            name=meta.name,
            version=meta.version or dataeval.__version__,
            execution_time=meta.execution_time,
            execution_duration=meta.execution_duration,
        )


class EvaluatorMetadata(ResultMetadata):
    """The JATIC envelope for an evaluator result."""

    evaluator: str = Field(default="", description="The evaluator type, e.g. `quality.duplicates`.")
    dataeval: DataEvalExecution = Field(
        default_factory=DataEvalExecution,
        description="DataEval's own record of the call. The parameters as written are in `resolved_config`.",
    )


@dataclass
class EvaluatorResult(Result[EvaluatorMetadata]):
    """One evaluator's result: DataEval's output, serialized, in the JATIC envelope.

    Carries no findings and no health status. An evaluator makes determinations under
    DataEval's thresholds; whether they are a problem is a workflow's verdict, so
    ``--fail-on-warning`` never reads this.

    ``output`` is DataEval's output as JSON. ``raw`` is the object DataEval returned, for
    Python callers, and is never serialized.
    """

    kind: ClassVar[TaskKind] = "evaluator"

    output: dict[str, Any]
    raw: "Output[Any] | None" = None

    def _report_title(self) -> str:
        """The result's name."""
        return self.name

    def _report_envelope(self) -> list[str]:
        """The shared envelope, with the DataEval version that made the determinations."""
        from dataeval_flow.workflow._text_report import _WIDTH

        block = _metadata_block(self.metadata)
        version = f"  DataEval:     {self.metadata.dataeval.version}"
        return [*block[:-1], version, block[-1]] if block else [version, "-" * _WIDTH]

    def _report_output(self, *, detailed: bool) -> list[str]:
        """DataEval's output as it came; when not *detailed*, tables stop at ``ROW_LIMIT`` rows."""
        from dataeval_flow.evaluator._report import render_output

        return render_output(self.output, detailed=detailed)

    def _dict_body(self) -> dict[str, object]:
        """DataEval's output, under ``output``."""
        return {"output": self.output}
