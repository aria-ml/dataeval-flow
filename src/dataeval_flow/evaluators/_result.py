"""Evaluator results: DataEval's output as JSON, in the JATIC envelope, with no health status."""

__all__ = ["DataEvalExecution", "EvaluatorMetadata", "EvaluatorResult"]

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar

from pydantic import BaseModel, Field

from dataeval_flow._blocks import Block, Fields
from dataeval_flow._result import Result, ResultMetadata, _envelope_items
from dataeval_flow._tables import table_limits

if TYPE_CHECKING:
    from dataeval.protocols import AnnotatedDataset
    from dataeval.types import ExecutionMetadata

    from dataeval_flow.config._schemas._task import TaskKind


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

    evaluator: str = Field(default="", description="The evaluator type, e.g. `duplicates`.")
    dataeval: DataEvalExecution = Field(
        default_factory=DataEvalExecution,
        description=(
            "DataEval's own record of the call: its `name`, `version`, `execution_time` and `execution_duration`. "
            "The parameters as written are in `resolved_config`."
        ),
    )


TOutput = TypeVar("TOutput")


class EvaluatorResult(Result[EvaluatorMetadata, TOutput]):
    """One evaluator's result: DataEval's output, in the JATIC envelope.

    Carries no findings and no health status. An evaluator makes determinations under
    DataEval's thresholds; whether they are a problem is a workflow's verdict, so
    ``--fail-on-warning`` never reads this.

    ``output`` is the object DataEval returned, so its own methods work. Its JSON form is what
    :meth:`to_dict` and :meth:`export` produce. ``metadata`` is the JATIC envelope, naming the
    evaluator and DataEval's own record of the call. ``kind`` is ``"evaluator"``.

    Parameters
    ----------
    serialized : Mapping[str, Any], optional
        DataEval's output as JSON, which :meth:`to_dict` and :meth:`report` render.
    source_names : Sequence[str], optional
        The names the run's sources are known by, in order: a task's source names, or a chain's node addresses.
        The result's own report section names its items by them, for their thumbnails. The other
        parameters are :class:`~dataeval_flow.Result`'s.

    Subclassing
    -----------
    Subclass it once per evaluator, parameterized by the DataEval output class the evaluator returns, and name
    the subclass as the result type argument of the evaluator's :class:`EvaluatorConfig`. The subclass needs no
    body: Flow builds every instance, successful or failed, so an evaluator never constructs one.
    ``isinstance`` then narrows a :class:`~dataeval_flow.Result` to it and types ``output`` as DataEval's output.

    Examples
    --------
    >>> from typing import Any
    >>> from dataeval.quality import OutliersOutput
    >>> from dataeval_flow.evaluators import EvaluatorResult
    >>> class BrightnessResult(EvaluatorResult[OutliersOutput[Any]]):
    ...     pass

    A task's result narrows to it:

    >>> from dataeval_flow import run_tasks
    >>> result = run_tasks(config)["brightness"]  # doctest: +SKIP
    >>> if isinstance(result, BrightnessResult) and result.success:  # doctest: +SKIP
    ...     print(result.output.aggregate_by_item())
    """

    kind: "ClassVar[TaskKind]" = "evaluator"
    metadata_type: ClassVar[type[ResultMetadata]] = EvaluatorMetadata

    def __init__(
        self,
        *,
        type: str,  # noqa: A002
        success: bool,
        metadata: EvaluatorMetadata,
        output: TOutput | None = None,
        errors: Sequence[str] = (),
        dataset: "AnnotatedDataset[Any] | None" = None,
        sources: "dict[str, AnnotatedDataset[Any]] | None" = None,
        serialized: Mapping[str, Any] | None = None,
        source_names: Sequence[str] = (),
    ) -> None:
        super().__init__(
            type=type,
            success=success,
            metadata=metadata,
            output=output,
            errors=errors,
            dataset=dataset,
            sources=sources,
        )
        self._serialized: dict[str, Any] | None = dict(serialized) if serialized is not None else None
        self._source_names: tuple[str, ...] = tuple(source_names)
        # Its own section is built when the report renders, after the run: it lists rows within the run's limits.
        self._limits = table_limits()

    @classmethod
    def failed(cls, *, type: str, errors: Sequence[str]) -> Self:  # noqa: A002
        """A failed result of this class, its envelope naming the evaluator that failed."""
        return cls(type=type, success=False, metadata=EvaluatorMetadata(evaluator=type), errors=errors)

    def _report_title(self) -> str:
        """The evaluator's type."""
        return self.type

    def _report_envelope(self) -> list[Block]:
        """The shared envelope, with the DataEval version that made the determinations."""
        items = [*_envelope_items(self.metadata), ("DataEval", self.metadata.dataeval.version)]
        return [Fields(items=list(items))]

    def _report_output(self, *, detailed: bool) -> list[Block]:
        """This evaluator's own section where it has one, else DataEval's output as it came; when not *detailed*,
        the latter's tables stop at ``ROW_LIMIT`` rows."""
        from dataeval_flow._tables import limited_tables
        from dataeval_flow.evaluators._report import output_blocks

        serialized = self._serialized or {}
        if self._source_names:
            with limited_tables(self._limits):
                own = self._section(serialized, self._source_names, detailed=detailed)
            if own is not None:
                return own
        return output_blocks(serialized, detailed=detailed)

    def _section(
        self,
        output: Mapping[str, Any],  # noqa: ARG002
        sources: Sequence[str],  # noqa: ARG002
        *,
        detailed: bool,  # noqa: ARG002
    ) -> list[Block] | None:
        """This evaluator's own report section, from its JSON and the names of the sources it read; ``None`` for
        DataEval's output as it came. A result class overrides it to show its evaluator's output its own way."""
        return None

    def _dict_body(self) -> dict[str, object]:
        """DataEval's output as JSON, under ``output``."""
        return {"output": self._serialized}
