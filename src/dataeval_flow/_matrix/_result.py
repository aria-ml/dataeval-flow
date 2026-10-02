"""A task matrix's result: every run's result in order, and what the report and JSON make of them (task-matrix spec
§6, §7)."""

__all__ = ["MatrixResult", "MatrixRun"]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pydantic_core import to_jsonable_python

from dataeval_flow._result import Result, ResultKind, ResultMetadata, finite_json

if TYPE_CHECKING:
    from dataeval_flow._blocks import Asset, Block, Scalar, Section


@dataclass(frozen=True)
class MatrixRun:
    """One run of a task's matrix.

    Attributes
    ----------
    number
        The run's number, from 1.
    label
        Its grid's keys and values, ``key=value, key=value``.
    values
        The values it set, by key as written.
    result
        Its result: what the task would return run alone with these values.
    """

    number: int
    label: str
    values: dict[str, Any]
    result: "Result[Any, Any]"

    @property
    def status(self) -> str:
        """``failed``, ``warning`` or ``ok``, as the run's health says; an evaluator run is ``ok`` or ``failed``."""
        if not self.result.success:
            return "failed"
        health = getattr(self.result, "health", None)
        return str(health["status"]) if isinstance(health, dict) else "ok"

    @property
    def warning_count(self) -> int:
        """The run's warnings, as its result counts them; 0 for an evaluator's."""
        return int(getattr(self.result, "warning_count", 0))


@dataclass(frozen=True)
class MatrixOutput:
    """A successful matrix's runs."""

    runs: list[MatrixRun]


class MatrixResult(Result[ResultMetadata, MatrixOutput]):
    """A task matrix's result: each run's result, in order, whether or not one failed.

    Any failed run fails it (``success`` is false, ``errors`` names each failed run), and its runs stay readable in
    :attr:`runs`. ``health["status"]`` is the worst of the runs, and :attr:`warning_count` adds up theirs. Its report
    opens with a table comparing the runs' findings; its JSON holds each run's result under ``runs``. It has no
    thumbnails of its own: each run keeps its own, since a run's items can differ from another's at one address.

    Fields
    ------
    keys
        The matrix's keys as written, in order, across grids.
    runs
        Each :class:`MatrixRun`, in order.

    Examples
    --------
    >>> from dataeval_flow import load_config, run_tasks
    >>> result = run_tasks(load_config("pipeline.yaml"))["clean"]  # doctest: +SKIP
    >>> result.runs[2].result  # run 3's result, a ChainResult for a preset  # doctest: +SKIP
    """

    kind: ClassVar[ResultKind] = "matrix"
    metadata_type: ClassVar[type[ResultMetadata]] = ResultMetadata

    def __init__(
        self,
        *,
        type: str,  # noqa: A002
        keys: Sequence[str],
        runs: Sequence[MatrixRun],
        metadata: ResultMetadata | None = None,
    ) -> None:
        runs = list(runs)
        success = all(run.result.success for run in runs)
        errors = [
            f"run {run.number} ({run.label}): {run.result.errors[0] if run.result.errors else 'failed'}"
            for run in runs
            if not run.result.success
        ]
        super().__init__(
            type=type,
            success=success,
            metadata=metadata if metadata is not None else ResultMetadata(),
            output=MatrixOutput(runs) if success else None,
            errors=errors,
        )
        self.keys: list[str] = list(keys)
        self.runs: list[MatrixRun] = runs

    @property
    def warning_count(self) -> int:
        """Every run's warnings, added up."""
        return sum(run.warning_count for run in self.runs)

    @property
    def health(self) -> dict[str, Any]:
        """``failed`` when a run failed, else ``warning`` when one warned, else ``ok``; with the counts behind it."""
        failed = [run.number for run in self.runs if run.status == "failed"]
        warnings = self.warning_count
        status = "failed" if failed else "warning" if warnings else "ok"
        return {"status": status, "warnings": warnings, "failed_runs": failed}

    def to_dict(self) -> dict[str, object]:
        """Kind, type, keys, envelope, health and every run's result, as its type writes it, then any errors.
        Non-finite floats are written as ``null``."""
        payload: dict[str, object] = {
            "kind": self.kind,
            "type": self.type,
            "keys": list(self.keys),
            "metadata": self.metadata.model_dump(mode="json"),
            "health": self.health,
            "runs": [
                {
                    "number": run.number,
                    "label": run.label,
                    "values": to_jsonable_python(run.values, fallback=str),
                    "result": run.result.to_dict(),
                }
                for run in self.runs
            ],
        }
        if self.errors:
            payload["errors"] = list(self.errors)
        return cast("dict[str, object]", finite_json(payload))

    def _report_title(self) -> str:
        return self.runs[0].result._report_title()  # noqa: SLF001 - the banner its runs draw

    def _report_ran(self) -> "tuple[str, Scalar]":
        label, ran = self.runs[0].result._report_ran()  # noqa: SLF001 - as its runs name what ran
        return label, f"{ran}, matrix of {len(self.runs)} runs"

    def _report_body(self, *, detailed: bool) -> "list[Block]":
        """The comparison and, when *detailed*, each run's report, whether or not a run failed."""
        from dataeval_flow._matrix._table import comparison_blocks

        return comparison_blocks(self, detailed=detailed)

    def _html_reports(self, *, detailed: bool) -> "list[tuple[Section, list[Asset]]]":
        """The comparison, then, when *detailed*, each run's report with its own thumbnails: a chain names an item by
        its node's address, which holds other items in a run reading another source."""
        from dataeval_flow._matrix._table import run_section

        runs = [(run_section(run), run.result.assets) for run in self.runs] if detailed else []
        return [(self._document(detailed=False), []), *runs]

    def _report_output(self, *, detailed: bool) -> "list[Block]":
        return self._report_body(detailed=detailed)

    def _dict_body(self) -> dict[str, object]:
        return {key: value for key, value in self.to_dict().items() if key not in ("kind", "metadata")}
