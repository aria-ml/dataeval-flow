"""A workflow's result: typed outputs, and the health verdict drawn from their findings."""

__all__ = ["WorkflowResult"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

from dataeval_flow.config.schemas import TaskKind
from dataeval_flow.result import Result
from dataeval_flow.workflow._text_report import _render_binning_section, _render_detail_section, _summary_line

if TYPE_CHECKING:
    from dataeval_flow.config.schemas import ResultMetadata

TMetadata = TypeVar("TMetadata", bound="ResultMetadata")
TData = TypeVar("TData", bound=BaseModel)


@dataclass
class WorkflowResult(Result[TMetadata], Generic[TMetadata, TData]):
    """A workflow's result: a :class:`~dataeval_flow.result.Result` with typed outputs and a health verdict.

    ``data`` holds the workflow's outputs, and :attr:`health` rolls its findings up into the
    verdict ``--fail-on-warning`` reads.  ``metadata`` carries the JATIC-required envelope (timestamp, tool info,
    dataset identifiers) plus any workflow-specific extras.  The orchestrator
    populates timing and dataset fields after execution; workflows construct
    the appropriate ``ResultMetadata`` subclass at creation time.

    Parameterize with metadata and data subclasses for typed access to
    workflow-specific fields, e.g.
    ``WorkflowResult[DataCleaningMetadata, DataCleaningOutputs]``.

    The optional ``dataset`` field holds the resolved, post-selection dataset
    used during workflow execution.  This is *not* serialized by
    :meth:`report`; it is provided purely for downstream programmatic use
    (visualization, filtering, export).

    The optional ``sources`` field maps source names to their resolved,
    post-selection datasets.  Multi-split workflows (e.g. data-analysis)
    populate this so callers can visualize images from any split without
    re-loading.
    """

    kind: ClassVar[TaskKind] = "workflow"

    data: TData

    @property
    def findings(self) -> list[Any]:
        """The report's findings, or an empty list where the workflow produced none."""
        report_obj = getattr(self.data, "report", None)
        return list(getattr(report_obj, "findings", []) or []) if report_obj is not None else []

    @property
    def warning_count(self) -> int:
        """How many findings breached their health threshold.

        The same count the report's health line renders, reached without parsing text —
        this is what an automated gate reads to decide whether a run is worth stopping on.
        """
        return sum(1 for f in self.findings if getattr(f, "severity", "info") == "warning")

    @property
    def health(self) -> dict[str, Any]:
        """Machine-readable roll-up of the run's findings.

        ``status`` is ``"warning"`` where any finding breached its threshold and ``"ok"``
        otherwise.  A warning is a prompt to look rather than a failure — ``success``
        stays the authority on whether the workflow ran.
        """
        warnings = self.warning_count
        return {
            "status": "warning" if warnings else "ok",
            "warnings": warnings,
            "findings": len(self.findings),
        }

    def _report_title(self) -> str:
        """The report summary, or the workflow's name where the workflow produced no report."""
        report_obj = getattr(self.data, "report", None)
        return self.name if report_obj is None else report_obj.summary

    def _report_output(self, *, detailed: bool) -> list[str]:
        """Findings with the health line, their details when *detailed*, then the metadata binning."""
        report_obj = getattr(self.data, "report", None)
        if report_obj is None:
            return ["  No report available."]
        findings = getattr(report_obj, "findings", [])
        lines = self._summary_lines(findings)
        if detailed:
            lines.extend(self._detail_lines(findings))
        lines.extend(
            _render_binning_section(self.metadata.metadata_binning, self.metadata.diagnostics, detailed=detailed)
        )
        return lines

    def _dict_body(self) -> dict[str, object]:
        """The health roll-up, then the workflow's own output fields."""
        return {"health": self.health, **self.data.model_dump(mode="json")}

    def _summary_lines(self, findings: list) -> list[str]:
        """Summary section with per-finding one-liners and health status."""
        if not findings:
            return ["  No findings to report."]

        warnings = self.warning_count
        lines = ["", "  SUMMARY", "  -------"]
        lines.extend(_summary_line(f) for f in findings)
        lines.append("")
        if warnings:
            lines.append(f"  Health: {warnings} warning(s) [!!] — review flagged findings")
        else:
            lines.append("  Health: All checks passed [ok]")
        return lines

    def _detail_lines(self, findings: list) -> list[str]:
        """Expanded detail sections for each finding."""
        lines: list[str] = []
        for finding in findings:
            lines.extend(_render_detail_section(finding))
        return lines
