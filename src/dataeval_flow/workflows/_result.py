"""A workflow's result: typed outputs, and the health verdict drawn from their findings."""

__all__ = ["WorkflowResult"]

from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from dataeval_flow._kind import type_arguments
from dataeval_flow._result import Result, ResultMetadata, TMetadata
from dataeval_flow._text_report import _render_binning_section, _render_detail_section, _summary_line
from dataeval_flow.workflows._base import Finding, WorkflowOutput

if TYPE_CHECKING:
    import dataeval_flow.workflows
    from dataeval_flow.config._schemas._task import TaskKind

TOutput = TypeVar("TOutput", bound="WorkflowOutput[Any, Any]")


class WorkflowResult(Result[TMetadata, TOutput]):
    """A workflow's result: a :class:`~dataeval_flow.Result` with typed output and a health verdict.

    ``output`` is the workflow's :class:`WorkflowOutput`, its raw output and report. :attr:`findings`,
    :attr:`warning_count` and :attr:`health` roll the report's findings up into the verdict ``--fail-on-warning``
    reads. ``metadata`` is the JATIC envelope plus any fields the workflow adds; Flow fills in its provenance
    after the run. ``kind`` is ``"workflow"``.

    Subclassing
    -----------
    Subclass it once per workflow, parameterized by the workflow's metadata and output classes, and name the
    subclass as the result type argument of the workflow's :class:`WorkflowConfig` and :class:`Workflow`. The
    subclass needs no body. ``isinstance`` then narrows a :class:`~dataeval_flow.Result` to it and types
    ``output`` and ``metadata`` all the way down. The metadata argument also binds ``metadata_type``, which Flow
    builds a failed result's metadata from, with no arguments.

    :meth:`Workflow.run` builds a successful result with the keyword arguments :class:`~dataeval_flow.Result`
    takes; Flow builds the failed ones, with :meth:`~dataeval_flow.Result.failed`.

    Examples
    --------
    >>> from pydantic import Field
    >>> from dataeval_flow import ResultMetadata
    >>> from dataeval_flow.workflows import WorkflowOutput, WorkflowRawOutput, WorkflowReport, WorkflowResult
    >>> class CountRaw(WorkflowRawOutput):
    ...     counts: dict[str, int] = Field(default_factory=dict, description="Items in each source.")
    >>> class CountOutput(WorkflowOutput[CountRaw, WorkflowReport]):
    ...     pass
    >>> class CountResult(WorkflowResult[ResultMetadata, CountOutput]):
    ...     pass
    >>> output = CountOutput(raw=CountRaw(dataset_size=3, counts={"train": 3}), report=WorkflowReport(summary="3"))
    >>> result = CountResult(type="example.count", success=True, output=output, metadata=ResultMetadata())
    >>> result.health
    {'status': 'ok', 'warnings': 0, 'findings': 0}

    A task's result narrows to it:

    >>> from dataeval_flow import run_tasks
    >>> result = run_tasks(config)["count"]  # doctest: +SKIP
    >>> if isinstance(result, CountResult) and result.success:  # doctest: +SKIP
    ...     print(result.output.raw.counts)
    """

    kind: "ClassVar[TaskKind]" = "workflow"
    metadata_type: ClassVar[type[ResultMetadata]] = ResultMetadata

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Bind ``metadata_type`` from the metadata class this result was parameterized with."""
        super().__init_subclass__(**kwargs)
        arguments = type_arguments(cls, WorkflowResult)
        if arguments and isinstance(arguments[0], type):
            cls.metadata_type = arguments[0]

    @property
    def findings(self) -> "list[dataeval_flow.workflows.Finding]":  # its public path, for the API pages
        """The report's findings, or an empty list where the run failed."""
        if not self.success:
            return []
        return list(self.output.report.findings)

    @property
    def warning_count(self) -> int:
        """How many findings breached their health threshold.

        The same count the report's health line renders, reached without parsing text —
        this is what an automated gate reads to decide whether a run is worth stopping on.
        """
        return sum(1 for f in self.findings if f.severity == "warning")

    @property
    def health(self) -> dict[str, Any]:
        """Machine-readable roll-up of the run's findings.

        ``status`` is ``"warning"`` where any finding breached its threshold, ``"ok"`` otherwise, and
        ``"failed"`` where the run did not complete. A warning is a prompt to look rather than a failure —
        ``success`` stays the authority on whether the workflow ran.
        """
        if not self.success:
            return {"status": "failed", "warnings": 0, "findings": 0}
        warnings = self.warning_count
        return {
            "status": "warning" if warnings else "ok",
            "warnings": warnings,
            "findings": len(self.findings),
        }

    def _report_title(self) -> str:
        """The report's summary."""
        return self.output.report.summary

    def _report_output(self, *, detailed: bool) -> list[str]:
        """Findings with the health line, their details when *detailed*, then the metadata binning."""
        findings = self.findings
        lines = self._summary_lines(findings)
        if detailed:
            lines.extend(self._detail_lines(findings))
        lines.extend(
            _render_binning_section(self.metadata.metadata_binning, self.metadata.diagnostics, detailed=detailed)
        )
        return lines

    def _dict_body(self) -> dict[str, object]:
        """The health roll-up, then the workflow's own output fields."""
        return {"health": self.health, **self.output.model_dump(mode="json")}

    def _summary_lines(self, findings: list[Finding]) -> list[str]:
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

    def _detail_lines(self, findings: list[Finding]) -> list[str]:
        """Expanded detail sections for each finding."""
        lines: list[str] = []
        for finding in findings:
            lines.extend(_render_detail_section(finding))
        return lines
