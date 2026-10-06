"""ViewModel for rendering workflow results in the TUI.

Transforms a workflow or evaluator result into view-ready structures.
No Textual dependency — consumed by the result modal and result cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataeval_flow._blocks import SummaryItem
from dataeval_flow._blocks._text import Frame, summary_line

__all__ = ["FindingSummary", "ResultViewModel"]


@dataclass
class FindingSummary:
    """View-ready summary of a single report finding."""

    title: str
    severity: str  # "ok" | "info" | "warning"
    brief: str


class ResultViewModel:
    """Transforms a task's ``Result`` into view-ready structures."""

    def __init__(self, result: Any) -> None:
        from dataeval_flow._matrix._result import MatrixResult
        from dataeval_flow.evaluators._result import EvaluatorResult
        from dataeval_flow.steps._result import ChainResult

        self._result = result
        self._is_evaluator = isinstance(result, EvaluatorResult)
        # A workflow's result holds its steps, not one report: its findings are its steps'.
        self._is_chain = isinstance(result, ChainResult)
        # A matrix result holds its runs, not one report: it shows its comparison and each run's report.
        self._is_matrix = isinstance(result, MatrixResult)
        self._findings = self._extract_findings()

    def _extract_findings(self) -> list[Any]:
        # A chain's steps that completed keep their findings, even where another step failed; no other result has any.
        return list(self._result.findings) if self._is_chain else []

    def output_text(self) -> str:
        """The result's output, as the text report renders it, every row included.

        An evaluator's output, or a workflow's or matrix's report body: a workflow's summary, then each step's
        section. Empty for any other result.
        """
        from dataeval_flow._blocks._text import Frame, render_text
        from dataeval_flow.evaluators._report import render_result_body

        if self._is_evaluator:
            return "\n".join(render_result_body(self._result, detailed=True))
        if self._is_chain or self._is_matrix:
            body = self._result._report_body(detailed=True)  # noqa: SLF001 - the report's body has no public accessor
            return "\n".join(render_text(body, Frame(indent="  ", depth=1)))
        return ""

    def status_tag(self) -> str:
        """The result card's status marker.

        A failed run of either kind shows a run-status marker, since a failure is not a finding.
        Otherwise, for a workflow this is the health verdict (``[ok]``/``[!!]``); an evaluator
        carries no health verdict, so a successful one is empty.
        """
        if not self._result.success:
            return " [bold red][failed][/bold red]"
        if self._is_evaluator:
            return ""
        return " [bold red][!!][/bold red]" if self.warning_count() else " [green][ok][/green]"

    # -- Summary -----------------------------------------------------------

    def summary_line(self) -> str:
        """One-line summary: finding count, warning count, duration."""
        if self._is_evaluator:
            from dataeval_flow.evaluators._report import serialized_of

            if not self._result.success:
                errors = self._result.errors
                parts = [f"failed: {errors[0]}" if errors else "failed"]
            else:
                output = serialized_of(self._result)
                count = len(output.get("rows", output.get("data", [])))
                noun = "row" if output.get("shape") == "table" else "value"
                parts = [f"{count} {noun}{'s' if count != 1 else ''}"]
            if self._result.metadata.execution_time_s is not None:
                parts.append(f"{self._result.metadata.execution_time_s:.1f}s")
            return ", ".join(parts)

        if self._is_matrix:
            parts = [f"{len(self._result.runs)} runs, {self.warning_count()} warning(s)"]
            if self._result.metadata.execution_time_s is not None:
                parts.append(f"{self._result.metadata.execution_time_s:.1f}s")
            return ", ".join(parts)

        findings = self._findings
        n = len(findings)
        warnings = self.warning_count()
        parts: list[str] = []
        parts.append(f"{n} finding{'s' if n != 1 else ''}")
        if warnings:
            parts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
        meta = self._result.metadata
        if meta.execution_time_s is not None:
            parts.append(f"{meta.execution_time_s:.1f}s")
        return ", ".join(parts)

    # -- Metadata ----------------------------------------------------------

    def metadata_lines(self) -> list[str]:
        """Human-readable metadata lines (timestamp, duration, source, model)."""
        meta = self._result.metadata
        lines: list[str] = []
        if meta.timestamp:
            lines.append(f"Timestamp:    {meta.timestamp.isoformat()}")
        if meta.execution_time_s is not None:
            lines.append(f"Duration:     {meta.execution_time_s:.2f}s")
        lines.extend(f"Source:       {desc}" for desc in getattr(meta, "source_descriptions", []))
        if meta.model_id:
            lines.append(f"Model:        {meta.model_id}")
        if meta.preprocessor_id:
            lines.append(f"Preprocessor: {meta.preprocessor_id}")
        return lines

    # -- Findings ----------------------------------------------------------

    def finding_count(self) -> int:
        """Number of findings."""
        return len(self._findings)

    def warning_count(self) -> int:
        """How many findings are warnings, as the result counted them; an evaluator's result has no findings."""
        return 0 if self._is_evaluator else int(self._result.warning_count)

    def finding_summaries(self) -> list[FindingSummary]:
        """Return view-ready summaries for all findings."""
        return [
            FindingSummary(
                title=finding.title,
                severity=getattr(finding, "severity", "info"),
                brief=finding.brief or "",
            )
            for finding in self._findings
        ]

    def finding_summary_markup(self, idx: int) -> str:
        """Rich-markup summary line for finding at *idx* (dotted summary style).

        A title and brief too long for one line wrap onto more, joined by ``\\n``; the last line
        carries the brief and the severity marker.
        """
        if 0 <= idx < len(self._findings):
            finding = self._findings[idx]
            item = SummaryItem(label=finding.title, value=finding.brief or "", severity=finding.severity)
            return "\n".join(line.rstrip() for line in summary_line(item, Frame(indent="  ")))
        return ""
