"""ViewModel for rendering workflow results in the TUI.

Transforms a workflow or evaluator result into view-ready structures.
No Textual dependency — consumed by the result modal and result cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeGuard

from dataeval_flow._blocks import Block, Paragraph, SummaryItem, Table
from dataeval_flow._blocks._table import cell_text
from dataeval_flow._blocks._text import Frame, summary_line

__all__ = ["FindingSummary", "ResultViewModel", "Segment", "table_data"]

# A run of blocks the detail view draws as text, or a table it shows as a native ``DataTable``.
Segment = Table | list[Block]


@dataclass
class FindingSummary:
    """View-ready summary of a single report finding."""

    title: str
    severity: str  # "ok" | "info" | "warning"
    brief: str


def _is_data_table(block: Block) -> TypeGuard[Table]:
    """A table a ``DataTable`` shows in full: rows, and nothing drawn, neither chart nor threshold marker.

    A table with a chart stays in the text, where its bars, stacks and threshold lines draw;
    a ``DataTable`` cell holds only text.
    """
    return isinstance(block, Table) and bool(block.rows) and all(column.kind == "text" for column in block.columns)


def table_data(table: Table) -> tuple[list[str], list[list[str]]]:
    """A data table's headers and rows as text, each cell printed as the text report prints it."""
    rows = [[cell_text(column, row.get(column.key)) for column in table.columns] for row in table.rows]
    return [column.header for column in table.columns], rows


class ResultViewModel:
    """Transforms a ``WorkflowResult`` into view-ready structures."""

    def __init__(self, result: Any) -> None:
        from dataeval_flow.evaluators._result import EvaluatorResult
        from dataeval_flow.steps._result import ChainResult

        self._result = result
        self._is_evaluator = isinstance(result, EvaluatorResult)
        # A custom workflow's result holds its steps, not one report: its findings are its steps'.
        self._is_chain = isinstance(result, ChainResult)
        self._findings = self._extract_findings()

    def _extract_findings(self) -> list[Any]:
        if self._is_chain:  # the steps that completed keep their findings, even where another step failed
            return list(self._result.findings)
        if self._is_evaluator or not self._result.success:
            return []
        return list(self._result.output.report.findings)

    @property
    def shows_output(self) -> bool:
        """Whether the detail view shows :meth:`output_text` in place of findings and health.

        An evaluator's result holds determinations only, and a custom workflow's holds its steps, each with its
        status, errors and output.
        """
        return self._is_evaluator or self._is_chain

    def output_text(self) -> str:
        """The rendered output :attr:`shows_output` names, as the text report renders it, every row included.

        An evaluator's output, or a custom workflow's report body: its summary, then each step's section. Empty for
        any other result.
        """
        from dataeval_flow._blocks._text import Frame, render_text
        from dataeval_flow.evaluators._report import render_result_body

        if self._is_evaluator:
            return "\n".join(render_result_body(self._result, detailed=True))
        if self._is_chain:
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

        findings = self._findings
        n = len(findings)
        warnings = sum(1 for f in findings if getattr(f, "severity", "info") == "warning")
        parts: list[str] = []
        parts.append(f"{n} finding{'s' if n != 1 else ''}")
        if warnings:
            parts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
        meta = self._result.metadata
        if meta.execution_time_s is not None:
            parts.append(f"{meta.execution_time_s:.1f}s")
        return ", ".join(parts)

    def report_summary(self) -> str:
        """A workflow type's own summary string (e.g. 'Data Cleaning Report'); empty for any other result."""
        if self._is_evaluator or self._is_chain or not self._result.success:
            return ""
        return self._result.output.report.summary

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
        """Number of findings with severity 'warning'."""
        return sum(1 for f in self._findings if getattr(f, "severity", "info") == "warning")

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

    def finding_blocks(self, idx: int) -> list[Block]:
        """The finding at *idx* as its detail draws it: the description as a lede, then its evidence."""
        if not 0 <= idx < len(self._findings):
            return []
        finding = self._findings[idx]
        lede: list[Block] = [Paragraph(text=finding.description)] if finding.description else []
        return [*lede, *finding.blocks]

    def finding_segments(self, idx: int) -> list[Segment]:
        """The finding's blocks in the order they draw: each data table on its own, the rest in runs of text."""
        segments: list[Segment] = []
        for block in self.finding_blocks(idx):
            if _is_data_table(block):
                segments.append(block)
            elif segments and isinstance(segments[-1], list):
                segments[-1].append(block)
            else:
                segments.append([block])
        return segments

    # -- Health summary ----------------------------------------------------

    def health_line(self) -> str:
        """Health status string for the summary section."""
        if self._is_evaluator:
            return ""
        warnings = self.warning_count()
        if warnings:
            return f"Health: {warnings} warning(s) — review flagged findings"
        return "Health: All checks passed"
