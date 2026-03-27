"""ViewModel for rendering workflow results in the TUI.

Transforms ``WorkflowResult`` data into view-ready structures.
No Textual dependency — consumed by the result modal and result cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataeval_flow.workflow._text_report import (
    _brief_value,
    _render_detail_section,
    _summary_line,
)

__all__ = ["FindingSummary", "ResultViewModel"]


@dataclass
class FindingSummary:
    """View-ready summary of a single report finding."""

    title: str
    severity: str  # "ok" | "info" | "warning"
    brief: str
    report_type: str
    has_table: bool


class ResultViewModel:
    """Transforms a ``WorkflowResult`` into view-ready structures."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self._findings = self._extract_findings()

    def _extract_findings(self) -> list[Any]:
        report_obj = getattr(self._result.data, "report", None)
        if report_obj is None:
            return []
        return list(getattr(report_obj, "findings", []))

    # -- Summary -----------------------------------------------------------

    def summary_line(self) -> str:
        """One-line summary: finding count, warning count, duration."""
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
        """The workflow's own summary string (e.g. 'Data Cleaning Report')."""
        report_obj = getattr(self._result.data, "report", None)
        if report_obj is None:
            return ""
        return getattr(report_obj, "summary", "")

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
        summaries: list[FindingSummary] = []
        for finding in self._findings:
            rt = finding.report_type
            summaries.append(
                FindingSummary(
                    title=finding.title,
                    severity=getattr(finding, "severity", "info"),
                    brief=_brief_value(finding),
                    report_type=rt,
                    has_table=rt in ("table", "pivot_table", "classwise_table", "chunk_table"),
                )
            )
        return summaries

    def finding_summary_markup(self, idx: int) -> str:
        """Rich-markup one-liner for finding at *idx* (dotted summary style)."""
        if 0 <= idx < len(self._findings):
            return _summary_line(self._findings[idx])
        return ""

    def finding_detail_markup(self, idx: int) -> str:
        """Rich-markup detail block for finding at *idx*."""
        if 0 <= idx < len(self._findings):
            lines = _render_detail_section(self._findings[idx])
            return "\n".join(lines)
        return ""

    def finding_table_data(self, idx: int) -> tuple[list[str], list[list[str]]] | None:
        """Extract structured table data for ``DataTable`` rendering.

        Returns ``(headers, rows)`` where each row is a list of strings,
        or ``None`` if the finding doesn't have tabular data.
        """
        if not (0 <= idx < len(self._findings)):
            return None

        finding = self._findings[idx]
        data = finding.data
        if not isinstance(data, dict):
            return None

        rt = finding.report_type

        if rt == "table":
            return self._extract_simple_table(data)
        if rt == "pivot_table":
            return self._extract_pivot_table(data)
        if rt in ("classwise_table", "chunk_table"):
            return self._extract_row_table(data)
        return None

    # -- Health summary ----------------------------------------------------

    def health_line(self) -> str:
        """Health status string for the summary section."""
        warnings = self.warning_count()
        if warnings:
            return f"Health: {warnings} warning(s) — review flagged findings"
        return "Health: All checks passed"

    # -- Table extraction helpers ------------------------------------------

    @staticmethod
    def _extract_simple_table(data: dict[str, Any]) -> tuple[list[str], list[list[str]]] | None:
        """Extract from ``table`` report type (dict of name→count)."""
        table_data: dict[str, int] = data.get("table_data", {})
        if not table_data:
            return None
        headers_raw = data.get("table_headers", ("Name", "Value"))
        headers: list[str] = [str(h) for h in headers_raw]
        rows = [[str(k), str(v)] for k, v in sorted(table_data.items(), key=lambda x: -x[1])]
        return headers, rows

    @staticmethod
    def _extract_pivot_table(data: dict[str, Any]) -> tuple[list[str], list[list[str]]] | None:
        """Extract from ``pivot_table`` report type."""
        rows_data: list[dict[str, Any]] = data.get("table_data", [])
        headers: list[str] = data.get("table_headers", [])
        if not rows_data or not headers:
            return None

        key_aliases: dict[str, str] = {"%": "pct", "Class Name": "class_name", "Count": "count"}
        keys = [key_aliases.get(h, h) for h in headers]

        rows: list[list[str]] = []
        for row in rows_data:
            cells: list[str] = []
            for key in keys:
                val = row.get(key, "")
                if key == "pct" and isinstance(val, (int, float)):
                    cells.append(f"{val:.1f}%")
                else:
                    cells.append(str(val) if val is not None else "")
            rows.append(cells)
        return headers, rows

    @staticmethod
    def _extract_row_table(data: dict[str, Any]) -> tuple[list[str], list[list[str]]] | None:
        """Extract from ``classwise_table`` or ``chunk_table`` (list of row dicts)."""
        rows_data: list[dict[str, Any]] = data.get("table_rows", [])
        if not rows_data:
            return None
        headers = list(rows_data[0].keys())
        rows: list[list[str]] = []
        for row in rows_data:
            cells: list[str] = []
            for h in headers:
                val = row.get(h, "")
                if isinstance(val, float):
                    cells.append(f"{val:.4f}")
                else:
                    cells.append(str(val) if val is not None else "")
            rows.append(cells)
        return headers, rows
