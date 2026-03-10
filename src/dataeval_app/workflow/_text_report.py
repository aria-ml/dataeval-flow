"""Text report rendering helpers for executive-summary output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataeval_app.workflow.base import Reportable

__all__ = ["_WIDTH", "_render_detail_section", "_summary_line"]

# Width of the report (matches the === bars).
_WIDTH = 64
# Maximum bar chart width in characters.
_BAR_MAX = 30
# Unicode left-filling fractional block characters (index 1 = 1/8 .. 7 = 7/8).
_FRAC_BLOCKS = " \u258f\u258e\u258d\u258c\u258b\u258a\u2589"


# ---------------------------------------------------------------------------
# Summary line (one per finding, for the SUMMARY section)
# ---------------------------------------------------------------------------


def _brief_value(finding: Reportable) -> str:
    """Extract a short value string from a finding's data for the summary line."""
    data = finding.data
    if isinstance(data, dict):
        brief = data.get("brief")
        if brief is not None:
            return str(brief)
    return ""


def _summary_line(finding: Reportable) -> str:
    """Build a single dotted summary line for a finding."""
    label = finding.title
    value = _brief_value(finding)
    severity = getattr(finding, "severity", "info")
    marker = "  [!]" if severity == "warning" else ""

    # Dotted fill between label and value
    dots_len = _WIDTH - 4 - len(label) - len(value) - len(marker)
    dots = " " + "." * max(dots_len - 2, 1) + " " if dots_len > 3 else " "
    return f"  {label}{dots}{value}{marker}"


# ---------------------------------------------------------------------------
# Detail section rendering
# ---------------------------------------------------------------------------


def _section_header(title: str, right_text: str = "") -> list[str]:
    """Render a section header with === bars."""
    lines: list[str] = [""]
    lines.append("=" * _WIDTH)
    if right_text:
        padding = _WIDTH - 2 - len(title) - len(right_text)
        lines.append(f"  {title}{' ' * max(padding, 1)}{right_text}")
    else:
        lines.append(f"  {title}")
    lines.append("=" * _WIDTH)
    return lines


def _render_detail_section(finding: Reportable) -> list[str]:
    """Render a full detail section for a finding."""
    data = finding.data
    brief = _brief_value(finding)
    lines = _section_header(finding.title.upper(), brief)

    if finding.description:
        lines.append(f"  {finding.description}")

    if not isinstance(data, dict):
        # Plain text or non-dict data
        if isinstance(data, str) and data:
            lines.append(f"  {data}")
        return lines

    rt = finding.report_type

    if rt == "pivot_table":
        lines.extend(_render_pivot_table(data))
    elif rt == "table":
        lines.extend(_render_table(data))
    elif rt == "key_value":
        lines.extend(_render_key_value(data))
    else:
        # text / image / unknown — just show description (already added above)
        pass

    return lines


# ---------------------------------------------------------------------------
# Type-specific renderers
# ---------------------------------------------------------------------------


def _render_key_value(data: dict[str, Any]) -> list[str]:
    """Render key_value findings — metric tables, detail lines, generic pairs."""
    lines: list[str] = []

    # Outlier per-metric breakdown
    per_metric = data.get("per_metric")
    if per_metric and isinstance(per_metric, dict):
        lines.append("")
        col1 = "Metric"
        col2 = "Count"
        w1 = max(len(col1), *(len(k) for k in per_metric))
        lines.append(f"  {col1:<{w1}}  {col2}")
        lines.append(f"  {'-' * w1}  -----")
        for metric, count in sorted(per_metric.items(), key=lambda x: -x[1]):
            lines.append(f"  {metric:<{w1}}  {count:>5}")
        total_flags = data.get("total_flags", 0)
        unique_count = data.get("count", 0)
        if total_flags > unique_count:
            subject = data.get("multi_metric_subject", "items")
            lines.append("")
            lines.append(f"  (Some {subject} trigger multiple metrics.)")

    # Generic detail lines (workflow-provided)
    detail_lines = data.get("detail_lines", [])
    if detail_lines:
        lines.append("")
        lines.extend(f"  {line}" for line in detail_lines)

    return lines


def _render_pivot_table(data: dict[str, Any]) -> list[str]:
    """Render pivot-table findings — multi-column tables like classwise outliers.

    ``table_headers`` are display names; row dicts use field keys.  A
    ``field_keys`` list in *data* maps each header to its row-dict key
    (defaults to headers themselves).  The ``%`` header is special-cased
    to read the ``pct`` field and format with one decimal + ``%`` suffix.
    """
    lines: list[str] = []
    rows: list[dict[str, Any]] = data.get("table_data", [])
    headers: list[str] = data.get("table_headers", [])
    if not rows or not headers:
        return lines

    # Map display header → row-dict key
    key_aliases: dict[str, str] = {"%": "pct"}
    keys = [key_aliases.get(h, h) for h in headers]

    # Format cell values: pct → "12.3%", others → str
    def _fmt(key: str, val: object) -> str:
        if key == "pct" and isinstance(val, (int, float)):
            return f"{val:.1f}%"
        return str(val) if val is not None else ""

    # Pre-format all cells for width calculation
    formatted: list[list[str]] = [[_fmt(k, row.get(k, "")) for k in keys] for row in rows]

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for cells in formatted:
        for i, cell in enumerate(cells):
            col_widths[i] = max(col_widths[i], len(cell))

    # Header
    lines.append("")
    header_line = "  " + "  ".join(
        f"{h:<{w}}" if i == 0 else f"{h:>{w}}" for i, (h, w) in enumerate(zip(headers, col_widths, strict=False))
    )
    lines.append(header_line)
    lines.append("  " + "  ".join("-" * w for w in col_widths))

    # Data rows
    for cells in formatted:
        parts: list[str] = []
        for i, (cell, w) in enumerate(zip(cells, col_widths, strict=False)):
            parts.append(f"{cell:<{w}}" if i == 0 else f"{cell:>{w}}")
        lines.append("  " + "  ".join(parts))

    return lines


def _render_table(data: dict[str, Any]) -> list[str]:
    """Render table findings — generic table with bar chart."""
    lines: list[str] = []
    table_data: dict[str, int] = data.get("table_data", {})
    if not table_data:
        return lines

    max_count = max(table_data.values())
    headers = data.get("table_headers", ("Name", "Value"))
    col1, col2 = headers[0], headers[1]
    w1 = max(len(col1), *(len(k) for k in table_data))
    w2 = max(len(col2), len(str(max_count)))

    lines.append("")
    lines.append(f"  {col1:<{w1}}  {col2:>{w2}}")
    lines.append(f"  {'-' * w1}  {'-' * w2}")

    for cls, count in sorted(table_data.items(), key=lambda x: -x[1]):
        bar_len = (count / max_count) * _BAR_MAX if max_count else 0
        full = int(bar_len)
        frac = bar_len - full
        bar = "\u2588" * full
        # Append fractional block (1/8 to 7/8) for the remainder
        frac_idx = int(frac * 8)
        if frac_idx > 0:
            bar += _FRAC_BLOCKS[frac_idx]
        lines.append(f"  {cls:<{w1}}  {count:>{w2}}  {bar}")

    # Generic footer lines (workflow-provided)
    footer_lines = data.get("footer_lines", [])
    if footer_lines:
        lines.append("")
        lines.extend(f"  {line}" for line in footer_lines)

    return lines
