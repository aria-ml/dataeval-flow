"""Text report rendering helpers for executive-summary output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataeval_flow.workflow.base import Reportable

__all__ = ["_WIDTH", "_render_config_section", "_render_detail_section", "_summary_line"]

# Width of the report (matches the === bars).
_WIDTH = 80
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
    marker = {"warning": "  [!!]", "ok": "  [ok]", "info": "  [..]"}.get(severity, "  [..]")

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
    elif rt == "chunk_table":
        lines.extend(_render_chunk_table(data))
    elif rt == "classwise_table":
        lines.extend(_render_classwise_table(data))
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

    # Fallback: render remaining scalar key-value pairs as a simple table
    handled = {"brief", "per_metric", "detail_lines", "total_flags", "count", "multi_metric_subject"}
    generic = {k: v for k, v in data.items() if k not in handled and not isinstance(v, (dict, list))}
    if generic:
        lines.append("")
        w_key = max(len(str(k)) for k in generic)
        for key, val in generic.items():
            lines.append(f"  {key:<{w_key}}  {val}")

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
    key_aliases: dict[str, str] = {"%": "pct", "Class Name": "class_name", "Count": "count"}
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

    # Footer lines (workflow-provided)
    footer_lines = data.get("footer_lines", [])
    if footer_lines:
        lines.append("")
        lines.extend(f"  {line}" for line in footer_lines)

    return lines


def _render_chunk_table(data: dict[str, Any]) -> list[str]:
    """Render chunked drift results — bar chart with threshold scale."""
    lines: list[str] = []
    rows: list[dict[str, Any]] = data.get("table_rows", [])
    if not rows:
        return lines

    # --- Bar chart with threshold scale ---
    bar_width = _BAR_MAX

    # Collect all values for scale calculation
    distances = [r["Distance"] for r in rows]
    upper_thresholds = [r["UpperThreshold"] for r in rows if r.get("UpperThreshold") is not None]
    lower_thresholds = [r["LowerThreshold"] for r in rows if r.get("LowerThreshold") is not None]

    # Thresholds (use first row's as representative)
    lower_thresh = lower_thresholds[0] if lower_thresholds else None
    upper_thresh = upper_thresholds[0] if upper_thresholds else None

    # Scale range: extend to cover both thresholds and all distances
    all_vals = distances + upper_thresholds + lower_thresholds
    scale_min = min(min(all_vals), 0.0)
    scale_max = max(all_vals) if all_vals else 1.0
    scale_range = scale_max - scale_min or 1.0

    def _val_to_pos(val: float) -> int:
        return int(((val - scale_min) / scale_range) * bar_width)

    # Threshold positions on the bar
    lower_pos = _val_to_pos(lower_thresh) if lower_thresh is not None else None
    upper_pos = _val_to_pos(upper_thresh) if upper_thresh is not None else None

    # Column widths
    w_chunk = max(5, *(len(str(r["Chunk"])) for r in rows))
    w_dist = max(8, *(len(f"{r['Distance']:.4f}") for r in rows))

    lines.append("")
    hdr = f"  {'Chunk':<{w_chunk}}  {'Distance':>{w_dist}}  {'':>{bar_width}}  Status"
    lines.append(hdr)
    lines.append(f"  {'-' * w_chunk}  {'-' * w_dist}  {'-' * bar_width}  ------")

    for row in rows:
        dist = row["Distance"]
        status = row["Status"]

        # Bar: █ from zero to distance, ░ elsewhere
        zero_pos = max(0, min(_val_to_pos(0.0), bar_width))
        dist_pos = max(0, min(_val_to_pos(dist), bar_width))
        bar_chars = ["\u2591"] * bar_width
        lo, hi = min(zero_pos, dist_pos), max(zero_pos, dist_pos)
        for p in range(lo, hi):
            bar_chars[p] = "\u2588"
        bar = "".join(bar_chars)

        lines.append(f"  {row['Chunk']:<{w_chunk}}  {dist:>{w_dist}.4f}  {bar}  {status}")

    # --- Threshold scale line: "(lower)|--------|(upper)" ---
    # The lower label can extend left into the prefix area (Threshold + Distance cols).
    if lower_pos is not None or upper_pos is not None:
        lower_label = f"({lower_thresh:.4f})" if lower_thresh is not None else ""
        upper_label = f"({upper_thresh:.4f})" if upper_thresh is not None else ""

        # prefix_width = indentation + Chunk col + gap + Distance col + gap before bar
        prefix_width = 2 + w_chunk + 2 + w_dist + 2
        # "Threshold" label takes up the first part of the prefix
        thresh_label = "  Threshold"
        min_prefix = len(thresh_label) + 1  # at least one space after "Threshold"

        if lower_pos is not None and upper_pos is not None:
            lp = min(max(lower_pos, 0), bar_width - 1)
            up = min(max(upper_pos, 0), bar_width - 1)
            if lp == up:
                scale_core = f"{lower_label}|{upper_label}"
            else:
                gap = "-" * max(0, up - lp - 1)
                scale_core = f"{lower_label}|{gap}|{upper_label}"
            # Position of first | in scale_core
            pipe_idx = len(lower_label)
            # We want pipe_idx to land at (prefix_width + lp) in the full line
            line_start = prefix_width + lp - pipe_idx
        elif upper_pos is not None:
            up = min(max(upper_pos, 0), bar_width - 1)
            scale_core = "-" * up + f"|{upper_label}"
            line_start = prefix_width
        else:
            lp = min(max(lower_pos, 0), bar_width - 1)  # type: ignore[arg-type]
            scale_core = f"{lower_label}|"
            pipe_idx = len(lower_label)
            line_start = prefix_width + lp - pipe_idx

        # Build the full line, ensuring "Threshold" label is visible
        line_start = max(line_start, min_prefix)
        full_line = thresh_label + " " * (line_start - len(thresh_label)) + scale_core
        lines.append(full_line)

    return lines


def _render_classwise_table(data: dict[str, Any]) -> list[str]:
    """Render classwise drift results — bar chart per class."""
    lines: list[str] = []
    rows: list[dict[str, Any]] = data.get("table_rows", [])
    if not rows:
        return lines

    bar_width = _BAR_MAX

    # Collect distances for scale
    distances = [abs(r["Distance"]) for r in rows]
    scale_max = max(distances) if distances else 1.0
    scale_max = scale_max or 1.0  # avoid division by zero

    def _val_to_pos(val: float) -> int:
        return int((val / scale_max) * bar_width)

    # Column widths
    w_class = max(5, *(len(str(r["Class"])) for r in rows))
    w_dist = max(8, *(len(f"{r['Distance']:.4f}") for r in rows))

    # Optional p_val column
    has_pval = any(r.get("PVal") is not None for r in rows)
    w_pval = 6
    if has_pval:
        w_pval = max(w_pval, *(len(f"{r['PVal']:.2f}") for r in rows if r.get("PVal") is not None))

    lines.append("")
    hdr = f"  {'Class':<{w_class}}  {'Distance':>{w_dist}}"
    sep = f"  {'-' * w_class}  {'-' * w_dist}"
    if has_pval:
        hdr += f"  {'PVal':>{w_pval}}"
        sep += f"  {'-' * w_pval}"
    hdr += f"  {'':>{bar_width}}  Status"
    sep += f"  {'-' * bar_width}  ------"
    lines.append(hdr)
    lines.append(sep)

    for row in rows:
        dist = abs(row["Distance"])
        status = row["Status"]
        drifted = status == "DRIFT"

        # Bar: █ for distance, ░ for remainder
        dist_pos = max(0, min(_val_to_pos(dist), bar_width))
        fill_char = "\u2588" if drifted else "\u2591"
        bar = fill_char * dist_pos + "\u2591" * (bar_width - dist_pos)

        line = f"  {row['Class']:<{w_class}}  {row['Distance']:>{w_dist}.4f}"
        if has_pval:
            pval = row.get("PVal")
            line += f"  {pval:>{w_pval}.2f}" if pval is not None else f"  {'':>{w_pval}}"
        line += f"  {bar}  {status}"
        lines.append(line)

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


# ---------------------------------------------------------------------------
# Configuration section
# ---------------------------------------------------------------------------


def _render_config_section(resolved_config: dict[str, Any]) -> list[str]:
    """Render a CONFIGURATION section showing the fully resolved config."""
    if not resolved_config:
        return []

    lines = _section_header("CONFIGURATION")
    _format_value(lines, resolved_config, indent=2, max_width=_WIDTH)
    return lines


_INDENT_STEP = 2


def _flow_repr(obj: Any) -> str:
    """Render a value as a compact, unquoted, single-line string.

    Dicts use ``{k: v, ...}`` syntax, lists use ``[v, ...]``, and
    contiguous int lists collapse to ``range(...)`` shorthand.
    """
    if isinstance(obj, dict):
        inner = ", ".join(f"{k}: {_flow_repr(v)}" for k, v in obj.items())
        return "{" + inner + "}"
    if isinstance(obj, list):
        if obj and all(isinstance(i, int) for i in obj):
            compact = _compact_indices(obj)
            if compact != str(obj):
                return compact
        return "[" + ", ".join(_flow_repr(v) for v in obj) + "]"
    return str(obj)


def _format_value(lines: list[str], obj: Any, indent: int, max_width: int) -> None:
    """Recursively format *obj*, using flow style when it fits in *max_width*.

    Dicts and lists are rendered block-style (one key/item per line) only
    when their flow representation would exceed *max_width*.  Otherwise
    the value is kept inline.
    """
    prefix = " " * indent

    if isinstance(obj, dict):
        _format_dict(lines, obj, indent, max_width)
    elif isinstance(obj, list):
        _format_list(lines, obj, indent, max_width)
    else:
        lines.append(f"{prefix}{obj}")


def _format_dict(lines: list[str], obj: dict[str, Any], indent: int, max_width: int) -> None:
    """Format a dict, keeping values inline when they fit."""
    prefix = " " * indent
    for key, val in obj.items():
        flow = _flow_repr(val)
        if len(f"{prefix}{key}: {flow}") <= max_width:
            lines.append(f"{prefix}{key}: {flow}")
        else:
            lines.append(f"{prefix}{key}:")
            _format_value(lines, val, indent + _INDENT_STEP, max_width)


def _format_list(lines: list[str], obj: list[Any], indent: int, max_width: int) -> None:
    """Format a list, putting the first dict key on the ``- `` line."""
    prefix = " " * indent
    for item in obj:
        flow = _flow_repr(item)
        if len(f"{prefix}- {flow}") <= max_width:
            lines.append(f"{prefix}- {flow}")
        elif isinstance(item, dict) and item:
            _format_list_dict_item(lines, item, indent, max_width)
        else:
            lines.append(f"{prefix}-")
            _format_value(lines, item, indent + _INDENT_STEP, max_width)


def _format_list_dict_item(lines: list[str], item: dict[str, Any], indent: int, max_width: int) -> None:
    """Format a dict inside a list, inlining the first key on the ``- `` line."""
    prefix = " " * indent
    it = iter(item.items())
    first_key, first_val = next(it)
    first_flow = _flow_repr(first_val)
    if len(f"{prefix}- {first_key}: {first_flow}") <= max_width:
        lines.append(f"{prefix}- {first_key}: {first_flow}")
    else:
        lines.append(f"{prefix}- {first_key}:")
        _format_value(lines, first_val, indent + _INDENT_STEP * 2, max_width)
    # Remaining keys align under the first key (one indent step past the "- ")
    rest_indent = indent + _INDENT_STEP
    _format_dict(lines, dict(it), rest_indent, max_width)


def _compact_indices(indices: list[int]) -> str:
    """Collapse a contiguous int list into range shorthand for display."""
    if not indices:
        return "[]"
    if len(indices) < 2:
        return str(indices)
    step = indices[1] - indices[0]
    if step == 0:
        return str(indices)
    stop = indices[-1] + (1 if step > 0 else -1)
    if indices == list(range(indices[0], stop, step)):
        if step == 1:
            return f"range({indices[0]}, {stop})"
        return f"range({indices[0]}, {stop}, {step})"
    return str(indices)
