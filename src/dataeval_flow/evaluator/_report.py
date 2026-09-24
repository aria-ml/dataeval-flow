"""Plain-text rendering of an evaluator result: the envelope, then DataEval's output as it came."""

__all__ = ["ROW_LIMIT", "render_output", "render_result_body", "render_rows"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from dataeval_flow.result import _failure_lines
from dataeval_flow.workflow._text_report import (
    _WIDTH,
    _flow_repr,
    _format_value,
    _section_header,
)

if TYPE_CHECKING:
    from dataeval_flow.evaluator.result import EvaluatorResult

#: Rows a table shows on the console before the rest are elided; ``result.txt`` shows them all.
ROW_LIMIT = 20
#: Widest a cell renders before it is cut with an ellipsis.
_CELL_WIDTH = 40
#: Array values shown before the rest are elided.
_ARRAY_HEAD = 10


def render_result_body(result: "EvaluatorResult", *, detailed: bool) -> list[str]:
    """Render the result's report body: the output for a success, or ``FAILED`` plus each error otherwise."""
    return render_output(result.output, detailed=detailed) if result.success else _failure_lines(result.errors)


def render_output(output: dict[str, Any], *, detailed: bool) -> list[str]:
    """Render serialized DataEval output under an ``OUTPUT`` header."""
    lines = _section_header("OUTPUT", _brief(output))
    lines.extend(_render_shape(output, detailed=detailed))
    return lines


def render_rows(columns: Sequence[str], rows: Sequence[dict[str, Any]], *, limit: int | None) -> list[str]:
    """Render table rows as aligned columns, eliding rows past *limit* and cutting wide cells."""
    if not rows:
        return ["", "  (no rows)"]
    shown = rows if limit is None else rows[:limit]
    cells = [[_cell(row.get(column)) for column in columns] for row in shown]
    widths = [max(len(column), *(len(row[i]) for row in cells)) for i, column in enumerate(columns)]
    lines = [
        "",
        "  " + "  ".join(column.ljust(width) for column, width in zip(columns, widths, strict=True)),
        "  " + "  ".join("-" * width for width in widths),
    ]
    lines.extend("  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)) for row in cells)
    if limit is not None and len(rows) > limit:
        lines.append(f"  … and {len(rows) - limit} more rows (run with -v, or read result.txt, to see them all)")
    return [line.rstrip() for line in lines]


def _brief(output: dict[str, Any]) -> str:
    shape = output.get("shape")
    if shape == "table":
        count = len(output["rows"])
        return f"{count} row{'s' if count != 1 else ''}"
    if shape == "array":
        count = len(output["data"])
        return f"{count} value{'s' if count != 1 else ''}"
    return ""


def _render_shape(value: Any, *, detailed: bool) -> list[str]:
    shape = value.get("shape") if isinstance(value, dict) else None
    if shape == "table":
        return render_rows(value["columns"], value["rows"], limit=None if detailed else ROW_LIMIT)
    if shape == "array":
        return _render_array(value["data"])
    if shape == "mapping":
        return _render_mapping(value["data"], detailed=detailed)
    lines: list[str] = []
    _format_value(lines, value, indent=2, max_width=_WIDTH)
    return lines


def _render_mapping(data: dict[str, Any], *, detailed: bool) -> list[str]:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict) and value.get("shape") == "table":
            lines.extend(["", f"  {key}:"])
            lines.extend(render_rows(value["columns"], value["rows"], limit=None if detailed else ROW_LIMIT))
        else:
            _format_value(lines, {key: value}, indent=2, max_width=_WIDTH)
    return lines


def _render_array(values: Sequence[Any]) -> list[str]:
    head = _flow_repr(list(values[:_ARRAY_HEAD]))
    more = f" … and {len(values) - _ARRAY_HEAD} more" if len(values) > _ARRAY_HEAD else ""
    return ["", f"  {len(values)} values: {head}{more}"]


def _cell_repr(value: Any) -> str:
    """Render a cell value like :func:`_flow_repr`, but without collapsing a contiguous
    int list to ``range(...)`` — a table cell shows the actual values, not a summary."""
    if isinstance(value, dict):
        inner = ", ".join(f"{k}: {_cell_repr(v)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_cell_repr(v) for v in value) + "]"
    return str(value)


def _cell(value: Any) -> str:
    text = "" if value is None else _cell_repr(value)
    return text if len(text) <= _CELL_WIDTH else text[: _CELL_WIDTH - 1] + "…"
