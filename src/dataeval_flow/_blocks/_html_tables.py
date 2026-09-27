"""Tables as HTML: each cell printed as the text report prints it, charts drawn, and every raw value kept."""

__all__ = ["draw_table"]

import math

from dataeval_flow._blocks._draw import fmt_num
from dataeval_flow._blocks._html_base import HtmlContext, escape, num, pct, series_class
from dataeval_flow._blocks._models import Cell, Column, Table
from dataeval_flow._blocks._table import _finite, _formatted, _markers, _scale, cell_text, numbers

_CHARTS = ("bar", "stacked", "sparkline")


def _raw(value: Cell) -> str:
    """The cell's own value for ``data-value``, so a reader can sort by it rather than by its display."""
    if value is None:
        return ""
    if isinstance(value, list):
        values = numbers(value)
        return "" if values is None else ",".join(str(v) for v in values)
    return escape(value)


def _legend(column: Column) -> str:
    return " ".join(
        f'<span class="swatch {series_class(i)}"></span>{escape(name)}' for i, name in enumerate(column.series)
    )


def _header(column: Column) -> str:
    if column.kind == "stacked" and not column.header:
        return _legend(column)
    return escape(column.header)


def _align(column: Column, index: int) -> str:
    if column.kind in _CHARTS:
        return "chart"
    return column.align or ("left" if index == 0 else "right")


def _bar(value: Cell, low: float, high: float) -> str:
    """A bar from zero across the column's scale; nothing for a zero or a value with no position."""
    if not _finite(value) or not value:
        return ""
    span = (high - low) or 1.0
    zero = (0.0 - low) / span * 100
    at = (float(value) - low) / span * 100
    start, stop = min(zero, at), max(zero, at)
    return f'<span class="bar" style="margin-left:{pct(start)};width:{pct(stop - start)}"></span>'


def _stacked(value: Cell, peak: float) -> str:
    values = numbers(value)
    if values is None or not peak or not all(math.isfinite(v) for v in values):
        return ""
    # One flex row, so a tiny segment's minimum width squeezes its neighbours rather than wrapping the last.
    segments = "".join(
        f'<span class="seg {series_class(i)}" style="width:{pct(v / peak * 100)}"></span>'
        for i, v in enumerate(values)
        if v > 0
    )
    return f'<span class="stack">{segments}</span>' if segments else ""


def _sparkline(value: Cell) -> str:
    values = numbers(value)
    if not values:
        return ""
    counts = [v if math.isfinite(v) else 0.0 for v in values]
    peak = max(counts) or 1.0
    bars = "".join(
        f'<rect x="{i}" y="{num(1 - c / peak)}" width="1" height="{num(c / peak)}"/>' for i, c in enumerate(counts)
    )
    return f'<svg class="spark" viewBox="0 0 {len(counts)} 1" preserveAspectRatio="none">{bars}</svg>'


def _caption(table: Table) -> str:
    """Each bar column's markers, named and formatted as its values are."""
    parts: list[str] = []
    for column in table.columns:
        markers = _markers(column) if column.kind == "bar" else []
        names = list(dict.fromkeys(name for name, _ in markers))
        for name in names:
            values = [_formatted(column, v) or fmt_num(v) for n, v in markers if n == name]
            parts.append(f"{escape(name)}: {escape(', '.join(values))}")
    return f"<caption>{' · '.join(parts)}</caption>" if parts else ""


def _extent(table: Table, column: Column) -> tuple[float, float]:
    """What a chart column's cells are drawn against: a bar's scale, or ``(0, peak)`` for a stacked row's total."""
    if column.kind == "bar":
        return _scale(table, column)
    if column.kind == "stacked":
        sums = [sum(v) for row in table.rows if (v := numbers(row.get(column.key))) is not None]
        return 0.0, max((s for s in sums if math.isfinite(s)), default=0)
    return 0.0, 0.0


def _cell(column: Column, value: Cell, index: int, extent: tuple[float, float]) -> str:
    align = _align(column, index)
    if column.kind == "bar":
        content = _bar(value, *extent)
    elif column.kind == "stacked":
        content = _stacked(value, extent[1])
    elif column.kind == "sparkline":
        content = _sparkline(value)
    else:
        content = "<br>".join(escape(line) for line in cell_text(column, value).split("\n"))
    return f'<td class="{align}" data-value="{_raw(value)}">{content}</td>'


def draw_table(block: Table, _ctx: HtmlContext) -> str:
    if not block.rows:
        return ""
    columns = block.columns
    head = ""
    if any(_header(column) for column in columns):
        cells = "".join(f'<th class="{_align(c, i)}">{_header(c)}</th>' for i, c in enumerate(columns))
        head = f"<thead><tr>{cells}</tr></thead>"
    # Once per column, not per cell: a scale reads every row, so per cell a table would cost rows squared.
    extents = [_extent(block, column) for column in columns]
    body = "".join(
        "<tr>" + "".join(_cell(c, row.get(c.key), i, extents[i]) for i, c in enumerate(columns)) + "</tr>"
        for row in block.rows
    )
    return f"<table>{_caption(block)}{head}<tbody>{body}</tbody></table>"
