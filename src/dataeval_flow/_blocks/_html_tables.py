"""Tables as HTML: each cell printed as the text report prints it, charts drawn, and every raw value kept."""

__all__ = ["draw_table"]

import math

from dataeval_flow._blocks._flags import card, flags_in, ordered, tag_text
from dataeval_flow._blocks._html_base import HtmlContext, escape, num, pct, series_class
from dataeval_flow._blocks._html_scale import draw_scale
from dataeval_flow._blocks._models import Cell, Column, Flag, Table
from dataeval_flow._blocks._table import _finite, _markers, _scale, cell_text, numbers

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
    if column.kind == "flags":
        return "flags"
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


def _marks(column: Column, low: float, high: float) -> str:
    """A bar column's markers as dashed lines through the bar, each where it falls on the column's scale."""
    span = (high - low) or 1.0
    return "".join(
        f'<span class="mark" style="left:{pct(min(max((value - low) / span * 100, 0.0), 100.0))}"></span>'
        for _, value in _markers(column)
    )


def _tag(flag: Flag) -> str:
    """One flag as a tag reading as its value against its limit, with its card opening on hover or keyboard focus."""
    title, rows = card(flag)
    detail = "".join(
        f'<span class="tip-row"><span>{escape(label)}</span><span>{escape(value)}</span></span>'
        for label, value in rows
    )
    return (
        f'<span class="tag" tabindex="0">{escape(tag_text(flag))}'
        f'<span class="tip"><span class="tip-title">{escape(title)}</span>{detail}</span></span>'
    )


def _scale_row(table: Table, extents: list[tuple[float, float]]) -> str:
    """The row closing a table whose bars have markers: each such column's scale, and blank cells elsewhere."""
    marked = [column.kind == "bar" and bool(_markers(column)) for column in table.columns]
    if not any(marked):
        return ""
    cells = "".join(
        f'<td class="chart">{draw_scale(column, extents[i])}</td>' if marked[i] else "<td></td>"
        for i, column in enumerate(table.columns)
    )
    return f'<tfoot><tr class="scale">{cells}</tr></tfoot>'


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
    if column.kind == "flags":
        flags = flags_in(value)
        tags = " ".join(_tag(flag) for flag in ordered(flags))
        return f'<td class="flags" data-sort="{len(flags)}">{tags}</td>'
    if column.kind == "bar":
        drawn = _bar(value, *extent) + _marks(column, *extent)
        content = f'<span class="track">{drawn}</span>' if drawn else ""
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
    scale = _scale_row(block, extents)
    # A scale's labels are placed for a 12rem chart column, so a table with one keeps its charts that wide.
    opening = '<table class="scaled">' if scale else "<table>"
    return f"{opening}{head}<tbody>{body}</tbody>{scale}</table>"
