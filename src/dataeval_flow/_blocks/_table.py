"""Table layout for the text renderer: column widths, chart cells, and the scale line under a bar."""

__all__ = ["cell_text", "draw_table", "natural_widths", "numbers", "shared_widths", "shown_rows", "tables"]

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard

from dataeval_flow._blocks._draw import BAR_CELLS, CHART_MIN, fmt_num, hbar, shape_cells
from dataeval_flow._blocks._flags import flags_in, ordered, tag_text
from dataeval_flow._blocks._models import Block, Cell, Column, Section, Table

_GAP = "  "
# Cells a sparkline column is drawn at when the row has room.  Matched to the resolution
# describe_binning records, so the column draws what was measured rather than a merge of it.
SPARKLINE_CELLS = 40
# Segment glyphs for a stacked bar, in series order.
_STACK_GLYPHS = "█░▒▓"
_CHARTS = ("bar", "stacked", "sparkline")


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: object) -> TypeGuard[int | float]:
    """A number a chart can place: a NaN or an infinity has no position, so it draws blank."""
    return _is_number(value) and math.isfinite(value)


def numbers(value: Cell) -> list[float] | None:
    """A chart cell's numbers, a sparkline's counts or a stacked bar's segments; ``None`` where it holds none.

    A list cell may hold flags instead, which no chart draws.
    """
    if not isinstance(value, list):
        return None
    floats = [float(v) for v in value if isinstance(v, int | float)]
    return floats if len(floats) == len(value) else None


def _markers(column: Column) -> list[tuple[str, float]]:
    return [(name, value) for name, value in column.markers if math.isfinite(value)]


def _sci(value: Cell, text: str) -> str:
    """*text*, unless it put a float in scientific notation: that always reads ``1.20e-05``.

    Three digits of mantissa, as ``fmt_num`` writes the small magnitudes it leaves in scientific notation.
    """
    return f"{value:.2e}" if isinstance(value, float) and "e" in text else text


def _formatted(column: Column, value: float) -> str | None:
    """*value* through the column's template, or ``None`` when there is none or it does not fit."""
    if not column.format:
        return None
    try:
        return _sci(value, column.format.format(value))
    except (ValueError, IndexError, KeyError, TypeError):
        return None  # a template that does not fit this value: the caller shows the value itself


def _text_lines(column: Column, value: Cell) -> list[str]:
    """A text cell's display lines: formatted when numeric and a format is set, split on ``\\n``.

    A flags cell prints one tag per line, by name.
    """
    if column.kind == "flags":
        return [tag_text(flag) for flag in ordered(flags_in(value))] or [""]
    if value is None:
        return [""]
    if _is_number(value) and (text := _formatted(column, value)) is not None:
        return [text]
    return _sci(value, str(value)).split("\n")


def cell_text(column: Column, value: Cell) -> str:
    """A text column's cell as the table prints it: formatted when numeric, empty for ``None``."""
    return "\n".join(_text_lines(column, value))


def _legend(column: Column) -> str:
    return "  ".join(f"{_STACK_GLYPHS[i % len(_STACK_GLYPHS)]} {name}" for i, name in enumerate(column.series))


def _header(column: Column) -> str:
    if column.kind == "stacked" and not column.header:
        return _legend(column)
    return column.header


def shown_rows(table: Table) -> list[dict[str, Cell]]:
    """The rows a renderer with little room shows: the table's preview, or every row where it has none."""
    return table.rows if table.preview is None else table.rows[: table.preview]


def natural_widths(table: Table) -> tuple[int, ...]:
    """How wide each column needs to be to hold its header and cells before any shrinking."""
    widths: list[int] = []
    for column in table.columns:
        if column.kind == "sparkline":
            widths.append(SPARKLINE_CELLS)
        elif column.kind in ("bar", "stacked"):
            widths.append(max(BAR_CELLS, len(_header(column))))
        else:
            cells = [line for row in shown_rows(table) for line in _text_lines(column, row.get(column.key))]
            widths.append(max([len(column.header), *(len(line) for line in cells)]))
    return tuple(widths)


def _without_images(table: Table) -> Table:
    """The table as text draws it: without its image columns, since a row's other cells name its items."""
    if all(column.kind != "image" for column in table.columns):
        return table
    return table.model_copy(update={"columns": [column for column in table.columns if column.kind != "image"]})


def _signature(table: Table) -> tuple[Any, ...]:
    return tuple((column.key, column.kind, column.header) for column in table.columns)


def tables(blocks: Sequence[Block]) -> list[Table]:
    """Every table among *blocks*, in document order, including those inside sections."""
    found: list[Table] = []
    for block in blocks:
        if isinstance(block, Table):
            found.append(block)
        elif isinstance(block, Section):
            found.extend(tables(block.blocks))
    return found


def shared_widths(blocks: Sequence[Block]) -> dict[tuple[Any, ...], tuple[int, ...]]:
    """Column widths for every group of two or more tables with identical columns among *blocks*.

    Tables read against each other, such as one per split, only compare when they land on the
    same columns, so each group is laid out at the widest any of its tables needs.
    """
    groups: dict[tuple[Any, ...], list[tuple[int, ...]]] = {}
    for table in map(_without_images, tables(blocks)):
        groups.setdefault(_signature(table), []).append(natural_widths(table))
    return {sig: tuple(map(max, *widths)) for sig, widths in groups.items() if len(widths) > 1}


def _fit(columns: Sequence[Column], widths: list[int], room: int) -> list[int]:
    """Shrink chart columns until the row fits: sparklines first, then bars, never below ten cells.

    A bar keeps the width of its heading too, such as a stacked bar's legend: the heading is drawn
    whole, and a column narrower than it would push every heading after it off its column.
    """
    over = sum(widths) + len(_GAP) * (len(widths) - 1) - room
    for kinds in (("sparkline",), ("bar", "stacked")):
        for index, column in enumerate(columns):
            if over <= 0:
                return widths
            floor = CHART_MIN if column.kind == "sparkline" else max(CHART_MIN, len(_header(column)))
            if column.kind in kinds and widths[index] > floor:
                take = min(over, widths[index] - floor)
                widths[index] -= take
                over -= take
    return widths


def _scale(table: Table, column: Column) -> tuple[float, float]:
    """The range a bar column is drawn over: zero or below, up to the largest value or marker or zero.

    Zero is always on the scale, since every bar runs from it: with only negative values, the one
    nearest zero draws the shortest bar rather than none.
    """
    values = [float(v) for row in table.rows if _finite(v := row.get(column.key))]
    values.extend(value for _, value in _markers(column))
    if not values:
        return 0.0, 1.0
    return min(min(values), 0.0), max(max(values), 0.0)


def _bar(value: Cell, low: float, high: float, width: int) -> str:
    """A bar from zero: eighth blocks when the scale starts at zero, whole cells when it runs below."""
    if not _finite(value):
        return ""
    number = float(value)
    if low == 0:
        return hbar(number, high, width)
    span = (high - low) or 1.0
    zero = min(max(int((0.0 - low) / span * width), 0), width)
    at = min(max(int((number - low) / span * width), 0), width)
    start, stop = min(zero, at), max(zero, at)
    return " " * start + "█" * (stop - start)


def _stacked(value: Cell, peak: float, width: int) -> str:
    """Segments drawn one glyph each, every nonzero segment keeping at least one cell."""
    values = numbers(value)
    if values is None or not peak or not all(math.isfinite(v) for v in values):
        return ""
    counts = [max(1 if v > 0 else 0, round(v / peak * width)) for v in values]
    while sum(counts) > width:
        counts[counts.index(max(counts))] -= 1
    return "".join(_STACK_GLYPHS[i % len(_STACK_GLYPHS)] * n for i, n in enumerate(counts))


def _stack_peak(table: Table, column: Column) -> float:
    """The largest finite row total a stacked column holds: the length its longest bar is drawn at."""
    sums = [sum(v) for row in table.rows if (v := numbers(row.get(column.key))) is not None]
    return max((total for total in sums if math.isfinite(total)), default=0)


def _chart(column: Column, value: Cell, width: int, scale: tuple[float, float]) -> str:
    """One chart cell. *scale* is the column's ``(low, high)``, measured once for every row."""
    if column.kind == "sparkline":
        return shape_cells(values, width) if (values := numbers(value)) is not None else ""
    if column.kind == "stacked":
        return _stacked(value, scale[1], width)
    return _bar(value, *scale, width)


def _marker_line(column: Column, scale: tuple[float, float], offset: int, width: int, indent: str) -> str:
    """``Threshold (lo)|-----|(hi)``, with the pipes under where the markers fall on the bar."""
    low, high = scale
    span = (high - low) or 1.0
    markers = _markers(column)
    names = list(dict.fromkeys(name for name, _ in markers))
    label = indent + "/".join(names)

    def _pos(value: float) -> int:
        return min(max(int((value - low) / span * width), 0), width - 1)

    def _text(value: float) -> str:
        text = _formatted(column, value)
        return "(" + (fmt_num(value) if text is None else text) + ")"

    ordered = sorted(markers, key=lambda marker: marker[1])
    if len(ordered) >= 2:
        lower, upper = ordered[0][1], ordered[-1][1]
        at_lower, at_upper = _pos(lower), _pos(upper)
        lower_text, upper_text = _text(lower), _text(upper)
        between = "" if at_lower == at_upper else "-" * max(0, at_upper - at_lower - 1) + "|"
        core = f"{lower_text}|{between}{upper_text}"
        start = offset + at_lower - len(lower_text)
    else:
        at = _pos(ordered[0][1])
        core = "-" * at + f"|{_text(ordered[0][1])}"
        start = offset
    start = max(start, len(label) + 1)
    return label + " " * (start - len(label)) + core


def draw_table(
    table: Table, *, indent: str, room: int, layouts: Mapping[tuple[Any, ...], tuple[int, ...]] | None = None
) -> list[str]:
    """Draw *table* as aligned columns within *room* characters after *indent*.

    Cells never wrap: a wrapped row reads as another row.  When the row is too wide, chart
    columns shrink; past their minimum, the table overflows. Image columns are left out: a row's other
    cells name its items.
    """
    table = _without_images(table)
    if not table.rows or not table.columns:
        return []
    columns = table.columns
    shared = (layouts or {}).get(_signature(table))
    widths = _fit(columns, list(shared or natural_widths(table)), room)
    aligns = [
        "left" if column.kind in (*_CHARTS, "flags") else column.align or ("left" if index == 0 else "right")
        for index, column in enumerate(columns)
    ]

    def _join(cells: Sequence[str]) -> str:
        parts = [
            cell.ljust(width) if align == "left" else cell.rjust(width)
            for cell, width, align in zip(cells, widths, aligns, strict=True)
        ]
        return indent + _GAP.join(parts)

    lines: list[str] = []
    headers = [_header(column) for column in columns]
    if any(headers):
        lines.append(
            _join([h[:w] if c.kind == "sparkline" else h for h, w, c in zip(headers, widths, columns, strict=True)])
        )
        lines.append(indent + _GAP.join("-" * width for width in widths))
    # Each chart column's scale spans every row, so it is measured once rather than per cell.
    scales = {
        index: (0.0, _stack_peak(table, column)) if column.kind == "stacked" else _scale(table, column)
        for index, column in enumerate(columns)
        if column.kind in ("bar", "stacked")
    }
    rows = shown_rows(table)
    for row in rows:
        cells = [
            [_chart(column, row.get(column.key), width, scales.get(index, (0.0, 0.0)))]
            if column.kind in _CHARTS
            else _text_lines(column, row.get(column.key))
            for index, (column, width) in enumerate(zip(columns, widths, strict=True))
        ]
        depth = max(len(cell) for cell in cells)
        lines.extend(_join([cell[sub] if sub < len(cell) else "" for cell in cells]) for sub in range(depth))
    if (hidden := len(table.rows) - len(rows)) > 0:
        lines.append(f"{indent}… {hidden:,} more row{'s' if hidden != 1 else ''}, in the HTML and JSON reports")
    for index, column in enumerate(columns):
        if column.kind == "bar" and _markers(column):
            offset = len(indent) + sum(widths[:index]) + len(_GAP) * index
            lines.append(_marker_line(column, scales[index], offset, widths[index], indent))
    return [line.rstrip() for line in lines]
