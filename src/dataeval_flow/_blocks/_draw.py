"""Glyph drawing for the text renderer: bars, sparklines, box plots, and compact value text.

Every character drawn here is East Asian *Ambiguous*, so ``docs/build_font_subset.py`` must cover it:
keep the two in step.
"""

__all__ = [
    "BAR_CELLS",
    "CHART_MIN",
    "box_plot",
    "compact_indices",
    "flow_repr",
    "fmt_num",
    "format_value",
    "hbar",
    "ratio_line",
    "resample",
    "shape_cells",
    "sparkline",
]

import math
from collections.abc import Sequence
from typing import Any

# Width of a bar column, in cells.
BAR_CELLS = 30
# Narrowest a chart may be drawn before it stops reading as one.
CHART_MIN = 10
# Left-filling fractional blocks, indexed by eighths (1 = 1/8 .. 7 = 7/8).
_FRAC_BLOCKS = " ▏▎▍▌▋▊▉"
# Vertical eighths. Index 0 is a space: blank means exactly zero, and every nonzero value
# floors at index 1.
_SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"
_INDENT_STEP = 2


def fmt_num(value: Any) -> str:
    """Format an observed bound compactly without losing the digits that distinguish it.

    Four significant figures suits most factors and hides everything informative about the
    ones with large magnitudes: a capture time in epoch milliseconds sits near 1.787e15, so
    a span printed that way reads ``[1.787e+15, 1.787e+15]`` however wide it is.  Large
    values therefore write out in full, which is the same trade DataEval makes when it
    names the bins these spans sit beside.
    """
    if not isinstance(value, float):
        return str(value)
    if value != value or value in (float("inf"), float("-inf")):
        return f"{value:g}"
    if abs(value) >= 1e6:
        return f"{value:.0f}" if value == int(value) else f"{value:.2f}"
    text = f"{value:.4g}"
    # Where it does fall to scientific notation, a mantissa of three digits is enough.
    return f"{value:.2e}" if "e" in text else text


def resample(counts: Sequence[float], cells: int) -> list[float]:
    """Spread ``counts`` over ``cells`` by area, merging or stretching as needed.

    A record wider than the column is *merged* rather than sampled: a spike that fell in one
    of two combined cells survives, where taking every other cell would drop it.  A record
    coarser than the column is stretched instead.  Resampled by area, not by integer
    grouping: the record rarely divides by the column width, and slicing it would hand one
    cell two source cells and its neighbour one, so a column of identical counts would draw
    ragged, and nothing would tell the reader the raggedness is the drawing rather than the
    data.
    """
    counts = _finite(counts)
    total = len(counts)
    if not total:
        return [0.0] * cells
    step = total / cells
    merged: list[float] = []
    for index in range(cells):
        low, high = index * step, (index + 1) * step
        merged.append(
            sum(
                counts[source] * (min(high, source + 1) - max(low, source))
                for source in range(int(low), min(int(high - 1e-9) + 1, total))
            )
        )
    return merged


def _finite(values: Sequence[float]) -> list[float]:
    """*values* with any NaN or infinity drawn as nothing: a count that is not a number has no height."""
    return [value if math.isfinite(value) else 0.0 for value in values]


def _eighths(values: Sequence[float]) -> str:
    """One vertical-eighths glyph per value, scaled to the peak.  Blank is exactly zero."""
    values = _finite(values)
    peak = max(values, default=0)
    if not peak:
        return " " * len(values)
    return "".join(_SPARK_BLOCKS[0] if v == 0 else _SPARK_BLOCKS[max(1, round(v / peak * 8))] for v in values)


def shape_cells(counts: Sequence[float], cells: int) -> str:
    """Draw ``counts`` into exactly ``cells`` cells of vertical eighths, whatever their length."""
    return _eighths(resample(counts, cells))


def sparkline(counts: Sequence[float]) -> str:
    """One glyph per count.  Blank is exactly zero; every nonzero count shows."""
    if not max(_finite(counts), default=0):
        return ""
    return _eighths(counts)


def hbar(value: float, peak: float, cells: int = BAR_CELLS) -> str:
    """A horizontal bar in eighth blocks, floored at the narrowest block for any nonzero value."""
    if not value or not peak or not (math.isfinite(value) and math.isfinite(peak)):
        return ""
    eighths = round(value / peak * cells * 8)
    full, rem = divmod(eighths, 8)
    bar = "█" * full + (_FRAC_BLOCKS[rem] if rem else "")
    return bar or _FRAC_BLOCKS[1]


def box_plot(
    histogram: Sequence[float], low: float, q1: float, median: float, q3: float, high: float
) -> tuple[str, str, str] | None:
    """A histogram line, the box beneath it, and the quartile legend, or ``None`` for an empty histogram."""
    width = len(histogram)
    if not width or not max(histogram):
        return None
    bars = _eighths(histogram)
    span = high - low

    def _at(value: float) -> int:
        return 0 if not span else min(max(int(round((value - low) / span * (width - 1))), 0), width - 1)

    cells = ["─"] * width
    # Whisker caps first, box over them. A box that reaches an end *is* the finding: a quarter
    # of the rows sitting on the extreme leaves no whisker on that side, so the box has to be
    # able to cover a cap rather than be overwritten by it. Both extremes are labelled either
    # side of the line, so nothing is lost when a cap is covered.
    cells[0], cells[-1] = "├", "┤"
    # The box never rounds away either: on a heavily skewed column the interquartile range can
    # be a fraction of a cell, and a plot drawn as two bare whiskers reads as broken rather
    # than as skewed.
    for i in range(_at(q1), max(_at(q3), _at(q1)) + 1):
        cells[i] = "█"
    # Light rather than heavy: U+2503 is absent from Liberation Mono and several other
    # common monospace faces, and a glyph the font lacks is drawn from a fallback whose
    # advance width is its own, which shifts every cell after it out of alignment.
    cells[_at(median)] = "│"
    lo_label, hi_label = fmt_num(low), fmt_num(high)
    pad = " " * len(lo_label)
    legend = f"p25 {fmt_num(q1)} · p50 {fmt_num(median)} · p75 {fmt_num(q3)}"
    return f"{lo_label} {bars} {hi_label}", f"{pad} {''.join(cells)}", legend


def ratio_line(parts: Sequence[tuple[str, int]]) -> str:
    """One line showing how a whole splits into parts; the bar shows the first part's share.

    The bar compares parts, so it is drawn only where two or more are nonzero: a whole of one
    kind drawn as a bar renders an empty track, which reads as a measurement that came back
    zero.  Neither side rounds away: 198 against 2 is 19.8 blocks, and a bar rounded to a full
    twenty would render a 99% split exactly like a whole.
    """
    total = sum(count for _, count in parts)
    if not total:
        return ""
    listed = ", ".join(f"{count:,} {label}" for label, count in parts if count)
    if sum(1 for _, count in parts if count) < 2:
        return listed
    # A first part of zero draws an empty track: flooring it at a block would show a share it lacks.
    floor = 1 if parts[0][1] else 0
    filled = min(max(round(parts[0][1] / total * 20), floor), 19)
    return f"{'█' * filled}{'░' * (20 - filled)}  {listed}"


def flow_repr(obj: Any) -> str:
    """Render a value as a compact, unquoted, single-line string.

    Dicts use ``{k: v, ...}`` syntax, lists use ``[v, ...]``, and contiguous int lists
    collapse to ``range(...)`` shorthand.
    """
    if isinstance(obj, dict):
        inner = ", ".join(f"{k}: {flow_repr(v)}" for k, v in obj.items())
        return "{" + inner + "}"
    if isinstance(obj, list):
        # A bool is an int, but a list of flags is not a run of indices: [True, False] stays as it is.
        if obj and all(isinstance(i, int) and not isinstance(i, bool) for i in obj):
            compact = compact_indices(obj)
            if compact != str(obj):
                return compact
        return "[" + ", ".join(flow_repr(v) for v in obj) + "]"
    return str(obj)


def format_value(lines: list[str], obj: Any, indent: int, max_width: int) -> None:
    """Append *obj* to *lines*, using flow style for any value that fits in *max_width*.

    Dicts and lists are rendered block-style (one key or item per line) only when their
    flow representation would exceed *max_width*.
    """
    if isinstance(obj, dict):
        _format_dict(lines, obj, indent, max_width)
    elif isinstance(obj, list):
        _format_list(lines, obj, indent, max_width)
    else:
        lines.append(f"{' ' * indent}{obj}")


def _format_dict(lines: list[str], obj: dict[str, Any], indent: int, max_width: int) -> None:
    prefix = " " * indent
    for key, val in obj.items():
        flow = flow_repr(val)
        if len(f"{prefix}{key}: {flow}") <= max_width:
            lines.append(f"{prefix}{key}: {flow}")
        else:
            lines.append(f"{prefix}{key}:")
            format_value(lines, val, indent + _INDENT_STEP, max_width)


def _format_list(lines: list[str], obj: list[Any], indent: int, max_width: int) -> None:
    prefix = " " * indent
    for item in obj:
        flow = flow_repr(item)
        if len(f"{prefix}- {flow}") <= max_width:
            lines.append(f"{prefix}- {flow}")
        elif isinstance(item, dict) and item:
            _format_list_dict_item(lines, item, indent, max_width)
        else:
            lines.append(f"{prefix}-")
            format_value(lines, item, indent + _INDENT_STEP, max_width)


def _format_list_dict_item(lines: list[str], item: dict[str, Any], indent: int, max_width: int) -> None:
    """Format a dict inside a list, inlining the first key on the ``- `` line."""
    prefix = " " * indent
    it = iter(item.items())
    first_key, first_val = next(it)
    first_flow = flow_repr(first_val)
    if len(f"{prefix}- {first_key}: {first_flow}") <= max_width:
        lines.append(f"{prefix}- {first_key}: {first_flow}")
    else:
        lines.append(f"{prefix}- {first_key}:")
        format_value(lines, first_val, indent + _INDENT_STEP * 2, max_width)
    # Remaining keys align under the first key (one indent step past the "- ")
    _format_dict(lines, dict(it), indent + _INDENT_STEP, max_width)


def compact_indices(indices: list[int]) -> str:
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
