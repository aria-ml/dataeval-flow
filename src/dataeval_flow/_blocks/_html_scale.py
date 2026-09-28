"""The scale under a bar column: its ends and markers, labelled so that no two labels overlap.

The page can't measure text without a script, so each label's width is estimated from its length
as a share of the chart column's fixed width. Labels take the first of up to three lines where they
clear every label already there; one that clears none is listed under the scale instead, so a
crowd of close markers never draws on top of itself.
"""

__all__ = ["Placed", "draw_scale", "place_labels"]

from collections.abc import Sequence
from typing import NamedTuple

from dataeval_flow._blocks._draw import fmt_num
from dataeval_flow._blocks._html_base import escape, pct
from dataeval_flow._blocks._models import Column
from dataeval_flow._blocks._table import _formatted, _markers

# One character's width as a share of the chart column (12rem wide, and never narrower under a scale;
# labels set at 0.72rem), a little over its average so that the estimate errs towards leaving room;
# the gap two labels keep; and how many lines the labels may take before the rest are listed.
_CHAR = 3.6
_GAP = 2.0
_LINES = 3
_LINE_HEIGHT = 1.2


class Placed(NamedTuple):
    """A label as drawn: where it starts and how wide it is, both as a share of the scale, and its line.

    ``order`` is the label's place among those asked for, so a caller can tell which one it is.
    """

    start: float
    width: float
    text: str
    line: int
    order: int


def place_labels(labels: Sequence[tuple[float, str]]) -> tuple[list[Placed], list[str]]:
    """Place each ``(position, text)`` label, in the order given, centred on its position and kept on the scale.

    Each takes the first line where it clears every label already on it by the gap. Those that
    clear none are returned apart, to be listed under the scale.
    """
    placed: list[Placed] = []
    listed: list[str] = []
    lines: list[list[tuple[float, float]]] = [[] for _ in range(_LINES)]
    for order, (position, text) in enumerate(labels):
        width = min(len(text) * _CHAR, 100.0)
        start = min(max(position - width / 2, 0.0), 100.0 - width)
        end = start + width
        for line, spans in enumerate(lines):
            if all(end + _GAP <= left or start >= right + _GAP for left, right in spans):
                spans.append((start, end))
                placed.append(Placed(start, width, text, line, order))
                break
        else:
            listed.append(text)
    return placed, listed


def _shown(column: Column, value: float) -> str:
    return _formatted(column, value) or fmt_num(value)


def draw_scale(column: Column, extent: tuple[float, float]) -> str:
    """The scale under a bar column with markers: a tick and a label at each end and at each marker."""
    low, high = extent
    span = (high - low) or 1.0

    def _at(value: float) -> float:
        return min(max((value - low) / span * 100, 0.0), 100.0)

    markers = sorted(((_at(value), f"{name} {_shown(column, value)}") for name, value in _markers(column)))
    labels = [(0.0, _shown(column, low)), (100.0, _shown(column, high)), *markers]
    placed, listed = place_labels(labels)
    ticks = "".join(f'<span class="tick" style="left:{pct(position)}"></span>' for position, _ in labels)
    # The two ends come first among the labels; every label after them is a marker.
    spans = "".join(
        f'<span class="{"label marker" if label.order >= 2 else "label"}" '
        f'style="left:{pct(label.start)};top:{label.line * _LINE_HEIGHT:g}em">{escape(label.text)}</span>'
        for label in placed
    )
    lines = 1 + max((label.line for label in placed), default=0)
    also = f'<div class="also">also: {escape(", ".join(listed))}</div>' if listed else ""
    return f'<div class="axis" style="height:{lines * _LINE_HEIGHT + 0.4:g}em">{ticks}{spans}</div>{also}'
