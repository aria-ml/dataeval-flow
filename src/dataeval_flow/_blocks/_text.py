"""The text renderer: report blocks drawn as fixed-width lines.

Layout lives in the :class:`Frame` a block is drawn into, never on the block: the width, the prefix
every line carries, and how deeply the block is nested.  A section hands its children a nested frame,
so a paragraph three sections deep wraps to the width that is left without anyone computing it.
"""

__all__ = ["DEFAULT_WIDTH", "MIN_WIDTH", "Frame", "render_text"]

import math
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from dataeval_flow._blocks._draw import (
    CHART_MIN,
    box_plot,
    fmt_num,
    format_value,
    ratio_line,
    shape_cells,
    sparkline,
)
from dataeval_flow._blocks._models import (
    Block,
    BulletList,
    Code,
    Distribution,
    Fields,
    Paragraph,
    Proportion,
    Section,
    Summary,
    SummaryItem,
    Table,
    Tree,
)
from dataeval_flow._blocks._table import draw_table, shared_widths

# Total line length when the caller names none: a terminal's classic width.
DEFAULT_WIDTH = 80
# Narrowest report a caller may ask for.  Below it, a section three deep has too little room
# for a chart at its minimum beside the labels it needs.
MIN_WIDTH = 40
_STEP = "  "
_MARKERS = {"warning": "  [!!]", "ok": "  [ok]", "info": "  [..]"}


@dataclass(frozen=True)
class Frame:
    """Where a block is drawn: the line width, the prefix every line carries, and the nesting depth."""

    width: int = DEFAULT_WIDTH
    indent: str = ""
    depth: int = 0
    # Column widths shared by tables with identical columns inside one top-level section, keyed
    # by the columns' signature, so tables read against each other land on the same columns.
    layouts: Mapping[tuple[Any, ...], tuple[int, ...]] = field(default_factory=dict)

    @property
    def room(self) -> int:
        """The width left after the indent."""
        return self.width - len(self.indent)

    def nested(self) -> "Frame":
        """The frame a nested section's children are drawn in: one step further in, one level deeper."""
        return replace(self, indent=self.indent + _STEP, depth=self.depth + 1)


def render_text(blocks: Sequence[Block], frame: Frame = Frame()) -> list[str]:  # noqa: B008 - Frame is frozen
    """Draw *blocks* into *frame*, one blank line between siblings, with no trailing whitespace."""
    lines: list[str] = []
    for block in blocks:
        drawn = DRAW[block.type](block, frame)
        if not drawn:
            continue
        if lines:
            lines.append("")
        lines.extend(drawn)
    return [line.rstrip() for line in lines]


def _wrap(text: str, first: str, rest: str, width: int) -> list[str]:
    """Wrap one line of prose.  Words and hyphenated identifiers never split; an over-long one overflows."""
    return textwrap.wrap(
        text,
        width=width,
        initial_indent=first,
        subsequent_indent=rest,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [first.rstrip()]


def _words(text: str, width: int) -> list[str]:
    """*text* broken into lines of at most *width*, never splitting a word; ``[text]`` when it is empty."""
    return textwrap.wrap(text, width, break_long_words=False, break_on_hyphens=False) or [text]


# -- Sections -----------------------------------------------------------------------------------


def _section(block: Section, frame: Frame) -> list[str]:
    if frame.depth == 0:
        return _banner(block, frame)
    if frame.depth == 1:
        return _major(block, frame)
    return _minor(block, frame)


def _banner(block: Section, frame: Frame) -> list[str]:
    """The report itself: its title between rules, its content, and a closing rule."""
    rule = "=" * frame.width
    prefix = frame.indent + _STEP
    title = [
        line for part in block.title.split("\n") for line in _wrap(part.strip().upper(), prefix, prefix, frame.width)
    ]
    inner = replace(frame, indent=frame.indent + _STEP, depth=1)
    return ["", rule, *title, rule, *render_text(block.blocks, inner), "", rule]


def _major(block: Section, frame: Frame) -> list[str]:
    """A top-level section: its title and brief between full-width rules, its content at the title's indent."""
    rule = "=" * frame.width
    title = block.title.upper()
    brief = block.brief or ""
    padding = frame.room - len(title) - len(brief)
    if brief and padding >= 1:
        head = [f"{frame.indent}{title}{' ' * padding}{brief}"]
    else:
        # Title and brief cannot share a line: the title wraps, and the brief follows it right-aligned.
        head = _wrap(title, frame.indent, frame.indent, frame.width)
        if brief:
            head.extend(frame.indent + part.rjust(frame.room) for part in _words(brief, frame.room))
    inner = replace(frame, depth=2, layouts=shared_widths(block.blocks))
    return [rule, *head, rule, *render_text(block.blocks, inner)]


def _minor(block: Section, frame: Frame) -> list[str]:
    """A nested section: ``title — brief``, its content one step further in."""
    head = block.title + (f" — {block.brief}" if block.brief else "")
    return [*_wrap(head, frame.indent, frame.indent + _STEP, frame.width), *render_text(block.blocks, frame.nested())]


# -- Prose ----------------------------------------------------------------------------------------


def _paragraph(block: Paragraph, frame: Frame) -> list[str]:
    return [line for part in block.text.split("\n") for line in _wrap(part, frame.indent, frame.indent, frame.width)]


def _bullets(block: BulletList, frame: Frame) -> list[str]:
    return [line for item in block.items for line in _wrap(item, frame.indent + "- ", frame.indent + "  ", frame.width)]


def _fields(block: Fields, frame: Frame) -> list[str]:
    """``label: value`` lines, values aligned; a value's ``\\n`` and its wrapping hang under the value column."""
    if not block.items:
        return []
    label_width = max(len(label) + 1 for label, _ in block.items)
    lines: list[str] = []
    for label, value in block.items:
        head = f"{frame.indent}{label + ':':<{label_width}} "
        hang = " " * len(head)
        text = "" if value is None else str(value)
        for index, part in enumerate(text.split("\n")):
            lines.extend(_wrap(part, head if index == 0 else hang, hang, frame.width))
    return lines


def _code(block: Code, frame: Frame) -> list[str]:
    return [frame.indent + line for line in block.text.splitlines()]


def _tree(block: Tree, frame: Frame) -> list[str]:
    lines: list[str] = []
    format_value(lines, block.value, indent=len(frame.indent), max_width=frame.width)
    return lines


# -- Summaries and charts -------------------------------------------------------------------------


def _summary_line(item: SummaryItem, frame: Frame) -> list[str]:
    """``label ....... value  [!!]``, ending two columns short of the width.

    A label too long for the line wraps, and its last line carries the leader, value and marker.
    When the value and marker leave the label too little room even for that, the label takes
    whole lines and the value follows it, right-aligned, wrapping itself if it must.
    """
    marker = _MARKERS.get(item.severity, "  [..]")
    room = frame.width - len(frame.indent) - 2
    tail = len(item.value) + len(marker)

    def _line(label: str) -> str:
        dots_len = room - len(label) - tail
        dots = " " + "." * max(dots_len - 2, 1) + " " if dots_len > 3 else " "
        return f"{frame.indent}{label}{dots}{item.value}{marker}"

    if room - len(item.label) - tail > 3:
        return [_line(item.label)]
    label_room = room - tail - 5
    if label_room > 10:
        wrapped = _words(item.label, label_room)
        return [*(frame.indent + part for part in wrapped[:-1]), _line(wrapped[-1])]
    lines = [frame.indent + part for part in _words(item.label, room)]
    values = _words(item.value, room - len(marker))
    values[-1] += marker
    lines.extend(frame.indent + part.rjust(room) for part in values)
    return lines


def _summary(block: Summary, frame: Frame) -> list[str]:
    return [line for item in block.items for line in _summary_line(item, frame)]


def _proportion(block: Proportion, frame: Frame) -> list[str]:
    line = ratio_line(block.parts)
    return [frame.indent + line] if line else []


def _distribution(block: Distribution, frame: Frame) -> list[str]:
    """A box plot when the quantiles are known, a sparkline otherwise, fitted to the frame."""
    counts = block.histogram
    q = block.quantiles
    # A quantile that is not a number places nothing, so the plot falls back to the counts alone.
    if q is not None and all(math.isfinite(v) for v in (q.low, q.q1, q.median, q.q3, q.high)):
        drawn = box_plot(counts, q.low, q.q1, q.median, q.q3, q.high, frame.room)
        return [frame.indent + line for line in drawn] if drawn else []
    peak = max(counts, default=0)
    if not peak:
        return []
    low = min(count for count in counts if count)
    tail = f" n={fmt_num(low)}–{fmt_num(peak)}"
    cells = min(len(counts), max(CHART_MIN, frame.room - len(tail)))
    line = sparkline(counts) if cells == len(counts) else shape_cells(counts, cells)
    # Padded to eleven and then spaced, rather than to twelve: a factor with more buckets
    # than that draws a longer sparkline, and padding alone would run it into the count.
    return [f"{frame.indent}{line:<11}{tail}"]


def _table(block: Table, frame: Frame) -> list[str]:
    return draw_table(block, indent=frame.indent, room=frame.room, layouts=frame.layouts)


DRAW: dict[str, Callable[[Any, Frame], list[str]]] = {
    "section": _section,
    "paragraph": _paragraph,
    "bullet_list": _bullets,
    "fields": _fields,
    "table": _table,
    "proportion": _proportion,
    "distribution": _distribution,
    "code": _code,
    "tree": _tree,
    "summary": _summary,
}
