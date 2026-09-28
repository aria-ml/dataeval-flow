"""What every HTML drawer shares: where a block is drawn, and escaping and formatting its text."""

__all__ = ["Draw", "HtmlContext", "Render", "badge", "escape", "inline", "num", "pct", "series_class"]

import html
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from dataeval_flow._blocks._draw import fmt_num
from dataeval_flow._blocks._models import Block

_CODE_SPAN = re.compile(r"`([^`]+)`")


class Render(Protocol):
    """Draws a container's child blocks one level deeper.

    *anchors* gives each child the ``id`` it draws with, in order, where the container named them; *links*
    replaces the card ids its descendants' summaries link to. Children inherit the links otherwise.
    """

    def __call__(
        self,
        children: Sequence[Block],
        *,
        anchors: Sequence[str | None] | None = None,
        links: tuple[str, ...] | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class HtmlContext:
    """Where a block is drawn: its section depth, and how to draw a container's child blocks one level deeper.

    ``anchor`` is the ``id`` this block draws with, where its container gave it one: a report on a page
    of several, or a finding's card. ``links`` is the current report's card ids, in the order of its
    findings, which the report's summary links its rows to.
    """

    depth: int
    render: Render
    anchor: str | None = None
    links: tuple[str, ...] = ()


Draw = Callable[[Any, HtmlContext], str]


def escape(text: object) -> str:
    """*text* safe inside an element or an attribute value: every string a result supplies goes through here."""
    return html.escape(str(text), quote=True)


def inline(text: str) -> str:
    """Escaped prose: backtick spans as code, each ``\\n`` a line break."""
    lines = [_CODE_SPAN.sub(r"<code>\1</code>", escape(line)) for line in text.split("\n")]
    return "<br>".join(lines)


def badge(severity: str) -> str:
    return f'<span class="badge {escape(severity)}">{escape(severity)}</span>'


def series_class(index: int) -> str:
    """A series' colour class: the page defines four, ``.s0`` to ``.s3``, and a fifth wraps round to the first."""
    return f"s{index % 4}"


def pct(value: float) -> str:
    return f"{fmt_num(value)}%"


def num(value: float) -> str:
    return f"{value:.4g}"
