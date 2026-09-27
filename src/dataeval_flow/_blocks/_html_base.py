"""What every HTML drawer shares: where a block is drawn, and escaping and formatting its text."""

__all__ = ["Draw", "HtmlContext", "badge", "escape", "inline", "num", "pct", "series_class"]

import html
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from dataeval_flow._blocks._draw import fmt_num
from dataeval_flow._blocks._models import Block

_CODE_SPAN = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class HtmlContext:
    """Where a block is drawn: its section depth, and how to draw a container's child blocks one level deeper."""

    depth: int
    render: Callable[[Sequence[Block]], str]


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
