"""Report blocks as HTML: one fragment per block, and one self-contained page to hold them.

The page is what a person opens, forwards and prints, so it loads nothing: no script and no
URL, which keeps it readable offline and printable as it shows, and so convertible to PDF.
Every string a result supplies is escaped, attribute values included: class names and factor
values are dataset content, and pages get shared.

Each block type draws through :data:`DRAW`, keyed by its ``type`` tag. ``render_html`` takes
replacements for any of them, so a chart backend can draw ``distribution`` or ``table`` its own
way while every other block keeps the default.
"""

__all__ = ["DRAW", "Draw", "HtmlContext", "html_page", "render_html"]

import math
from collections.abc import Mapping, Sequence

from dataeval_flow._blocks._draw import fmt_num, format_value
from dataeval_flow._blocks._html_base import Draw, HtmlContext, badge, escape, inline, num, pct, series_class
from dataeval_flow._blocks._html_style import STYLE
from dataeval_flow._blocks._html_tables import draw_table
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
    Tree,
)
from dataeval_flow._blocks._text import DEFAULT_WIDTH


def render_html(blocks: Sequence[Block], *, draw: Mapping[str, Draw] | None = None) -> str:
    """*blocks* as an HTML fragment, one element per block, siblings on their own lines.

    *draw* replaces the drawing of a block type by its ``type`` tag. A replacement returns an
    HTML fragment and owns escaping its own output.
    """
    table = {**DRAW, **(draw or {})}
    return _render(blocks, 0, table)


def _render(blocks: Sequence[Block], depth: int, table: Mapping[str, Draw]) -> str:
    ctx = HtmlContext(depth=depth, render=lambda children: _render(children, depth + 1, table))
    return "\n".join(part for block in blocks if (part := table[block.type](block, ctx)))


def html_page(title: str, blocks: Sequence[Block]) -> str:
    """One complete page holding *blocks*: its own stylesheet, and nothing it has to fetch."""
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(title)}</title>",
            f"<style>{STYLE}</style>",
            "</head>",
            "<body>",
            "<main>",
            render_html(blocks),
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


# -- Text ---------------------------------------------------------------------------------------


def _section(block: Section, ctx: HtmlContext) -> str:
    level = min(ctx.depth + 1, 4)
    title = "<br>".join(escape(part.strip()) for part in block.title.split("\n"))
    brief = f' <span class="brief">{escape(block.brief)}</span>' if block.brief else ""
    mark = f" {badge(block.severity)}" if block.severity else ""
    classes = f"section {block.severity}" if block.severity else "section"
    children = ctx.render(block.blocks)
    body = f"\n{children}" if children else ""
    return f'<section class="{classes}"><h{level}>{title}{brief}{mark}</h{level}>{body}</section>'


def _paragraph(block: Paragraph, _ctx: HtmlContext) -> str:
    return f"<p>{inline(block.text)}</p>"


def _bullets(block: BulletList, _ctx: HtmlContext) -> str:
    return "<ul>" + "".join(f"<li>{inline(item)}</li>" for item in block.items) + "</ul>"


def _fields(block: Fields, _ctx: HtmlContext) -> str:
    rows = "".join(
        f"<dt>{escape(label)}</dt><dd>{'' if value is None else inline(str(value))}</dd>"
        for label, value in block.items
    )
    return f'<dl class="fields">{rows}</dl>' if rows else ""


def _code(block: Code, _ctx: HtmlContext) -> str:
    language = f' class="language-{escape(block.language)}"' if block.language else ""
    return f"<pre><code{language}>{escape(block.text)}</code></pre>"


def _tree(block: Tree, _ctx: HtmlContext) -> str:
    lines: list[str] = []
    format_value(lines, block.value, indent=0, max_width=DEFAULT_WIDTH)
    return f'<pre class="tree">{escape(chr(10).join(lines))}</pre>'


def _summary(block: Summary, _ctx: HtmlContext) -> str:
    rows = "".join(
        f"<tr><td>{escape(item.label)}</td><td>{escape(item.value)}</td><td>{badge(item.severity)}</td></tr>"
        for item in block.items
    )
    return f'<table class="summary"><tbody>{rows}</tbody></table>'


# -- Charts -------------------------------------------------------------------------------------


def _proportion(block: Proportion, _ctx: HtmlContext) -> str:
    total = sum(count for _, count in block.parts)
    if not total:
        return ""
    listed = escape(", ".join(f"{count:,} {label}" for label, count in block.parts if count))
    if sum(1 for _, count in block.parts if count) < 2:
        return f'<div class="proportion">{listed}</div>'
    segments = "".join(
        f'<span class="seg {series_class(i)}" style="width:{pct(count / total * 100)}"></span>'
        for i, (_, count) in enumerate(block.parts)
        if count
    )
    return f'<div class="proportion"><span class="stack">{segments}</span> {listed}</div>'


def _distribution(block: Distribution, _ctx: HtmlContext) -> str:
    """A histogram, and beneath it in the same drawing the box plot when the quantiles are known."""
    counts = block.histogram
    peak = max(counts, default=0)
    if not peak:
        return ""
    n = len(counts)
    bars = "".join(
        f'<rect class="hist" x="{i}" y="{num(1 - c / peak)}" width="1" height="{num(c / peak)}"/>'
        for i, c in enumerate(counts)
    )
    q = block.quantiles
    if q is None or not all(math.isfinite(v) for v in (q.low, q.q1, q.median, q.q3, q.high)):
        low = min(count for count in counts if count)
        svg = f'<svg class="dist" viewBox="0 0 {n} 1" preserveAspectRatio="none">{bars}</svg>'
        return f'<figure class="distribution">{svg}<figcaption>n={low}–{peak}</figcaption></figure>'
    span = q.high - q.low

    def _x(value: float) -> str:
        return num(n / 2 if not span else (value - q.low) / span * n)

    box = (
        f'<line class="whisker" x1="0" y1="1.35" x2="{n}" y2="1.35"/>'
        f'<rect class="box" x="{_x(q.q1)}" y="1.2" width="{num(float(_x(q.q3)) - float(_x(q.q1)))}" height="0.3"/>'
        f'<line class="median" x1="{_x(q.median)}" y1="1.15" x2="{_x(q.median)}" y2="1.55"/>'
    )
    svg = f'<svg class="dist" viewBox="0 0 {n} 1.6" preserveAspectRatio="none">{bars}{box}</svg>'
    legend = f"{fmt_num(q.low)}–{fmt_num(q.high)} · p25 {fmt_num(q.q1)} · p50 {fmt_num(q.median)} · p75 {fmt_num(q.q3)}"
    return f'<figure class="distribution">{svg}<figcaption>{escape(legend)}</figcaption></figure>'


DRAW: dict[str, Draw] = {
    "section": _section,
    "paragraph": _paragraph,
    "bullet_list": _bullets,
    "fields": _fields,
    "table": draw_table,
    "proportion": _proportion,
    "distribution": _distribution,
    "code": _code,
    "tree": _tree,
    "summary": _summary,
}
