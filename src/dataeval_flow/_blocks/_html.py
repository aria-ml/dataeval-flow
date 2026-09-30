"""Report blocks as HTML: one fragment per block, and one self-contained page to hold them.

The page is what a person opens, forwards and prints, so it loads nothing: no URL, and one
inline script that only adds to a page complete without it. That keeps it readable offline, in a
viewer that blocks scripts, and in print, and so convertible to PDF.
Every string a result supplies is escaped, attribute values included: class names and factor
values are dataset content, and pages get shared.

Each block type draws through :data:`DRAW`, keyed by its ``type`` tag. ``render_html`` takes
replacements for any of them, so a chart backend can draw ``distribution`` or ``table`` its own
way while every other block keeps the default.
"""

__all__ = ["DRAW", "Draw", "HtmlContext", "html_page", "render_html"]

import math
import re
from collections.abc import Mapping, Sequence

from dataeval_flow._blocks._draw import fmt_num, format_value
from dataeval_flow._blocks._html_base import Draw, HtmlContext, badge, escape, inline, num, pct, series_class
from dataeval_flow._blocks._html_style import SCRIPT, STYLE
from dataeval_flow._blocks._html_tables import draw_table
from dataeval_flow._blocks._models import (
    Asset,
    Block,
    BulletList,
    Code,
    Distribution,
    Fields,
    ItemRef,
    Paragraph,
    Proportion,
    Section,
    Summary,
    Tree,
)
from dataeval_flow._blocks._text import DEFAULT_WIDTH

_NOT_A_NAME = re.compile(r"[^a-z0-9]+")


def render_html(
    blocks: Sequence[Block], *, draw: Mapping[str, Draw] | None = None, assets: Sequence[Sequence[Asset]] = ()
) -> str:
    """*blocks* as an HTML fragment, one element per block, siblings on their own lines.

    *draw* replaces the drawing of a block type by its ``type`` tag. A replacement returns an
    HTML fragment and owns escaping its own output. ``assets[i]`` are the thumbnails
    ``blocks[i]``'s image cells show, as each result's report shows its own: two tasks may read
    one source through two draws of a random view, so one index can be two images. An item
    without a thumbnail is named instead, as is every item of a block past the end of *assets*.
    """
    table = {**DRAW, **(draw or {})}
    # On a page of several reports each is numbered, as the page's contents list numbers them.
    several = len(_reports(blocks)) > 1
    anchors: list[str | None] = []
    count = 0
    for block in blocks:
        if several and isinstance(block, Section):
            count += 1
            anchors.append(f"r{count}")
        else:
            anchors.append(None)
    scopes = [{asset.item: asset for asset in owned} for owned in assets]
    parts = (
        _render([block], 0, table, [anchor], scopes[i] if i < len(scopes) else {})
        for i, (block, anchor) in enumerate(zip(blocks, anchors, strict=True))
    )
    return "\n".join(part for part in parts if part)


def _render(
    blocks: Sequence[Block],
    depth: int,
    table: Mapping[str, Draw],
    anchors: Sequence[str | None] | None,
    assets: Mapping[ItemRef, Asset],
) -> str:
    def _children(children: Sequence[Block], *, anchors: Sequence[str | None] | None = None) -> str:
        return _render(children, depth + 1, table, anchors, assets)

    parts = (
        table[block.type](block, HtmlContext(depth, _children, anchors[i] if anchors else None, assets))
        for i, block in enumerate(blocks)
    )
    return "\n".join(part for part in parts if part)


def _reports(blocks: Sequence[Block]) -> list[Section]:
    """The page's reports: the sections at its top level, one per result."""
    return [block for block in blocks if isinstance(block, Section)]


def _is_finding(block: Block) -> bool:
    """Whether *block*, directly under a report, is one of its findings: a section carrying a verdict."""
    return isinstance(block, Section) and block.severity is not None


def _is_group(block: Block) -> bool:
    """Whether *block*, directly under a report, groups findings: a section with no verdict of its own that holds
    nothing but sections carrying one, such as a chain's findings for one split."""
    return (
        isinstance(block, Section)
        and block.severity is None
        and bool(block.blocks)
        and all(isinstance(child, Section) and child.severity is not None for child in block.blocks)
    )


def _slug(title: str) -> str:
    return _NOT_A_NAME.sub("-", title.lower()).strip("-") or "finding"


def _is_summary(block: Block) -> bool:
    """Whether *block*, directly under a report, is its summary: the section holding its findings' lines."""
    return isinstance(block, Section) and any(isinstance(child, Summary) for child in block.blocks)


def _cards(report: Section, prefix: str) -> list[str | None]:
    """The ``id`` of each of the report's findings' cards, aligned with its blocks; ``None`` for everything else.

    A card is named after its finding's title, numbered from 2 where that name is taken, even by a
    card whose own title ends in a number, and prefixed with its report's anchor where the page holds
    several reports, so every ``id`` is its own. A group of findings takes an ``id`` the same way, and its cards'
    ids begin with it.
    """
    taken: set[str] = set()
    cards: list[str | None] = []
    for block in report.blocks:
        if not (isinstance(block, Section) and (_is_finding(block) or _is_group(block))):
            cards.append(None)
            continue
        name = f"{prefix}{_slug(block.title)}"
        card, number = name, 1
        while card in taken:
            number += 1
            card = f"{name}-{number}"
        taken.add(card)
        cards.append(card)
    return cards


def _heading(title: str) -> str:
    """A title as the page shows it: escaped."""
    return escape(title.strip())


def _contents(reports: Sequence[Section]) -> str:
    """A list of the page's reports, each linking to its own, where the page holds more than one."""
    if len(reports) < 2:
        return ""
    items = "".join(
        f'<li><a href="#r{number}">{escape(report.title.strip())}</a></li>' for number, report in enumerate(reports, 1)
    )
    return f'<nav class="contents"><h2>Reports</h2><ol>{items}</ol></nav>'


def html_page(title: str, blocks: Sequence[Block], assets: Sequence[Sequence[Asset]] = ()) -> str:
    """One complete page holding *blocks*: its own stylesheet and script, and nothing it has to fetch.

    ``assets[i]`` are the thumbnails ``blocks[i]``'s image cells show, embedded as ``data:`` URIs.
    """
    contents = _contents(_reports(blocks))
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
            *([contents] if contents else []),
            render_html(blocks, assets=assets),
            "</main>",
            f"<script>{SCRIPT}</script>",
            "</body>",
            "</html>",
            "",
        ]
    )


# -- Text ---------------------------------------------------------------------------------------


def _section(block: Section, ctx: HtmlContext) -> str:
    """A report at the top of the page, a finding's card under it, and a plain section anywhere else."""
    if ctx.depth == 0:
        return _report(block, ctx)
    level = min(ctx.depth + 1, 4)
    brief = f' <span class="brief">{escape(block.brief)}</span>' if block.brief else ""
    mark = f" {badge(block.severity)}" if block.severity else ""
    heading = f"<h{level}>{_heading(block.title)}{brief}{mark}</h{level}>"
    children = ctx.render(block.blocks) if not (ctx.depth == 1 and _is_group(block)) else _group_cards(block, ctx)
    body = f"\n{children}" if children else ""
    if ctx.depth == 1 and block.severity:
        return _card(block, heading, body, ctx.anchor)
    if ctx.depth == 1 and block.reference:
        return f'<details class="panel"><summary>{heading}</summary>{body}</details>'
    classes = f"section {block.severity}" if block.severity else "section"
    named = f' id="{escape(ctx.anchor)}"' if ctx.anchor else ""
    return f'<section class="{classes}"{named}>{heading}{body}</section>'


def _card(block: Section, heading: str, body: str, anchor: str | None) -> str:
    """A finding as a card: a warning is open on arrival, as it asks for a look; the rest stay one line."""
    card = f' id="{escape(anchor)}"' if anchor else ""
    opened = " open" if block.severity == "warning" else ""
    return f'<details class="card {block.severity}"{card}{opened}><summary>{heading}</summary>{body}</details>'


def _group_cards(group: Section, ctx: HtmlContext) -> str:
    """A group's findings, each a card like a top-level finding, its ``id`` after the group's."""
    taken: set[str] = set()
    parts: list[str] = []
    for child in (block for block in group.blocks if isinstance(block, Section)):
        name = f"{ctx.anchor}-{_slug(child.title)}" if ctx.anchor else _slug(child.title)
        anchor, number = name, 1
        while anchor in taken:
            number += 1
            anchor = f"{name}-{number}"
        taken.add(anchor)
        brief = f' <span class="brief">{escape(child.brief)}</span>' if child.brief else ""
        heading = f"<h3>{_heading(child.title)}{brief} {badge(child.severity or 'info')}</h3>"
        inner = ctx.render(child.blocks)
        parts.append(_card(child, heading, f"\n{inner}" if inner else "", anchor))
    return "\n".join(parts)


def _health(report: Section) -> str:
    """The report's verdict as a badge: the warnings its summary counted, or passed; none for a report without one."""
    summary = next(
        (
            child
            for block in report.blocks
            if isinstance(block, Section)
            for child in block.blocks
            if isinstance(child, Summary)
        ),
        None,
    )
    if summary is not None and summary.failed:
        return f'<span class="badge failed">failed: {escape(", ".join(summary.failed))}</span>'
    if summary is None or not summary.items:
        return ""
    if not summary.warnings:
        return '<span class="badge ok">passed</span>'
    return f'<span class="badge warning">{summary.warnings} warning{"s" if summary.warnings != 1 else ""}</span>'


def _report(block: Section, ctx: HtmlContext) -> str:
    """A result's report: a header with its title and verdict, its envelope as the provenance, then the rest.

    Where it has finding cards, they stand for its summary: each card's title, brief and badge say what
    its summary line says, and the header's badge says what the health line does. The short form has no
    cards, so its summary stays.
    """
    brief = f' <span class="brief">{escape(block.brief)}</span>' if block.brief else ""
    verdict = badge(block.severity) if block.severity else _health(block)
    head = f'<header class="report-head"><h1>{_heading(block.title)}{brief}</h1>{verdict}</header>'
    cards = _cards(block, f"{ctx.anchor}-" if ctx.anchor else "")
    rest = list(block.blocks)
    provenance = ""
    if rest and isinstance(first := rest[0], Fields):
        provenance = _fields(first, ctx, css="fields provenance")
        rest, cards = rest[1:], cards[1:]
    if any(_is_finding(child) or _is_group(child) for child in rest):
        kept = [i for i, child in enumerate(rest) if not _is_summary(child)]
        rest, cards = [rest[i] for i in kept], [cards[i] for i in kept]
    children = ctx.render(rest, anchors=cards)
    body = "".join(f"\n{part}" for part in (provenance, children) if part)
    opening = f'<article class="report" id="{escape(ctx.anchor)}">' if ctx.anchor else '<article class="report">'
    return f"{opening}{head}{body}</article>"


def _paragraph(block: Paragraph, _ctx: HtmlContext) -> str:
    return f"<p>{inline(block.text)}</p>"


def _bullets(block: BulletList, _ctx: HtmlContext) -> str:
    return "<ul>" + "".join(f"<li>{inline(item)}</li>" for item in block.items) + "</ul>"


def _fields(block: Fields, _ctx: HtmlContext, css: str = "fields") -> str:
    rows = "".join(
        f"<dt>{escape(label)}</dt><dd>{'' if value is None else inline(str(value))}</dd>"
        for label, value in block.items
    )
    return f'<dl class="{css}">{rows}</dl>' if rows else ""


def _code(block: Code, _ctx: HtmlContext) -> str:
    language = f' class="language-{escape(block.language)}"' if block.language else ""
    return f"<pre><code{language}>{escape(block.text)}</code></pre>"


def _tree(block: Tree, _ctx: HtmlContext) -> str:
    lines: list[str] = []
    format_value(lines, block.value, indent=0, max_width=DEFAULT_WIDTH)
    return f'<pre class="tree">{escape(chr(10).join(lines))}</pre>'


def _summary(block: Summary, _ctx: HtmlContext) -> str:
    rows = ""
    group = ""
    for item in block.items:
        if item.group != group:
            group = item.group
            rows += f'<tr class="group"><th colspan="3">{escape(group)}</th></tr>' if group else ""
        rows += f"<tr><td>{escape(item.label)}</td><td>{escape(item.value)}</td><td>{badge(item.severity)}</td></tr>"
    warnings = block.warnings
    counted = f"{warnings} warning{'s' if warnings != 1 else ''}"
    if block.failed:
        steps = inline(", ".join(f"`{step}`" for step in block.failed))
        noun = "Step" if len(block.failed) == 1 else "Steps"
        also = f"; {counted} to review" if warnings else ""
        health = f'<p class="health failed">{noun} {steps} failed{also}</p>'
    elif not rows:
        return ""
    elif warnings:
        health = f'<p class="health warning">{counted} — review the flagged findings</p>'
    else:
        health = '<p class="health ok">All checks passed</p>'
    table = f'<table class="summary"><tbody>{rows}</tbody></table>' if rows else ""
    return f"{table}{health}"


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
