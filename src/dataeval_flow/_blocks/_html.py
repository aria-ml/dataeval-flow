"""Report blocks as HTML: one fragment per block, and one self-contained page to hold them.

The page is what a person opens, forwards and prints, so it loads nothing: no script and no
URL, which keeps it readable offline and printable as it shows, and so convertible to PDF.
Every string a result supplies is escaped, attribute values included: class names and factor
values are dataset content, and pages get shared.

Each block type draws through :data:`DRAW`, keyed by its ``type`` tag. ``render_html`` takes
replacements for any of them, so a chart backend can draw ``distribution`` or ``table`` its own
way while every other block keeps the default.
"""

__all__ = ["DRAW", "Draw", "html_page", "render_html"]

import html
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dataeval_flow._blocks._draw import fmt_num, format_value
from dataeval_flow._blocks._models import (
    Block,
    BulletList,
    Cell,
    Code,
    Column,
    Distribution,
    Fields,
    Paragraph,
    Proportion,
    Section,
    Summary,
    Table,
    Tree,
)
from dataeval_flow._blocks._table import _finite, _formatted, _markers, _scale, cell_text, numbers
from dataeval_flow._blocks._text import DEFAULT_WIDTH

_CHARTS = ("bar", "stacked", "sparkline")
_CODE_SPAN = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class HtmlContext:
    """Where a block is drawn: its section depth, and how to draw a container's child blocks one level deeper."""

    depth: int
    render: Callable[[Sequence[Block]], str]


Draw = Callable[[Any, HtmlContext], str]


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
            f"<title>{_escape(title)}</title>",
            f"<style>{_STYLE}</style>",
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


def _escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def _inline(text: str) -> str:
    """Escaped prose: backtick spans as code, each ``\\n`` a line break."""
    lines = [_CODE_SPAN.sub(r"<code>\1</code>", _escape(line)) for line in text.split("\n")]
    return "<br>".join(lines)


def _section(block: Section, ctx: HtmlContext) -> str:
    level = min(ctx.depth + 1, 4)
    title = "<br>".join(_escape(part.strip()) for part in block.title.split("\n"))
    brief = f' <span class="brief">{_escape(block.brief)}</span>' if block.brief else ""
    badge = f" {_badge(block.severity)}" if block.severity else ""
    classes = f"section {block.severity}" if block.severity else "section"
    children = ctx.render(block.blocks)
    body = f"\n{children}" if children else ""
    return f'<section class="{classes}"><h{level}>{title}{brief}{badge}</h{level}>{body}</section>'


def _badge(severity: str) -> str:
    return f'<span class="badge {_escape(severity)}">{_escape(severity)}</span>'


def _paragraph(block: Paragraph, _ctx: HtmlContext) -> str:
    return f"<p>{_inline(block.text)}</p>"


def _bullets(block: BulletList, _ctx: HtmlContext) -> str:
    return "<ul>" + "".join(f"<li>{_inline(item)}</li>" for item in block.items) + "</ul>"


def _fields(block: Fields, _ctx: HtmlContext) -> str:
    rows = "".join(
        f"<dt>{_escape(label)}</dt><dd>{'' if value is None else _inline(str(value))}</dd>"
        for label, value in block.items
    )
    return f'<dl class="fields">{rows}</dl>' if rows else ""


def _code(block: Code, _ctx: HtmlContext) -> str:
    language = f' class="language-{_escape(block.language)}"' if block.language else ""
    return f"<pre><code{language}>{_escape(block.text)}</code></pre>"


def _tree(block: Tree, _ctx: HtmlContext) -> str:
    lines: list[str] = []
    format_value(lines, block.value, indent=0, max_width=DEFAULT_WIDTH)
    return f'<pre class="tree">{_escape(chr(10).join(lines))}</pre>'


def _summary(block: Summary, _ctx: HtmlContext) -> str:
    rows = "".join(
        f"<tr><td>{_escape(item.label)}</td><td>{_escape(item.value)}</td><td>{_badge(item.severity)}</td></tr>"
        for item in block.items
    )
    return f'<table class="summary"><tbody>{rows}</tbody></table>'


# -- Tables -------------------------------------------------------------------------------------


def _series(index: int) -> str:
    """A series' colour class: the page defines four, ``.s0`` to ``.s3``, and a fifth wraps round to the first."""
    return f"s{index % 4}"


def _pct(value: float) -> str:
    return f"{fmt_num(value)}%"


def _raw(value: Cell) -> str:
    """The cell's own value for ``data-value``, so a reader can sort by it rather than by its display."""
    if value is None:
        return ""
    if isinstance(value, list):
        values = numbers(value)
        return "" if values is None else ",".join(str(v) for v in values)
    return _escape(value)


def _legend(column: Column) -> str:
    return " ".join(f'<span class="swatch {_series(i)}"></span>{_escape(name)}' for i, name in enumerate(column.series))


def _header(column: Column) -> str:
    if column.kind == "stacked" and not column.header:
        return _legend(column)
    return _escape(column.header)


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
    return f'<span class="bar" style="margin-left:{_pct(start)};width:{_pct(stop - start)}"></span>'


def _stacked(value: Cell, peak: float) -> str:
    values = numbers(value)
    if values is None or not peak or not all(math.isfinite(v) for v in values):
        return ""
    # One flex row, so a tiny segment's minimum width squeezes its neighbours rather than wrapping the last.
    segments = "".join(
        f'<span class="seg {_series(i)}" style="width:{_pct(v / peak * 100)}"></span>'
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
        f'<rect x="{i}" y="{_num(1 - c / peak)}" width="1" height="{_num(c / peak)}"/>' for i, c in enumerate(counts)
    )
    return f'<svg class="spark" viewBox="0 0 {len(counts)} 1" preserveAspectRatio="none">{bars}</svg>'


def _num(value: float) -> str:
    return f"{value:.4g}"


def _caption(table: Table) -> str:
    """Each bar column's markers, named and formatted as its values are."""
    parts: list[str] = []
    for column in table.columns:
        markers = _markers(column) if column.kind == "bar" else []
        names = list(dict.fromkeys(name for name, _ in markers))
        for name in names:
            values = [_formatted(column, v) or fmt_num(v) for n, v in markers if n == name]
            parts.append(f"{_escape(name)}: {_escape(', '.join(values))}")
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
        content = "<br>".join(_escape(line) for line in cell_text(column, value).split("\n"))
    return f'<td class="{align}" data-value="{_raw(value)}">{content}</td>'


def _table(block: Table, _ctx: HtmlContext) -> str:
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


# -- Charts -------------------------------------------------------------------------------------


def _proportion(block: Proportion, _ctx: HtmlContext) -> str:
    total = sum(count for _, count in block.parts)
    if not total:
        return ""
    listed = _escape(", ".join(f"{count:,} {label}" for label, count in block.parts if count))
    if sum(1 for _, count in block.parts if count) < 2:
        return f'<div class="proportion">{listed}</div>'
    segments = "".join(
        f'<span class="seg {_series(i)}" style="width:{_pct(count / total * 100)}"></span>'
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
        f'<rect class="hist" x="{i}" y="{_num(1 - c / peak)}" width="1" height="{_num(c / peak)}"/>'
        for i, c in enumerate(counts)
    )
    q = block.quantiles
    if q is None or not all(math.isfinite(v) for v in (q.low, q.q1, q.median, q.q3, q.high)):
        low = min(count for count in counts if count)
        svg = f'<svg class="dist" viewBox="0 0 {n} 1" preserveAspectRatio="none">{bars}</svg>'
        return f'<figure class="distribution">{svg}<figcaption>n={low}–{peak}</figcaption></figure>'
    span = q.high - q.low

    def _x(value: float) -> str:
        return _num(n / 2 if not span else (value - q.low) / span * n)

    box = (
        f'<line class="whisker" x1="0" y1="1.35" x2="{n}" y2="1.35"/>'
        f'<rect class="box" x="{_x(q.q1)}" y="1.2" width="{_num(float(_x(q.q3)) - float(_x(q.q1)))}" height="0.3"/>'
        f'<line class="median" x1="{_x(q.median)}" y1="1.15" x2="{_x(q.median)}" y2="1.55"/>'
    )
    svg = f'<svg class="dist" viewBox="0 0 {n} 1.6" preserveAspectRatio="none">{bars}{box}</svg>'
    legend = f"{fmt_num(q.low)}–{fmt_num(q.high)} · p25 {fmt_num(q.q1)} · p50 {fmt_num(q.median)} · p75 {fmt_num(q.q3)}"
    return f'<figure class="distribution">{svg}<figcaption>{_escape(legend)}</figcaption></figure>'


DRAW: dict[str, Draw] = {
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

_STYLE = """
:root { --ink: #1f2328; --muted: #59636e; --rule: #d1d9e0; --ok: #1a7f37; --info: #0969da;
  --warning: #bc4c00; --s0: #0969da; --s1: #d4a72c; --s2: #8250df; --s3: #1a7f37; }
body { margin: 0; color: var(--ink); background: #fff;
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 64rem; margin: 0 auto; padding: 1.5rem; }
h1, h2, h3, h4 { margin: 1.5rem 0 0.5rem; line-height: 1.25; }
h1 { font-size: 1.6rem; border-bottom: 2px solid var(--ink); padding-bottom: 0.3rem; }
h2 { font-size: 1.25rem; border-bottom: 1px solid var(--rule); padding-bottom: 0.2rem; }
h3 { font-size: 1.05rem; } h4 { font-size: 0.95rem; }
.brief { font-weight: normal; color: var(--muted); margin-left: 0.5rem; }
.badge { font-size: 0.75rem; font-weight: 600; border-radius: 1rem; padding: 0.05rem 0.5rem;
  margin-left: 0.5rem; color: #fff; background: var(--info); vertical-align: middle; }
.badge.ok { background: var(--ok); } .badge.warning { background: var(--warning); }
code, pre, table { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "DejaVu Sans Mono", monospace; }
code { font-size: 0.9em; background: #f6f8fa; padding: 0 0.2em; border-radius: 3px; }
pre { background: #f6f8fa; padding: 0.75rem; overflow-x: auto; font-size: 0.85rem; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 0.5rem 0; font-size: 0.85rem; }
caption { caption-side: bottom; text-align: left; color: var(--muted); padding-top: 0.25rem; }
th, td { padding: 0.2rem 0.6rem; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { border-bottom: 2px solid var(--rule); }
.left { text-align: left; } .right { text-align: right; }
td.chart { width: 12rem; min-width: 8rem; }
.bar, .seg { display: inline-block; height: 0.8em; min-width: 1px; vertical-align: middle; }
.bar { background: var(--s0); }
.seg.s0, .swatch.s0 { background: var(--s0); } .seg.s1, .swatch.s1 { background: var(--s1); }
.seg.s2, .swatch.s2 { background: var(--s2); } .seg.s3, .swatch.s3 { background: var(--s3); }
.swatch { display: inline-block; width: 0.7em; height: 0.7em; margin: 0 0.25em 0 0.5em; }
.stack { display: inline-flex; width: 12rem; vertical-align: middle; }
svg.spark { width: 10rem; height: 1em; fill: var(--s0); vertical-align: middle; }
svg.dist { width: 100%; max-width: 32rem; height: 4rem; fill: var(--s0); }
svg.dist .box { fill: var(--s1); }
svg.dist line { stroke: var(--ink); stroke-width: 2px; vector-effect: non-scaling-stroke; }
figure { margin: 0.5rem 0; } figcaption { color: var(--muted); font-size: 0.85rem; }
dl.fields { display: grid; grid-template-columns: max-content auto; gap: 0.1rem 1rem; margin: 0.5rem 0; }
dl.fields dt { font-weight: 600; } dl.fields dd { margin: 0; }
@media print {
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body { font-size: 10pt; }
  main { max-width: none; padding: 0; }
  tr, pre, figure, dl, .proportion { break-inside: avoid; }
  h1, h2, h3, h4 { break-after: avoid; }
}
"""
