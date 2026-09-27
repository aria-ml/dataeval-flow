"""The page's stylesheet, kept apart from the drawing code as the text it is."""

__all__ = ["STYLE"]

STYLE = """
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
