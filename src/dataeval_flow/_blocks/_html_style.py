"""The page's stylesheet and its one script, kept apart from the drawing code as the text they are.

The script only adds to a page that is complete without it: it makes tables sortable and long ones
filterable, adds expand-all and collapse-all for the findings, lets Esc put back an enlarged thumbnail,
and opens every finding and shows every filtered-out row before printing.
It loads nothing and builds every control from text, so no result string is ever read as markup.
"""

__all__ = ["SCRIPT", "STYLE"]

# The light palette, written once: the screen's default, and what print always uses whatever the reader's setting.
_LIGHT = """
  --bg: #ffffff; --card: #ffffff; --soft: #f6f8fa; --ink: #111827; --muted: #6b7280; --rule: #e5e7eb;
  --link: #1d4ed8; --ok: #16a34a; --info: #2563eb; --warning: #d97706; --mark: #dc2626;
  --s0: #2563eb; --s1: #d97706; --s2: #7c3aed; --s3: #16a34a;
  --tag: #eef2ff; --tag-ink: #1e1b4b;
"""

_DARK = """
  --bg: #0f1115; --card: #171a21; --soft: #1d212a; --ink: #e5e7eb; --muted: #9ca3af; --rule: #2a2f3a;
  --link: #93c5fd; --ok: #22c55e; --info: #60a5fa; --warning: #f59e0b; --mark: #f87171;
  --s0: #60a5fa; --s1: #f59e0b; --s2: #a78bfa; --s3: #22c55e;
  --tag: #232a3d; --tag-ink: #c7d2fe;
"""

_SCREEN = """
body { margin: 0; color: var(--ink); background: var(--bg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 72rem; margin: 0 auto; padding: 1.5rem; }
a { color: var(--link); }
h1, h2, h3, h4 { margin: 1.25rem 0 0.5rem; line-height: 1.25; }
h1 { font-size: 1.6rem; } h2 { font-size: 1.2rem; } h3 { font-size: 1.05rem; } h4 { font-size: 0.95rem; }
nav.contents { border: 1px solid var(--rule); border-radius: 7px; padding: 0.25rem 1rem; margin-bottom: 1.5rem; }
nav.contents h2 { font-size: 1rem; margin: 0.5rem 0; }
.controls { display: flex; justify-content: flex-end; gap: 0.5rem; margin-bottom: 0.5rem; }
.controls button { font: inherit; font-size: 0.8rem; padding: 0.2rem 0.7rem; cursor: pointer;
  color: var(--ink); background: var(--soft); border: 1px solid var(--rule); border-radius: 4px; }
.report { margin-bottom: 2.5rem; }
.report-head { display: flex; align-items: baseline; gap: 0.75rem; flex-wrap: wrap;
  border-bottom: 2px solid var(--ink); padding-bottom: 0.3rem; }
.report-head h1 { margin: 0; }
dl.fields.provenance { display: grid; grid-template-columns: max-content 1fr; gap: 0.15rem 1rem;
  margin: 0.6rem 0 1.2rem; padding: 0.5rem 0.8rem; background: var(--soft); border-radius: 6px;
  color: var(--muted); font-size: 0.85rem; }
dl.provenance dt { font-weight: 600; color: var(--ink); }
dl.provenance dd { margin: 0; overflow-wrap: anywhere; }
section.section > h2 { border-bottom: 1px solid var(--rule); padding-bottom: 0.2rem; }
details.card, details.panel { background: var(--card); border: 1px solid var(--rule);
  border-left: 4px solid var(--info); border-radius: 7px; padding: 0.3rem 1rem 0.6rem; margin: 0.9rem 0; }
details.card.ok { border-left-color: var(--ok); } details.card.warning { border-left-color: var(--warning); }
details.panel { border-left-color: var(--rule); margin-top: 1.5rem; }
:is(details.card, details.panel) > summary { cursor: pointer; list-style: none; }
:is(details.card, details.panel) > summary::-webkit-details-marker { display: none; }
:is(details.card, details.panel) > summary h2 { display: inline; font-size: 1.1rem; margin: 0; }
:is(details.card, details.panel) > summary::before { content: "▸"; color: var(--muted); margin-right: 0.4rem; }
:is(details.card, details.panel)[open] > summary::before { content: "▾"; }
.brief { font-weight: normal; color: var(--muted); margin-left: 0.5rem; }
.badge { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.02em; border-radius: 1rem; padding: 0.05rem 0.55rem;
  margin-left: 0.5rem; color: #fff; background: var(--info); vertical-align: middle; }
.badge.ok { background: var(--ok); } .badge.warning { background: var(--warning); }
.badge.failed { background: var(--mark); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "DejaVu Sans Mono", monospace; }
code { font-size: 0.9em; background: var(--soft); padding: 0 0.2em; border-radius: 3px; }
pre { background: var(--soft); padding: 0.75rem; overflow-x: auto; font-size: 0.85rem; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 0.5rem 0; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
th, td { padding: 0.25rem 0.6rem; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { color: var(--muted); font-weight: 600; border-bottom: 2px solid var(--rule); text-align: left; }
th.sortable { cursor: pointer; user-select: none; }
th[aria-sort="ascending"]::after { content: " ▲"; } th[aria-sort="descending"]::after { content: " ▼"; }
th:focus-visible, .tag:focus-visible { outline: 2px solid var(--info); outline-offset: 1px; }
.left, th.flags, td.flags { text-align: left; } .right { text-align: right; }
table.summary td a { color: var(--ink); font-weight: 600; }
table.summary tr.group th { padding-top: 0.6rem; color: var(--ink); text-align: left; }
p.health { margin: 0.6rem 0; padding: 0.4rem 0.8rem; font-weight: 600; background: var(--soft);
  border-left: 4px solid var(--info); border-radius: 4px; }
p.health.ok { border-left-color: var(--ok); } p.health.warning { border-left-color: var(--warning); }
p.health.failed { border-left-color: var(--mark); }
.filter { display: flex; align-items: center; gap: 0.6rem; margin: 0.5rem 0 0.2rem; }
.filter input { font: inherit; font-size: 0.8rem; min-width: 14rem; padding: 0.15rem 0.5rem;
  color: var(--ink); background: var(--soft); border: 1px solid var(--rule); border-radius: 4px; }
.filter .count { color: var(--muted); font-size: 0.8rem; }
td.chart { width: 12rem; min-width: 8rem; } table.scaled td.chart { min-width: 12rem; }
.track { position: relative; display: block; height: 0.9em; }
.bar, .seg { display: inline-block; height: 0.8em; min-width: 1px; vertical-align: middle; }
.bar { background: var(--s0); border-radius: 1px; }
.mark { position: absolute; top: -0.3em; bottom: -0.3em; border-left: 2px dashed var(--mark); }
.seg.s0, .swatch.s0 { background: var(--s0); } .seg.s1, .swatch.s1 { background: var(--s1); }
.seg.s2, .swatch.s2 { background: var(--s2); } .seg.s3, .swatch.s3 { background: var(--s3); }
.swatch { display: inline-block; width: 0.7em; height: 0.7em; margin: 0 0.25em 0 0.5em; }
.stack { display: inline-flex; width: 12rem; vertical-align: middle; }
tr.scale td { border-bottom: 0; padding-top: 0; }
.axis { position: relative; border-top: 1px solid var(--muted); font-size: 0.72rem; color: var(--muted); }
.axis .tick { position: absolute; top: -4px; height: 4px; border-left: 1px solid var(--muted); }
.axis .label { position: absolute; white-space: nowrap; padding-top: 2px; line-height: 1.2em; }
.axis .marker { color: var(--mark); font-weight: 600; }
.also { font-size: 0.72rem; color: var(--mark); }
.tag { position: relative; display: inline-block; border-radius: 3px; padding: 0 0.35rem; margin: 0 0.2rem 0.2rem 0;
  white-space: nowrap; cursor: default; color: var(--tag-ink); background: var(--tag); }
.tip { display: none; position: absolute; z-index: 10; left: 0; top: 100%; margin-top: 4px; min-width: 16rem;
  padding: 0.4rem 0.6rem; font-size: 0.8rem; white-space: normal; color: var(--ink); background: var(--card);
  border: 1px solid var(--rule); border-radius: 6px; box-shadow: 0 4px 14px rgb(0 0 0 / 0.18); }
.tag:hover .tip, .tag:focus .tip { display: block; }
.tip-title { display: block; font-weight: 600; margin-bottom: 0.2rem; }
.tip-row { display: flex; justify-content: space-between; gap: 1rem; }
.tip-row > span:first-child { color: var(--muted); }
td.image { text-align: left; }
td.image .item { color: var(--muted); }
details.thumb { display: inline-block; width: 48px; height: 48px; margin: 0 2px 2px 0; vertical-align: middle; }
details.thumb > summary { display: block; list-style: none; cursor: zoom-in; }
details.thumb > summary::-webkit-details-marker { display: none; }
details.thumb img { display: block; width: 48px; height: 48px; object-fit: contain; border-radius: 3px;
  background: var(--soft); outline: 1px solid var(--rule); outline-offset: -1px; }
details.thumb[open] > summary::before { content: ""; position: fixed; inset: 0; z-index: 20;
  background: rgb(0 0 0 / 0.6); cursor: zoom-out; }
details.thumb[open] img { position: fixed; z-index: 21; top: 50%; left: 50%; transform: translate(-50%, -50%);
  width: 192px; height: 192px; image-rendering: pixelated; background: var(--card); cursor: zoom-out;
  box-shadow: 0 8px 30px rgb(0 0 0 / 0.5); }
svg.spark { width: 10rem; height: 1em; fill: var(--s0); vertical-align: middle; }
svg.dist { width: 100%; max-width: 32rem; height: 4rem; fill: var(--s0); }
svg.dist .box { fill: var(--s1); }
svg.dist line { stroke: var(--ink); stroke-width: 2px; vector-effect: non-scaling-stroke; }
figure { margin: 0.5rem 0; } figcaption { color: var(--muted); font-size: 0.85rem; }
dl.fields { display: grid; grid-template-columns: max-content auto; gap: 0.1rem 1rem; margin: 0.5rem 0; }
dl.fields dt { font-weight: 600; } dl.fields dd { margin: 0; }
"""

_PRINT = """
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .controls, .filter, .tip { display: none !important; }
  body { font-size: 10pt; }
  main { max-width: none; padding: 0; }
  a { color: inherit; text-decoration: none; }
  tr, pre, figure, dl, .proportion { break-inside: avoid; }
  h1, h2, h3, h4, :is(details.card, details.panel) > summary { break-after: avoid; }
  details.thumb[open] > summary::before { display: none; }
  details.thumb[open] img { position: static; transform: none; width: 48px; height: 48px; box-shadow: none; }
"""

STYLE = (
    ":root { color-scheme: light dark;"
    + _LIGHT
    + "}\n@media (prefers-color-scheme: dark) { :root {"
    + _DARK
    + "} }"
    + _SCREEN
    + "@media print { :root { color-scheme: light;"
    + _LIGHT
    + "}"
    + _PRINT
    + "}\n"
)

SCRIPT = """
(() => {
  "use strict";
  const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
  // A stray NaN or infinity has no numeric place, so it sorts as blank (last), not as text.
  const NON_FINITE = /^[+-]?(nan|infinity|inf)$/i;

  // What a cell sorts by: the key a renderer gave it, else its raw value, else its text.
  const sortValue = (row, column) => {
    const cell = row.cells[column];
    if (!cell) return "";
    const value = cell.dataset.sort ?? cell.dataset.value ?? cell.textContent.trim();
    return NON_FINITE.test(value) ? "" : value;
  };

  // Two numbers compare as numbers and two texts by the collator, and a number comes before a text.
  const compare = (x, y) => {
    const [a, b] = [Number(x), Number(y)];
    const [p, q] = [Number.isFinite(a), Number.isFinite(b)];
    return p && q ? a - b : p !== q ? q - p : collator.compare(x, y);
  };

  // Headers sort on a click, or Enter or Space: ascending, then descending, then the original order.
  const makeSortable = (table) => {
    const head = table.tHead;
    const body = table.tBodies[0];
    if (!head || !body || body.rows.length < 2) return;
    Array.from(body.rows).forEach((row, index) => { row.dataset.order = String(index); });
    const headers = Array.from(head.rows[0].cells);
    headers.forEach((header, column) => {
      header.tabIndex = 0;
      header.classList.add("sortable");
      header.setAttribute("aria-sort", "none");
      const sort = () => {
        const state = header.getAttribute("aria-sort");
        const next = state === "none" ? "ascending" : state === "ascending" ? "descending" : "none";
        headers.forEach((other) => other.setAttribute("aria-sort", "none"));
        header.setAttribute("aria-sort", next);
        const rows = Array.from(body.rows);
        const direction = next === "descending" ? -1 : 1;
        rows.sort((a, b) => {
          if (next === "none") return Number(a.dataset.order) - Number(b.dataset.order);
          const x = sortValue(a, column);
          const y = sortValue(b, column);
          if (x === "" || y === "") return (x === "") - (y === "");
          return direction * compare(x, y);
        });
        body.append(...rows);
      };
      header.addEventListener("click", sort);
      header.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          sort();
        }
      });
    });
  };

  // What a reader sees of a node: its text, less any hover card's, which shows only on hover.
  const shownText = (node) => {
    if (node.nodeType === 3) return node.data;
    if (node.nodeType !== 1 || node.classList.contains("tip")) return "";
    return Array.from(node.childNodes, shownText).join("");
  };

  // A table of more than ten rows gets a box that keeps the rows whose text holds what's typed.
  const makeFilterable = (table) => {
    const body = table.tBodies[0];
    if (!body || body.rows.length <= 10) return;
    // Each row's text as its cells show it, a tab between cells, so a query never runs across two.
    const texts = new Map(
      Array.from(body.rows, (row) => [row, Array.from(row.cells, shownText).join("\\t").toLowerCase()]),
    );
    const box = document.createElement("div");
    box.className = "filter";
    const input = document.createElement("input");
    input.type = "search";
    input.placeholder = "Filter rows";
    input.setAttribute("aria-label", "Filter rows");
    const count = document.createElement("span");
    count.className = "count";
    const total = body.rows.length;
    const update = () => {
      const query = input.value.trim().toLowerCase();
      let shown = 0;
      for (const row of body.rows) {
        const match = !query || texts.get(row).includes(query);
        row.hidden = !match;
        if (match) shown += 1;
      }
      count.textContent = query ? `${shown} of ${total} rows` : `${total} rows`;
    };
    input.addEventListener("input", update);
    box.append(input, count);
    table.before(box);
    update();
    // A filtered table prints whole, since print hides the count that says rows are missing.
    window.addEventListener("beforeprint", () => {
      for (const row of body.rows) row.hidden = false;
    });
    window.addEventListener("afterprint", update);
  };

  // Expand all and Collapse all, above the reports, when the page has findings or panels to open and close.
  const addExpanders = () => {
    const cards = document.querySelectorAll("details.card, details.panel");
    const main = document.querySelector("main");
    if (!cards.length || !main) return;
    const bar = document.createElement("div");
    bar.className = "controls";
    for (const [label, open] of [["Expand all", true], ["Collapse all", false]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.addEventListener("click", () => cards.forEach((card) => { card.open = open; }));
      bar.append(button);
    }
    main.prepend(bar);
  };

  // Esc puts back an enlarged thumbnail, as a click anywhere does.
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    document.querySelectorAll("details.thumb[open]").forEach((thumb) => { thumb.open = false; });
  });

  // Every finding prints open, and those the reader had closed close again afterwards. Thumbnails print
  // as they are, since an enlarged one would cover the page.
  let closed = [];
  window.addEventListener("beforeprint", () => {
    closed = Array.from(document.querySelectorAll("details:not([open]):not(.thumb)"));
    closed.forEach((details) => { details.open = true; });
  });
  window.addEventListener("afterprint", () => {
    closed.forEach((details) => { details.open = false; });
    closed = [];
  });

  for (const table of document.querySelectorAll("main table:not(.summary)")) {
    makeSortable(table);
    makeFilterable(table);
  }
  addExpanders();
})();
"""
