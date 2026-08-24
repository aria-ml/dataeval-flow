# What's New

Release guides for DataEval Flow. Each one covers a single release: what changed, why it matters, and what you
have to do about it. The {doc}`Change Log <../changelog>` is the exhaustive, authoritative list of every entry —
these pages are the narrower story of upgrading.

Reading in order? Start at the release you are on and work forward; each guide's upgrade checklist assumes the
one before it has been applied.

## Releases

| Release | Date | Headline |
| --- | --- | --- |
| {doc}`v0.2.1 <v0.2.1>` | 2026-08-21 | CUDA container variants renamed; YOLO per-split loading restored |
| {doc}`v0.2 <v0.2>` | 2026-08-19 | DataEval v1.1, data coverage workflow, metadata encoding policies |
| {doc}`v0.1 <v0.1>` | 2026-03-30 | First release — the workflow orchestration framework |

```{toctree}
:hidden:

v0.2.1
v0.2
v0.1
```

:::{list-table}
:widths: 25 75
:header-rows: 0

- - {doc}`v0.2.1 <v0.2.1>`
  - A patch release. The published CUDA image tags moved to `cu126` / `cu130`, and YOLO datasets can select a
    single split again.
- - {doc}`v0.2 <v0.2>`
  - The upgrade to DataEval v1.1, a new `data-coverage` workflow, committed metadata encoding policies, and a
    round of configuration and result-field renames. **Breaking** — read this one before upgrading from v0.1.
- - {doc}`v0.1 <v0.1>`
  - The first public release: the workflow registry and task runner, seven workflows, the TUI and CLI, dataset
    adapters, caching, and signed containers.

:::

## Where to go instead

- {doc}`Change Log <../changelog>` — every change, per release, in full.
- {doc}`Installation <../installation>` — every supported install path and extra.
- {doc}`Quickstart <../quickstart>` — install and run a first evaluation end to end.
