# What's New

Release guides for DataEval Flow. Each one covers a release and the patches that followed it: what changed, why it
matters, and what you have to do about it. The {doc}`Change Log <../changelog>` is the exhaustive, authoritative list of
every entry. These pages cover the upgrade path.

Read the guides in order. Start at the release you are on and work forward. Each guide's upgrade checklist assumes the
one before it has been applied.

## Releases

| Release | Date | Headline |
| --- | --- | --- |
| {doc}`v0.3 <v0.3>` | 2026-11-12 | Presets as chains of steps, the `audit` verdict, HTML reports, DataEval v1.2 |
| {doc}`v0.2 <v0.2>` | 2026-08-19 | DataEval v1.1, data coverage workflow, metadata encoding policies; patches to v0.2.5 |
| {doc}`v0.1 <v0.1>` | 2026-03-30 | First release, the workflow orchestration framework |

```{toctree}
:hidden:

v0.3
v0.2
v0.1
```

:::{list-table}
:widths: 25 75
:header-rows: 0

- - {doc}`v0.3 <v0.3>`
  - Every workflow type becomes a preset, a chain of steps that you can read and change. `audit` gives a verdict
    that a training job can gate on. The release also adds HTML reports, content digests, a preview HTTP service and
    DataEval v1.2. It breaks v0.2 configs and scripts, so read it before you upgrade from v0.2.
- - {doc}`v0.2 <v0.2>`
  - The upgrade to DataEval v1.1, a new `data-coverage` workflow, committed metadata encoding policies, and a
    round of configuration and result-field renames. It is a breaking release, so read it before upgrading from v0.1.
    The page also covers the patches v0.2.1 to v0.2.5: the CUDA image rename, the CLI additions and environment
    variables, the new image repository and `max_processes`.
- - {doc}`v0.1 <v0.1>`
  - The first public release: the workflow registry and task runner, seven workflows, the TUI and CLI, dataset
    adapters, caching, and signed containers.

:::

## Where to go next

- {doc}`Change Log <../changelog>` — Read every change, per release, in full.
- {doc}`Installation <../installation>` — See every supported install path and extra.
- {doc}`Quickstart <../quickstart>` — Install and run a first evaluation end to end.
