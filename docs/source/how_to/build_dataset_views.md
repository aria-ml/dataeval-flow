# Narrow a dataset with views

A first pass over 500 images, a run restricted to two classes and a reproducible random sample are all
{term}`views <View>`. A view is a named, ordered pipeline of dataset operations that a source applies before a workflow
sees the data. The [`view`](../reference/transforms.md#view) transform applies the same operations inside a chain of
steps. A view also conforms a dataset to a shared vocabulary, which lets you
[merge sources into one dataset](merge_sources.md). This guide needs a pipeline with a dataset and a source.

## Define a view and reference it

Declare a view once and reference it by name from a `source`:

```yaml
datasets:
  - name: skysealand_train
    format: coco
    path: data/skysealand_datamaite_train

views:
  - name: first500
    operations:
      - type: Limit
        params:
          size: 500

sources:
  - name: skysealand_src
    dataset: skysealand_train
    view: first500
```

Operations run in the order listed. Each consumes the output of the previous one. A `source` without a `view` uses
the whole dataset.

:::{warning}
`Limit` on its own samples the storage order, not the dataset. Collections are often written grouped, by capture
site, class or acquisition date. A bare `Limit` keeps whatever is at the front. On SkySeaLand, the first 500 frames
put `ship` at 11% of annotations against 20% for the whole split, enough to move every statistic computed after.

Put a seeded `Shuffle` in front of the `Limit`:

```yaml
views:
  - name: sample500
    operations:
      - type: Shuffle
        params:
          seed: 0
      - type: Limit
        params:
          size: 500
```

`Shuffle` without a `seed` is not reproducible. A pipeline-level `seed:` does not reach view operations, so declare
the seed here.
:::

## Choose an operation

Operations are pass-throughs to `dataeval.data`, so the type names and parameters are DataEval's:

| Operation | Effect |
| --- | --- |
| `Limit` | Keep the first `n` items |
| `Indices` | Keep an explicit set of positions |
| `Filter` | Keep the listed classes, the items whose metadata factors match a `where` predicate, or both |
| `ClassBalance` | Even out per-class counts |
| `Shuffle` | Randomize order (seedable) |
| `Reverse` | Reverse order |
| `Relabel` | Remap the label space |
| `DetectionCrops` | Turn detection boxes into per-object crops |
| `Crop` | Crop every item to a fixed region |
| `Resize` | Resize every item to a fixed size |
| `SelectChannels` | Keep a chosen subset of channels |
| `TorchvisionTransform` | Apply a `torchvision` transform as a view operation |

Flow resolves the `type` by name against `dataeval.data`. It does not check the name against a fixed list, so any
operation that module exports is usable, including operations added by a DataEval release newer than this table.

`ClassBalance` had a counting bug fixed in DataEval v1.1, so a view that uses it can select differently than under
v0.1. `ClassFilter` still works but is deprecated in favor of `Filter`. Replace
`{type: ClassFilter, params: {classes: [0, 3]}}` with `{type: Filter, params: {classes: [0, 3]}}`, and replace
`filter_detections: false` with `drop: items`.

The [DataEval `dataeval.data` reference](https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/data/index.html)
lists the parameters of each operation.

## Order the operations

Order matters. Filtering before limiting gives 500 items of the classes you care about. Limiting first gives whatever
500 is at the front of the file.

```yaml
views:
  - name: two_classes_sample
    operations:
      - type: Filter
        params:
          classes: [0, 3]
      - type: Shuffle
        params:
          seed: 42
      - type: Limit
        params:
          size: 500
```

Always pass a `seed` to `Shuffle`. Without one the view differs between runs, so the result is not reproducible. It
also defeats caching, because the [cache key](reuse_results_with_cache.md) includes the view.

A `Shuffle` seed pins only that operation's ordering. To pin every stochastic component of the run, such as clustering,
random splits and sampling, set the pipeline-level `seed` as well:

```yaml
seed: 42

views:
  - name: two_classes_sample
    operations: [...]
```

See [Reproducibility](../concepts/Reproducibility.md) for how the two interact.

## Filter by metadata factor

`Filter`'s `where` is a SQL `WHERE` clause over the dataset's metadata factors, and `classes` is ANDed with it:

```yaml
views:
  - name: night_cars
    operations:
      - type: Filter
        params:
          where: "time_of_day = 'night' AND altitude > 100"
          classes: [1]
```

An item missing a factor never matches. `altitude IS NULL` keeps those items. Double-quote a factor name that holds
spaces (`"time of day" = 'night'`). For detections, `drop` decides what a detection-level predicate removes. `both`
(the default) keeps the images holding a match and only their matching detections, `items` keeps those images whole,
and `detections` keeps every image but only its matching detections.

## Select an index range

`Indices` accepts a range shorthand, so you do not enumerate the positions:

```yaml
      - type: Indices
        params:
          indices: { start: 500, stop: 550 }        # 500…549

      - type: Indices
        params:
          indices: { start: 0, stop: 100, step: 2 } # every other item
```

The keys match Python's `range()`: `start` and `stop` are required, `step` is optional. The explicit list form is
still accepted. A range that expands past 1,000,000 elements is rejected. Load those indices from a file.

## Build a view in Python

`ViewConfig` and `ViewOperation` build the same view. The {doc}`API Reference <../reference/autoapi/dataeval_flow/index>`
lists their fields.

```python
from dataeval_flow.config import PipelineConfig, SourceConfig, ViewConfig, ViewOperation

view = ViewConfig(
    name="sample500",
    operations=[
        ViewOperation(type="Shuffle", params={"seed": 0}),
        ViewOperation(type="Limit", params={"size": 500}),
    ],
)
source = SourceConfig(name="skysealand_src", dataset="skysealand_train", view="sample500")
```

## Next steps

- [Clean a dataset](../notebooks/data_cleaning.py) — Sample a dataset with a seeded view, then remove its outliers and
  duplicates.
- [Reproducibility](../concepts/Reproducibility.md) — Why a seeded, declarative view is part of a defensible result.
- [Merge sources into one dataset](merge_sources.md) — Conform two datasets to a shared vocabulary and combine them.
