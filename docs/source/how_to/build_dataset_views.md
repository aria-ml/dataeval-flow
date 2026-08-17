# Narrow a dataset with views

You do not always want to evaluate a whole dataset. A first pass over 500 images, a run restricted to two classes, a
reproducible random sample — all of these are {term}`views <View>`: named, ordered pipelines of dataset operations
applied before a workflow sees the data.

## Used in these tutorials

- {doc}`Clean a dataset <../notebooks/data_cleaning>`
- {doc}`Analyze dataset quality across splits <../notebooks/data_analysis>`

## Define a view and reference it

Views are declared once and referenced by name from a `source`:

```yaml
datasets:
  - name: cppe5_train
    format: huggingface
    path: data/cppe5/train
    task: object_detection

views:
  - name: first500
    operations:
      - type: Limit
        params:
          size: 500

sources:
  - name: cppe5_src
    dataset: cppe5_train
    view: first500
```

Operations run in the order listed, each consuming the output of the last. A `source` without a `view` uses the whole
dataset.

## Available operations

Operations are pass-throughs to `dataeval.data`, so the type names and parameters are DataEval's:

| Operation | Effect |
| --- | --- |
| `Limit` | Keep the first *n* items |
| `Indices` | Keep an explicit set of positions |
| `ClassFilter` | Keep only the listed classes |
| `ClassBalance` | Even out per-class counts |
| `Shuffle` | Randomize order (seedable) |
| `Reverse` | Reverse order |
| `Relabel` | Remap the label space |
| `DetectionCrops` | Turn detection boxes into per-object crops |
| `Crop` | Crop every item to a fixed region |
| `Resize` | Resize every item to a fixed size |
| `SelectChannels` | Keep a chosen subset of channels |
| `TorchvisionTransform` | Apply a `torchvision` transform as a view operation |

The `type` is resolved by name against `dataeval.data` rather than checked against a fixed list, so **anything that
module exports is usable** — including operations added by a DataEval release newer than this table. `ClassBalance`
in particular had a counting bug fixed in DataEval v1.1, so a view using it can select differently than it did
before; see {doc}`../migration/v1.1`.

See the [DataEval `dataeval.data` reference](https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/data/index.html)
for the full parameter list of each.

## Combine operations

Order matters. Filtering before limiting gives you 500 items *of the classes you care about*; limiting first gives you
whatever 500 came out of the front of the file.

```yaml
views:
  - name: two_classes_sample
    operations:
      - type: ClassFilter
        params:
          classes: [0, 3]
      - type: Shuffle
        params:
          seed: 42
      - type: Limit
        params:
          size: 500
```

Always pass a `seed` to `Shuffle`. Without one the view differs between runs, which makes the result
non-reproducible — and, because the cache key includes the view, defeats caching as well.

A `Shuffle` seed pins only that operation's ordering. To pin every stochastic component of the run — clustering,
random splits, sampling — set the pipeline-level `seed` as well:

```yaml
seed: 42

views:
  - name: two_classes_sample
    operations: [...]
```

See [Reproducibility](../concepts/Reproducibility.md) for how the two interact.

## Select an index range without enumerating it

`Indices` accepts a range shorthand so a contiguous span does not have to be written out:

```yaml
      - type: Indices
        params:
          indices: { start: 500, stop: 550 }        # 500…549

      - type: Indices
        params:
          indices: { start: 0, stop: 100, step: 2 } # every other item
```

The keys match Python's `range()`: `start` and `stop` are required, `step` is optional. The explicit list form is
still accepted. A range that expands past 1,000,000 elements is rejected — load those indices from a file instead.

## Build a view in Python

```python
from dataeval_flow.config import PipelineConfig, SourceConfig, ViewConfig, ViewOperation

view = ViewConfig(
    name="first500",
    operations=[ViewOperation(type="Limit", params={"size": 500})],
)
source = SourceConfig(name="cppe5_src", dataset="cppe5_train", view="first500")
```

## How views interact with the cache

The {term}`cache <Caching>` key includes a hash of the applied view, so two sources that differ only by view do not
collide, and changing a view invalidates only that view's artifacts. This is why a seeded `Shuffle` is worth the
keystrokes: an unseeded one produces a new view on every run and nothing is ever reused. See
{doc}`reuse_results_with_cache`.

## A note on the legacy vocabulary

Older configs used `selections` / `selection` / `steps` where current ones use `views` / `view` / `operations`. The old
keys are still accepted and emit a `DeprecationWarning`; `SelectionConfig` and `SelectionStep` remain importable as
aliases of `ViewConfig` and `ViewOperation`. New configs should use the current names.

## Related material

- [Reproducibility](../concepts/Reproducibility.md) — why a seeded, declarative view is part of a defensible result
- {doc}`reuse_results_with_cache` — how the view participates in the cache key
- {doc}`API Reference <../reference/autoapi/dataeval_flow/index>` — `ViewConfig` and `ViewOperation`
