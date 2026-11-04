# Name the setting a matrix key varies

A key in a task's `matrix:` is a dotted path to the setting each run varies.
[Sweep settings with a matrix](run_a_matrix.md) varies the settings of the entry a task runs. This guide reaches the
other settings: the task's `sources` and `extractor`, the items of a list, the keys of a mapping, the entries a
[custom workflow](write_a_custom_workflow.md) runs, and an extractor's settings. It also lists the keys the load
refuses. You need a task with a `matrix:`, the extractor `bovw_ext`, and a [`quality`](../reference/presets.md#quality)
entry such as this one, which the examples vary:

```yaml
workflows:
  - name: cleaning
    type: quality
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: adaptive
```

## Pick the form of a key

A key is a dotted path, read as the first of these forms that matches:

| Key | Reaches |
| --- | --- |
| `sources` | The task's `sources`. A value is a source name or a list of them. |
| `extractor` | The task's `extractor`. A value is an extractor name, or `null` for none. |
| `evaluators.<name>.<path>` | A setting of the `evaluators:` entry `<name>` |
| `workflows.<name>.<path>` | A setting of the `workflows:` entry `<name>` |
| `extractors.<name>.<path>` | A setting of the `extractors:` entry `<name>` |
| `steps.<step>.<setting>` | A setting of a transform, combine or check step in the custom workflow the task runs |
| anything else | A path into the entry the task runs |

A path walks fields by name: `checks.image-outliers.warning` sets one field of `quality`'s `checks`. The sections below
give an example of each other form.

## Reach a list item by name

Where a path meets a list of entries, such as a `shift` entry's `detectors`, it picks the item by its `name`. An entry
without a `name` is named by its type, so an unnamed detector is reached by its type:

```yaml
workflows:
  - name: ood
    type: shift
    detectors:
      - {name: knn, type: ood-kneighbors, k: 10}
      - {type: ood-domain-classifier, n_folds: 5}

tasks:
  - name: ood_check
    workflow: ood
    sources: [train, operational]
    extractor: bovw_ext
    matrix:
      detectors.knn.k: [5, 10, 20]
      detectors.ood-domain-classifier.n_folds: [3, 5]
```

A path may end on a whole item, written whole, and the item keeps its name:
`detectors.knn: [{type: ood-kneighbors, k: 5}, {type: ood-kneighbors, k: 20, distance_metric: euclidean}]`. Items are
not reached by their position. A name no item has fails the load, listing the names there are:
`` `detectors` in shift has no item named `knn2` (it has `knn`, `ood-domain-classifier`) ``.

## Reach a key of a mapping

A setting that maps names to values, such as an `outliers` evaluator's per-metric `outlier_threshold`, is walked by key,
and a key the mapping lacks is added for the run:

```yaml
evaluators:
  - name: outl
    type: outliers
    flags: [pixel, visual]
    outlier_threshold:
      brightness: [zscore, 3.0]
      contrast: [zscore, 3.0]

tasks:
  - name: outliers_only
    evaluator: outl
    sources: train
    matrix:
      outlier_threshold.brightness: [[zscore, 2.0], [zscore, 3.0], [iqr, 1.5]]
```

Each value is the whole threshold for `brightness`, a method and its bound, so the values are lists. A path through a
setting left unset fails the load, as there is nothing to walk into. Without `outlier_threshold` on the entry, the load
says `` `outlier_threshold` in outliers is unset, so `outlier_threshold.brightness` has nothing to set; vary
`outlier_threshold` whole ``.

## Vary the task's sources and extractor

`sources` runs the task on each source, or list of sources, in turn. `extractor` runs it with each extractor:

```yaml
extractors:
  - name: resnet
    model: onnx
    model_path: ./models/resnet50-v2-7.onnx
    batch_size: 32

tasks:
  - name: clean_each
    workflow: cleaning
    sources: train
    matrix:
      sources: [train, validation, test]
  - name: ood_by_extractor
    workflow: ood
    sources: [train, operational]
    matrix:
      extractor: [bovw_ext, resnet]
```

A value that names no source or extractor fails the load. `ood_by_extractor` sets no `extractor:` of its own, which is
fine since every run has one: a task as written is not checked, only its runs.

## Reach the entries a custom workflow runs

A custom workflow's evaluator steps run `evaluators:` entries, whose settings live in those entries. Vary them with
`evaluators.<name>.<path>`, and a transform's, combine's or check's settings with `steps.<step>.<setting>`:

```yaml
evaluators:
  - {name: knn, type: ood-kneighbors, k: 10}

workflows:
  - name: novelty
    inputs: [reference, test]
    steps:
      - {name: knn, evaluator: knn, input: [reference, test]}
      - {name: knn_check, check: ood, input: knn}

tasks:
  - name: novelty_check
    workflow: novelty
    sources: [train, operational]
    extractor: bovw_ext
    matrix:
      evaluators.knn.k: [5, 10, 20]
      steps.knn_check.warning: [5.0, 10.0]
```

`steps.` reaches a setting whether the step writes it or leaves it at its default, as `knn_check` leaves `warning` at
10.0. Changing an entry changes it for every step in the run that reads it: two steps naming `knn` both see the new `k`.
`workflows.<name>.<path>` reaches a `workflows:` entry the same way, such as a `quality` entry a custom workflow runs as
a step. `steps.knn_check.warning` and `workflows.novelty.steps.knn_check.warning` are one key.

## Vary an extractor's settings

`extractors.<name>.<path>` varies an extractor's settings:

```yaml
tasks:
  - name: ood_vocab
    workflow: ood
    sources: [train, operational]
    extractor: bovw_ext
    matrix:
      extractors.bovw_ext.vocab_size: [256, 512, 1024]
```

## Check what a key may not vary

A matrix varies settings. It may not change which entry is which, or how a chain is wired, since every other key relies
on them. The load refuses:

- an entry's `name` and `type` (an extractor's `name` and `model`, a custom workflow's `inputs`), and a list item's
  `name`: `` `type` varies `type`, which says which entry this is: a matrix varies settings, not identities ``;
- on a custom workflow's step, its kind (`evaluator`, `workflow`, `transform`, `combine`, `check`), `name`, `input`,
  `by`, `optional`, `pairs` and `extractor`;
- an `export` step's `to`, since each run already writes under a directory of its own (see
  [Gate a matrix run and read its files](gate_a_matrix_run.md));
- `steps.<step>.…` where the step runs an evaluator or workflow entry, with the key to use instead:
  `` `steps.knn.k`: step 'knn' runs the evaluator entry `knn`, whose settings live there: vary `evaluators.knn.k` ``;
- the settings of an extractor given as a Python object, not by its settings;
- a setting the entry does not have: `` quality has no setting `outlier_thresh` ``;
- the `datasets:`, `sources:`, `views:`, `preprocessors:`, `metadata:`, `stats:` and `ontologies:` pools. An entry's
  reference to one of them by name can vary, such as a `quality` entry's `stats:`;
- anything but `sources`, where the task reads an evaluator that allows only that. `feasibility` does: its runs would be
  separate calls over the same data, and choosing among them needs `alpha` divided by their number, which Flow cannot
  know you mean. Separate sources are separate populations, so `sources` may still vary.

## Make a key reach every run

A key that names another entry must name one each of its runs reads: the entry the task runs, the entries its custom
workflow's steps run, and the extractors any of them use. On an `ood_compare` task running `ood`, the matrix
`{extractor: [bovw_ext, resnet], extractors.bovw_ext.vocab_size: [256, 512]}` means to compare two extractors and two
BoVW vocabularies. But the runs with `resnet` read no `bovw_ext`, so `vocab_size` would change nothing in them, and the
load refuses:

```text
Task 'ood_compare' has a matrix that can't run:
- run 3 (extractor=resnet, extractors.bovw_ext.vocab_size=256): the run reads no extractor `bovw_ext`, so
  `extractors.bovw_ext.vocab_size` would change nothing; give it a grid of its own with the runs that read it
- run 4 (extractor=resnet, extractors.bovw_ext.vocab_size=512): the run reads no extractor `bovw_ext`, so
  `extractors.bovw_ext.vocab_size` would change nothing; give it a grid of its own with the runs that read it
```

Two grids say what was meant, three runs:

```yaml
tasks:
  - name: ood_compare
    workflow: ood
    sources: [train, operational]
    matrix:
      - extractor: [bovw_ext]
        extractors.bovw_ext.vocab_size: [256, 512]
      - extractor: [resnet]
```

## Next steps

- [Chain steps into a workflow of your own](write_a_custom_workflow.md) — Write the custom workflows whose steps a
  matrix can vary.
- [Gate a matrix run and read its files](gate_a_matrix_run.md) — Set the exit code, export a dataset for each run and
  read the matrix in the result files.
