# Transform Catalog

A transform is a step, in a preset's chain or a custom workflow, that makes Datasets: it views, wraps, merges, splits,
selects from, removes from or relabels the Datasets earlier steps made, or writes one to disk. See [Workflows as Chains
of Steps](../concepts/WorkflowsAsChains.md) for how steps chain, and [Chain steps into a workflow of your
own](../how_to/write_a_custom_workflow.md) for a worked example. Each entry's **Used in** names the presets that run the
transform; where it names none, chain it in a workflow of your own. Each example assumes the pipeline defines
`datasets:`, the sources `train`, `test`, `validation`, `operational`, `labeled` and `unlabeled`, and the extractor
`bovw_ext`, as [Evaluator recipes](../how_to/evaluator_recipes.md) does.

## At a glance

| Type | Reads | Makes | Runs |
| --- | --- | --- | --- |
| `view` | `input`: a Dataset | a Dataset | `dataeval.data` view operations |
| `wrap` | `input`: an object-detection Dataset | a classification Dataset, one item per detection; with `other_kinds: pass`, another kind unchanged | `dataeval.data.DetectionCrops` |
| `merge` | `input`: two or more Datasets | a Dataset | `dataeval.data.merge_datasets` |
| `split` | `input`: a Dataset | `train`, `val` and `test` | `dataeval.data.split_dataset` |
| `kfold` | `input`: a Dataset | `train` and `val`, one per fold, and `test` | `dataeval.data.split_dataset` |
| `select` | `input`: a Dataset; `ranking`: a `prioritization` Output | a Dataset | `dataeval.data.Indices` |
| `remove` | `input`: a Dataset; `plans`: `duplicates` or `outliers` Outputs | a Dataset | `deduplicate` and `prune`, then `dataeval.data.Indices(plan, exclude=True)` |
| `conform` | `input`: a Dataset; `alignment`: a `label-alignment` Output | a Dataset | `dataeval.data.Relabel` |
| `export` | `input`: an object-detection Dataset | an export record | the datamaite writers that top-level `exports:` use |

## How settings work

A transform step writes its settings beside it, in the step entry. Every step also holds `name` and its kind key,
here `transform:` naming the type, and may hold `optional:`. The catalog describes those once, in
[Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md), rather than per transform.

The fields that feed a transform's ports hold addresses: an input, an earlier step, one of its outputs such as
`split.train`, or one element of a list such as `kfold.train[0]`. A step with one output is addressed by its name. An
unknown setting, or a value its field does not take, fails the config load. `dataeval-flow steps <type>` prints a
transform's ports and the JSON Schema of its settings.

## Shaping a Dataset

### `view`

Applies DataEval view operations, such as ClassFilter, Relabel, Limit or Indices.

It applies them as a source's `view:` does.

- **Reads:** `input`, a Dataset of any kind.
- **Makes:** one Dataset of the same kind.

**Settings** ({py:class}`~dataeval_flow.steps.transforms.ViewTransformConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to view |
| `operations` | a list of operations, each `{type, params}` as a `views:` entry writes them | none | The `dataeval.data` operations to apply, in order |
| `view` | the name of a `views:` entry | none | A `views:` entry whose operations to apply instead |

Name exactly one of `operations` and `view`. A `view:` name is replaced by that entry's operations when the config
loads, so editing the entry changes the Dataset's key. An operation that needs a particular Dataset kind is refused
before any step runs when its input is another kind.

- **Used in:** [`data-splitting`](presets.md#data-splitting)

```yaml
workflows:
  - name: example
    inputs: [data]
    steps:
      - name: first-hundred
        transform: view
        input: data
        operations:
          - {type: Limit, params: {size: 100}}
```

### `wrap`

Wraps a Dataset in a DataEval wrapper that changes its kind, such as DetectionCrops.

It runs `dataeval.data.DetectionCrops`. `min_size` drops detections whose box's shorter side is under that many pixels.
By default, wrapping a Dataset of another kind is refused before any step runs. With `other_kinds: pass` a chain can
wrap detection data and read classification data as it is: `data-coverage` does. The video wrappers come once Flow can
load a tracking dataset.

- **Reads:** `input`, an object-detection Dataset.
- **Makes:** a classification Dataset with one item per detection the wrapper keeps, labelled with the detection's
  class. With `other_kinds: pass`, a Dataset of another kind is handed on unchanged.

**Settings** ({py:class}`~dataeval_flow.steps.transforms.WrapConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to wrap |
| `wrapper` | `DetectionCrops` | required | The DataEval wrapper |
| `params` | a mapping of the wrapper's keyword arguments | `{}` | For `DetectionCrops`: `region`, `padding`, `min_size`, `square` and `fill` |
| `other_kinds` | `refuse` or `pass` | `refuse` | A Dataset the wrapper does not take: refused before the run, or passed on unchanged, keeping its kind and its source's cached embeddings |

- **Used in:** [`audit`](presets.md#audit), [`data-coverage`](presets.md#data-coverage)

```yaml
workflows:
  - name: example
    inputs: [data]
    steps:
      - name: crops
        transform: wrap
        input: data
        wrapper: DetectionCrops
        params: {min_size: 32}
        other_kinds: pass
```

## Combining and splitting

### `merge`

Concatenates Datasets that share a label vocabulary, in order.

It runs `dataeval.data.merge_datasets`. The inputs must share `index2label`.
Conform each onto one ontology first: DataEval refuses inputs whose vocabularies differ, which fails the step. Naming
fewer than two fails the config load.

- **Reads:** `input`, two or more Datasets.
- **Makes:** one Dataset.

**Settings** ({py:class}`~dataeval_flow.steps.transforms.MergeConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | a list of two or more addresses | required | The Datasets to concatenate, in order |

- **Used in:** none; chain it in a [workflow of your own](../how_to/write_a_custom_workflow.md)

```yaml
workflows:
  - name: example
    inputs: [street, aerial]
    steps:
      - {name: merged, transform: merge, input: [street, aerial]}
```

### `split`

Splits a Dataset into train, val and test, optionally stratified or grouped.

It runs DataEval's `split_dataset` over the Dataset's metadata.

- **Reads:** `input`, a Dataset of any kind.
- **Makes:** `train`, `val` and `test`, each a view of the input, addressed as `<step>.train`, `<step>.val` and
  `<step>.test`.

**Settings** ({py:class}`~dataeval_flow.steps.transforms.SplitConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to split |
| `test_frac` | a number from 0 up to, not including, 1 | `0.0` | The share held out as `test` |
| `val_frac` | a number from 0 up to, not including, 1 | `0.0` | The share held out as `val` |
| `stratify` | `true` or `false` | `false` | Whether each part keeps the input's class proportions |
| `split_on` | a list of metadata factor names | none | Factors whose values never straddle parts, such as a scene or site. Classification data only: DataEval ignores it on detection data, with a warning in the log |
| `metadata` | the name of a `metadata:` policy | DataEval's defaults | The policy the Dataset's metadata is built under |

Set `test_frac`, `val_frac` or both; together they must leave something to train on. A part whose fraction is 0 is
empty, and a step that reads it fails the config load. With `test_frac` alone, DataEval's one holdout becomes `test`.

- **Used in:** [`data-splitting`](presets.md#data-splitting)

```yaml
workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: split, transform: split, input: data, test_frac: 0.2, val_frac: 0.1, stratify: true}
      - {name: view-train, transform: view, input: split.train, operations: [{type: Limit, params: {size: 100}}]}
```

### `kfold`

Splits a Dataset into k train and val folds, and one test.

It runs DataEval's `split_dataset` over the Dataset's metadata.

- **Reads:** `input`, a Dataset of any kind.
- **Makes:** `train` and `val`, each a list keyed `0` to `folds - 1`, and `test`. A step that reads `<step>.train` runs
  once per fold; `<step>.train[0]` reads the first fold alone.

**Settings** ({py:class}`~dataeval_flow.steps.transforms.KFoldConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to split |
| `folds` | a whole number, 2 or more | required | How many train and val pairs |
| `test_frac` | a number from 0 up to, not including, 1 | `0.0` | The share held out as `test` |
| `stratify` | `true` or `false` | `false` | Whether each part keeps the input's class proportions |
| `split_on` | a list of metadata factor names | none | Factors whose values never straddle parts, such as a scene or site. Classification data only: DataEval ignores it on detection data, with a warning in the log |
| `metadata` | the name of a `metadata:` policy | DataEval's defaults | The policy the Dataset's metadata is built under |

`test` is empty when `test_frac` is 0, and a step that reads it then fails the config load, as does a fold key outside
`0` to `folds - 1`. To rebalance each fold's training set, follow `kfold` with a `view` step that reads `<step>.train`
and applies `ClassBalance`.

- **Used in:** [`data-splitting`](presets.md#data-splitting)

```yaml
workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: folds, transform: kfold, input: data, folds: 5, test_frac: 0.2, stratify: true}
      - {name: first-train, transform: view, input: "folds.train[0]", operations: [{type: Limit, params: {size: 100}}]}
```

## Acting on an evaluator's output

`select`, `remove` and `conform` apply an evaluator step's output to a Dataset, and only to the Dataset it was computed
on. The step their `ranking:`, `plans:` or `alignment:` names must have read exactly their `input`, except that a
`prioritization` ranking may also read a reference set after it, so for `select` only its first input must be
`input`. Anything else fails the config load.

### `select`

Keeps the top of a Prioritize ranking of the same Dataset.

It keeps the items in ranked order, and runs `View(input, Indices(ranking.indices[:n]))`.

- **Reads:** `input`, a Dataset; `ranking`, a `prioritization` Output.
- **Makes:** one Dataset.

**Settings** ({py:class}`~dataeval_flow.steps.transforms.SelectConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to select from |
| `ranking` | the address of a `prioritization` step | required | The ranking, computed on `input` |
| `n` | a whole number, 1 or more | none | How many items to keep |
| `fraction` | a number above 0, up to 1 | none | The share of items to keep, rounded up |

Name exactly one of `n` and `fraction`.

- **Used in:** [`data-prioritization`](presets.md#data-prioritization)

```yaml
evaluators:
  - {name: prioritization, type: prioritization, order: hard_first}

workflows:
  - name: example
    inputs: [pool]
    steps:
      - {name: prioritization, evaluator: prioritization, input: pool}
      - {name: top, transform: select, input: pool, ranking: prioritization, fraction: 0.1}
```

### `remove`

Removes the items, detections or tracks that Duplicates and Outliers plans name.

It runs `View(input, Indices(plan, exclude=True))`.

- **Reads:** `input`, a Dataset; `plans`, one or more `duplicates` or `outliers` Outputs.
- **Makes:** one Dataset.

**Settings** ({py:class}`~dataeval_flow.steps.transforms.RemoveConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to remove from |
| `plans` | a mapping from the address of an evaluator step to its plan method's arguments | required | The Outputs whose plans to apply, at least one |

Each `plans` entry's arguments are passed to DataEval's plan method:

- **A `duplicates` step:** `DuplicatesOutput.deduplicate`, taking `dup_types`, `keep`, `exclude_groups` and
  `levels`.
- **An `outliers` step:** `OutliersOutput.prune`, taking `metrics` and `min_flags`.

`{}` takes DataEval's defaults: for duplicates, every exact copy but the first is removed. Each argument's name and
value is checked against the method when the config loads. The plans combine, so whatever any of them names is removed,
whether whole items or single detections. The step's report section counts what it removed at each level. The
Dataset's key follows the plan applied, not the arguments behind it, so two settings that remove the same rows key
alike.

- **Used in:** [`data-cleaning`](presets.md#data-cleaning), [`data-prioritization`](presets.md#data-prioritization)

```yaml
evaluators:
  - {name: dupes, type: duplicates}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: dupes, evaluator: dupes, input: data}
      - name: clean
        transform: remove
        input: data
        plans:
          dupes: {keep: first}
```

### `conform`

Relabels a Dataset onto an ontology by its label alignment, refusing loss beyond `allow`.

It runs `View(input, Relabel(remap, target=ontology))`.

- **Reads:** `input`, a Dataset; `alignment`, a `label-alignment` Output.
- **Makes:** one Dataset.

**Settings** ({py:class}`~dataeval_flow.steps.transforms.ConformConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to relabel |
| `alignment` | the address of a `label-alignment` step | required | The alignment, computed on `input` |
| `allow` | `lossless`, `lossy` or `partial` | `lossless` | The most loss accepted |
| `class_remap` | a mapping from a source class to a concept, by id or by a label no other concept shares | `{}` | Overrides of the alignment's remap |

`allow` levels are ordered, each accepting what the one before it does:

- `lossless` maps each class to a concept of its own;
- `lossy` lets several classes collapse onto one concept;
- `partial` also drops classes that align to nothing.

An alignment beyond `allow` fails the step, naming what collapses or aligns to nothing. An override can settle a class
that aligned to nothing, and so change how much loss the step needs. An override for a class the input does not have
fails the step, naming the classes it has. Two conforms onto one ontology give their outputs the same `index2label`,
which `merge` needs. The step adds a record of the remap it applied to the result's `label_space`, and its report
section lists each collapse and each dropped class.

- **Used in:** none; chain it in a [workflow of your own](../how_to/write_a_custom_workflow.md)

```yaml
ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}

evaluators:
  - {name: align, type: label-alignment, ontology: vehicles}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: align, evaluator: align, input: data}
      - {name: conformed, transform: conform, input: data, alignment: align, allow: lossy}
```

## Writing out

### `export`

Writes an object-detection Dataset to disk as COCO, YOLO or another datamaite format.

It writes through the same datamaite writers as top-level `exports:`. Handed a list (a list
input, a list output such as `kfold.train`, or a step run once per element), it writes each element under its key:
`datasets/<to>/<key>/`, such as `datasets/t.dataset/0/` for fold 0. A key that is not one plain directory name fails its
element, and the others are still written.

- **Reads:** `input`, one object-detection Dataset.
- **Makes:** an export record: `path`, `format`, `mode`, `items` and `provenance`.

**Settings** ({py:class}`~dataeval_flow.steps.transforms.ExportTransformConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to write |
| `format` | `coco`, `yolo`, `huggingface_vision` or `visdrone` | `coco` | The format to write |
| `mode` | `error`, `replace` or `append` | `error` | What to do when the destination holds a dataset: refuse, replace it, or add to it |
| `ontology` | the name of an `ontologies:` entry | none | The ontology to record in the provenance |
| `to` | one directory name, not `.` or `..` | `<task>.<step>` | The directory under `<output>/datasets/` |

The step writes to `<output>/datasets/<to>/`, with a `provenance.json` that records each source the Dataset descends
from, with the dataset and view it read, and the lineage of the Dataset written. Its `label_space` list is shaped as a
top-level export's: the sources' own Relabel records, then each `conform` on the way, in chain order, with its remap
and ontology digest. A Dataset a chain made is written with its pixels encoded; a chain input whose view left its
pixels alone is written by reference to its image files, as a top-level export is. A run with no output directory skips
the step, and that is not a failure. A Dataset of another kind is refused before any step runs. Two destinations that
coincide anywhere in the run, top-level `exports:` included, fail the config load.
[Export a dataset](../how_to/export_a_dataset.md) covers the formats and modes in full.

- **Used in:** none; chain it in a [workflow of your own](../how_to/write_a_custom_workflow.md)

```yaml
datasets:
  - {name: street, format: coco, path: ./street}

sources:
  - {name: street, dataset: street}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: dataset, transform: export, input: data, format: yolo, mode: replace}

tasks:
  - {name: build, workflow: example, sources: [street]}
```
