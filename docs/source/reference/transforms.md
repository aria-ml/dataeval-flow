# Transform Catalog

A transform is a step of a custom workflow that makes Datasets: it views, wraps, merges, splits, selects from,
removes from or relabels the Datasets earlier steps made, or writes one to disk. See
[Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) for how steps chain, and
[Chain steps into a workflow of your own](../how_to/write_a_custom_workflow.md) for a worked example.

## At a glance

| Type | Reads | Makes | Runs |
| --- | --- | --- | --- |
| `view` | `input`: a Dataset | a Dataset | `dataeval.data` view operations |
| `wrap` | `input`: an object-detection Dataset | a classification Dataset, one item per detection | `dataeval.data.DetectionCrops` |
| `merge` | `input`: two or more Datasets | a Dataset | `dataeval.data.merge_datasets` |
| `split` | `input`: a Dataset | `train`, `val` and `test` | `dataeval.data.split_dataset` |
| `kfold` | `input`: a Dataset | `train` and `val`, one per fold, and `test` | `dataeval.data.split_dataset` |
| `select` | `input`: a Dataset; `ranking`: a `scope.prioritize` Output | a Dataset | `dataeval.data.Indices` |
| `remove` | `input`: a Dataset; `plans`: `quality.duplicates` or `quality.outliers` Outputs | a Dataset | `deduplicate` and `prune`, then `dataeval.data.Indices(plan, exclude=True)` |
| `conform` | `input`: a Dataset; `alignment`: a `scope.label-alignment` Output | a Dataset | `dataeval.data.Relabel` |
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

Applies DataEval view operations to a Dataset, as a source's `view:` does. Configured by
{py:class}`~dataeval_flow.steps.transforms.ViewTransformConfig`.

Reads `input`, a Dataset of any kind. Makes one Dataset of the same kind.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to view |
| `operations` | a list of operations, each `{type, params}` as a `views:` entry writes them | none | The `dataeval.data` operations to apply, in order |
| `view` | the name of a `views:` entry | none | A `views:` entry whose operations to apply instead |

Name exactly one of `operations` and `view`. A `view:` name is replaced by that entry's operations when the config
loads, so editing the entry changes the Dataset's key. An operation that needs a particular Dataset kind is refused
before any step runs when its input is another kind.

### `wrap`

Wraps a Dataset in a DataEval wrapper that changes its kind. Configured by
{py:class}`~dataeval_flow.steps.transforms.WrapConfig`; runs `dataeval.data.DetectionCrops`.

Reads `input`, an object-detection Dataset. Makes a classification Dataset with one item per detection the wrapper
keeps, labelled with the detection's class.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to wrap |
| `wrapper` | `DetectionCrops` | required | The DataEval wrapper |
| `params` | a mapping of the wrapper's keyword arguments | `{}` | For `DetectionCrops`: `region`, `padding`, `min_size`, `square` and `fill` |

`min_size` drops detections whose box's shorter side is under that many pixels, as `data-coverage`'s `crop_min_size`
does. Wrapping a Dataset of another kind is refused before any step runs. The video wrappers come once Flow can load a
tracking dataset.

## Combining and splitting

### `merge`

Concatenates Datasets that share a label vocabulary, in the order named. Configured by
{py:class}`~dataeval_flow.steps.transforms.MergeConfig`; runs `dataeval.data.merge_datasets`.

Reads `input`, two or more Datasets. Makes one Dataset.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | a list of two or more addresses | required | The Datasets to concatenate, in order |

The inputs must share `index2label`. Conform each onto one ontology first: DataEval refuses inputs whose vocabularies
differ, which fails the step. Naming fewer than two fails the config load.

### `split`

Splits a Dataset into train, val and test, from DataEval's `split_dataset` over the Dataset's metadata. Configured
by {py:class}`~dataeval_flow.steps.transforms.SplitConfig`.

Reads `input`, a Dataset of any kind. Makes `train`, `val` and `test`, each a view of the input, addressed as
`<step>.train`, `<step>.val` and `<step>.test`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to split |
| `test_frac` | a number from 0 up to, not including, 1 | `0.0` | The share held out as `test` |
| `val_frac` | a number from 0 up to, not including, 1 | `0.0` | The share held out as `val` |
| `stratify` | `true` or `false` | `false` | Whether each part keeps the input's class proportions |
| `split_on` | a list of metadata factor names | none | Factors whose values never straddle parts, such as a scene or site |
| `metadata` | the name of a `metadata:` policy | DataEval's defaults | The policy the Dataset's metadata is built under |

Set `test_frac`, `val_frac` or both; together they must leave something to train on. A part whose fraction is 0 is
empty, and a step that reads it fails the config load. With `test_frac` alone, DataEval's one holdout becomes `test`.

### `kfold`

Splits a Dataset into `folds` train and val pairs, and one test, from DataEval's `split_dataset` over the Dataset's
metadata. Configured by {py:class}`~dataeval_flow.steps.transforms.KFoldConfig`.

Reads `input`, a Dataset of any kind. Makes `train` and `val`, each a list keyed `0` to `folds - 1`, and `test`. A
step that reads `<step>.train` runs once per fold; `<step>.train[0]` reads the first fold alone.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to split |
| `folds` | a whole number, 2 or more | required | How many train and val pairs |
| `test_frac` | a number from 0 up to, not including, 1 | `0.0` | The share held out as `test` |
| `stratify` | `true` or `false` | `false` | Whether each part keeps the input's class proportions |
| `split_on` | a list of metadata factor names | none | Factors whose values never straddle parts, such as a scene or site |
| `metadata` | the name of a `metadata:` policy | DataEval's defaults | The policy the Dataset's metadata is built under |

`test` is empty when `test_frac` is 0, and a step that reads it then fails the config load, as does a fold key
outside `0` to `folds - 1`. To rebalance each fold's training set, follow `kfold` with a `view` step that reads
`<step>.train` and applies `ClassBalance`.

## Acting on an evaluator's output

`select`, `remove` and `conform` apply an evaluator step's output to a Dataset, and only to the Dataset it was computed
on. The step their `ranking:`, `plans:` or `alignment:` names must have read exactly their `input`; anything else
fails the config load.

### `select`

Keeps the top of a `scope.prioritize` ranking of the same Dataset, in ranked order. Configured by
{py:class}`~dataeval_flow.steps.transforms.SelectConfig`; runs `View(input, Indices(ranking.indices[:n]))`.

Reads `input`, a Dataset, and `ranking`, a `scope.prioritize` Output. Makes one Dataset.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to select from |
| `ranking` | the address of a `scope.prioritize` step | required | The ranking, computed on `input` |
| `n` | a whole number, 1 or more | none | How many items to keep |
| `fraction` | a number above 0, up to 1 | none | The share of items to keep, rounded up |

Name exactly one of `n` and `fraction`. A `scope.prioritize` step may read a reference set after the Dataset it
ranks, so only its first input must be `input`.

### `remove`

Removes what Duplicates and Outliers removal plans name, from the Dataset they were computed on. Configured by
{py:class}`~dataeval_flow.steps.transforms.RemoveConfig`; runs `View(input, Indices(plan, exclude=True))`.

Reads `input`, a Dataset, and `plans`, one or more `quality.duplicates` or `quality.outliers` Outputs. Makes one
Dataset.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to remove from |
| `plans` | a mapping from the address of an evaluator step to its plan method's arguments | required | The Outputs whose plans to apply, at least one |

Each `plans` entry's arguments are passed to DataEval's plan method:

- **A `quality.duplicates` step:** `DuplicatesOutput.deduplicate`, taking `dup_types`, `keep`, `exclude_groups` and
  `levels`.
- **A `quality.outliers` step:** `OutliersOutput.prune`, taking `metrics` and `min_flags`.

`{}` takes DataEval's defaults: for duplicates, every exact copy but the first is removed. Each argument's name and
value is checked against the method when the config loads. The plans combine, so whatever any of them names is removed,
whether whole items or single detections. The step's report section counts what it removed at each level. The
Dataset's key follows the plan applied, not the arguments behind it, so two settings that remove the same rows key
alike.

### `conform`

Relabels a Dataset onto an ontology by its label alignment, refusing loss beyond `allow`. Configured by
{py:class}`~dataeval_flow.steps.transforms.ConformConfig`; runs `View(input, Relabel(remap, target=ontology))`.

Reads `input`, a Dataset, and `alignment`, a `scope.label-alignment` Output. Makes one Dataset.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to relabel |
| `alignment` | the address of a `scope.label-alignment` step | required | The alignment, computed on `input` |
| `allow` | `lossless`, `lossy` or `partial` | `lossless` | The most loss accepted |
| `class_remap` | a mapping from a source class to a concept, by id or by a label no other concept shares | `{}` | Overrides of the alignment's remap |

`allow` levels are ordered, each accepting what the one before it does:

- `lossless` maps each class to a concept of its own;
- `lossy` lets several classes collapse onto one concept;
- `partial` also drops classes that align to nothing.

An alignment beyond `allow` fails the step, naming what collapses or aligns to nothing. An override can settle a class
that aligned to nothing, and so change how much loss the step needs. Two conforms onto one ontology give their outputs
the same `index2label`, which `merge` needs. The step adds a record of the remap it applied to the result's
`label_space`, and its report section lists each collapse and each dropped class.

## Writing out

### `export`

Writes an object-detection Dataset to disk, and records what it wrote. Configured by
{py:class}`~dataeval_flow.steps.transforms.ExportStepConfig`; writes through the same datamaite writers as
top-level `exports:`.

Reads `input`, an object-detection Dataset. Makes an export record: `path`, `format`, `mode`, `items` and
`provenance`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset to write |
| `format` | `coco`, `yolo`, `huggingface_vision` or `visdrone` | `coco` | The format to write |
| `mode` | `error`, `replace` or `append` | `error` | What to do when the destination holds a dataset: refuse, replace it, or add to it |
| `ontology` | the name of an `ontologies:` entry | none | The ontology to record in the provenance |
| `to` | one directory name, not `.` or `..` | `<task>.<step>` | The directory under `<output>/datasets/` |

The step writes to `<output>/datasets/<to>/`, with a `provenance.json` that records the lineage of the Dataset written
and each `conform` on the way, with its remap and ontology digest. A Dataset a chain made is written with its pixels
encoded; a chain input whose view left its pixels alone is written by reference to its image files, as a top-level
export is. A run with no output directory skips the step, and that is not a failure. A Dataset of another kind is
refused before any step runs. Two destinations that coincide anywhere in the run, top-level `exports:` included, fail
the config load. [Export a dataset](../how_to/export_a_dataset.md) covers the formats and modes in full.
