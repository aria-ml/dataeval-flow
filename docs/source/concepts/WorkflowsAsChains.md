# Workflows as Chains of Steps

A workflow type such as `data-cleaning` runs one fixed analysis. A **custom workflow** is one you write: a chain of
steps, each reading what an earlier step made. One custom workflow can conform two corpora onto an ontology, merge
them, remove the duplicates, check what is left, split it and write it out, all from YAML.

## A workflow is a chain of steps

A `workflows:` entry is one of two things:

- **A workflow type,** named by `type:`, with that type's settings.
- **A chain,** with `inputs:` and `steps:` and no `type:`.

```yaml
workflows:
  - name: basic_clean               # a workflow type
    type: data-cleaning
    outlier_method: adaptive
    outlier_flags: [dimension, pixel, visual]

  - name: clean_export              # a chain
    inputs: [data]
    steps:
      - {name: dupes, evaluator: dupes, input: data}
      - {name: clean, transform: remove, input: data, plans: {dupes: {keep: first}}}
      - {name: corpus, transform: export, input: clean}

tasks:
  - {name: prep, workflow: clean_export, sources: [train]}
```

A task runs either one with `workflow:`. A chain's `inputs:` are slots, and the task's `sources:` bind to them in
order, so one chain can run on different sources from different tasks. The last slot may be a list,
`{name: cameras, list: true}`, which binds every source left over, keyed by source name.

Steps run in the order they are written, and a step reads only the inputs and the steps above it. A chain whose steps
do not connect fails when the config loads, before any data is read.

## Kinds of step

Each step names exactly one kind:

| Kind key | Names | Makes |
| --- | --- | --- |
| `evaluator:` | an `evaluators:` entry | that evaluator's DataEval output |
| `workflow:` | a `workflows:` entry with a `type:` | that workflow's result, with its findings |
| `transform:` | a built-in transform, with its settings beside it | one or more Datasets, or an export record |

An evaluator or workflow step takes its settings from the entry it names. The step itself holds only what it reads,
and optionally `extractor:` and `optional:`. A chain cannot run as a step of another chain.

Transform steps make Datasets:

| Transform | Reads | Makes |
| --- | --- | --- |
| `view` | a Dataset | that Dataset through DataEval view operations, inline (`operations:`) or from `views:` (`view:`) |
| `merge` | two or more Datasets that share a label vocabulary | one Dataset, in the order named |
| `split` | a Dataset | `train`, `val` and `test`, from `test_frac`, `val_frac` or both |
| `kfold` | a Dataset | `train` and `val`, each a list with one Dataset per fold, and `test` |
| `wrap` | an object-detection Dataset | a classification Dataset with one item per detection (`wrapper: DetectionCrops`) |
| `select` | a Dataset, and a `scope.prioritize` ranking of it (`ranking:`) | the first `n`, or `fraction`, of the ranking |
| `remove` | a Dataset, and Duplicates or Outliers outputs computed on it (`plans:`) | the Dataset without what the plans name |
| `conform` | a Dataset, and a `scope.label-alignment` of it (`alignment:`) | the Dataset relabelled onto the ontology |
| `export` | an object-detection Dataset | a corpus on disk under the run's output directory, and a record of it |

`remove`, `select` and `conform` apply an evaluator's output to a Dataset, and only to the Dataset it was computed on.
The step that `plans:`, `ranking:` or `alignment:` names must have read exactly the Dataset the transform's own
`input:` names. The config refuses anything else when it loads. A `scope.prioritize` ranking may also read a reference
set, so for `select` only the first Dataset it read must match.

`dataeval-flow steps` lists every step a chain can use, and `dataeval-flow steps NAME` prints one step's ports and
settings.

Only workflow types judge health. A chain's findings are those of the workflow-type steps it ran, so a chain of
evaluators and transforms reports no warnings, and its health is `ok` unless a step fails. Check steps, which will
judge what evaluators find, come later.

## Addresses and lists

A step names what it reads by address:

| Address | Reads |
| --- | --- |
| `clean` | an input, or a step with one output |
| `split.train` | one output of a step with several |
| `kfold.train[0]` | one element of a list, by its key |
| `cameras[cam1]` | one source of a list input, by source name |

A step with one output is addressed by its name alone, even when that output is a list: `clean.output` is refused. A
step with several outputs is addressed by one of them: `split` alone is refused, and `split.train` is not.

Lists come from a list input, keyed by source name, and from `kfold`'s `train` and `val`, keyed `0` to `k-1`. A step
that reads one Dataset, handed a list, runs once per element, and its output is a list with the same keys:

```yaml
workflows:
  - name: folds
    inputs: [data]
    steps:
      - {name: kfold, transform: kfold, input: data, folds: 5}
      - {name: fold_balance, evaluator: balance, input: kfold.train}    # five runs, keyed 0 to 4
      - name: first_dupes                                               # one run, on the first fold
        evaluator: dupes
        input: kfold.train[0]
```

In YAML's flow style, inside braces or brackets, quote an address that holds a key, since a bracket there opens a
YAML list: `{name: first_dupes, evaluator: dupes, input: "kfold.train[0]"}`, or `input: [reference, "cameras[cam1]"]`.
Block style, as above, needs no quotes.

Several lists across one step zip by key: element `cam1` of one runs with element `cam1` of the other. A single
Dataset beside a list is repeated for each element, as a drift reference is here:

```yaml
workflows:
  - name: per_camera
    inputs: [reference, {name: cameras, list: true}]
    steps:
      - {name: drift, evaluator: mmd, input: [reference, cameras]}     # one run per camera

tasks:
  - {name: watch, workflow: per_camera, sources: [train, cam1, cam2], extractor: bovw_ext}
```

A key missing from one of the lists skips that element, and the reason names the key. Lists do not nest: a step that
outputs lists refuses a list where it reads one Dataset.

## Derived data

Evaluators read statistics, metadata and embeddings. In a chain, Flow derives them from the Dataset each step reads,
as it does from a source. Each Dataset derives its own, so a `bias.balance` step on `clean` reads the metadata of the
cleaned corpus. Removing items makes a new Dataset, and everything is derived again from it.

The same Dataset under the same policy is derived once. Two steps that read `merged`'s statistics share one
computation. With a disk cache (`--cache`), a Dataset a chain made is cached under its own key, as a source is, so a
rerun that makes the same Dataset reads its derived data back.

Policies stay where they are: a metadata or statistics policy is named on the `evaluators:` or `workflows:` entry, and
the step derives under it. `split` and `kfold` take theirs as `metadata:` on the step itself. Embeddings come from the
task's `extractor:`, or from a step's own `extractor:`, which overrides the task's for that step.

## Failures

A step ends `ok`, `failed` or `skipped`. A failed step skips every step that reads it, directly or through other
steps, and the reason names what failed: "needs `clean`, which failed". Steps that do not depend on it still run. In a
list, each element has its own status, and a failed element skips only the elements that read it.

`optional: true` records a step's failure as a skip that carries the error, and the task does not fail because of it.
The steps that read it are still skipped.

A required step's failure fails the task: its health status is `failed`, and the CLI exits 1, as for any failed task.
The steps that ran stay in the result, with what they made. Reading `result.output` on a failed result raises, as for
any result, but `result.steps` holds every step either way.

An `export` step in a run with no output directory is skipped with the reason "the run has no output directory". That
is not a failure: a run from Python writes nothing unless it is given somewhere to write.

A config error still stops the run before any step runs, and produces no result. Addresses and settings are checked
when the config loads. Dataset kinds are checked once the sources load, since DataEval reads a Dataset's kind from its
first item: an `export` of a classification Dataset, say, stops the run there, before an hour of statistics.

## Lineage

Each Dataset in a chain has a lineage record, in `result.metadata.lineage` and under `metadata.lineage` in the JSON.
It holds the Dataset's address (`name`), the step that made it and that step's `type`, the addresses it was made from
(`inputs`), the `source` bound to it when it is an input, its number of `items`, and a 12-character `digest`:

```json
{"name": "clean", "step": "clean", "type": "remove", "inputs": ["merged"], "source": null,
 "digest": "1bf4dcfcadf4", "items": 47}
```

The report opens each step's section with a lineage line naming every address the step read. Each Dataset among them
is walked back through the first Dataset it was made from, to the source bound to it, and an Output is named as it is.
For a `remove` step that read `merged` and the `dupes` Output, and for a step that read `split.train`:

```text
On `merged` ← `street_conformed` ← `street` (street_2024), `dupes`
On `split.train` ← `clean` ← `merged` ← `street_conformed` ← `street` (street_2024)
```

The digest is what lets two results be compared. Two results that give a Dataset the same digest read the same data:
the same sources, through the same steps and settings, to the same content. The content is what a step resolved from
the data: the plan `remove` applied, the indices `select`, `split` and `kfold` chose, and the remap `conform` applied.
So two `remove` steps whose different plan arguments remove the same items give the same digest, and removing one
detection changes the digest though the number of items stays the same.

An `export` step writes the lineage of the Dataset it wrote, and each `conform` on the way with its remap and ontology
digest, into the corpus's `provenance.json`.

See [Chain steps into a workflow of your own](../how_to/write_a_custom_workflow.md) to write one, and
[Workflows and Evaluators](WorkflowsAndEvaluators.md) for what evaluators and workflow types each decide.
