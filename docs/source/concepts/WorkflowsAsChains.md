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
| `workflow:` | a `workflows:` entry with a `type:` | that workflow's result, with its findings; or a preset's steps, [below](#workflow-types-as-presets) |
| `transform:` | a built-in transform, with its settings beside it | one or more Datasets, or an export record |
| `combine:` | a registered combine, with its settings beside it | an Output a check reads, made from Outputs |
| `check:` | a registered check, with its thresholds beside it | findings, each `ok`, `info` or `warning` |

An evaluator or workflow step takes its settings from the entry it names. The step itself holds only what it reads,
and optionally `extractor:` and `optional:`. A custom workflow cannot run as a step of another.

Transform steps make Datasets. The [Transform Catalog](../reference/transforms.md) lists each one's settings:

| Transform | Reads | Makes |
| --- | --- | --- |
| `view` | a Dataset | that Dataset through DataEval view operations, inline (`operations:`) or from `views:` (`view:`) |
| `merge` | two or more Datasets that share a label vocabulary | one Dataset, in the order named |
| `split` | a Dataset | `train`, `val` and `test`, from `test_frac`, `val_frac` or both |
| `kfold` | a Dataset | `train` and `val`, each a list with one Dataset per fold, and `test` |
| `wrap` | an object-detection Dataset | a classification Dataset with one item per detection (`wrapper: DetectionCrops`) |
| `select` | a Dataset, and a `prioritize` ranking of it (`ranking:`) | the first `n`, or `fraction`, of the ranking |
| `remove` | a Dataset, and Duplicates or Outliers outputs computed on it (`plans:`) | the Dataset without what the plans name |
| `conform` | a Dataset, and a `label-alignment` of it (`alignment:`) | the Dataset relabelled onto the ontology |
| `export` | an object-detection Dataset | a corpus on disk under the run's output directory, and a record of it |

`remove`, `select` and `conform` apply an evaluator's output to a Dataset, and only to the Dataset it was computed on.
The step that `plans:`, `ranking:` or `alignment:` names must have read exactly the Dataset the transform's own
`input:` names. The config refuses anything else when it loads. One element of a step that ran once per element of a
list was computed on that element: `dupes[0]`, where `dupes` read `kfold.train`, applies to `kfold.train[0]`. A
`prioritize` ranking may also read a reference set, so for `select` only the first Dataset it read must match.

`dataeval-flow steps` lists every step a chain can use, and `dataeval-flow steps NAME` prints one step's ports and
settings.

## Judging what a chain found

Evaluators judge nothing: they report what DataEval determined. A **check** step judges it. It reads Outputs,
compares them with thresholds written beside it, and makes findings, each `ok`, `info` or `warning`. A **combine** step
makes an Output a check reads, from Outputs and the Datasets they were computed on, such as outliers counted per class.
The [Check and Combine Catalog](../reference/checks.md) lists the built-in ones.

```yaml
evaluators:
  - {name: labels, type: label-health}

workflows:
  - name: judged
    inputs: [data]
    steps:
      - {name: dupes, evaluator: dupes, input: data}
      - {name: labels, evaluator: labels, input: data}
      - {name: duplicates, check: duplicate-rate, input: dupes, near: 2.0}
      - {name: imbalance, check: class-imbalance, input: labels}
```

A chain's health rolls up over its checks' findings, and over the findings of the workflow-type steps it runs. Its
status is `warning` where any finding is a warning, and `failed` where a step that is not `optional` failed. The
result's JSON lists the check findings at the top, each naming its step. A workflow-type step's findings stay in that
step. A threshold of `null` judges nothing: the finding is still made, as `info`.

A check is never skipped because an input produced nothing. It makes one `info` finding briefed `not assessed`,
saying which input holds nothing and why, so the report shows what could not be judged.

The report gives each finding a section of its own, with the evidence it judged below it: `duplicates`' finding holds
the `dupes` step's duplicate groups. The steps no finding shows follow, then a table of every step.
[Read evaluation outputs](../how_to/read_evaluation_outputs.md) describes the layout.

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

A key is checked as soon as it is known. A fold key outside `0` to `k-1` fails the config load. A source name is known
once a task binds its sources, so `cameras[cam3]` fails the load of a task that binds only `cam1` and `cam2` to
`cameras`, and a run from Python that binds them fails before any step runs.

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
Block style, as above, needs no quotes. A file that leaves one unquoted fails to load, and the error says which address
to quote.

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

A key missing from one of the lists skips that element, and the reason names the key. A check run once per element
makes each element's findings, and the report groups them under the element's key. Lists do not nest: a step that
outputs lists refuses a list where it reads one Dataset. An `export` handed a list writes each element in a directory of
its own, named by its key.

## Workflow types as presets

A workflow type can be a **preset**: its settings expand to a chain of steps. `data-cleaning` and `data-prioritization`
are presets. Data-cleaning's evaluators find outliers and duplicates, its checks judge them against `health_thresholds`,
and its `clean` step removes what they flagged. The [Check and Combine
Catalog](../reference/checks.md#data-cleaning-is-this-chain) lists the chain. The other workflow types will follow.
Until then, each runs as one step that makes its result, and its findings stay in that step.

Run as a task, a preset returns a `ChainResult` under its own type id, such as `data-cleaning`, holding each step of
its chain. Run as a step of a custom workflow, as `{name: cleaning, workflow: basic_clean, input: data}` runs the
`basic_clean` entry above, its steps run in your chain as `cleaning/outliers`, `cleaning/dupes` and so on. The step's
`optional:` holds for each of them, and its `extractor:` for each that reads embeddings. Its checks' findings are your
chain's, listed at the top of the JSON, each naming its step, such as `cleaning/image-outliers`.

Only a preset's declared outputs can be addressed, and always by name: `cleaning.clean` reads the cleaned Dataset,
while `cleaning` alone, `cleaning.dupes` and `cleaning/dupes` are refused. Handed a list, a preset runs its whole chain
once per element, so `cleaning.clean` is a list with the same keys. A preset's last input can be a list, as
data-prioritization's `pools` is, and a step running it binds that input to a list, such as
`input: [ref, cleaning.clean]`.

## Derived data

Evaluators read statistics, metadata and embeddings. In a chain, Flow derives them from the Dataset each step reads,
as it does from a source. Each Dataset derives its own, so a `balance` step on `clean` reads the metadata of the
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
 "digest": "ca2f57f5f4ad", "items": 47}
```

The report's Steps table names, under *Reads*, every address each step read, one per line. Each Dataset among them is
walked back through the first Dataset it was made from, to the source bound to it, and an Output is named as it is.
For a `remove` step that read `merged` and the `dupes` Output:

```text
`merged` ← `street_conformed` ← `street` (street_2024)
`dupes`
```

A step that read `split.train` reads
`` `split.train` ← `clean` ← `merged` ← `street_conformed` ← `street` (street_2024) ``. A step run once per element
of a list reads the list, walked back through its elements to the source of each: `` `cameras` (cam1, cam2) ``.

The digest is what lets two results be compared. Two results that give a Dataset the same digest read the same data:
the same sources, through the same steps and settings, to the same content. The content is what a step resolved from
the data: the plan `remove` applied, the indices `select`, `split` and `kfold` chose, and the remap `conform` applied.
So two `remove` steps whose different plan arguments remove the same items give the same digest, and removing one
detection changes the digest though the number of items stays the same.

An `export` step writes into the corpus's `provenance.json` each source the Dataset descends from, with the dataset and
view it read; the lineage of the Dataset it wrote; and a `label_space` list shaped as a top-level export's. The list
holds each source's own Relabel records first, as a top-level export of that source writes them, then one record per
`conform` on the way, in chain order, with its remap and ontology digest and the step's address as its `source`.

See [Chain steps into a workflow of your own](../how_to/write_a_custom_workflow.md) to write one, and
[Workflows and Evaluators](WorkflowsAndEvaluators.md) for what evaluators and workflow types each decide.
