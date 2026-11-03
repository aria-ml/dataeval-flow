# Chain steps into a workflow of your own

A workflow type such as [`quality`](../reference/presets.md#quality) runs one fixed analysis. When the analysis you need
is a sequence, such as relabeling two datasets onto one vocabulary, merging them, dropping the duplicates and then
checking what is left, write it as a custom workflow: a `workflows:` entry whose `steps:` each read what an earlier step
made. This guide builds one section by section. It aligns classes with
[`label-alignment`](../reference/evaluators.md#label-alignment), then chains the
[`conform`](../reference/transforms.md#conform), [`merge`](../reference/transforms.md#merge),
[`remove`](../reference/transforms.md#remove) and [`export`](../reference/transforms.md#export) transforms.
[Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) explains the ideas behind it. You need two
object-detection datasets in COCO format.

## Conform two datasets and merge them

Two object-detection datasets name their classes differently: `street_2024` labels `car` and `person`, and `drone_2025`
labels `car`, `truck` and `pedestrian`. To merge them, relabel both onto one vocabulary first. Declare the datasets, an
ontology for the vocabulary ([Declare an ontology](declare_an_ontology.md) covers it), the evaluator that aligns a
dataset's classes to it, and a workflow that aligns, conforms and merges:

```yaml
datasets:
  - name: street_2024
    format: coco
    path: ./data/street_2024
  - name: drone_2025
    format: coco
    path: ./data/drone_2025

sources:
  - name: street_2024
    dataset: street_2024
  - name: drone_2025
    dataset: drone_2025

ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}
      - {id: Person, label: Person, synonyms: [person, pedestrian]}

evaluators:
  - name: align
    type: label-alignment
    ontology: vehicles

workflows:
  - name: combine
    inputs: [street, aerial]
    steps:
      - {name: align_street, evaluator: align, input: street}
      - {name: align_aerial, evaluator: align, input: aerial}
      - {name: street_conformed, transform: conform, input: street, alignment: align_street}
      - {name: aerial_conformed, transform: conform, input: aerial, alignment: align_aerial}
      - {name: merged, transform: merge, input: [street_conformed, aerial_conformed]}

tasks:
  - name: build
    workflow: combine
    sources: [street_2024, drone_2025]
```

Keep these points in mind:

- `inputs:` names the workflow's own slots. The task's `sources:` bind to them in order: `street_2024` to `street`, and
  `drone_2025` to `aerial`.
- Each step has a `name`, one kind key (`evaluator:` or `transform:`), and what it reads. `input:` takes an address: an
  input, or an earlier step by its name.
- The `align` entry is defined once and runs twice. Its settings live in the entry, and each step says only what it
  reads.
- `conform` reads a Dataset and the alignment computed on that Dataset, under `alignment:`. Naming an alignment of any
  other Dataset fails the config load.
- `merge` takes two or more Datasets that share a label vocabulary, which conforming both onto `vehicles` gives them.

## Run the workflow

```bash
dataeval-flow -c pipeline.yaml -o out/
```

The config is checked as it loads: every address must name an input or an earlier step, and every setting must be one
its transform takes. Then the steps run in order, and `aerial_conformed` fails. The console prints the report's short
form: how many steps ran, the summary, and a line per step with why it made nothing where it did not. Below its banner
and above its configuration:

```text
  Steps: 5 (3 ran, 1 failed, 1 skipped)

================================================================================
  SUMMARY
================================================================================
  No findings to report.

  Health: failed [!!] — step `aerial_conformed` failed

================================================================================
  STEPS
================================================================================
  Step              Status   Note
  ----------------  -------  ---------------------------------------------------
  align_street      ok

  align_aerial      ok

  street_conformed  ok

  aerial_conformed  failed   ValueError: The alignment is lossy, beyond `allow:
                             lossless`: car, truck collapse onto Vehicle. Set
                             `allow: lossy` to accept it, or settle classes with
                             `class_remap:`.

  merged            skipped  needs `aerial_conformed`, which failed: ValueError:
                             The alignment is lossy, beyond `allow: lossless`:
                             car, truck collapse onto Vehicle. Set `allow:
                             lossy` to accept it, or settle classes with
                             `class_remap:`.
```

The note is too long for its column, so the table wraps its cells and leaves a blank line between rows.
`out/results/result.txt`, and the console with `-v`, hold the full report, which gives each step a section headed by its
type's title and its name:

```text
================================================================================
  CONFORM · AERIAL_CONFORMED                                              failed
================================================================================
  Failed
    ValueError: The alignment is lossy, beyond `allow: lossless`: car, truck
    collapse onto Vehicle. Set `allow: lossy` to accept it, or settle classes
    with `class_remap:`.

================================================================================
  MERGE · MERGED                                                         skipped
================================================================================
  Skipped: needs `aerial_conformed`, which failed: ValueError: The alignment is
  lossy, beyond `allow: lossless`: car, truck collapse onto Vehicle. Set `allow:
  lossy` to accept it, or settle classes with `class_remap:`.
```

`drone_2025` names `car` and `truck`, and both align to `Vehicle`. Merging them loses a distinction, so `conform`
refuses until the config says it accepts that. `merged` reads the failed step, so it is skipped. The steps that do not
depend on it, such as `align_street` and `street_conformed`, still ran. The task has failed, so the run exits 1.

## Accept the collapse, clean, and export

Add a duplicates evaluator, and grow the workflow:

```yaml
evaluators:
  - name: dupes
    type: duplicates

workflows:
  - name: combine
    inputs: [street, aerial]
    steps:
      - {name: align_street, evaluator: align, input: street}
      - {name: align_aerial, evaluator: align, input: aerial}
      - {name: street_conformed, transform: conform, input: street, alignment: align_street}
      - name: aerial_conformed
        transform: conform
        input: aerial
        alignment: align_aerial
        allow: lossy                  # car and truck both become Vehicle
      - {name: merged, transform: merge, input: [street_conformed, aerial_conformed]}
      - {name: dupes, evaluator: dupes, input: merged}
      - name: clean
        transform: remove
        input: merged
        plans:
          dupes: {keep: first}
      - {name: dataset, transform: export, input: clean, format: coco}
```

`allow:` says how much loss a `conform` step accepts. Each level accepts everything the one before it does:

| `allow` | Accepts |
| --- | --- |
| `lossless` (default) | Each class maps to a concept of its own |
| `lossy` | Several classes may collapse onto one concept |
| `partial` | Several classes may collapse onto one concept, and classes that align to nothing are dropped; the report says what was dropped |

`class_remap:` on the step overrides the alignment for a class, mapping it to a concept by its id, or by a label no
other concept shares. The [`conform`](../reference/transforms.md#conform) entry covers what an override can settle and
the error for a class the Dataset does not have.

`plans:` names the evaluator steps whose findings `remove` applies, by address. Each must be a `duplicates` or
`outliers` step computed on the same Dataset as the `remove` step's `input`. Each holds the arguments of DataEval's plan
method: `deduplicate` for duplicates (`dup_types`, `keep`, `exclude_groups`, `levels`), and `prune` for outliers
(`metrics`, `min_flags`). Their names and values are checked when the config loads. `{}` takes DataEval's defaults,
which for duplicates removes every exact copy but the first. Several plans combine: whatever any of them names is
removed.

`export` writes `clean` under the run's output directory, at `out/datasets/<task>.<step>/`: here
`out/datasets/build.dataset/`. Handed a list, such as `kfold.train`, it writes each element under its key:
`out/datasets/build.dataset/0/` and so on. `to:` names another directory, and `mode:` says what to do when it already
holds a dataset (`error` by default, `replace` or `append`). A run without `-o` writes nothing, and the export step is
skipped with the reason "the run has no output directory". The [`export`](../reference/transforms.md#export) entry
covers the `provenance.json` beside the dataset and how its pixels are written, and
[Export a dataset](export_a_dataset.md) covers the formats and the modes.

## Check coverage and bias, then split

Add an extractor, two evaluators, and the steps that read the cleaned dataset:

```yaml
extractors:
  - name: bovw_ext
    model: bovw
    vocab_size: 512
    batch_size: 32

evaluators:
  - name: coverage
    type: coverage
  - name: balance
    type: balance

workflows:
  - name: combine
    inputs: [street, aerial]
    steps:
      - {name: align_street, evaluator: align, input: street}
      - {name: align_aerial, evaluator: align, input: aerial}
      - {name: street_conformed, transform: conform, input: street, alignment: align_street}
      - name: aerial_conformed
        transform: conform
        input: aerial
        alignment: align_aerial
        allow: lossy
      - {name: merged, transform: merge, input: [street_conformed, aerial_conformed]}
      - {name: dupes, evaluator: dupes, input: merged}
      - name: clean
        transform: remove
        input: merged
        plans:
          dupes: {keep: first}
      - {name: dataset, transform: export, input: clean, format: coco}
      - {name: crops, transform: wrap, input: clean, wrapper: DetectionCrops, params: {min_size: 32}}
      - {name: coverage, evaluator: coverage, input: crops}
      - {name: balance, evaluator: balance, input: clean}
      - {name: split, transform: split, input: clean, test_frac: 0.2, val_frac: 0.1}
      - {name: train_balance, evaluator: balance, input: split.train}

tasks:
  - name: build
    workflow: combine
    sources: [street_2024, drone_2025]
    extractor: bovw_ext
```

Keep these points in mind:

- `wrap` turns each detection into an item of its own, which is what makes coverage per detection and per class. On the
  detection dataset itself, `coverage` would measure whole images as one class, and warn that it has no class breakdown.
- `params:` passes DataEval's `DetectionCrops` arguments. `min_size: 32` drops boxes whose shorter side is under 32
  pixels. The [`scope`](../reference/presets.md#scope) preset passes its `wrap.params.min_size` to its own `wrap` step
  the same way. A tiny crop carries no SIFT features for BoVW to describe.
- Coverage embeds the crops with the task's extractor, which every step uses unless it names its own `extractor:`. BoVW
  needs no model file. [An ONNX model](../notebooks/onnx_embeddings.py) is the higher-fidelity choice once you have one.
- `balance` and `train_balance` run one evaluator entry on two Datasets. Each derives the metadata of the Dataset it
  reads, so `train_balance` measures the training split alone.
- `split` has three outputs, so a step names the one it reads: `split.train`, `split.val` or `split.test`. With
  `test_frac` alone, `val` is empty, and a step that reads `split.val` fails the config load.

To judge the parts of a split as an audit does, see [Check a set of splits](check_a_set_of_splits.md).

## Judge what the workflow found

The evaluators report, and nothing judges. Checks do: add a label-health evaluator, and three steps at the end of the
workflow:

```yaml
evaluators:
  - name: labels
    type: label-health

workflows:
  - name: combine
    inputs: [street, aerial]
    steps:
      - {name: align_street, evaluator: align, input: street}
      - {name: align_aerial, evaluator: align, input: aerial}
      - {name: street_conformed, transform: conform, input: street, alignment: align_street}
      - name: aerial_conformed
        transform: conform
        input: aerial
        alignment: align_aerial
        allow: lossy
      - {name: merged, transform: merge, input: [street_conformed, aerial_conformed]}
      - {name: dupes, evaluator: dupes, input: merged}
      - name: clean
        transform: remove
        input: merged
        plans:
          dupes: {keep: first}
      - {name: dataset, transform: export, input: clean, format: coco}
      - {name: crops, transform: wrap, input: clean, wrapper: DetectionCrops, params: {min_size: 32}}
      - {name: coverage, evaluator: coverage, input: crops}
      - {name: balance, evaluator: balance, input: clean}
      - {name: split, transform: split, input: clean, test_frac: 0.2, val_frac: 0.1}
      - {name: train_balance, evaluator: balance, input: split.train}
      - {name: labels, evaluator: labels, input: clean}
      - {name: merged_duplicates, check: image-duplicates, input: dupes}
      - {name: imbalance, check: class-imbalance, input: labels, warning: 3.0}
```

`merged_duplicates` judges the duplicates in the merged dataset, before `remove`. It warns where more than 0% of the
images are exact duplicates, or 5% near duplicates. `imbalance` warns where the cleaned dataset's largest class
outnumbers its smallest by more than 3 to 1. The task's health now says `warning` where either does, and
`--fail-on-warning` fails the run. The report gives each finding a section, with the step it judged below it: the Image
Duplicates finding holds `dupes`' duplicate groups, and the Class Imbalance finding holds `labels`' class counts. The
[Check Catalog](../reference/checks.md) lists every check and its thresholds. To group the findings under headings or
give the workflow a verdict, see [Judge a custom workflow like a preset](judge_a_workflow_like_a_preset.md).

## Run `quality` as a step

A workflow type that is a preset, such as `quality`, runs as a step with its whole chain: its evaluators, the checks
that judge them against its `checks`, and a `clean` step that removes each flagged image and box, and each duplicate but
the first. Name the entry with `workflow:`, and read the cleaned Dataset as `cleaning.clean`:

```yaml
workflows:
  - name: tidy
    type: quality
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: adaptive

  - name: street_clean
    inputs: [data]
    steps:
      - {name: cleaning, workflow: tidy, input: data}
      - {name: dataset, transform: export, input: cleaning.clean, format: coco}

tasks:
  - name: street
    workflow: street_clean
    sources: [street_2024]
```

`cleaning` runs quality's steps as `cleaning/outliers`, `cleaning/duplicates` and so on, to `cleaning/clean`. In the
report, each of its checks' findings has a section, with the steps it judged below it, headed such as
`From Outliers · cleaning/outliers`, and `cleaning/clean` has one of its own, `Remove · cleaning/clean`. Its checks'
findings count toward the task's health, as the findings of the checks in "Judge what the workflow found" do. Only
`clean` can be read from outside, and only as `cleaning.clean`: `cleaning` alone and `cleaning.duplicates` fail the
config load. [Workflow types as presets](../concepts/WorkflowsAsChains.md#workflow-types-as-presets) says more.

## Read the result

The output below comes from running the workflow as "Accept the collapse, clean, and export" leaves it, through
`dataset`, on two small synthetic datasets of 24 images each: `street_2024` names `car` and `person` and copies one
image, and `drone_2025` names `car`, `truck` and `pedestrian`. The steps that "Check coverage and bias, then split" and
"Judge what the workflow found" add were not part of that run, so nothing here reports coverage, balance or a split, and
no check judges a finding.

The console prints `Steps: 8 ran`, `No findings to report.` and a line per step, each `ok`: with no finding and no
failed step, there is no health to state. The full report, in `out/results/result.txt`, gives each step a section of its
own, since no finding shows any as evidence. The section of `clean` says what it kept and what each plan named, and the
section of `dataset` says where it wrote:

```text
================================================================================
  REMOVE · CLEAN
================================================================================
  Kept 47 of 48 images. Removed 1 image: 1 named by `dupes`.

================================================================================
  EXPORT · DATASET
================================================================================
  Path:   out/datasets/build.dataset
  Format: coco
  Images: 47
```

A Steps table follows the steps' sections, before the configuration. It gives each step's type and status, why it made
nothing where it did not, and each Dataset it read, walked back to its source: `dataset` reads
`` `clean` ← `merged` ← `street_conformed` ← `street` (street_2024) ``. The HTML report's table adds each step's title,
which the text report leaves to the step's section.

In `out/results/result.json`, the task's entry holds `steps`, each step by name with its kind, type, status and the
addresses it read. A Dataset a step made is written as its size and digest, never as data. `metadata.lineage` records
each Dataset in the chain. Trimmed to the `clean` step:

```json
{
  "kind": "workflow",
  "metadata": {"lineage": [{"name": "clean", "step": "clean", "type": "remove", "inputs": ["merged"],
                            "source": null, "digest": "ca2f57f5f4ad", "items": 47}]},
  "health": {"status": "ok", "warnings": 0, "findings": 0, "failed_steps": []},
  "steps": {
    "clean": {"kind": "transform", "type": "remove", "status": "ok", "inputs": ["merged", "dupes"],
              "output": {"items": 47, "digest": "ca2f57f5f4ad"},
              "details": {"removed": {"items": 1, "detections": 0, "tracks": 0, "frames": 0},
                          "by_plan": {"dupes": {"items": 1}}}}
  }
}
```

`details` counts what `remove` removed at each level, and `by_plan` what each plan named, at the levels where it named
something. Plans may name the same item, so their counts can add up to more than `removed`.

That run's health status is `ok`: it ran evaluators and transforms only, and evaluators judge nothing. A chain's health
rolls up its checks' findings and those of the workflow types it runs as steps, and is `failed` when a required step
fails.

From Python, a custom workflow's task returns a `ChainResult`. Its `steps` hold each step's live output, and hold every
step that ran even when a later one failed:

```python
from pathlib import Path

from dataeval_flow import load_config, run_tasks
from dataeval_flow.steps import ChainResult

config = load_config(Path("pipeline.yaml"))
result = run_tasks(config, tasks="build", data_dir=Path("."), output_dir=Path("out"))["build"]
assert isinstance(result, ChainResult)

clean = result.steps["clean"].output  # the cleaned dataset, a DataEval View
train = result.steps["split"].output["train"]  # a step with several outputs holds each by name
for record in result.metadata.lineage:
    print(record.name, record.items, record.digest)
```

`clean` is a DataEval `View` over the merged dataset, so you can go on to train on it or evaluate it from Python.
Without `output_dir`, the export step is skipped and nothing is written.

## List the steps you can chain

```bash
dataeval-flow steps
```

Each line names a step's kind and type, the ports it reads and makes, and what it does. `dataeval-flow steps remove`
prints one step's catalog entry, including its friendly title, the one its report headings use, and the schema of its
settings, and `--json` prints the whole catalog. When two kinds share a name, write it as `KIND:NAME`, as in
`transform:split`. From Python, `list_steps()` in `dataeval_flow.steps` returns the same catalog.

The catalogs list each step with its settings: the [Evaluator Catalog](../reference/evaluators.md), the
[Transform Catalog](../reference/transforms.md), the [Check Catalog](../reference/checks.md) and the
[Combine Catalog](../reference/combines.md). An evaluate step or a check also takes `by:`, which runs it once per class:
see [Run a step once per class](run_a_step_once_per_class.md).

## Next steps

- [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) — Learn how steps, addresses and lists, derived data,
  failures and lineage work.
- [Reuse a cleaning chain on new data](reuse_a_workflow.md) — Keep a chain in your config and run it on each new
  dataset.
- [Save a workflow from Python](save_a_workflow_from_python.md) — Build a chain in Python and save it as YAML.
