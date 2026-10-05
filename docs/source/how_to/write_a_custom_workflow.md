# Chain steps into a workflow of your own

A workflow type such as `data-cleaning` runs one fixed analysis. When the analysis you need is a sequence, such as
relabelling two collections onto one vocabulary, merging them, dropping the duplicates and then checking what is left,
write it as a custom workflow: a `workflows:` entry whose `steps:` each read what an earlier step made. This guide
builds one in three stages. [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) explains the ideas behind
it.

## 1. Conform two datasets and merge them

Two object-detection collections name their classes differently: `street_2024` labels `car` and `person`, and
`drone_2025` labels `car`, `truck` and `pedestrian`. To merge them, relabel both onto one vocabulary first. Declare the
datasets, an ontology for the vocabulary, the evaluator that aligns a dataset's classes to it, and a workflow that
aligns, conforms and merges:

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

Notice:

- `inputs:` names the workflow's own slots. The task's `sources:` bind to them in order: `street_2024` to `street`,
  and `drone_2025` to `aerial`.
- Each step has a `name`, one kind key (`evaluator:` or `transform:`), and what it reads. `input:` takes an address:
  an input, or an earlier step by its name.
- The `align` entry is defined once and runs twice. Its settings live in the entry, and each step says only what it
  reads.
- `conform` reads a Dataset and the alignment computed on that Dataset, under `alignment:`. Naming an alignment of any
  other Dataset fails the config load.
- `merge` takes two or more Datasets that share a label vocabulary, which conforming both onto `vehicles` gives them.

## 2. Run it

```bash
dataeval-flow -c pipeline.yaml -o out/
```

The config is checked as it loads: every address must name an input or an earlier step, and every setting must be one
its transform takes. Then the steps run in order, and `aerial_conformed` fails. The console prints the report's short
form: how many steps ran, the summary, and a line per step with why it made nothing where it did not. Below its
banner and above its configuration:

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

  merged            skipped  needs `aerial_conformed`, which failed
```

The note is too long for its column, so the table wraps its cells and leaves a blank line between rows.
`out/results/result.txt`, and the console with `-v`, hold the full report, which gives each step a section headed by
its type's title and its name:

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
  Skipped: needs `aerial_conformed`, which failed
```

`drone_2025` names `car` and `truck`, and both align to `Vehicle`. Merging them loses a distinction, so `conform`
refuses until the config says it accepts that. `merged` reads the failed step, so it is skipped. The steps that do not
depend on it, such as `align_street` and `street_conformed`, still ran. The task has failed, so the run exits 1.

## 3. Accept the collapse, clean, and export

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
other concept shares. An override can settle a class that aligned to nothing, and so change how much loss the step needs
to accept. An override for a class the Dataset does not have fails the step, naming the classes it has, so a misspelled
class cannot pass unnoticed.

`plans:` names the evaluator steps whose findings `remove` applies, by address. Each must be a `duplicates` or
`outliers` step computed on the same Dataset as the `remove` step's `input`. Each holds the arguments of
DataEval's plan method: `deduplicate` for duplicates (`dup_types`, `keep`, `exclude_groups`, `levels`), and `prune` for
outliers (`metrics`, `min_flags`). Their names and values are checked when the config loads. `{}` takes DataEval's
defaults, which for duplicates removes every exact copy but the first. Several plans combine: whatever any of them
names is removed.

`export` writes `clean` under the run's output directory, at `out/datasets/<task>.<step>/`: here
`out/datasets/build.dataset/`. Handed a list, such as `kfold.train`, it writes each element under its key:
`out/datasets/build.dataset/0/` and so on. `to:` names another directory, and `mode:` says what to do when it already
holds a dataset (`error` by default, `replace` or `append`). Beside the dataset, `provenance.json` records the sources it
came from, with the dataset and view each read, and the lineage of what was written. Its `label_space` list holds each
source's Relabel, then each conform on the way, with its remap and ontology digest. A Dataset a chain made is written
with its pixels encoded, not by reference to the image files, even where no step changed them. A run without `-o`
writes nothing, and the export step is skipped with the reason "the run has no output directory".

## 4. Check coverage and bias, and split

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

Notice:

- `wrap` turns each detection into an item of its own, which is what makes coverage per detection and per class. On
  the detection dataset itself, `coverage` would measure whole images as one class, and warn that it has no class
  breakdown.
- `params:` passes DataEval's `DetectionCrops` arguments. `min_size: 32` drops boxes whose shorter side is under 32
  pixels; `data-coverage` passes its `crops.min_size` to its own `wrap` step the same way. A tiny crop carries no SIFT
  features for BoVW to describe.
- Coverage embeds the crops with the task's extractor, which every step uses unless it names its own `extractor:`.
  BoVW needs no model file; [an ONNX model](../notebooks/onnx_embeddings.py) is the higher-fidelity choice once you
  have one.
- `balance` and `train_balance` run one evaluator entry on two Datasets. Each derives the metadata of the Dataset it
  reads, so `train_balance` measures the training split alone.
- `split` has three outputs, so a step names the one it reads: `split.train`, `split.val` or `split.test`. With
  `test_frac` alone, `val` is empty, and a step that reads `split.val` fails the config load.

## 5. Judge what it found

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
`--fail-on-warning` fails the run. The report gives each finding a section, with the step it judged below it: the
Duplicates finding holds `dupes`' duplicate groups, and the Class Imbalance finding holds `labels`' class counts.
The [Check and Combine Catalog](../reference/checks.md) lists every check and its thresholds.

## 6. Run data-cleaning as a step

A workflow type that is a preset, such as `data-cleaning`, runs as a step with its whole chain: its evaluators, the
checks that judge them against its `checks`, and a `clean` step that removes each flagged image and box,
and each duplicate but the first. Name the entry with `workflow:`, and read the cleaned Dataset as `cleaning.clean`:

```yaml
workflows:
  - name: tidy
    type: data-cleaning
    outlier_method: adaptive
    outlier_flags: [dimension, pixel, visual]

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

`cleaning` runs data-cleaning's steps as `cleaning/outliers`, `cleaning/dupes` and so on, to `cleaning/clean`. In the
report, each of its checks' findings has a section, with the steps it judged below it, headed such as
`From Outliers · cleaning/outliers`, and `cleaning/clean` has one of its own, `Remove · cleaning/clean`. Its checks'
findings count toward the task's health, as section 5's do.
Only `clean` can be read from outside, and only as `cleaning.clean`: `cleaning` alone and `cleaning.dupes` fail the
config load. [Workflow types as presets](../concepts/WorkflowsAsChains.md#workflow-types-as-presets) says more.

## 7. Read the result

The output below comes from running the workflow as section 3 leaves it, through `dataset`, on two small synthetic
datasets of 24 images each: `street_2024` names `car` and `person` and copies one image, and `drone_2025` names `car`,
`truck` and `pedestrian`. The steps sections 4 and 5 add were not part of that run, so nothing here reports coverage,
balance or a split, and no check judges a finding.

The console prints `Steps: 8 ran`, `No findings to report.` and a line per step, each `ok`: with no finding and no
failed step, there is no health to state. The full report, in `out/results/result.txt`, gives each step a section of
its own, since no finding shows any as evidence. `clean`'s says what it kept and what each plan named, and
`dataset`'s where it wrote:

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

A Steps table follows the steps' sections, before the configuration. It gives each step's type and status, why it
made nothing where it did not, and each Dataset it read, walked back to its source: `dataset` reads
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
rolls up its checks' findings and those of the workflow types it runs as steps, and is `failed` when a required step fails.

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

## 8. See which steps you can chain

```bash
dataeval-flow steps
```

Each line names a step's kind and type, the ports it reads and makes, and what it does. `dataeval-flow steps remove`
prints one step's catalog entry, including its friendly title, the one its report headings use, and the schema of its
settings, and `--json` prints the whole catalog. When two
kinds share a name, write it as `KIND:NAME`, as in `transform:split`. From Python, `list_steps()` in
`dataeval_flow.steps` returns the same catalog.

## 9. Save a workflow from Python

`CustomWorkflowConfig` builds a workflow in Python, and `save` writes it into a config file:

```python
from dataeval_flow.steps import CustomWorkflowConfig, StepEntry

workflow = CustomWorkflowConfig(
    name="clean_and_export",
    inputs=["data"],
    steps=[
        StepEntry(name="dupes", evaluator="dupes", input="data"),
        StepEntry(name="clean", transform="remove", input="data", plans={"dupes": {"keep": "first"}}),
        StepEntry(name="dataset", transform="export", input="clean", format="coco"),
    ],
)
workflow.save("config/workflows.yaml")  # the config/ directory must already exist
```

`save` adds the workflow to the file, or replaces the entry of the same name, and keeps the file's other keys and
workflows. It creates a file that does not exist, but not a missing directory, so save to a path whose directory
exists. It refuses a file that is not a pipeline config. PyYAML keeps no comments, so a file `save` rewrites loses
them. `workflow.to_yaml()` returns the same entry as text:

```yaml
workflows:
- name: clean_and_export
  inputs:
  - data
  steps:
  - name: dupes
    evaluator: dupes
    input: data
  - name: clean
    transform: remove
    input: data
    plans:
      dupes:
        keep: first
  - name: dataset
    transform: export
    input: clean
    format: coco
```

Building the workflow checks its slots, its step names, and each transform's settings. Whether `dupes` names a real
`evaluators:` entry, and whether the steps connect, is checked when the file loads into a pipeline. `load_config`
reads a folder as one pipeline, so `dataeval-flow -c config/` runs a `workflows.yaml` beside your other config files.
Loading a config and saving it, from the TUI or the config builder, keeps every custom workflow as written.

`save` and `to_yaml` also take `definitions`, the entries the steps name, as `run` does:
`workflow.save("config/workflows.yaml", definitions=[DuplicatesConfig(name="dupes")])` writes the `dupes` entry
beside the workflow, so the file loads and runs on its own. [Reuse a cleaning chain on new data](reuse_a_workflow.md)
keeps a chain that way and runs it on each new dataset.

## 10. Run a step once per class

An evaluate step or a check with `by:` runs once per class, or once per group of classes, and keeps every run in one
Output. It suits an evaluator that compares two sources, such as a drift detector, when you want to know which
classes differ:

```yaml
evaluators:
  - {name: mmd, type: drift-mmd}

workflows:
  - name: class_drift
    inputs: [reference, {name: tests, list: true}]
    steps:
      - {name: mmd-classes, evaluator: mmd, input: [reference, tests], by: class}
      - {name: mmd-classes-check, check: drift, input: mmd-classes, by: class}
      - name: mmd-groups
        evaluator: mmd
        input: [reference, tests]
        by:
          class:
            groups: {vehicles: [car, truck, van], people: [person]}
            min_items: 2
      - {name: mmd-groups-check, check: drift, input: mmd-groups, by: class}
```

Notice:

- `by: class` keys each class by its name, taken from the first input's `index2label`, and runs the step on that
  class's items alone, in ascending class index.
- `groups:` keys each named group instead, in the order written. A group lists classes by name or by index, and a name
  that resolves to nothing fails the step. Groups may overlap, and a class in no group is left out and listed as
  skipped.
- With several inputs, a key runs only where every input holds at least `min_items` of it, 2 unless you set it.
  Otherwise the key is skipped, and the Output says why.
- A key whose run raises, such as a class too small for the evaluator, is skipped with its error, and the other keys
  still run. The step fails only when every key raises.
- The step's output is a `PerClassOutput`: `outputs` holds each key's own Output, and `skipped` holds each key left
  out with its reason. Only a check with `by:` can read it.
- A check's `by: class` takes no settings, since its keys come from the step it reads. It runs once for each key and
  rolls the findings into one, titled with the per-key title and `by class` or `by group`, which names the keys that
  warned and each key it did not assess.
- `by:` slices embeddings and labels, so it needs a Dataset with one label per item, as image classification has.
  The load refuses an evaluator that reads statistics, metadata or clusters. For detection data, run `wrap` with
  `DetectionCrops` first.

[Monitor drift with steps](monitor_drift.md) uses `by:` with drift detectors, and compares one group against another.

## See also

- [Monitor drift with steps](monitor_drift.md) — merge test sources, compare classes or groups, and drift on crops
- [Reuse a cleaning chain on new data](reuse_a_workflow.md) — keep a chain and run it on each new dataset
- [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) — steps, addresses and lists, derived data,
  failures and lineage
- [Declare an ontology](declare_an_ontology.md) — the vocabulary `label-alignment` aligns to
- [Export a dataset](export_a_dataset.md) — formats, modes, and what an export records and drops
- [Evaluator Catalog](../reference/evaluators.md) — every evaluator a step can run
- [Transform Catalog](../reference/transforms.md) — every transform a step can run, with its settings
- [Check and Combine Catalog](../reference/checks.md) — every check and combine a step can run, with its thresholds
