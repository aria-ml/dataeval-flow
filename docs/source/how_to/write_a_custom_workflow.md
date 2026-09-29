# Chain steps into a workflow of your own

A workflow type such as `data-cleaning` runs one fixed analysis. When the analysis you need is a sequence, such as
relabelling two collections onto one vocabulary, merging them, dropping the duplicates and then checking what is left,
write it as a custom workflow: a `workflows:` entry whose `steps:` each read what an earlier step made. This guide
builds one in three stages. [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) explains the ideas behind
it.

## 1. Conform two corpora and merge them

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
    type: scope.label-alignment
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
its transform takes. Then the steps run in order, and `aerial_conformed` fails. The report gives each step a section:

```text
================================================================================
  AERIAL_CONFORMED (CONFORM)                                              failed
================================================================================
  On `aerial` (drone_2025), `align_aerial`

  Failed
    ValueError: The alignment is lossy, beyond `allow: lossless`: car, truck
    collapse onto Vehicle. Set `allow: lossy` to accept it, or settle classes
    with `class_remap:`.

================================================================================
  MERGED (MERGE)                                                         skipped
================================================================================
  On `street_conformed` ← `street` (street_2024), `aerial_conformed`

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
    type: quality.duplicates

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
      - {name: corpus, transform: export, input: clean, format: coco}
```

`allow:` says how much loss a `conform` step accepts. Each level accepts everything the one before it does:

| `allow` | Accepts |
| --- | --- |
| `lossless` (default) | Each class maps to a concept of its own |
| `lossy` | Several classes may collapse onto one concept |
| `partial` | Several classes may collapse onto one concept, and classes that align to nothing are dropped; the report says what was dropped |

`class_remap:` on the step overrides the alignment for a class, mapping it to a concept by its id, or by a label no
other concept shares. An override can settle a class that aligned to nothing, and so change how much loss the step
needs to accept.

`plans:` names the evaluator steps whose findings `remove` applies, by address. Each must be a `quality.duplicates` or
`quality.outliers` step computed on the same Dataset as the `remove` step's `input`. Each holds the arguments of
DataEval's plan method: `deduplicate` for duplicates (`dup_types`, `keep`, `exclude_groups`, `levels`), and `prune` for
outliers (`metrics`, `min_flags`). Their names and values are checked when the config loads. `{}` takes DataEval's
defaults, which for duplicates removes every exact copy but the first. Several plans combine: whatever any of them
names is removed.

`export` writes `clean` under the run's output directory, at `out/datasets/<task>.<step>/`: here
`out/datasets/build.corpus/`. It writes one Dataset: a list, such as `kfold.train`, fails the config load, so export one
element of it, such as `kfold.train[0]`. `to:` names another directory, and `mode:` says what to do when it already
holds a dataset (`error` by default, `replace` or `append`). Beside the corpus, `provenance.json` records the lineage of
what was written and each conform on the way, with its remap. A Dataset a chain made is written with its pixels encoded,
not by reference to the image files, even where no step changed them. A run without `-o` writes nothing, and the export
step is skipped with the reason "the run has no output directory".

## 4. Check coverage and bias, and split

Add an extractor, two evaluators, and the steps that read the cleaned corpus:

```yaml
extractors:
  - name: bovw_ext
    model: bovw
    vocab_size: 512
    batch_size: 32

evaluators:
  - name: coverage
    type: scope.coverage
  - name: balance
    type: bias.balance

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
      - {name: corpus, transform: export, input: clean, format: coco}
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
  the detection corpus itself, `scope.coverage` would measure whole images as one class, and warn that it has no class
  breakdown.
- `params:` passes DataEval's `DetectionCrops` arguments. `min_size: 32` drops boxes whose shorter side is under 32
  pixels, as `data-coverage`'s `crop_min_size` does: a tiny crop carries no SIFT features for BoVW to describe.
- Coverage embeds the crops with the task's extractor, which every step uses unless it names its own `extractor:`.
  BoVW needs no model file; [an ONNX model](../notebooks/onnx_embeddings.py) is the higher-fidelity choice once you
  have one.
- `balance` and `train_balance` run one evaluator entry on two Datasets. Each derives the metadata of the Dataset it
  reads, so `train_balance` measures the training split alone.
- `split` has three outputs, so a step names the one it reads: `split.train`, `split.val` or `split.test`. With
  `test_frac` alone, `val` is empty, and a step that reads `split.val` fails the config load.

## 5. Read the result

The text report opens with a summary of the steps, then gives each step a section of its own, headed by the Dataset it
read walked back to its source:

```text
================================================================================
  TRAIN_BALANCE (BIAS.BALANCE)
================================================================================
  On `split.train` ← `clean` ← `merged` ← `street_conformed` ← `street`
  (street_2024)
```

In `out/results/result.json`, the task's entry holds `steps`, each step by name with its kind, type, status and the
addresses it read. A Dataset a step made is written as its size and digest, never as data. `metadata.lineage` records
each Dataset in the chain. Trimmed to the `clean` step, from a small run:

```json
{
  "kind": "workflow",
  "metadata": {"lineage": [{"name": "clean", "step": "clean", "type": "remove", "inputs": ["merged"],
                            "source": null, "digest": "1bf4dcfcadf4", "items": 47}]},
  "health": {"status": "ok", "warnings": 0, "findings": 0, "failed_steps": []},
  "steps": {
    "clean": {"kind": "transform", "type": "remove", "status": "ok", "inputs": ["merged", "dupes"],
              "output": {"items": 47, "digest": "1bf4dcfcadf4"},
              "details": {"removed": {"items": 1, "detections": 0, "tracks": 0, "frames": 0}}}
  }
}
```

The health status is `ok`. Evaluators judge nothing, so a chain's health reports only the findings of the workflow
types it runs as steps, and is `failed` when a required step fails.

From Python, a custom workflow's task returns a `ChainResult`. Its `steps` hold each step's live output, and hold every
step that ran even when a later one failed:

```python
from pathlib import Path

from dataeval_flow import load_config, run_tasks
from dataeval_flow.steps import ChainResult

config = load_config(Path("pipeline.yaml"))
result = run_tasks(config, tasks="build", data_dir=Path("."), output_dir=Path("out"))["build"]
assert isinstance(result, ChainResult)

clean = result.steps["clean"].output  # the cleaned corpus, a DataEval View
train = result.steps["split"].output["train"]  # a step with several outputs holds each by name
for record in result.metadata.lineage:
    print(record.name, record.items, record.digest)
```

`clean` is a DataEval `View` over the merged corpus, so you can go on to train on it or evaluate it from Python.
Without `output_dir`, the export step is skipped and nothing is written.

## 6. See which steps you can chain

```bash
dataeval-flow steps
```

Each line names a step's kind and type, the ports it reads and makes, and what it does. `dataeval-flow steps remove`
prints one step's catalog entry, including the schema of its settings, and `--json` prints the whole catalog. When two
kinds share a name, write it as `KIND:NAME`, as in `transform:split`. From Python, `list_steps()` in
`dataeval_flow.steps` returns the same catalog.

## 7. Save a workflow from Python

`CustomWorkflowConfig` builds a workflow in Python, and `save` writes it into a config file:

```python
from dataeval_flow.steps import CustomWorkflowConfig, StepEntry

workflow = CustomWorkflowConfig(
    name="clean_and_export",
    inputs=["data"],
    steps=[
        StepEntry(name="dupes", evaluator="dupes", input="data"),
        StepEntry(name="clean", transform="remove", input="data", plans={"dupes": {"keep": "first"}}),
        StepEntry(name="corpus", transform="export", input="clean", format="coco"),
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
  - name: corpus
    transform: export
    input: clean
    format: coco
```

Building the workflow checks its slots, its step names, and each transform's settings. Whether `dupes` names a real
`evaluators:` entry, and whether the steps connect, is checked when the file loads into a pipeline. `load_config`
reads a folder as one pipeline, so `dataeval-flow -c config/` runs a `workflows.yaml` beside your other config files.
Loading a config and saving it, from the TUI or the config builder, keeps every custom workflow as written.

## See also

- [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) — steps, addresses and lists, derived data,
  failures and lineage
- [Declare an ontology](declare_an_ontology.md) — the vocabulary `scope.label-alignment` aligns to
- [Export a dataset](export_a_dataset.md) — formats, modes, and what an export records and drops
- [Evaluator Catalog](../reference/evaluators.md) — every evaluator a step can run
- [Transform Catalog](../reference/transforms.md) — every transform a step can run, with its settings
