# Reuse a cleaning chain on new data

A custom workflow names no data. Its `inputs:` are slots, and a task binds them to sources. So a chain you wrote to
prepare one dataset, such as dropping its outliers and duplicates, runs unchanged on the next. This guide keeps such a
chain in three ways: in your config beside a task per dataset, in a file of its own that each project loads, and
saved from Python. [Chain steps into a workflow of your own](write_a_custom_workflow.md) covers writing one.

## 1. Keep the chain in your config, with a task per dataset

The chain below drops each image with an outlier flag and each duplicate but the first of its group, then exports what
is left. The evaluator entries its steps name sit beside it:

```yaml
datasets:
  - {name: batch_09, format: coco, path: ./data/batch_09}
  - {name: batch_10, format: coco, path: ./data/batch_10}

sources:
  - {name: batch_09, dataset: batch_09}
  - {name: batch_10, dataset: batch_10}

evaluators:
  - {name: outliers, type: outliers, flags: [pixel, visual], outlier_threshold: zscore}
  - {name: dupes, type: duplicates}

workflows:
  - name: my_clean
    inputs: [data]
    steps:
      - {name: outliers, evaluator: outliers, input: data}
      - {name: dupes, evaluator: dupes, input: data}
      - name: clean
        transform: remove
        input: data
        plans:
          dupes: {dup_types: [exact, near], keep: first}
          outliers: {min_flags: 1}
      - {name: prepared, transform: export, input: clean, format: coco}

tasks:
  - {name: clean_09, workflow: my_clean, sources: [batch_09]}
  - {name: clean_10, workflow: my_clean, sources: [batch_10]}
```

Each task runs the whole chain on its own source. Run with an output directory, and each batch's cleaned copy lands
under `datasets/clean_09.prepared/` and `datasets/clean_10.prepared/`. When the next batch arrives, add its dataset,
its source and one task line. The chain does not change.

## 2. Keep the chain in a file of its own

Move the `evaluators:` and `workflows:` entries into a file of their own, and keep each project's `datasets:`,
`sources:` and `tasks:` in another, in one folder:

```text
config/
  10-my_clean.yaml   # evaluators: and workflows:, the chain
  20-batches.yaml    # datasets:, sources: and tasks: for this project
```

`dataeval-flow -c config/`, or `load_config("config/")` in Python, reads the folder as one pipeline, merging its files
in name order. To reuse the chain in another project, copy or link `10-my_clean.yaml` into that project's config
folder. Two files that define one name in the same section are refused when the folder loads, so give the chain's
entries names your projects won't use for anything else.

## 3. Save the chain from Python, and run it on new data

Build the chain in Python and try it on the data at hand. `run` takes the entries its steps name as `definitions`:

```python
from dataeval_flow import run
from dataeval_flow.evaluators.quality import DuplicatesConfig, OutliersConfig
from dataeval_flow.steps import CustomWorkflowConfig, StepEntry

outliers = OutliersConfig(name="outliers", flags=["pixel", "visual"], outlier_threshold="zscore")
dupes = DuplicatesConfig(name="dupes")
my_clean = CustomWorkflowConfig(
    name="my_clean",
    inputs=["data"],
    steps=[
        StepEntry(name="outliers", evaluator="outliers", input="data"),
        StepEntry(name="dupes", evaluator="dupes", input="data"),
        StepEntry(
            name="clean",
            transform="remove",
            input="data",
            plans={"dupes": {"dup_types": ["exact", "near"], "keep": "first"}, "outliers": {"min_flags": 1}},
        ),
    ],
)
result = run(my_clean, september, definitions=[outliers, dupes])  # september: any dataset in memory
```

When it does what you want, save it with the same `definitions`. The file then holds the whole block, so it loads and
runs without the code that built it:

```python
my_clean.save("config/my_clean.yaml", definitions=[outliers, dupes])  # the config/ directory must already exist
```

It writes each entry with its name, its type and the settings that differ from their defaults, as you would write it
by hand. `my_clean.to_yaml(definitions=[outliers, dupes])` returns the same block as text:

```yaml
evaluators:
- name: outliers
  type: outliers
  flags:
  - pixel
  - visual
  outlier_threshold: zscore
- name: dupes
  type: duplicates
workflows:
- name: my_clean
  inputs:
  - data
  steps:
  - name: outliers
    evaluator: outliers
    input: data
  - name: dupes
    evaluator: dupes
    input: data
  - name: clean
    transform: remove
    input: data
    plans:
      dupes:
        dup_types:
        - exact
        - near
        keep: first
      outliers:
        min_flags: 1
```

Later, on the next batch, load the block and run it:

```python
from dataeval_flow import load_config, run

recipe = load_config("config/my_clean.yaml")
(my_clean,) = recipe.workflows
result = run(my_clean, october, definitions=[*recipe.evaluators])  # october: the next dataset
cleaned = result.steps["clean"].output  # the October batch without what the chain removed
```

`save` replaces an entry of the same name in each section, and keeps the file's other keys and entries. It does not
check that `definitions` holds every entry the steps name, since a chain saved into one file of a config folder may
rely on entries in another. Loading the file names any entry that is missing.

## See also

- [Chain steps into a workflow of your own](write_a_custom_workflow.md) — writing a chain, running it, and reading its
  result
- [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) — inputs, addresses and lists, derived data and
  failures
- [Export a dataset](export_a_dataset.md) — formats, modes, and what an export records
- [Preset Catalog](../reference/presets.md#quality) — `quality`, the built-in preset that judges what
  it finds before removing it
