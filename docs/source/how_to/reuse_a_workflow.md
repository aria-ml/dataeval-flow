# Reuse a cleaning chain on new data

A custom workflow names no data. Its `inputs:` are slots, and a task binds them to sources. So a chain you wrote to
prepare one dataset, such as dropping its outliers and duplicates, runs unchanged on the next. This guide keeps such a
chain in your config, beside a task per dataset, and in a file of its own that each project loads.
[Save a workflow from Python](save_a_workflow_from_python.md) keeps it from Python. The chain below runs
[`outliers`](../reference/evaluators.md#outliers) and [`duplicates`](../reference/evaluators.md#duplicates), then the
[`remove`](../reference/transforms.md#remove) and [`export`](../reference/transforms.md#export) transforms.
[Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) explains inputs, addresses and lists, and
[Chain steps into a workflow of your own](write_a_custom_workflow.md) covers writing a chain. To judge what the chain
finds before it removes anything, use the built-in [`quality`](../reference/presets.md#quality) preset instead. You need
one dataset for each batch, in COCO format.

## Keep the chain in your config, with a task per dataset

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
under `datasets/clean_09.prepared/` and `datasets/clean_10.prepared/`. When the next batch arrives, add its dataset, its
source and one task line. The chain does not change.

## Keep the chain in a file of its own

Move the `evaluators:` and `workflows:` entries into a file of their own, and keep each project's `datasets:`,
`sources:` and `tasks:` in another, in one folder:

```text
config/
  10-my_clean.yaml   # evaluators: and workflows:, the chain
  20-batches.yaml    # datasets:, sources: and tasks: for this project
```

`dataeval-flow -c config/`, or `load_config("config/")` in Python, reads the folder as one pipeline, merging its files
in name order. To reuse the chain in another project, copy or link `10-my_clean.yaml` into that project's config folder.
Two files that define one name in the same section are refused when the folder loads, so give the chain's entries names
your projects will not use for anything else.

## Next steps

- [Save a workflow from Python](save_a_workflow_from_python.md) — Build the chain in Python, save it with its
  evaluators, and run it on new data.
- [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) — Learn how inputs, addresses and lists, derived data
  and failures work in a chain.
- [Export a dataset](export_a_dataset.md) — Choose the formats and modes of the export that ends the chain.
