# Gate a matrix run and read its files

A matrix task changes how a run exits and what it writes. This guide sets the exit code with `fail_on`, exports a
dataset for each run with the [`export`](../reference/transforms.md#export) step, and reads the matrix in `result.json`
and the other result files. [Export a dataset](export_a_dataset.md) covers the formats and modes of an export. You need
a task with a `matrix:`, as in [Sweep settings with a matrix](run_a_matrix.md), and the
[`quality`](../reference/presets.md#quality) entry that the examples vary:

```yaml
workflows:
  - name: cleaning
    type: quality
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: adaptive
```

## Gate the run

By default a failed run fails the command (exit 1), and warnings do not. `result: fail_on: warning`, or
`--fail-on-warning`, exits 3 when any run warns. For an exploratory matrix, where most runs are meant to warn,
`fail_on: never` stops a failed or warning run from failing the command, though a verdict short of `result: require:`
still exits 4:

```yaml
result:
  fail_on: never
```

## Export a dataset for each run

An `export` step inside a run writes under `datasets/<to>/run-<n>/`, where `n` is the run's number in the table, and
under `datasets/<to>/run-<n>/<key>/` where it writes a list, once per element. Numbers name the directories, since a
source list or a float does not always make a safe directory name. The run's values are in the result. Its
`provenance.json` names it `<to>/run-<n>`. Here each run writes its cleaned dataset, to
`out/datasets/clean_export.dataset/run-1/` and `run-2/`:

```yaml
workflows:
  - name: tidy
    inputs: [data]
    steps:
      - {name: cleaning, workflow: cleaning, input: data}
      - {name: dataset, transform: export, input: cleaning.clean, format: coco}

tasks:
  - name: clean_export
    workflow: tidy
    sources: train
    matrix:
      workflows.cleaning.outliers.outlier_threshold: [[adaptive, 3.5], [adaptive, 4.5]]
```

The runs cannot clash with each other, but another task writing to the same `to` still fails the load. A matrix may not
vary `to` itself: `steps.dataset.to` fails the load, since each run already writes under its own `run-<n>/`.

## Read the matrix in result.json

In `result.json`, the task's entry holds the matrix. Each run's `result` is its type's JSON, its `assets` included: each
run keeps the thumbnails of the items its own report names, and the matrix has none of its own. A chain names an item by
its Dataset's address, such as `data` for a preset's input, whichever source the task binds there. Where the runs read
other sources, or other items through a varied view, one address holds other images in each run. `errors` is there only
where a run failed. From the run in [Sweep settings with a matrix](run_a_matrix.md), trimmed:

```json
{
  "clean": {
    "kind": "matrix",
    "type": "quality",
    "keys": ["outliers.outlier_threshold"],
    "metadata": {"source_descriptions": ["train (ds[sample])"], "model_id": null, "metadata_binning": null,
                 "resolved_config": {"task": {"name": "clean", "workflow": "cleaning", "sources": "train",
                                              "matrix": {"outliers.outlier_threshold": [["adaptive", 2.5], ["adaptive", 3.5], ["adaptive", 4.5]]}},
                                     "sources": ["train"], "seed": 42}},
    "health": {"status": "warning", "warnings": 9, "failed_runs": []},
    "runs": [
      {"number": 1, "label": "outliers.outlier_threshold=[adaptive, 2.5]",
       "values": {"outliers.outlier_threshold": ["adaptive", 2.5]},
       "result": {"kind": "workflow", "metadata": {}, "health": {}, "steps": {}, "findings": [], "assets": []}}
    ]
  }
}
```

The table is not stored: it reads off the runs. The matrix's `metadata` names `model_id` only where every run used one
extractor, and leaves the encoding fields `null`, since each run records its own.

## Find the other files

- A JUnit report has a test suite per run, named `clean · run 3`, its findings as cases.
- A Markdown report gives the task's health, the table and the failed runs' errors.
- With `per_task: true`, a matrix task's files hold its matrix result, as any task's hold its result.
- `encoding.json`, and `dataeval-flow encoding result.json`, describe a matrix task by its runs' descriptor where they
  agree. Where its runs were encoded differently, the command refuses the task, naming the runs that differ, and the run
  writes no `encoding.json`, as when two tasks disagree.

## Next steps

- [Read evaluation outputs](read_evaluation_outputs.md) — Read the report, its severities and the JSON that a lone task
  writes.
- [Export a dataset](export_a_dataset.md) — Choose the format and mode of an export and read what it records.
