# Read the result envelope

Every run returns a {term}`result envelope <Result Envelope>`: the findings plus the provenance that makes them
auditable. To read it from a program, export it as JSON or YAML and read the keys below.
[Provenance](../concepts/Provenance.md) explains why a result carries its lineage and what the envelope records. The
examples assume a finished run, `result = run_task(config, task)`.
[Read a finding's report blocks](read_report_blocks.md) describes the blocks of each finding.

## Export the result

`export()` writes the structured result, findings plus provenance, to disk:

```python
result.export("./results")  # writes ./results/results.json
result.export("./run.yaml", fmt="yaml")  # explicit file, YAML
payload = result.export()  # no path: returns the serialized string
data = result.to_dict()  # no serialization: a plain dict
```

A directory (or an extension-less path) gets `results.<ext>` written inside it. Anything with a suffix is treated as a
file, and parent directories are created as needed.

From the CLI, `--output` sets the directory the envelope and reports are written to:

```bash
dataeval-flow --config params.yaml --data . --output ./results
```

To reach the numbers behind a finding without serializing the result, see
[Get the numbers behind a finding](get_the_numbers_behind_a_finding.md).

## Read the envelope shape

The serialized envelope has five top-level keys, one of them `kind`, which is `"workflow"` here, and a sixth, `assets`,
where its report pictures any items:

```json
{
  "kind":     "workflow",
  "metadata": { "timestamp": "...", "tool": "dataeval-flow", "resolved_config": {} },
  "health":   { "status": "ok", "warnings": 0, "findings": 7 },
  "raw":      { },
  "report":   { "summary": "...", "findings": [] }
}
```

`metadata` is the provenance envelope, `health` the roll-up of the findings' severities, `raw` the typed numeric
outputs, and `report` the same findings the text report renders: a summary string plus a list of findings, each with a
`title`, `severity`, `brief`, `description`, and `blocks`, its evidence as typed report blocks, one object per block
with its `type`. `assets` holds the thumbnails, as the last section of
[Read a finding's report blocks](read_report_blocks.md) describes. `kind` distinguishes this from an evaluator's
envelope, covered in the "Read an evaluator's result" section.

A chain's result has `steps` and a top-level `findings` in place of `raw` and `report`. A custom workflow's result is
one, and so is a `quality` result: `quality` is a {term}`preset <Preset>`, whose settings expand to a chain of steps.

```json
{
  "kind":     "workflow",
  "metadata": {
    "timestamp": "...",
    "workflow":  "clean",
    "lineage":   [ { "name": "data", "source": "ref-src", "items": 261 },
                   { "name": "clean", "step": "clean", "items": 162 } ]
  },
  "health":   { "status": "warning", "warnings": 4, "findings": 4, "failed_steps": [] },
  "steps":    { "outliers": { "kind": "evaluator", "type": "outliers", "status": "ok" } },
  "findings": [ { "step": "image-outliers", "title": "Image Outliers", "severity": "warning" } ]
}
```

That is `quality`'s result on MILCO's reference campaigns, from
{doc}`View a report as HTML <../notebooks/view_html_reports>`, trimmed. `steps` holds each step by name, in run order,
with its kind, type, status, the addresses it read and what it made. `findings` lists the check steps' findings, each
shaped as [Read a finding's report blocks](read_report_blocks.md) describes and naming its step under `step`. `health`
also lists the steps that failed. {doc}`write_a_custom_workflow` shows more of a chain's JSON.

`health.status` is `"warning"` where any finding breached its threshold and `"ok"` otherwise, unless the run failed. A
failed workflow's status is `"failed"`, and so is the status of a chain whose required step failed. Warnings answer a
different question from whether the workflow ran: a task that failed produces errors, not warnings.

## Read the provenance fields

The `metadata` block is what makes a finding auditable and interoperable with other JATIC tools:

| Field | What it records |
| --- | --- |
| `version` | Envelope schema version |
| `timestamp` | UTC time the workflow ran |
| `execution_time_s` | Wall-clock duration |
| `tool` / `tool_version` | `dataeval-flow` and the exact version that produced the result |
| `library_versions` | The installed version of DataEval, PyTorch, NumPy, Pillow, datamaite, OpenCV (where installed) and the extractor's runtime |
| `dataset_id` | Identifier(s) of the evaluated dataset(s) |
| `source_descriptions` | Human-readable description of each resolved source |
| `selection_id` | Identifier for the {term}`view <View>` applied to the dataset |
| `label_source` | Where labels came from |
| `model_id` / `preprocessor_id` | The extractor model and preprocessing pipeline used |
| `resolved_config` | The fully resolved configuration, after merge and defaults |
| `metadata_binning` | How each factor was typed and discretized. See {doc}`get_the_numbers_behind_a_finding`. |
| `diagnostics` | Library warnings raised during the run |

`resolved_config` is the field that makes a run repeatable. It is the configuration as executed. Keep the
envelope and you can reproduce the run without the original config file.

Workflows extend this envelope with their own fields, so `metadata` carries more than the table above. A `quality`
result, for example, is a chain's, and records `workflow`, the entry's name, and `lineage`, each Dataset in the chain
with what made it. It records no metadata encoding, so its `metadata_binning` is `null`. Treat the table as the
guaranteed floor, not the full set: each result class in the
{doc}`API Reference <../reference/autoapi/dataeval_flow/index>`, such as {py:class}`~dataeval_flow.steps.ChainResult`,
lists the `metadata` fields it adds under "Fields".

## Read an evaluator's result

An evaluator's `result.json` entry has a different shape: `"kind": "evaluator"`, and no findings, severities, or
`health` at all. DataEval's own output sits under `output`, as a table, mapping, or array, whatever shape that evaluator
produces:

```json
{
  "kind": "evaluator",
  "metadata": { "evaluator": "duplicates", "dataeval": { "version": "1.1.1" } },
  "output": { "shape": "table", "columns": ["group_id", "..."], "rows": [] }
}
```

From Python, an evaluator's `result.output` is DataEval's own output object rather than a JSON-ready dict. See
{doc}`run_a_single_evaluator` and the [Evaluator Catalog](../reference/evaluators.md) for what each evaluator returns.

## Read a matrix result

A task with a `matrix:` runs its entry once per combination of values, and its `result.json` entry holds every run's
result, with a `kind` of `"matrix"`:

```json
{
  "kind":     "matrix",
  "type":     "quality",
  "keys":     ["outliers.outlier_threshold"],
  "metadata": { "timestamp": "...", "resolved_config": { "task": {}, "sources": ["train"], "seed": 42 } },
  "health":   { "status": "warning", "warnings": 9, "failed_runs": [] },
  "runs":     [ { "number": 1, "label": "outliers.outlier_threshold=[adaptive, 2.5]",
                  "values": { "outliers.outlier_threshold": ["adaptive", 2.5] },
                  "result": { "kind": "workflow", "metadata": {}, "health": {}, "steps": {}, "findings": [],
                              "assets": [] } } ]
}
```

`type` is what one run's result would say, and `keys` lists the matrix's keys as written. `health` holds the worst of
the runs' statuses (`failed`, then `warning`, then `ok`), the sum of their warnings, and the numbers of the runs that
failed. Each run's `result` is the JSON the task would write run alone with the run's `values`, its `assets` included.
Each run keeps the thumbnails of the items its own report names, since a chain names an item by its Dataset's address,
which holds other images in a run that reads another source. The matrix has no `assets` of its own. `errors`, where a
run failed, holds one line per failed run. The `metadata` records the task as written, matrix included, and the sources
its runs read; its encoding fields are `null`, since each run records its own. The comparison table is not stored, since
it reads off the runs.

From Python, the result is a `MatrixResult`, and `result.runs[2].result` is run 3's result, which you narrow with
`isinstance` as you would a lone task's. {doc}`run_a_matrix` covers the table and the other files.

## Next steps

- [Provenance](../concepts/Provenance.md) — Learn why a result must carry its lineage, and what the envelope records.
- [Reproducibility](../concepts/Reproducibility.md) — Learn how declarative configuration and config-keyed caching make
  a result repeatable.
- [Read a finding's report blocks](read_report_blocks.md) — Read the typed blocks that hold a finding's evidence.
