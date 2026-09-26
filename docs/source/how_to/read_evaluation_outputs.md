# Read evaluation outputs

A workflow run produces two things: a human-readable report and a machine-readable {term}`result envelope
<Result Envelope>`. This guide covers what each contains, how to get at the raw numbers behind a finding, and what the
provenance fields mean when you have to defend a result.

## Used in these tutorials

Every tutorial ends by reading its results, so this guide applies throughout. It is referenced directly from:

- {doc}`Clean a dataset <../notebooks/data_cleaning>`
- {doc}`Analyze dataset quality across splits <../notebooks/data_analysis>`
- {doc}`Assess dataset coverage <../notebooks/data_coverage>`
- {doc}`Monitor incoming data for drift <../notebooks/drift_monitoring>`
- {doc}`Detect out-of-distribution samples <../notebooks/ood_detection>`

## The text report

`report()` renders the findings for a run:

```python
result = run_task(task, config)
print(result.report())  # findings plus per-finding detail
print(result.report(detailed=False))  # summary only
```

The report is laid out in this order:

1. **Title** — the workflow's one-line summary.
2. **Provenance** — timestamp, duration, dataset and source descriptions, model and preprocessor identifiers.
3. **Summary** — one line per finding, then a health line.
4. **Detail** — a section per finding: its description, then its evidence as paragraphs, labelled values,
   tables and charts. This is the only part `detailed=False` suppresses.
5. **Metadata factors** — how the run encoded its metadata, when it used any.
6. **Resolved configuration** — the configuration as actually executed. Always rendered, at both detail levels.

The report is 80 columns wide. Pass `width=` (at least 40) to draw it narrower or wider: prose wraps, and charts
shrink to fit. From the CLI, `--report-width` sets it, or the `DATAEVAL_REPORT_WIDTH` environment variable.

### Severity and the health line

Each finding carries a severity of `ok`, `info`, or `warning`. A finding becomes a `warning` when it breaches its
{term}`health threshold <Health Threshold>`; otherwise it stays at `info`. The health line summarizes the run:

```text
  Health: 2 warning(s) [!!] — review flagged findings
  Health: All checks passed [ok]
```

A warning is a prompt to look, not a failure. The thresholds encode *your* risk tolerance — see
{doc}`configure_outlier_detection` for how to set them.

The same roll-up is available without parsing the report text:

```python
result.warning_count  # 2
result.health  # {"status": "warning", "warnings": 2, "findings": 7}
```

From the CLI, `--fail-on-warning` turns that roll-up into an exit code, so a pipeline can stop on a
run whose findings breached their thresholds:

```bash
dataeval-flow --config params.yaml --output ./results --fail-on-warning
```

Without the flag the warnings are still logged, and the run exits `0` — only a task that *failed* is
fatal by default.

## The result envelope

`export()` writes the structured result — findings plus provenance — to disk:

```python
result.export("./results")  # writes ./results/results.json
result.export("./run.yaml", fmt="yaml")  # explicit file, YAML
payload = result.export()  # no path: returns the serialized string
data = result.to_dict()  # no serialization: a plain dict
```

Path handling: a directory (or an extension-less path) gets `results.<ext>` written inside it; anything with a suffix
is treated as a file, and parent directories are created as needed.

From the CLI, `--output` sets the directory the envelope and reports are written to:

```bash
dataeval-flow --config params.yaml --data . --output ./results
```

### Envelope shape

The serialized envelope has five top-level keys, and a `kind` of `"workflow"`:

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
outputs, and `report` the same findings the text report renders — summary string plus a list of findings, each
with a `title`, `severity`, `brief`, `description`, and `blocks`: its evidence as typed report blocks, one object
per block with its `type`. `kind` distinguishes this from an evaluator's envelope, covered next.

### Findings and their report blocks

Each finding in `report.findings` has five keys:

| Key | Holds |
| --- | --- |
| `title` | A short label: the finding's summary line and the heading of its detail. |
| `severity` | `ok`, `info` or `warning`. |
| `brief` | The value on the summary line, or `null`. |
| `description` | A sentence or two of plain prose that leads the detail, or `null`. |
| `blocks` | The evidence: report blocks, in reading order. |

```json
{"severity": "info", "title": "Label Distribution", "brief": "3 classes, 60 items, imbalance 4.0:1",
 "description": "3 classes, 60 items.",
 "blocks": [
   {"type": "table",
    "columns": [{"key": "name", "header": "Class"},
                {"key": "value", "header": "Count"},
                {"key": "value", "kind": "bar"}],
    "rows": [{"name": "cat", "value": 40}, {"name": "dog", "value": 10}, {"name": "eel", "value": 10}]},
   {"type": "paragraph", "text": "Imbalance ratio: 4.0 (max/min)"}]}
```

A block is an object whose `type` says what it holds. A field at its default is left out, so a reader fills in the
defaults shown in parentheses:

| `type` | Fields |
| --- | --- |
| `section` | `title`; `brief` (`null`); `severity` (`null`), one of `ok`, `info`, `warning`; `blocks` (`[]`), nested blocks |
| `paragraph` | `text`: prose, where a backtick span is inline code and `\n` a line break |
| `bullet_list` | `items`: strings |
| `fields` | `items`: `[label, value]` pairs, in order; a value is a string, number, boolean or `null` |
| `table` | `columns` and `rows`, below |
| `proportion` | `parts`: `[label, count]` pairs making up one whole |
| `distribution` | `histogram`: counts per bin, in order; `quantiles` (`null`): `low`, `q1`, `median`, `q3`, `high` |
| `code` | `text`; `language` (`null`) |
| `tree` | `value`: any JSON value, such as a configuration |
| `summary` | `items`: each a `label`, a `value` (`""`) and a `severity` (`"info"`) |

A table's `rows` are objects keyed by each column's `key`. A column has:

| Field | Holds |
| --- | --- |
| `key` | The row key it reads. Two columns may share one, such as a count and its bar. |
| `header` (`""`) | The column's heading. |
| `kind` (`"text"`) | `text`, `bar`, `stacked` or `sparkline`. |
| `align` (`null`) | `left` or `right`; `null` puts the first column left and the rest right. |
| `format` (`null`) | A Python `str.format` template for a numeric cell, such as `"{:.1f}%"`. |
| `series` (`[]`) | A stacked column's segment names, in cell order. |
| `markers` (`[]`) | A bar column's labelled reference values, `[name, value]`, such as drift thresholds. |

A cell is a string, number, boolean or `null`, or, in a `stacked` or `sparkline` column, a list of numbers. A
number stays a number, and its column's `format` says how it displays, so a reader can sort and chart it. A cell
is a string only when it combines values, such as `"12 (30%)"`.

Flow defines the block types, and a later version may add one. A reader that meets a `type` it doesn't know should
show a one-line placeholder naming it and carry on, rather than fail, so an older reader keeps working on a newer
result.

`health.status` is `"warning"` where any finding breached its threshold and `"ok"` otherwise. It answers a
different question from whether the workflow *ran*: a task that failed produces errors, not warnings.

### Provenance fields

The `metadata` block is what makes a finding auditable and interoperable with other JATIC tools:

| Field | What it records |
| --- | --- |
| `version` | Envelope schema version |
| `timestamp` | UTC time the workflow ran |
| `execution_time_s` | Wall-clock duration |
| `tool` / `tool_version` | `dataeval-flow` and the exact version that produced the result |
| `dataset_id` | Identifier(s) of the evaluated dataset(s) |
| `source_descriptions` | Human-readable description of each resolved source |
| `selection_id` | Identifier for the {term}`view <View>` applied to the dataset |
| `label_source` | Where labels came from |
| `model_id` / `preprocessor_id` | The extractor model and preprocessing pipeline used |
| `resolved_config` | The fully resolved configuration, after merge and defaults |
| `metadata_binning` | How each factor was typed and discretized — see below |
| `diagnostics` | Library warnings raised during the run |

`resolved_config` is the field that makes a run repeatable. It is the configuration as actually executed. Keep the
envelope and you can reproduce the run without the original config file.

Workflows extend this envelope with their own fields, so `metadata` carries more than the table above. A
`data-cleaning` result, for example, also records `mode`, `evaluators`, `flagged_indices`, `clean_indices`, and
`removed_count`. Treat the table as the guaranteed floor, not the full set: each workflow's result class in the
{doc}`API Reference <../reference/autoapi/dataeval_flow/index>`, such as
{py:class}`~dataeval_flow.workflows.data_cleaning.DataCleaningResult`, lists the `metadata` fields it adds under
**Fields**.

## Evaluator results

An evaluator's `result.json` entry has a different shape: `"kind": "evaluator"`, and no findings, severities, or
`health` at all. DataEval's own output sits under `output`, as a table, mapping, or array — whatever shape that
evaluator produces:

```json
{
  "kind": "evaluator",
  "metadata": { "evaluator": "quality.duplicates", "dataeval": { "version": "1.1.1" } },
  "output": { "shape": "table", "columns": ["group_id", "..."], "rows": [] }
}
```

From Python, an evaluator's `result.output` is DataEval's own output object rather than a JSON-ready dict. See
[Run a single evaluator](run_a_single_evaluator.md) and the [Evaluator Catalog](../reference/evaluators.md) for what
each evaluator returns.

## Getting at the raw numbers

The report is a rendering; the numbers behind it live on the result object. `result.output.raw` holds the typed,
workflow-specific outputs:

```python
result = run_task(task, config)

# data-cleaning
flagged = result.output.raw.img_outliers

# data-coverage
onto_findings = result.output.raw.ontology
```

Each workflow declares its own raw output, so field names differ by workflow. Each workflow's result class in the
{doc}`API Reference <../reference/autoapi/dataeval_flow/index>`, such as
{py:class}`~dataeval_flow.workflows.data_cleaning.DataCleaningResult`, lists every `output.raw` field and what it holds
under **Fields**. Narrow a result to that class with `isinstance`, and your editor and type checker know the fields too.

### How metadata factors were treated

Bias, balance, diversity, and coverage analyses read factors as *codes* — a continuous factor cut into intervals, a
categorical one mapped to ordinals — so what a run decided about binning is part of what the result means.
`result.metadata.metadata_binning` records it: per factor, the type, the {term}`level <Metadata Level>` it was
binned at, whether it was binned or digitized, and the observed range and population of every bin.
`result.metadata.diagnostics` carries the library warnings the run raised. Both render in the text report under
**METADATA FACTORS**.

Per-factor summaries in `raw` carry the same shape of information alongside the values: `level` and `is_binned` per
factor, plus `dropped_factors` naming vector-valued statistics (`histogram`, `percentiles`, `center`) that have no
single-column form and so never became factors at all. `invalid_box` is carried through as a factor; the other hash
columns are discarded.

{doc}`configure_metadata_binning` covers how to control any of this.

:::{note}
`level` means two different things in a result, on two different axes. Per-factor metadata summaries report a
{term}`metadata level <Metadata Level>` — `sequence`, `unit`, `track`, or `instance`. Duplicate groups report `item`
or `target`, which is unrelated and unchanged. Classwise outlier pivots report `count_basis` (`image` or
`annotation`) rather than `level`, precisely so the three cannot be read as one.
:::

## Getting at more of the run

Two more fields are useful for follow-up work and are deliberately *not* serialized into the envelope:

- `result.dataset` — the resolved, post-view dataset the workflow ran on, for pulling up the images behind a finding.
- `result.sources` — for multi-split workflows such as `data-analysis`, a mapping of source name to resolved dataset.

```python
for issue in result.output.raw.img_outliers["issues"]:
    image, target, meta = result.dataset[issue["item_index"]]
```

## Checking whether a run succeeded

```python
if not result.success:
    for err in result.errors:
        print(err)
```

In the container, success or failure is also reported through the process exit code — see
{doc}`containerized_workflows` and the {doc}`Container Reference <../reference/containers>`.

## Related material

- [Provenance](../concepts/Provenance.md) — why a result must carry its lineage, and what the envelope records
- [Reproducibility](../concepts/Reproducibility.md) — how declarative configuration and config-keyed caching make a
  result repeatable
- {doc}`configure_outlier_detection` — setting the thresholds that drive severity
