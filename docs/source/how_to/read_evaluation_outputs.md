# Read evaluation outputs

A workflow run produces two things: a human-readable report and a machine-readable {term}`result envelope
<Result Envelope>`. This guide covers what each contains, how to get at the raw numbers behind a finding, and what the
provenance fields mean when you have to defend a result later.

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

The report is laid out in five blocks:

1. **Title** — the workflow's one-line summary.
2. **Provenance** — timestamp, duration, dataset and source descriptions, model and preprocessor identifiers.
3. **Summary** — one line per finding, then a health line.
4. **Detail** — an expanded section per finding. This is the only block `detailed=False` suppresses.
5. **Resolved configuration** — the configuration as actually executed. Always rendered, at both detail levels.

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
result.health         # {"status": "warning", "warnings": 2, "findings": 7}
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

The serialized envelope has three top-level keys:

```json
{
  "metadata": { "timestamp": "...", "tool": "dataeval-flow", "resolved_config": {} },
  "health":   { "status": "ok", "warnings": 0, "findings": 7 },
  "raw":      { },
  "report":   { "summary": "...", "findings": [] }
}
```

`metadata` is the provenance envelope, `health` the roll-up of the findings' severities, `raw` the typed numeric
outputs, and `report` the same findings the text report renders — summary string plus a list of findings, each
with a `title`, `severity`, and `data`.

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

`resolved_config` is the field that makes a run repeatable: it is the configuration as actually executed, not as
written. Keep the envelope and you can reproduce the run without the original config file.

Workflows extend this envelope with their own fields, so `metadata` carries more than the table above. A
`data-cleaning` result, for example, also records `mode`, `evaluators`, `flagged_indices`, `clean_indices`, and
`removed_count`. Treat the table as the guaranteed floor, not the full set — the
{doc}`API Reference <../reference/autoapi/dataeval_flow/index>` lists each workflow's metadata model.

## Getting at the raw numbers

The report is a rendering; the numbers behind it live on the result object. `result.data.raw` holds the typed,
workflow-specific outputs:

```python
result = run_task(task, config)

# data-cleaning
flagged = result.data.raw.img_outliers

# data-coverage
onto_findings = result.data.raw.ontology
```

Each workflow declares its own raw output model, so field names differ by workflow — the
{doc}`API Reference <../reference/autoapi/dataeval_flow/index>` lists them per workflow.

### How metadata factors were treated

Bias, balance, diversity, and coverage analyses read factors as *codes* — a continuous factor cut into intervals, a
categorical one mapped to ordinals — so what a run decided about binning is part of what the result means.
`result.metadata.metadata_binning` records it: per factor, the type, the {term}`level <Metadata Level>` it was
binned at, whether it was binned or digitized, and the observed range and population of every bin.
`result.metadata.diagnostics` carries the library warnings the run raised. Both render in the text report under
**METADATA FACTORS**.

Per-factor summaries in `raw` carry the same shape of information alongside the values: `level` and `is_binned` per
factor, plus `dropped_factors` naming vector-valued statistics (`histogram`, `percentiles`, `center`) that have no
single-column form and so never became factors at all. `invalid_box` is carried through as a factor rather than
discarded with the hash columns.

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
for idx in result.data.raw.img_outliers:
    image, target, meta = result.dataset[idx]
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
