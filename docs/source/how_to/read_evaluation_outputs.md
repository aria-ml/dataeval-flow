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
print(result.report(detailed=False))  # the short form, which the console prints without -v
```

A workflow's report is laid out in this order:

1. **Banner** — the friendly title of what ran, such as `Drift Monitoring`. The text report prints it in capitals;
   HTML keeps its case, and its page title adds the entry where it differs from the id, as
   `Drift Monitoring — drift`.
2. **Provenance** — opens with a line naming what ran: `Workflow: drift (drift-monitoring)`, the config entry beside
   its type id, or `Evaluator: dupes (duplicates)` for an evaluator task. Where the entry is named after its type, the
   line is the id alone, as `Workflow: drift-monitoring`. Then the timestamp, duration, the sources the run read,
   model and preprocessor identifiers.
3. **Summary sentence** — the workflow's one-line account of the run, such as
   `Steps: 4 ran` for a preset, which runs a chain of steps.
4. **Summary** — one line per finding, then a health line. A run with no findings says `No findings to report.`
5. **Detail** — a section per finding: its description, then its evidence as paragraphs, labelled values,
   tables and charts.
6. **Metadata factors** — how the run encoded its metadata, when it used any.
7. **Configuration** — the configuration as actually executed, less the settings left unset. A setting written as
   `null` stays, since `null` means something, such as a health threshold switched off.

`detailed=False` leaves out the detail, and the metadata factors' *Per-factor detail*. The rest is rendered at both
detail levels.

The report is 80 columns wide. Pass `width=` (at least 40) to draw it narrower or wider: prose wraps, and charts
shrink to fit. A table still too wide wraps its text cells, with a blank line between its rows. From the CLI,
`--report-width` sets it, else the `DATAEVAL_REPORT_WIDTH` environment variable, else the pipeline's
`result: width`.

A long table, such as one row per flagged image, shows its first rows and a line counting the rest. Every row is in
the HTML report and in the JSON. Text has no pictures, so a table's thumbnails are left out, and the row's other
cells, such as its *Item*, name each item.

### A chain's report

A chain's report, a custom workflow's or a {term}`preset <Preset>`'s such as `data-cleaning`'s, puts each finding
beside the evidence it judged:

1. **Banner** and **provenance**, as above. A custom workflow's banner is its name, and its provenance opens with
   `Workflow: name (custom workflow)`.
2. **Steps** — how many ran: `Steps: 10 ran`, or `Steps: 5 (3 ran, 1 failed, 1 skipped)` where some did not.
3. **Summary** — one line per finding, then the health line.
4. **Findings** — a section per finding, headed by its title and brief. It holds the finding's own description and
   blocks, then the evidence: each step the finding read, headed *From* and the step's heading. A step's heading is
   its type's title, with the step's name beside it where the name is not the type id: `Outliers` for the step
   `outliers`, `Duplicates · dupes` for the step `dupes`. A combine shows nothing of its own, so the steps it read
   stand in for it. A step two findings read is shown under the first, and the second names it.
5. **Other steps** — a section for each step no finding showed that has something to show: what a transform made,
   an evaluator's output no check judged, or why a step failed or was skipped. A check that did not complete made
   no finding, so it is listed here.
6. **Steps** — a table of every step in run order: its name, type and status, where each Dataset it read came from,
   and why it made nothing where it did not. The HTML report's table also gives each step's title.
7. **Configuration**, as above.

In `data-cleaning`'s report on MILCO's reference campaigns, from {doc}`View a report as HTML
<../notebooks/view_html_reports>`, the Target Outliers finding read the `outliers` step, which the Image Outliers
section above it already shows, and the `labels` step, which no finding had shown yet:

```text
================================================================================
  TARGET OUTLIERS                                              32 targets (6.5%)
================================================================================
  Evidence: Outliers, under Image Outliers.

  From Label Health · labels
    Items:        261
    Classes:      2
    Labels:       492
    Empty images: 97
    Label source: annotations

    Class  Labels  Images
    -----  ------  ------
    MILCO     319     155
    NOMBO     173      70
```

Where checks ran once per element of a list, the findings are grouped by the element's key. The Summary names each
key above its findings, and each key has a section holding them. Here `data-cleaning` ran as a step on a list of
MILCO's two campaigns, `reference` and `operational`:

```text
================================================================================
  SUMMARY
================================================================================
  reference
    Image Outliers .................................... 11 images (4.2%)  [!!]
    Target Outliers .................................. 32 targets (6.5%)  [!!]
    Classwise Outliers ...... worst: NOMBO (7.5%), 2/2 classes over 3.0%  [!!]
    Duplicates ........................ 0 exact (0.0%), 150 near (57.5%)  [!!]
    Label Distribution ........... 2 classes, 261 items, imbalance 1.8:1  [..]
  operational
    Image Outliers .................................... 22 images (2.4%)  [..]
    Target Outliers ................................... 6 targets (3.4%)  [!!]
    Classwise Outliers ...... worst: NOMBO (5.2%), 1/2 classes over 3.0%  [!!]
    Duplicates ........................ 0 exact (0.0%), 787 near (86.6%)  [!!]
    Label Distribution ........... 2 classes, 909 items, imbalance 2.0:1  [..]

  Health: 7 warning(s) [!!] — review flagged findings
```

A chain's short form, `report(detailed=False)`, keeps the step count, the Summary and a compact Steps table of each
step's name, status and note, with no finding's detail and no evidence. The reference campaigns' run, less its
banner and configuration:

```text
  Steps: 10 ran

================================================================================
  SUMMARY
================================================================================
  Image Outliers ...................................... 11 images (4.2%)  [!!]
  Target Outliers .................................... 32 targets (6.5%)  [!!]
  Classwise Outliers ........ worst: NOMBO (7.5%), 2/2 classes over 3.0%  [!!]
  Duplicates .......................... 0 exact (0.0%), 150 near (57.5%)  [!!]
  Label Distribution ............. 2 classes, 261 items, imbalance 1.8:1  [..]

  Health: 4 warning(s) [!!] — review flagged findings

================================================================================
  STEPS
================================================================================
  Step             Status  Note
  ---------------  ------  ----
  outliers         ok
  labels           ok
  by-class         ok
  dupes            ok
  image-outliers   ok
  target-outliers  ok
  classwise        ok
  duplicates       ok
  imbalance        ok
  clean            ok
```

### Severity and the health line

Each finding carries a severity of `ok`, `info`, or `warning`. A finding becomes a `warning` when it breaches its
{term}`health threshold <Health Threshold>`; otherwise it stays at `info`. The health line summarizes the run,
counting its warnings or saying that every check passed:

```text
  Health: 1 warning(s) [!!] — review flagged findings
  Health: All checks passed [ok]
```

A chain whose required step failed has failed, whatever its warnings. Its health line names the steps that did, and
still counts the warnings:

```text
  Health: failed [!!] — step `coverage` failed; 1 warning(s) to review
```

A warning is a prompt to look, not a failure. The thresholds encode *your* risk tolerance — see
{doc}`configure_outlier_detection` for how to set them.

The same roll-up is available without parsing the report text:

```python
result.warning_count  # 1
result.health  # {"status": "warning", "warnings": 1, "findings": 1}
```

A chain's `health` adds `failed_steps`, the required steps that failed, and its `status` is `failed` where any did.

From the CLI, `--fail-on-warning`, or `fail_on: warning` in the pipeline's `result:` block, turns that roll-up
into an exit code, `3`, so a pipeline can stop on a run whose findings breached their thresholds:

```bash
dataeval-flow --config params.yaml --output ./results --fail-on-warning
```

Without the flag the warnings are still logged, and the run exits `0` — only a task that *failed* is
fatal by default.

## The HTML report

`to_html()` renders the same report as one self-contained page:

```python
from pathlib import Path

Path("report.html").write_text(result.to_html(), encoding="utf-8")
```

{doc}`View a report as HTML <../notebooks/view_html_reports>` shows the page rendered, so you can try it. The page
holds everything the text report holds, laid out for reading on screen:

- The header gives the report's title and verdict, with the entry and type id under them. The verdict is the
  number of warnings, `passed`, or `failed` and the required steps that failed. A page holding several tasks'
  reports lists them first.
- Each finding is a card that opens and closes, headed by its title, its value and its severity, so the cards read
  as the report's summary. A warning starts open and the rest start closed.
- The metadata factors, a chain's Steps table and the configuration close the report as panels of their own,
  closed until opened.
- *Expand all* and *Collapse all*, above the report, open or close every card and panel at once.
- A table's headers sort it on a click, and a table of more than ten rows gets a box that filters its rows. Cells
  keep their raw values, so a column of numbers sorts as numbers, and a column of flags by how many each row holds.
- An outlier's flags show as tags, each reading its value against the limit it crossed, such as
  `brightness 0.99 > 0.84`, and listed by name. Hovering a tag, or reaching it with the keyboard, shows where the
  value ranks in its population and the population's mean and standard deviation.
- A threshold is a dashed line across a chart's bars, labelled on a scale below the table.
- Each item a finding names shows its thumbnail in its row: a flagged image or box, a duplicate group, an OOD sample,
  an uncovered item, a prioritized item at either end of its ranking, an unlabelled image, and the items that hold a
  metadata value that doesn't read like the rest. Click one to enlarge it over the page, and click anywhere, or press
  Esc, to put it back. An item without a thumbnail is named instead.
- Histograms and sparklines are drawn as SVG, and the page follows the system's dark mode.

A chain's report, `data-cleaning`'s among them, draws each finding as a card too, and holds in it the evidence the
finding judged, each step headed *From* and the step's heading: data cleaning's Duplicates card holds the `dupes`
step's duplicate groups. A step two findings judged is shown in the first one's card, and the second names that card.
Findings grouped by key sit under a heading per key, each still a card. The chain's other steps, such as `clean`,
follow as sections, then the Steps table and the configuration as panels. The Steps table gives each step's title,
type and status, what it read, and why it made nothing where it did not. The short page, `to_html(detailed=False)`,
has no cards: it keeps the summary table, and an open Steps table of each step's status and note.

Flow takes the thumbnails once a run is done, from the datasets the run read: one per item, at most 192 pixels
across, and at most 200 per result. A pipeline's `result: max_images:` sets that limit: `0` embeds none, and `-1`
every item the report names. The
limit is shared evenly between the findings that name items, and a finding's share between its tables, rows in
order; a finding that needs fewer passes its spare to the rest. So with 200 and four such findings, each gets 50.
A box's thumbnail is cropped from its image with a margin around it. `--no-report-images`,
`DATAEVAL_REPORT_IMAGES=0`, or `report_images=False` on `run()`, `run_task()` and `run_tasks()` turn them off, and
then the run reads no item for them. Only images have thumbnails for now; any other kind of item is named.

It loads nothing, neither font nor URL. One inline script adds the sorting, the filter boxes and the expand-all
buttons. With scripts blocked, as some mail viewers and locked-down browsers block them, the page reads the same
without those controls.

The page prints (or saves as PDF from the browser's print dialog) in the light palette. Before it prints, its script
opens every finding and shows every row a filter hid. With scripts blocked, each finding and panel prints as the
reader left it. Hover cards don't print, and thumbnails print at their own size. In data cleaning, the `outliers`
step's limits tables give each metric's limits and its population's mean and standard deviation, and say `varies`
where its flags' figures differ. Percentiles, and data analysis's populations, show only in the hover cards and the
JSON.

The page is UTF-8, so write it with `encoding="utf-8"`. With `--output`, the CLI writes `results/result.html`, every
task's report on one page, unless the pipeline's `result:` block says otherwise (see
{doc}`Run workflows in a container <containerized_workflows>`).

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

The serialized envelope has five top-level keys, a sixth, `assets`, where its report pictures any items, and a `kind`
of `"workflow"`:

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
per block with its `type`. `assets` holds the thumbnails, as the end of the next section describes. `kind`
distinguishes this from an evaluator's envelope, covered next.

A chain's result has `steps` and a top-level `findings` in place of `raw` and `report`. A custom workflow's result
is one, and so is a `data-cleaning` result: `data-cleaning` is a {term}`preset <Preset>`, whose settings expand to a
chain of steps.

```json
{
  "kind":     "workflow",
  "metadata": {
    "timestamp": "...",
    "workflow":  "clean",
    "lineage":   [ { "name": "data", "source": "ref-src", "items": 261 },
                   { "name": "clean", "step": "clean", "items": 162 } ]
  },
  "health":   { "status": "warning", "warnings": 4, "findings": 5, "failed_steps": [] },
  "steps":    { "outliers": { "kind": "evaluator", "type": "outliers", "status": "ok" } },
  "findings": [ { "step": "image-outliers", "title": "Image Outliers", "severity": "warning" } ]
}
```

That is `data-cleaning`'s result on MILCO's reference campaigns, from {doc}`View a report as HTML
<../notebooks/view_html_reports>`, trimmed. `steps` holds each step by name, in run order, with its kind, type,
status, the addresses it read and what it made. `findings` lists the check steps' findings, each shaped as below and
naming its step under `step`. `health` also lists the steps that failed. {doc}`write_a_custom_workflow` shows more of
a chain's JSON.

### Findings and their report blocks

Each finding, in a workflow's `report.findings` or a chain's `findings`, has five keys, and a chain's names its step
under a sixth, `step`:

| Key | Holds |
| --- | --- |
| `title` | A short label: the finding's summary line and the heading of its detail. |
| `severity` | `ok`, `info` or `warning`. |
| `brief` | The value on the summary line, or `null`. |
| `description` | A sentence or two of plain prose that leads the detail, or `null` where the brief says it all. |
| `blocks` | The evidence: report blocks, in reading order. |

The same run's Label Distribution finding:

```json
{"severity": "info", "title": "Label Distribution", "brief": "2 classes, 261 items, imbalance 1.8:1",
 "description": null,
 "blocks": [
   {"type": "table",
    "columns": [{"key": "name", "header": "Class"},
                {"key": "value", "header": "Count"},
                {"key": "value", "kind": "bar"}],
    "rows": [{"name": "MILCO", "value": 319}, {"name": "NOMBO", "value": 173}]},
   {"type": "paragraph", "text": "Labels annotations"},
   {"type": "paragraph", "text": "Imbalance ratio: 1.8 (max/min)"}],
 "step": "imbalance"}
```

A block is an object whose `type` says what it holds. A field at its default is left out, so a reader fills in the
defaults shown in parentheses:

| `type` | Fields |
| --- | --- |
| `section` | `title`; `brief` (`null`); `severity` (`null`), one of `ok`, `info`, `warning`; `reference` (`false`), whether HTML folds it away as a panel, as it does the configuration; `blocks` (`[]`), nested blocks |
| `paragraph` | `text`: prose, where a backtick span is inline code and `\n` a line break |
| `bullet_list` | `items`: strings |
| `fields` | `items`: `[label, value]` pairs, in order; a value is a string, number, boolean or `null` |
| `table` | `columns` and `rows`, below; `preview` (`null`) |
| `proportion` | `parts`: `[label, count]` pairs making up one whole |
| `distribution` | `histogram`: counts per bin, in order; `quantiles` (`null`): `low`, `q1`, `median`, `q3`, `high` |
| `code` | `text`; `language` (`null`) |
| `tree` | `value`: any JSON value, such as a configuration |
| `summary` | `items`: each a `label`, a `value` (`""`), a `severity` (`"info"`) and a `group` (`""`), the heading it sits under, such as a split's key; `warnings`: how many findings are warnings; `failed` (`[]`): the required steps that failed |

A table's `rows` are objects keyed by each column's `key`. A column has:

| Field | Holds |
| --- | --- |
| `key` | The row key it reads. Two columns may share one, such as a count and its bar. |
| `header` (`""`) | The column's heading. |
| `kind` (`"text"`) | `text`, `bar`, `stacked`, `sparkline`, `flags` or `image`. |
| `align` (`null`) | `left` or `right`; `null` puts the first column left and the rest right. |
| `format` (`null`) | A Python `str.format` template for a numeric cell, such as `"{:.1f}%"`. |
| `series` (`[]`) | A stacked column's segment names, in cell order. |
| `markers` (`[]`) | A bar column's labelled reference values, `[name, value]`, such as drift thresholds. |
| `in_text` (`true`) | Whether the text report draws the column; HTML always does. A chain's Steps table sets it `false` on its *Title* column. |

A summary item's `group`, a summary's `failed` and a column's `in_text` are left out of the JSON at their defaults,
like every other field, so a result that uses none of them reads as it did before they were added.

A table's `preview` says how many rows a renderer with little room, such as the text report, shows before a line
counting the rest. `null` shows every row. A table of items, such as flagged images, previews 10 of at most 500 rows,
with a paragraph naming the rest; a pipeline's `result: preview_rows:` and `result: max_rows:` change those, and `-1`
lifts either.

A cell is a string, number, boolean or `null`. In a `stacked` or `sparkline` column it is a list of numbers, in a
`flags` column a list of flags, and in an `image` column an item reference, or a list of them for a group such as
duplicates. A number stays a number, and its column's `format` says how it displays, so a reader can sort and chart
it. A cell is a string only when it combines values, such as `"12 (30%)"`.

An item reference names one item as the run saw it:

| Field | Holds |
| --- | --- |
| `source` | The source it was read from, as the task or `run()` named it. |
| `index` | Its index in that source, after the source's view. |
| `target` (`null`) | A detection box, by its index in the item's annotation; `null` for the whole item. |

A flag is one measurement against the population it was judged in, such as a metric that marked an image an outlier:

| Field | Holds |
| --- | --- |
| `name` | What was measured, such as `brightness`. |
| `value` | This item's value. |
| `direction` | `upper` or `lower`: which limit the value crossed. |
| `bound` | The limit it crossed, or `null` when unknown. |
| `percentile` | Where the value ranks in its population, from 0 to 100, or `null` when unknown. |
| `mean`, `std` | The population's mean and standard deviation, each `null` when unknown. |

A figure is unknown when the run didn't record it, as a DataEval that predates these figures doesn't. A flag whose
`bound` is `null` had its `direction` go unrecorded too, so don't read it. The reports show such a flag as its value
alone, and the run logs a warning to upgrade DataEval.

The reports show a flag as its value against its bound, `>` for an upper limit and `<` for a lower, and list a
cell's flags by name. They don't rank one flag above another: a limit may be a plain value rather than a percentile,
and a value further past its limit isn't a worse item for it.

Flow defines the block types, and a later version may add one. A reader that meets a `type` it doesn't know should
show a one-line placeholder naming it and carry on, rather than fail, so an older reader keeps working on a newer
result.

A result's `assets` are the thumbnails of the items its image cells name, one per item:

| Field | Holds |
| --- | --- |
| `item` | The item reference it shows. |
| `media_type` | What `data` holds: `image/webp`, the one kind Flow makes today. |
| `width`, `height` | Its size in pixels, at most 192 on the long side. |
| `data` | The thumbnail, base64-encoded. |

```json
{"item": {"source": "train", "index": 41}, "media_type": "image/webp", "width": 192, "height": 144,
 "data": "UklGRjAAAABXRUJQ..."}
```

An item named in a cell may have no asset: thumbnails were turned off, the item was past a cap, or it couldn't be
read. Show its name instead, as the reports do. A later version may make other kinds of preview, so show the name
too for a `media_type` you can't draw.

`health.status` is `"warning"` where any finding breached its threshold and `"ok"` otherwise, unless the run
failed: a failed workflow's, and a chain's whose required step failed, is `"failed"`. Warnings answer a different
question from whether the workflow *ran*: a task that failed produces errors, not warnings.

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
`data-cleaning` result, for example, is a chain's, and records `workflow`, the entry's name, and `lineage`, each
Dataset in the chain with what made it. It records no metadata encoding, so its `metadata_binning` is `null`. Treat
the table as the guaranteed floor, not the full set: each result class in the
{doc}`API Reference <../reference/autoapi/dataeval_flow/index>`, such as
{py:class}`~dataeval_flow.steps.ChainResult`, lists the `metadata` fields it adds under **Fields**.

## Evaluator results

An evaluator's `result.json` entry has a different shape: `"kind": "evaluator"`, and no findings, severities, or
`health` at all. DataEval's own output sits under `output`, as a table, mapping, or array — whatever shape that
evaluator produces:

```json
{
  "kind": "evaluator",
  "metadata": { "evaluator": "duplicates", "dataeval": { "version": "1.1.1" } },
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

# data-coverage
onto_findings = result.output.raw.ontology
uncovered = result.output.raw.coverage.uncovered  # each uncovered item, its box and class, and its distance
```

Each workflow declares its own raw output, so field names differ by workflow. Each workflow's result class in the
{doc}`API Reference <../reference/autoapi/dataeval_flow/index>`, such as
{py:class}`~dataeval_flow.workflows.data_coverage.DataCoverageResult`, lists every `output.raw` field and what it holds
under **Fields**. Narrow a result to that class with `isinstance`, and your editor and type checker know the fields too.

A chain's result, a `data-cleaning` result among them, is a {py:class}`~dataeval_flow.steps.ChainResult` and has no
`output.raw`. Its `steps` hold each step's output, by step name: an evaluator step's is DataEval's own output, a
check's is its findings, and a transform's is the Dataset it made, a DataEval `View`:

```python
result = run_task(task, config)  # a data-cleaning task

outliers = result.steps["outliers"].output  # DataEval's Outliers output
flags = outliers.data()  # one row per flag: its item, its box if any, the metric, its value and the limit crossed
duplicates = result.steps["dupes"].output  # DataEval's Duplicates output
cleaned = result.steps["clean"].output  # without each flagged image and box, and each duplicate but the first
```

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

- `result.dataset` — the resolved, post-view dataset a one-source workflow ran on, for pulling up the images behind a
  finding.
- `result.sources` — for multi-split workflows such as `data-analysis`, and for every chain, `data-cleaning` among
  them, a mapping of source name to resolved dataset.

```python
dataset = result.sources["train"]  # a data-cleaning task on the source `train`
for item in result.steps["outliers"].output.data()["item_index"].unique().sort():
    image, target, meta = dataset[item]
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
