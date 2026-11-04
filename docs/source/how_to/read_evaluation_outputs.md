# Read evaluation outputs

A workflow run produces two things: a report for a person and a machine-readable
{term}`result envelope <Result Envelope>`. To read the report, print the text report or write the HTML page, then check
each finding's severity and the run's health. [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) explains
chains of steps.
[Read the result envelope](read_the_result_envelope.md) covers the envelope, and
[Get the numbers behind a finding](get_the_numbers_behind_a_finding.md) covers the raw numbers behind a finding. The
examples assume a finished run, `result = run_task(config, task)`.

## Print the text report

`report()` renders the findings for a run:

```python
result = run_task(config, task)
print(result.report())  # findings plus per-finding detail
print(result.report(detailed=False))  # the short form, which the console prints without -v
```

A workflow's report has these sections, in this order:

| Section | Holds |
| --- | --- |
| Banner | The friendly title of what ran, such as `Shift`. The text report prints it in capitals. HTML keeps its case, and its page title adds the entry where it differs from the id, as `Shift — drift`. |
| Provenance | A line naming what ran: `Workflow: drift (shift)`, the config entry beside its type id, or `Evaluator: dupes (duplicates)` for an evaluator task. Where the entry is named after its type, the line is the id alone, as `Workflow: shift`. Then the timestamp, duration, the sources the run read, and model and preprocessor identifiers. |
| Summary sentence | The workflow's one-line account of the run, such as `Steps: 4 ran` for a preset, which runs a chain of steps. |
| Summary | One line per finding, then a health line. A run with no findings says `No findings to report.` |
| Detail | A section per finding: its description, then its evidence as paragraphs, labeled values, tables and charts. |
| Metadata factors | How the run encoded its metadata, when it used any. |
| Configuration | The configuration as executed, less the settings left unset. A setting written as `null` stays, since `null` means something, such as a check's threshold switched off. |

`detailed=False` leaves out the detail, and the metadata factors' "Per-factor detail". The rest is rendered at both
detail levels.

The report is 80 columns wide. Pass `width=` (at least 40) to draw it narrower or wider: prose wraps, and charts shrink
to fit. A table still too wide wraps its text cells, with a blank line between its rows. From the CLI, `--report-width`
sets it, else the `DATAEVAL_REPORT_WIDTH` environment variable, else the pipeline's `result: width`.

A long table, such as one row per flagged image, shows its first rows and a line counting the rest. Every row is in the
HTML report and in the JSON. Text has no pictures, so a table's thumbnails are left out, and the row's other cells, such
as its Item cell, name each item.

## Read a chain's report

A chain's report, a custom workflow's or a {term}`preset <Preset>`'s such as `quality`'s, puts each finding beside the
evidence it judged. It has these sections, in this order:

| Section | Holds |
| --- | --- |
| Banner and provenance | As above. A custom workflow's banner is its name, and its provenance opens with `Workflow: name (custom workflow)`. |
| Step count | How many steps ran: `Steps: 10 ran`, or `Steps: 5 (3 ran, 1 failed, 1 skipped)` where some did not. |
| Summary | One line per finding, then the health line. |
| Findings | A section per finding, headed by its title and brief. It holds the finding's own description and blocks, then the evidence: each step the finding read, headed "From" and the step's heading. A step's heading is its type's title, with the step's name beside it where the name is not the type id: `Outliers` for the step `outliers`, `Duplicates · dupes` for a custom workflow's step `dupes`. A combine shows nothing of its own, so the steps it read stand in for it. A step two findings read is shown under the first, and the second names it. |
| Other steps | A section for each step no finding showed that has something to show: what a transform made, an evaluator's output no check judged, or why a step failed or was skipped. A check that did not complete made no finding, so it is listed here. |
| Steps | A table of every step in run order: its name, type and status, where each Dataset it read came from, and why it made nothing where it did not. The HTML report's table also gives each step's title. |
| Configuration | As above. |

In `quality`'s report on MILCO's reference campaigns, from
{doc}`View a report as HTML <../notebooks/view_html_reports>`, the Target Outliers finding read the `outliers` step,
which the Image Outliers section above it already shows, and the `label-health` step, which no finding had shown yet:

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

Where checks ran once per element of a list, the findings are grouped by the element's key. The Summary names each key
above its findings, and each key has a section holding them. Here `quality` ran as a step on a list of MILCO's two
campaigns, `reference` and `operational`:

```text
================================================================================
  SUMMARY
================================================================================
  reference
    Image Outliers .................................... 11 images (4.2%)  [!!]
    Target Outliers .................................. 32 targets (6.5%)  [!!]
    Class Outliers .......... worst: NOMBO (7.5%), 2/2 classes over 3.0%  [!!]
    Image Duplicates .................. 0 exact (0.0%), 150 near (57.5%)  [!!]
  operational
    Image Outliers .................................... 22 images (2.4%)  [..]
    Target Outliers ................................... 6 targets (3.4%)  [!!]
    Class Outliers .......... worst: NOMBO (5.2%), 1/2 classes over 3.0%  [!!]
    Image Duplicates .................. 0 exact (0.0%), 787 near (86.6%)  [!!]

  Health: 7 warning(s) [!!] — review flagged findings
```

A chain's short form, `report(detailed=False)`, keeps the step count, the Summary and a compact Steps table of each
step's name, status and note, with no finding's detail and no evidence. The reference campaigns' run, less its banner
and configuration:

```text
  Steps: 9 ran

================================================================================
  SUMMARY
================================================================================
  Image Outliers ...................................... 11 images (4.2%)  [!!]
  Target Outliers .................................... 32 targets (6.5%)  [!!]
  Class Outliers ............ worst: NOMBO (7.5%), 2/2 classes over 3.0%  [!!]
  Image Duplicates .................... 0 exact (0.0%), 150 near (57.5%)  [!!]

  Health: 4 warning(s) [!!] — review flagged findings

================================================================================
  STEPS
================================================================================
  Step                Status  Note
  ------------------  ------  ----
  outliers            ok
  label-health        ok
  outliers-by-class   ok
  duplicates          ok
  image-outliers      ok
  target-outliers     ok
  class-outliers  ok
  image-duplicates    ok
  clean               ok
```

## Check the severity and the health line

Each finding carries a {term}`severity <Severity>` of `ok`, `info`, or `warning`, which the glossary defines. A finding
is a `warning` when it passes a {term}`threshold <Threshold>` or meets another criterion its check warns on. The health
line summarizes the run, counting its warnings or saying that every check passed:

```text
  Health: 1 warning(s) [!!] — review flagged findings
  Health: All checks passed [ok]
```

A chain whose required step failed has failed, whatever its warnings. Its health line names the steps that did, and
still counts the warnings:

```text
  Health: failed [!!] — step `coverage` failed; 1 warning(s) to review
```

A warning is a prompt to look, not a failure. The thresholds encode your risk tolerance, and
{doc}`configure_outlier_detection` shows how to set them.

The same roll-up is available without parsing the report text:

```python
result.warning_count  # 1
result.health  # {"status": "warning", "warnings": 1, "findings": 1}
```

A chain's `health` adds `failed_steps`, the required steps that failed, and its `status` is `failed` where any did.

From the CLI, `--fail-on-warning`, or `fail_on: warning` in the pipeline's `result:` block, turns that roll-up into an
exit code, `3`, so a pipeline can stop on a run whose findings breached their thresholds:

```bash
dataeval-flow --config params.yaml --output ./results --fail-on-warning
```

Without the flag the warnings are still logged, and the run exits `0`. Only a task that failed is fatal by default.
{doc}`fail_a_ci_job_on_warnings` shows how to gate a CI job on the exit code.

## Check whether a run succeeded

```python
if not result.success:
    for err in result.errors:
        print(err)
```

In the container, success or failure is also reported through the process exit code. See
{doc}`containerized_workflows` and the {doc}`Container Reference <../reference/containers>`.

## Read the HTML report

`to_html()` renders the same report as one self-contained page:

```python
from pathlib import Path

Path("report.html").write_text(result.to_html(), encoding="utf-8")
```

The page is UTF-8, so write it with `encoding="utf-8"`. With `--output`, the CLI writes `results/result.html`, every
task's report on one page, unless the pipeline's `result:` block says otherwise (see
{doc}`Run workflows in containers <containerized_workflows>`).

{doc}`View a report as HTML <../notebooks/view_html_reports>` shows the page rendered, so you can try it, and shows how
to sort and filter its tables, read a flag and print it. The page holds everything the text report holds, laid out for
reading on screen. It also has these parts:

- An audit's header shows its verdict in the place of the health: the verdict's label, then its reasons.
- The metadata factors, a chain's Steps table and the configuration close the report as panels of their own, closed
  until opened.
- A chain's report draws findings grouped by key under a heading per key, each still a card. The short page,
  `to_html(detailed=False)`, has no cards: it keeps the summary table, and an open Steps table of each step's status
  and note.
- An outlier's flags show as tags, each reading its value against the limit it crossed, such as
  `brightness 0.99 > 0.84`, and a cell lists them by name.
- Each item a finding names shows its thumbnail in its row: a flagged image or box, a duplicate group, an OOD sample, an
  uncovered item, a prioritized item at either end of its ranking, an unlabeled image, and the items that hold a
  metadata value that does not read like the rest. An item without a thumbnail is named instead.
- Histograms and sparklines are drawn as SVG.

The page loads nothing, no font and no URL. One inline script adds the sorting, the filter boxes and the expand-all
buttons. With scripts blocked, as some mail viewers and locked-down browsers block them, the page reads the same
without those controls.

### Control the thumbnails

Flow takes the thumbnails once a run is done, from the datasets the run read: one per item, at most 192 pixels across,
and at most 200 per result. A pipeline's `result: max_images:` sets that limit: `0` embeds none, and `-1` every item the
report names. The limit is shared evenly between the findings that name items, and a finding's share between its
tables, rows in order. A finding that needs fewer passes its spare to the rest. So with 200 and four such findings, each
gets 50. A box's thumbnail is cropped from its image with a margin around it. `--no-report-images`,
`DATAEVAL_REPORT_IMAGES=0` and `report_images=False` on `run()` set it to `0`, and then the run reads no item for them.
Only images have thumbnails for now. Any other kind of item is named.

### Print the page

The page prints (or saves as PDF from the browser's print dialog) in the light palette. Before it prints, its script
opens every finding and shows every row a filter hid. With scripts blocked, each finding and panel prints as the reader
left it. Hover cards do not print, and thumbnails print at their own size. In data cleaning, the `outliers` step's
limits tables give each metric's limits and its population's mean and standard deviation, and say `varies` where its
flags' figures differ. Percentiles show only in the hover cards and the JSON.

## Next steps

- [Clean a dataset](../notebooks/data_cleaning.py) — Run a `quality` workflow and read its report.
- [Read the result envelope](read_the_result_envelope.md) — Read the JSON a run writes, finding by finding.
- [Get the numbers behind a finding](get_the_numbers_behind_a_finding.md) — Reach the step outputs and the dataset
  behind a finding.
