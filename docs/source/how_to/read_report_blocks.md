# Read a finding's report blocks

Each {term}`finding <Finding>` in a result holds its evidence as typed report blocks, the same blocks the text and HTML
reports draw. To show a result in your own viewer, read each finding's keys and its blocks from the JSON of a run.
[Read the result envelope](read_the_result_envelope.md) shows where the findings sit in the envelope.

## Read a finding's keys

Each finding, in a workflow's `report.findings` or a chain's `findings`, has five keys, and a chain's names its step
under a sixth, `step`:

| Key | Holds |
| --- | --- |
| `title` | A short label: the finding's summary line and the heading of its detail. |
| `severity` | `ok`, `info` or `warning`. |
| `brief` | The value on the summary line, or `null`. |
| `description` | A sentence or two of plain prose that leads the detail, or `null` where the brief says it all. |
| `blocks` | The evidence: report blocks, in reading order. |

A `bias` run on the MILCO reference campaigns of {doc}`View a report as HTML <../notebooks/view_html_reports>` judges
their class balance. Its Class Imbalance finding, with `info: 2` set:

```json
{"severity": "ok", "title": "Class Imbalance", "brief": "2 classes, 261 items, imbalance 1.8:1",
 "description": null,
 "blocks": [
   {"type": "table",
    "columns": [{"key": "name", "header": "Class"},
                {"key": "value", "header": "Count"},
                {"key": "value", "kind": "bar"}],
    "rows": [{"name": "MILCO", "value": 319}, {"name": "NOMBO", "value": 173}]},
   {"type": "paragraph", "text": "Labels annotations"},
   {"type": "paragraph", "text": "Imbalance ratio: 1.8 (max/min)"}],
 "step": "class-imbalance"}
```

## Read a block

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
| `markers` (`[]`) | A bar column's labeled reference values, `[name, value]`, such as drift thresholds. |
| `in_text` (`true`) | Whether the text report draws the column; HTML always does. A chain's Steps table sets it `false` on its Title column. |

A summary item's `group`, a summary's `failed` and a column's `in_text` are left out of the JSON at their defaults, like
every other field, so a result that uses none of them reads as it did before they were added.

A table's `preview` says how many rows a renderer with little room, such as the text report, shows before a line
counting the rest. `null` shows every row. A table of items, such as flagged images, previews 10 of at most 500 rows,
with a paragraph naming the rest. A pipeline's `result: preview_rows:` and `result: max_rows:` change those, and `-1`
lifts either.

A cell is a string, number, boolean or `null`. In a `stacked` or `sparkline` column it is a list of numbers, in a
`flags` column a list of flags, and in an `image` column an item reference, or a list of them for a group such as
duplicates. A number stays a number, and its column's `format` says how it displays, so a reader can sort and chart it.
A cell is a string only when it combines values, such as `"12 (30%)"`.

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

A figure is unknown when the run did not record it, as a DataEval that predates these figures does not. A flag whose
`bound` is `null` had its `direction` go unrecorded too, so do not read it. The reports show such a flag as its value
alone, and the run logs a warning to upgrade DataEval.

The reports show a flag as its value against its bound, `>` for an upper limit and `<` for a lower, and list a cell's
flags by name. They do not rank one flag above another: a limit may be a plain value rather than a percentile, and a
value further past its limit is not a worse item for it.

Flow defines the block types, and a later version may add one. A reader that meets a `type` it does not know should show
a one-line placeholder naming it and carry on, rather than fail, so an older reader keeps working on a newer result.

## Read the thumbnails

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

An item named in a cell may have no asset: thumbnails were turned off, the item was past a cap, or it could not be read.
Show its name instead, as the reports do. A later version may make other kinds of preview, so show the name too for a
`media_type` you cannot draw.

## Next steps

- [Read the result envelope](read_the_result_envelope.md) — Read the envelope that holds the findings.
- [Read evaluation outputs](read_evaluation_outputs.md) — Read the reports these blocks draw.
