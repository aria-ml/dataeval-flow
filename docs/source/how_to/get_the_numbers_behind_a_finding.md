# Get the numbers behind a finding

The report is a rendering, and the numbers behind it live on the result object. To reach them from Python, read the
output of a step of a chain's result, by step name. [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md)
explains steps and their outputs, and the [Evaluator Catalog](../reference/evaluators.md) lists what each evaluator
returns. You need a finished run, `result = run_task(config, task)`.

## Read the outputs of a quality result

A chain's result, a `quality` or `audit` result among them, is a {py:class}`~dataeval_flow.steps.ChainResult`. Its
`steps` hold each step's output, by step name: an evaluator step's is DataEval's own output, a check's is its findings,
and a transform's is the Dataset it made, a DataEval `View`:

```python
result = run_task(config, task)  # a quality task

outliers = result.steps["outliers"].output  # DataEval's Outliers output
flags = outliers.data()  # one row per flag: its item, its box if any, the metric, its value and the limit crossed
duplicates = result.steps["duplicates"].output  # DataEval's Duplicates output
cleaned = result.steps["clean"].output  # without each flagged image and box, and each duplicate but the first
```

## Read the outputs of a scope result

A `scope` result reads the same way:

```python
result = run_task(config, task)  # a scope task with an extractor

coverage = result.steps["coverage"].output  # DataEval's Coverage output: a row per class, and uncovered_indices
gaps = result.steps["factor-gaps"].output.gaps  # each under-represented class-factor-value combination
counts = result.steps["label-health"].output.data()["label_counts_per_class"]  # every declared class, at 0 where unseen
```

On detection data, `coverage` indexes the `crops` step's items, one per box: `result.steps["crops"].output.item_indices`
and `.target_indices`, indexed by `coverage.uncovered_indices`, give each uncovered crop's image and box.

A step that was skipped, as `coverage` is without an extractor, has no output. Its `reason` says why.

## Read the verdict and the digests of an audit result

An `audit` result also carries its verdict, and each split's digests in its `content-digest` steps. A step that reads a
list of Datasets, as `content-digest-evals` reads the evaluation splits, runs once for each, and its `elements` hold
each run by the split's name:

```python
result = run_task(config, task)  # an audit task over train, val and test

result.verdict.level  # "not-ready", "ready-with-caveats" or "ready"
result.verdict.blocking  # each blocking check's warning no acceptance covers: its check, step, title and brief
result.verdict.warnings  # each other warning no acceptance covers
recorded = result.steps["content-digest-train"].output.data()  # {"content": ..., "metadata": ..., "items": ...}
val = result.steps["content-digest-evals"].elements["val"].output.data()
```

[Gate training on an audit](gate_training_on_an_audit.md) compares those digests with the data a training job reads.

## Read how metadata factors were treated

Bias, balance, diversity, and coverage analyses read factors as codes: a continuous factor cut into intervals, a
categorical one mapped to ordinals. So what a run decided about binning is part of what the result means.
`result.metadata.metadata_binning` records it: per factor, the type, the {term}`level <Metadata Level>` it was binned
at, whether it was binned or digitized, and the observed range and population of every bin.
`result.metadata.diagnostics` carries the library warnings the run raised. Both render in the text report under
"METADATA FACTORS".

The `factor-summary` step's per-factor summaries, in `audit` and `bias`, carry the same shape of information alongside
the values: `level` and `is_binned` per factor, plus an entry of type `dropped`, with its reasons, for each column that
never became a factor at all, such as an identifier or a vector-valued statistic (`histogram`, `percentiles`, `center`)
with no single-column form. `invalid_box` is carried through as a factor; the other hash columns are discarded.

{doc}`configure_metadata_binning` covers how to control any of this.

:::{note}
`level` means two different things in a result, on two different axes. Per-factor metadata summaries report a
{term}`metadata level <Metadata Level>`: `sequence`, `unit`, `track`, or `instance`. Duplicate groups report `item` or
`target`, which is unrelated and unchanged. Classwise outlier pivots report `count_basis` (`image` or `annotation`)
rather than `level`, so the three cannot be read as one.
:::

## Reach the dataset behind a finding

Two more fields are useful for follow-up work and are deliberately not serialized into the envelope:

- `result.dataset` is the resolved, post-view dataset a one-source workflow ran on, for pulling up the images behind a
  finding.
- `result.sources` is, for every chain, `quality` and `audit` among them, and for a workflow that reads several sources,
  a mapping of source name to resolved dataset.

```python
dataset = result.sources["train"]  # a quality task on the source `train`
for item in result.steps["outliers"].output.data()["item_index"].unique().sort():
    image, target, meta = dataset[item]
```

## Next steps

- [Read the result envelope](read_the_result_envelope.md) — Read the same result as JSON.
- [Gate training on an audit](gate_training_on_an_audit.md) — Compare an audit's digests with the data a training job
  reads.
- [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) — Learn how steps, their outputs and their addresses
  fit together.
