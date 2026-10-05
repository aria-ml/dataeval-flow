# Workflows and Evaluators

DataEval Flow runs two kinds of thing, and they answer different questions. An
**evaluator** runs one DataEval evaluator and reports what it determined. A
**workflow** runs several, and its **checks** judge whether what they determined is a
problem.

## Three tiers

Each tier encodes more policy than the one below it.

| | DataEval core functions | Evaluators | Checks |
| --- | --- | --- | --- |
| Encodes | No policy: pure computation | DataEval's policy: a threshold or gate, and a determination made against it | Flow's policy on top: health and readiness, judged from those determinations against each check's thresholds |
| Answers | "What are the hashes? The statistics?" | "Which images are duplicates? Did the test set drift at p < 0.05?" | "Is this data clean enough? Is it ready to train on?" |
| Example | `phash`, `compute_stats` | `balance`, `coverage`, `drift-mmd` | `image-outliers` in `data-cleaning`, `drift` in `drift-monitoring` |
| Output | Raw numbers | Determinations (flags, groups, p-values) and the numbers behind them | Findings, each `ok`, `info` or `warning`, rolled up into a health status |
| Configured under | not exposed | `evaluators:`, run by a task's `evaluator:` | a preset's `checks:`, or a `check:` step written beside the evaluator steps of a custom workflow; the workflow is run by a task's `workflow:` |

## Determinations, not verdicts

An evaluator applies DataEval's threshold and says *what* it found: this group of
images is a near duplicate; this image's brightness is an outlier. It never says
whether that is a problem. That matches DataEval's own framing in
[Acting on Results](https://dataeval.readthedocs.io/en/latest/concepts/ActingOnResults.html):
every output is "a prompt to investigate, not a verdict".

A check adds the verdict. `data-cleaning` runs DataEval's Duplicates and
Outliers, and its checks compare what they found against their
{term}`thresholds <Threshold>`, set under its `checks:`, and make a finding that warns
where a threshold is passed. A check written as a step of a custom workflow judges the
same way; see [How thresholds work](../reference/checks.md#how-thresholds-work).
Health, readiness and warnings come from checks' findings, and so does
`--fail-on-warning`, which a workflow's warnings trip. An evaluator result has no
health status, and `--fail-on-warning` never trips on one.

## When to use which

Use an **evaluator** when you want one algorithm's answer, as DataEval gives it, to
read yourself or to feed your own tooling. For example: "which images in this set
are duplicates?"

Use a **workflow** when you want DataEval Flow to judge the answers for you: to
combine several evaluators, apply thresholds, and fail a pipeline that breaches
them.

Write a **custom workflow** when the analysis is a sequence of your own: a chain of
evaluator steps, workflow-type steps, and transforms that make the Datasets they
read, such as a merged dataset, the same dataset without its duplicates, or its
training split. A chain's health counts the findings of its check steps, which
hold an evaluator's output to thresholds, and those of the workflow types it runs.
[Workflows as Chains of Steps](WorkflowsAsChains.md) explains how a chain works.

## Why both can run DataEval's Duplicates

`duplicates` and `data-cleaning` both call DataEval's Duplicates. Their
cluster-mode results are merged with the statistical ones through the same shared
code, and their `from_stats` calls are tested to agree, so on the same data they
find the same groups. `data-cleaning` adds outlier detection, label statistics,
per-class breakdowns and the checks that judge them on top. `duplicates` stops at the
groups.

## Core functions are not exposed

DataEval's `core` functions compute without deciding anything: they carry no
threshold, so they have no determination to report. DataEval Flow does not expose
them, as evaluators or otherwise. They remain building blocks inside evaluators and
workflows.

See the [Evaluator Catalog](../reference/evaluators.md) for the evaluators this
release provides, and [Run a single evaluator](../how_to/run_a_single_evaluator.md)
to use one.
