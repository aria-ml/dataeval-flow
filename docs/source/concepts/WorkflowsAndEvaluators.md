# Workflows and Evaluators

DataEval Flow runs two kinds of thing, and they answer different questions. An
**evaluator** runs one DataEval evaluator and reports what it determined. A
**workflow** runs several, and judges whether what they determined is a problem.

## Three tiers

Each tier encodes more policy than the one below it.

| | DataEval core functions | Evaluators | Workflows |
| --- | --- | --- | --- |
| Encodes | No policy: pure computation | DataEval's policy: a threshold or gate, and a determination made against it | Flow's policy on top: health and readiness, judged from those determinations |
| Answers | "What are the hashes? The statistics?" | "Which images are duplicates? Did the test set drift at p < 0.05?" | "Is this data clean enough? Is it ready to train on?" |
| Example | `phash`, `compute_stats` | `quality.duplicates`, `quality.outliers` | `data-cleaning`, `drift-monitoring` |
| Output | Raw numbers | Determinations (flags, groups, p-values) and the numbers behind them | Findings with a health status |
| Configured under | not exposed | `evaluators:`, run by a task's `evaluator:` | `workflows:`, run by a task's `workflow:` |

## Determinations, not verdicts

An evaluator applies DataEval's threshold and says *what* it found: this group of
images is a near duplicate; this image's brightness is an outlier. It never says
whether that is a problem. That matches DataEval's own framing in
[Acting on Results](https://dataeval.readthedocs.io/en/latest/concepts/ActingOnResults.html):
every output is "a prompt to investigate, not a verdict".

A workflow adds the verdict. `data-cleaning` runs DataEval's Duplicates and
Outliers, compares what they found against its
[health thresholds](../reference/glossary.md), and marks a finding as a warning
when a threshold is breached. Health, readiness, warnings and `--fail-on-warning`
belong to workflows only. An evaluator result has no health status, and
`--fail-on-warning` never trips on one.

## When to use which

Use an **evaluator** when you want one algorithm's answer, as DataEval gives it, to
read yourself or to feed your own tooling. For example: "which images in this set
are duplicates?"

Use a **workflow** when you want DataEval Flow to judge the answers for you: to
combine several evaluators, apply thresholds, and fail a pipeline that breaches
them.

## Why both can run DataEval's Duplicates

`quality.duplicates` and `data-cleaning` both call DataEval's Duplicates. Their
cluster-mode results are merged with the statistical ones through the same shared
code, and their `from_stats` calls are tested to agree, so on the same data they
find the same groups. `data-cleaning` adds outlier detection, label statistics,
per-class breakdowns and a health verdict on top. `quality.duplicates` stops at the
groups.

## Core functions are not exposed

DataEval's `core` functions compute without deciding anything: they carry no
threshold, so they have no determination to report. DataEval Flow does not expose
them, as evaluators or otherwise. They remain building blocks inside evaluators and
workflows.

See the [Evaluator Catalog](../reference/evaluators.md) for the evaluators this
release provides, and [Run a single evaluator](../how_to/run_a_single_evaluator.md)
to use one.
