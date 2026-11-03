# Workflows and Evaluators

DataEval Flow runs two kinds of thing, and they answer different questions. An **evaluator** runs one DataEval evaluator
and reports what it determined. A **workflow** runs several, and its **checks** judge whether what they determined is a
problem.

## Three tiers

Each tier encodes more policy than the one below it.

| | DataEval core functions | Evaluators | Checks |
| --- | --- | --- | --- |
| Encodes | No policy: pure computation | DataEval's policy: a threshold or gate, and a determination made against it | Flow's policy on top: health and readiness, judged from those determinations against each check's thresholds |
| Answers | "What are the hashes? The statistics?" | "Which images are duplicates? Did the test set drift at p < 0.05?" | "Is this data clean enough? Is it ready to train on?" |
| Example | `phash`, `image_stats` | `balance`, `coverage`, `drift-mmd` | `image-outliers` in `quality`, `drift` in `shift` |
| Output | Raw numbers | Determinations (flags, groups, p-values) and the numbers behind them | Findings, each `ok`, `info` or `warning`, rolled up into a health status |
| Configured under | not exposed | `evaluators:`, run by a task's `evaluator:` | a preset's `checks:`, or a `check:` step written beside the evaluator steps of a custom workflow; the workflow is run by a task's `workflow:` |

## Determinations and judgments

An evaluator applies DataEval's threshold and says what it found. This group of images is a near duplicate. This image's
brightness is an outlier. It never says whether that is a problem. That matches DataEval's own framing in
[Acting on Results](https://dataeval.readthedocs.io/en/latest/concepts/ActingOnResults.html), where every output is "a
prompt to investigate, not a verdict".

A check adds the judgment. [`quality`](../reference/presets.md#quality) runs DataEval's Duplicates and Outliers. Its
checks compare what they found against their {term}`thresholds <Threshold>`, set under its `checks:`, and make a finding
that warns where a threshold is passed. A check written as a step of a custom workflow judges the same way. See
[How thresholds work](../reference/checks.md#how-thresholds-work). Health, readiness and warnings come from checks'
findings. `--fail-on-warning` also depends on them, because a workflow's warnings trip it. An evaluator result has no
health status, and `--fail-on-warning` never trips on one.

A preset may add a verdict over its checks. [`audit`](../reference/presets.md#audit) names its blocking checks under
`blocking:` and its accepted risks under `accepted:`. An unaccepted warning of a blocking check makes the verdict
`not-ready`, and an accepted warning leaves it `ready-with-caveats`. Health still counts every warning, accepted or not,
because it describes the data, and the verdict records the decision. See the
[Preset Catalog](../reference/presets.md#audit).

## When to use which

Use an evaluator when you want one algorithm's answer, as DataEval gives it, to read yourself or to feed your own
tooling. For example, "which images in this set are duplicates?"

Use a workflow when you want DataEval Flow to judge the answers. It combines several evaluators, applies thresholds, and
fails a pipeline that breaches them.

Write a custom workflow when the analysis is a sequence of your own. It is a chain of evaluator steps, workflow-type
steps, and transforms that make the Datasets they read, such as a merged dataset, the same dataset without its
duplicates, or its training split. A chain's health counts the findings of its check steps, which hold an evaluator's
output to thresholds, and those of the workflow types it runs. [Workflows as Chains of Steps](WorkflowsAsChains.md)
explains how a chain works.

## Why `duplicates` and `quality` find the same groups

`duplicates` and `quality` both call DataEval's Duplicates. Their cluster-mode results are merged with the statistical
ones through the same shared code, and their `from_stats` calls are tested to agree, so on the same data they find the
same groups. `quality` adds outlier detection, label statistics, per-class breakdowns and the checks that judge them on
top. `duplicates` stops at the groups.

## Core functions are not exposed

DataEval's `core` functions compute without deciding anything. They carry no threshold, so they have no determination to
report. DataEval Flow does not expose them, as evaluators or otherwise. They remain building blocks inside evaluators
and workflows.

## Related concept pages

- [Workflows as Chains of Steps](WorkflowsAsChains.md): How a chain of steps works.
- [Data Quality and Cleaning](DataQualityAndCleaning.md): The `quality` workflow, whose checks judge what DataEval's
  Duplicates and Outliers find.

## See this in practice

### How-to guides

- [Run a single evaluator](../how_to/run_a_single_evaluator.md): Run one DataEval evaluator, such as finding duplicates,
  and read its output with no health status.
- [Chain steps into a workflow of your own](../how_to/write_a_custom_workflow.md): Conform and merge two datasets,
  remove their duplicates, export the result, and check what is left, as one chain of steps.

### Reference

- [Evaluator Catalog](../reference/evaluators.md): The evaluators this release provides.

## References

1. DataEval, [Acting on Results](https://dataeval.readthedocs.io/en/latest/concepts/ActingOnResults.html). The framing
   of every output as a prompt to investigate, not a verdict.
