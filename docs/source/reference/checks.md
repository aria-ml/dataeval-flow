# Check and Combine Catalog

A **check** is a step of a custom workflow that judges what evaluators found. It reads their Outputs, and makes
findings: each `ok`, `info` or `warning`, rolled up into the task's health, where a warning counts toward
`--fail-on-warning`. A **combine** reads Outputs, and the Datasets they were computed on, and makes an Output a check
reads. See [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) for how steps chain, and the
[Transform Catalog](transforms.md) for the steps that make Datasets.

The built-in checks reproduce `data-cleaning`'s findings. A chain of them gives that workflow's findings, in the
same order, with the same severities.

## At a glance

| Type | Kind | Reads | Makes |
| --- | --- | --- | --- |
| `outlier-rate` | check | `input`: a `quality.outliers` Output | Image Outliers |
| `target-outlier-rate` | check | `input`: a `quality.outliers` Output run with `per_target: true`; `labels`: a `quality.label-health` Output | Target Outliers |
| `classwise-outlier-rate` | check | `input`: a `classwise-outliers` Output | Classwise Outliers |
| `duplicate-rate` | check | `input`: a `quality.duplicates` Output | Duplicates |
| `class-imbalance` | check | `input`: a `quality.label-health` Output | Label Distribution |
| `classwise-outliers` | combine | `input`: a Dataset; `outliers`: a `quality.outliers` Output computed on it | outliers per class |

## How thresholds work

A check's thresholds are written beside it, in the step entry, like a transform's settings. Each is a percentage or a
ratio, and a finding warns where the measured value passes it. `null` switches a threshold off: the finding is still
made, as `info`. The defaults are `data-cleaning`'s `health_thresholds`.

A check is never skipped because an input produced nothing. Where a step it reads failed or was skipped, it makes one
`info` finding in its own name, briefed `not assessed`, whose description names that input and why it holds nothing:
"Not assessed: `count` failed: RuntimeError: …". A check that reads a list on a port that takes one Output runs once
per element, and each finding names its element: `imbalance[train]`.

## Checks

### `outlier-rate`

The share of a Dataset's images with at least one image-level outlier flag. Configured by
{py:class}`~dataeval_flow.steps.checks.OutlierRateConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `quality.outliers` Output |
| `image` | a percentage, or `null` | `3.0` | Most images, as a percentage of the Dataset, that may be flagged before the finding warns |

With nothing flagged, the finding is `ok`.

### `target-outlier-rate`

The share of a Dataset's boxes with at least one outlier flag. Configured by
{py:class}`~dataeval_flow.steps.checks.TargetOutlierRateConfig`. It makes no finding where nothing was flagged per
box, as on a classification Dataset.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `quality.outliers` Output run with `per_target: true` |
| `labels` | an address | required | A `quality.label-health` Output on the same Dataset: its label count is the number of boxes |
| `target` | a percentage, or `null` | `3.0` | Most boxes, as a percentage of all, that may be flagged before the finding warns |

### `classwise-outlier-rate`

The worst class's share of outliers, how many classes pass the limit, and whether all classes together do.
Configured by {py:class}`~dataeval_flow.steps.checks.ClasswiseOutlierRateConfig`. Its evidence is the per-class table.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `classwise-outliers` Output |
| `total` | a percentage, or `null` | `3.0` | Most items or boxes, as a percentage of all, the outliers may take up before the finding warns; each class is counted against it too |

### `duplicate-rate`

The shares of a Dataset's images in exact and in near duplicate groups. Configured by
{py:class}`~dataeval_flow.steps.checks.DuplicateRateConfig`. It makes no finding where there are no duplicate
images.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `quality.duplicates` Output |
| `exact` | a percentage, or `null` | `0.0` | Most images that may sit in exact-duplicate groups before the finding warns |
| `near` | a percentage, or `null` | `5.0` | Most images that may sit in near-duplicate groups before the finding warns |

### `class-imbalance`

The largest class's label count over the smallest's. Configured by
{py:class}`~dataeval_flow.steps.checks.ClassImbalanceConfig`. It makes no finding where no item has a label, or the
Dataset declares no class. Its title reads "Label/Directory_Name Distribution" where the labels come from file paths.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `quality.label-health` Output |
| `ratio` | a ratio of at least 1, or `null` | `5.0` | Largest class count over smallest that may hold before the finding warns; an empty class always warns |

## Combines

### `classwise-outliers`

An Outliers Output pivoted by class: how many of each class's items, or boxes for detection, were flagged, as a count
and a share of the class, most flagged first, and the total. Configured by
{py:class}`~dataeval_flow.steps.combines.ClasswiseOutliersConfig`; makes a
{py:class}`~dataeval_flow.steps.combines.ClasswiseOutliers`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset the outliers were found in; its labels name each item's class |
| `outliers` | an address | required | A `quality.outliers` Output computed on exactly `input`; for a detection Dataset, with `per_target: true` |

On a detection Dataset it refuses outliers not computed per box (`per_target: true`), rather than report none.

The config refuses an `outliers` computed on another Dataset when it loads, as `remove` does.

## data-cleaning as a chain

These steps give `data-cleaning`'s findings, on the same Dataset, with its default thresholds:

```yaml
evaluators:
  - {name: outliers, type: quality.outliers, flags: [pixel, visual], outlier_threshold: zscore, per_target: true}
  - {name: dupes, type: quality.duplicates, merge_near_duplicates: true}
  - {name: labels, type: quality.label-health}

workflows:
  - name: cleaning_report
    inputs: [data]
    steps:
      - {name: outliers, evaluator: outliers, input: data}
      - {name: labels, evaluator: labels, input: data}
      - {name: by_class, combine: classwise-outliers, input: data, outliers: outliers}
      - {name: dupes, evaluator: dupes, input: data}
      - {name: image_outliers, check: outlier-rate, input: outliers}
      - {name: target_outliers, check: target-outlier-rate, input: outliers, labels: labels}
      - {name: classwise, check: classwise-outlier-rate, input: by_class}
      - {name: duplicates, check: duplicate-rate, input: dupes}
      - {name: imbalance, check: class-imbalance, input: labels}
```

The chain's report differs in layout: each evaluator's section holds the evidence, the flagged items and the duplicate
groups, and each check's section its finding.
