# Check and Combine Catalog

A **check** is a step of a custom workflow that judges what evaluators found. It reads their Outputs, and makes
findings: each `ok`, `info` or `warning`, rolled up into the task's health, where a warning counts toward
`--fail-on-warning`. A **combine** reads Outputs, and the Datasets they were computed on, and makes an Output a check
reads. See [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) for how steps chain, and the
[Transform Catalog](transforms.md) for the steps that make Datasets.

The built-in checks are the ones `data-cleaning` runs, whose findings are theirs, `metadata-issues`, which makes
`metadata-triage`'s, and `drift`, which judges `drift-monitoring`'s detectors. See [data-cleaning is this chain](#data-cleaning-is-this-chain).

## At a glance

| Type | Kind | Reads | Makes |
| --- | --- | --- | --- |
| `outlier-rate` | check | `input`: an `outliers` Output | Image Outliers |
| `target-outlier-rate` | check | `input`: an `outliers` Output run with `per_target: true`; `labels`: a `label-health` Output | Target Outliers |
| `classwise-outlier-rate` | check | `input`: a `classwise-outliers` Output | Classwise Outliers |
| `duplicate-rate` | check | `input`: a `duplicates` Output | Duplicates |
| `class-imbalance` | check | `input`: a `label-health` Output | Label Distribution |
| `drift` | check | `input`: a drift evaluator's Output | one finding: the verdict, or the chunks' verdicts |
| `metadata-issues` | check | `input`: a `factor-triage` Output | one finding per kind of issue, Suggested policy, Verified, Recommended policy |
| `classwise-outliers` | combine | `input`: a Dataset; `outliers`: an `outliers` Output computed on it | outliers per class |

## How thresholds work

A check's thresholds are written beside it, in the step entry, like a transform's settings. Each is a percentage or a
ratio, and a finding warns where the measured value passes it. `null` switches a threshold off: the finding is still
made, as `info`. The defaults are `data-cleaning`'s `health_thresholds`.

A check is never skipped because an input produced nothing. Where a step it reads failed or was skipped, it makes one
`info` finding briefed `not assessed`, titled with its `subject` where it takes one and with its own title otherwise,
followed by " by class" where it has `by: class`. Its description names that input and why it holds nothing: "Not
assessed: `count` failed: RuntimeError: …". A check that reads a list on a port that takes one Output runs once
per element, and each finding names its element under `step`, as `imbalance[train]`. The report groups those findings
by the element's key, `train`, in its summary and below it.

A check with `by: class` runs once per class, or per group of classes, of an Output made with the same `by:`, and
rolls the findings up into one, titled with the first's title and " by class". Its brief counts the classes that warn,
as `1/3 classes warn`, and its description names those that warned and those not assessed. See
[Write a custom workflow](../how_to/write_a_custom_workflow.md) for how to set it up.

## Checks

### `outlier-rate`

The share of a Dataset's images with at least one image-level outlier flag. Configured by
{py:class}`~dataeval_flow.steps.checks.OutlierRateConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An `outliers` Output |
| `image` | a percentage, or `null` | `3.0` | Most images, as a percentage of the Dataset, that may be flagged before the finding warns |

With nothing flagged, the finding is `ok`.

### `target-outlier-rate`

The share of a Dataset's boxes with at least one outlier flag. Configured by
{py:class}`~dataeval_flow.steps.checks.TargetOutlierRateConfig`. It makes no finding where nothing was flagged per
box, as on a classification Dataset.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An `outliers` Output run with `per_target: true` |
| `labels` | an address | required | A `label-health` Output on the same Dataset: its label count is the number of boxes |
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
| `input` | an address | required | A `duplicates` Output |
| `exact` | a percentage, or `null` | `0.0` | Most images that may sit in exact-duplicate groups before the finding warns |
| `near` | a percentage, or `null` | `5.0` | Most images that may sit in near-duplicate groups before the finding warns |

### `class-imbalance`

The largest class's label count over the smallest's. Configured by
{py:class}`~dataeval_flow.steps.checks.ClassImbalanceConfig`. It makes no finding where no item has a label, or the
Dataset declares no class. Its title reads "Label/Directory_Name Distribution" where the labels come from file paths.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-health` Output |
| `ratio` | a ratio of at least 1, or `null` | `5.0` | Largest class count over smallest that may hold before the finding warns; an empty class always warns |

### `metadata-issues`

The findings `metadata-triage` makes, from a `factor-triage` Output, in this order:

1. One finding per kind of issue `factor-triage` found. It is a warning where any issue of that kind is blocking,
   meaning the run did less than its configuration asked, and info otherwise.
2. "Suggested policy", holding the stanza to paste under `metadata:`.
3. "Verified", saying what each suggestion recovered, or "Verification failed", a warning, when verification raised.
4. "Recommended policy", opening with the caveat that a policy read from unrepresentative data can mislead, then the
   stanza that pins every factor the policy left unpinned; or "Recommendation failed", a warning, when reading the
   data back raised. Neither where the policy already pins everything.

It has no thresholds. Configured by {py:class}`~dataeval_flow.steps.checks.MetadataIssuesConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `factor-triage` Output |
| `max_examples` | an integer of at least 1 | `20` | Distinct values shown per kind per factor; display only |

### `drift`

Whether a drift detector found drift. Configured by {py:class}`~dataeval_flow.steps.checks.DriftCheckConfig`. Without
chunking, drift is a warning, or `info` where `warn_on_drift` is false. With chunking, the finding warns when the share
of drifted chunks or the longest run of drifted chunks reaches its limit, is `info` when some chunks drifted but
neither does, and is `ok` when no chunk drifted. With both limits `null` it judges nothing, and is `info` whether or
not a chunk drifted.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A drift evaluator's Output |
| `subject` | text, or `null` | `null` | The finding's title, a `not assessed` one's too; unset, the evaluator's title, followed by its entry's name where that differs from its type |
| `warn_on_drift` | true or false | `true` | Unchunked, and per class: whether drift warns, or is `info` |
| `chunk_percent` | a percentage, or `null` | `10.0` | Chunked: the share of drifted chunks at which the finding warns |
| `consecutive_chunks` | an integer of at least 1, or `null` | `3` | Chunked: the longest run of drifted chunks at which the finding warns |

## Combines

### `classwise-outliers`

An Outliers Output pivoted by class: how many of each class's items, or boxes for detection, were flagged, as a count
and a share of the class, most flagged first, and the total. Configured by
{py:class}`~dataeval_flow.steps.combines.ClasswiseOutliersConfig`; makes a
{py:class}`~dataeval_flow.steps.combines.ClasswiseOutliers`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset the outliers were found in; its labels name each item's class |
| `outliers` | an address | required | An `outliers` Output computed on exactly `input`; for a detection Dataset, with `per_target: true` |

On a detection Dataset it refuses outliers not computed per box (`per_target: true`), rather than report none.

The config refuses an `outliers` computed on another Dataset when it loads, as `remove` does.

## data-cleaning is this chain

`data-cleaning` is a preset: its settings expand to a chain of steps, run on the task's one source, `data`. With
`outlier_method: zscore`, `outlier_flags: [pixel, visual]` and the default thresholds, it runs these steps:

```yaml
evaluators:
  - {name: outliers, type: outliers, flags: [pixel, visual], outlier_threshold: zscore, per_target: true}
  - {name: dupes, type: duplicates, merge_near_duplicates: true}
  - {name: labels, type: label-health}

workflows:
  - name: cleaning
    inputs: [data]
    steps:
      - {name: outliers, evaluator: outliers, input: data}
      - {name: labels, evaluator: labels, input: data}
      - {name: by-class, combine: classwise-outliers, input: data, outliers: outliers}
      - {name: dupes, evaluator: dupes, input: data}
      - {name: image-outliers, check: outlier-rate, input: outliers}
      - {name: target-outliers, check: target-outlier-rate, input: outliers, labels: labels}
      - {name: classwise, check: classwise-outlier-rate, input: by-class}
      - {name: duplicates, check: duplicate-rate, input: dupes}
      - {name: imbalance, check: class-imbalance, input: labels}
      - name: clean
        transform: remove
        input: data
        plans:
          dupes: {dup_types: [exact, near], keep: first}
          outliers: {min_flags: 1}
```

Its other settings go to the evaluators: `outlier_threshold`, `outlier_cluster_threshold`,
`outlier_cluster_algorithm` and `outlier_n_clusters` to `outliers`, the `duplicate_*` settings to `dupes`, `metadata`
to `labels`, and `stats` to both `outliers` and `dupes`. Each `health_thresholds` entry is the threshold of the check
that judges it. `clean` removes each image and box with at least one outlier flag, and each exact or near duplicate
but the first of its group.

Its report gives each finding a section, with the evaluators it judged below it: the flagged images and boxes under
Image Outliers, and the duplicate groups under Duplicates. The class counts sit under the first finding that read
`labels`: Target Outliers where any box was flagged, else Label Distribution. A finding that read a step shown already
names the finding it is under, as Classwise Outliers names Image Outliers for the outliers `by-class` counted.
`clean`'s section follows, saying how many images it kept and what each plan named. On MILCO's reference campaigns,
as {doc}`View a report as HTML <../notebooks/view_html_reports>` runs it: "Kept 162 of 261 images. Removed 99 images
and 32 detections: 90 images named by `dupes`, 11 images and 32 detections by `outliers`." Two images were named by
both plans. A Steps table lists every step, what it read, and why it made nothing where it did not.

Run as a step of a custom workflow, `<step>.clean` reads the cleaned Dataset; see
[Workflow types as presets](../concepts/WorkflowsAsChains.md#workflow-types-as-presets).

## data-prioritization is this chain

`data-prioritization` is a preset too: its settings expand to a chain of steps, run on the task's sources, the first
the `reference` and the rest the `pools`. With `cleaning:` set (`outlier_method: zscore`,
`outlier_flags: [pixel, visual]`), `method: knn`, `k: 5` and `n: 200`, it runs these steps:

```yaml
evaluators:
  - {name: rank, type: prioritize, method: knn, k: 5, order: hard_first, policy: difficulty, num_bins: 50, n_init: auto}
  - {name: outliers, type: outliers, flags: [pixel, visual], outlier_threshold: zscore}
  - {name: dupes, type: duplicates, merge_near_duplicates: true}

workflows:
  - name: prioritization
    inputs: [reference, {name: pools, list: true}]
    steps:
      - {name: reference-outliers, evaluator: outliers, input: reference}
      - {name: reference-dupes, evaluator: dupes, input: reference}
      - name: reference-clean
        transform: remove
        input: reference
        plans:
          reference-dupes: {dup_types: [exact, near], keep: first}
          reference-outliers: {min_flags: 1}
      - {name: pool-outliers, evaluator: outliers, input: pools}
      - {name: pool-dupes, evaluator: dupes, input: pools}
      - name: pool-clean
        transform: remove
        input: pools
        plans:
          pool-dupes: {dup_types: [exact, near], keep: first}
          pool-outliers: {min_flags: 1}
      - {name: rank, evaluator: rank, input: [pool-clean, reference-clean]}
      - {name: selected, transform: select, input: pool-clean, ranking: rank, n: 200}
```

Without `cleaning:`, only `rank` and `selected` run, reading `pools` and `reference`. `duplicate_exact_only: true`
makes both plans' `dup_types` `[exact]`. With neither `n` nor `fraction`, `selected` keeps every item
(`fraction: 1.0`). The chain has no checks, so it makes no findings.

## metadata-triage is this chain

`metadata-triage` is a preset too: its settings expand to two steps on the task's one source, `data`. With its
defaults, it runs:

```yaml
evaluators:
  - {name: triage, type: factor-triage}

workflows:
  - name: triage_chain
    inputs: [data]
    steps:
      - {name: triage, evaluator: triage, input: data}
      - {name: issues, check: metadata-issues, input: triage}
```

- `metadata:`, `verify`, `default_bins` and `min_missing_fraction` are `triage`'s settings, and `max_examples` is
  `issues`'.
- Its findings are `issues`': one per kind of issue, then the suggested policy and what verification recovered.
- The chain makes no Dataset, so it declares no output.
- Its result's `metadata_binning` records the encoding `triage` read, which `dataeval-flow encoding` writes out.
