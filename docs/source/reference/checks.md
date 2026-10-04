# Check and Combine Catalog

A **check** is a step of a custom workflow that judges what evaluators found. It reads their Outputs, and makes
findings: each `ok`, `info` or `warning`, rolled up into the task's health, where a warning counts toward
`--fail-on-warning`. A **combine** reads Outputs, and the Datasets they were computed on, and makes an Output a check
reads. See [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) for how steps chain, and the
[Transform Catalog](transforms.md) for the steps that make Datasets.

The built-in checks are the ones `data-cleaning` runs, whose findings are theirs; `metadata-issues`, which makes
`metadata-triage`'s; `drift`, which judges `drift-monitoring`'s detectors; `ood`, which judges `ood-detection`'s
detectors; `stratification` and `uncovered-rate`, which judge `data-splitting`'s split and coverage;
`leaf-coverage`, `label-conformance`, `ontology-structure` and `mergeability`, which make `label-space`'s; and
`class-coverage`, `completeness-score`, `coverage-gaps` and `class-shortfall`, which make `data-coverage`'s with
`class-imbalance` and `uncovered-rate`. See [data-cleaning is this chain](#data-cleaning-is-this-chain).

## At a glance

| Type | Kind | Reads | Makes |
| --- | --- | --- | --- |
| `outlier-rate` | check | `input`: an `outliers` Output | Image Outliers |
| `target-outlier-rate` | check | `input`: an `outliers` Output run with `per_target: true`; `labels`: a `label-health` Output | Target Outliers |
| `classwise-outlier-rate` | check | `input`: a `classwise-outliers` Output | Classwise Outliers |
| `duplicate-rate` | check | `input`: a `duplicates` Output | Duplicates |
| `class-imbalance` | check | `input`: a `label-health` Output | Label Distribution |
| `class-sufficiency` | check | `input`: a `label-health` Output over train; `evals`: the evaluation splits' | Class Sufficiency |
| `untrained-classes` | check | `input`: a `label-health` Output over train; `evals`: the evaluation splits' | Untrained Classes |
| `stratification` | check | `input`: a `label-health` Output over the whole; `parts`: the parts'; `shown`: more, not judged | Stratification |
| `uncovered-rate` | check | `input`: a `coverage` Output | Uncovered Rate |
| `coverage-gaps` | check | `input`: a `factor-gaps` Output | Metadata Coverage Gaps |
| `completeness-score` | check | `input`: a `completeness` Output | Dimensional Completeness |
| `class-coverage` | check | `input`: a `coverage` Output | Embedding Coverage |
| `class-shortfall` | check | `input`: a `representation` Output with no ontology | Class Balance Worklist |
| `leaf-coverage` | check | `input`: a `representation` Output against a declared ontology | Label Space Coverage |
| `label-conformance` | check | `input`: a `label-reconciliation` Output | Label Conformance |
| `mergeability` | check | `input`: a `label-alignment` Output | Label Alignment |
| `ontology-structure` | check | `input`: an `ontology-validation` Output | Ontology Structure |
| `distribution-shift` | check | `input`: a `divergence` Output | Distribution Shift |
| `leakage` | check | `duplicates`: `duplicates` Outputs over two splits; `factors`: `factor-leakage` Outputs | Leakage |
| `drift` | check | `input`: a drift evaluator's Output | one finding: the verdict, or the chunks' verdicts |
| `ood-agreement` | check | `input`: an `ood-union` Output | Aggregate OOD (all detectors agree), Unique OOD Samples |
| `ood` | check | `input`: an OOD evaluator's Output | one finding: the images flagged of those assessed |
| `metadata-issues` | check | `input`: a `factor-triage` Output | one finding per kind of issue, Suggested policy, Verified, Recommended policy |
| `factor-gaps` | combine | `input`: a Dataset; `balance`: a `balance` Output computed on it | each factor's MI with the class, and the under-represented combinations |
| `classwise-outliers` | combine | `input`: a Dataset; `outliers`: an `outliers` Output computed on it | outliers per class |
| `ood-union` | combine | `input`: the OOD Outputs of one comparison of a test source with a reference | each flagged image as mutual, partial or unique, with its agreement score |
| `factor-predictors` | combine | `ood`: an `ood-union` or OOD Output; `reference`, `input`: the Datasets it was computed on | each factor's association with being flagged |
| `factor-deviation` | combine | the same | the factors setting each of the most out-of-distribution agreed images apart |

## How thresholds work

A check's thresholds are written beside it, in the step entry, like a transform's settings. Each is a percentage or a
ratio, and a finding warns where the measured value passes it. `null` switches a threshold off: the finding is still
made, as `info`. The defaults are `data-cleaning`'s `health_thresholds`.
A check with a criterion that has no threshold, such as an unmet share, an ambiguous name or an empty class, keeps
judging it, so its finding can still be `ok` or `warning` when its thresholds are `null`.

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

### `coverage-gaps`

Whether class-factor-value combinations are under-represented: a warning at `count` gaps or more, `info` with fewer,
`ok` with none, and the gaps as a table, largest deficit first. Configured by
{py:class}`~dataeval_flow.steps.checks.CoverageGapsConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `factor-gaps` Output |
| `count` | a count, or `null` | `3` | The number of gaps at which the finding warns, this many or more; `null` judges nothing |

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

The largest class's label count over the smallest's, taken over the classes with labels; a class with none is named and
warns unless `empty` is `false`. Configured by
{py:class}`~dataeval_flow.steps.checks.ClassImbalanceConfig`. It makes a finding whenever the Dataset has classes,
declared or observed, and lists the images with no labels. Its title reads "Label/Directory_Name Distribution" where
the labels come from file paths.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-health` Output |
| `ratio` | a ratio of at least 1, or `null` | `5.0` | Largest class count over smallest that may hold before the finding warns; an empty class warns unless `empty` is `false` |
| `info` | a ratio, or `null` | `null` | A ratio at or under which the finding is ok; must not exceed `ratio` |
| `empty` | `true` or `false` | `true` | Whether a declared class with no labels warns; `false` leaves it to `untrained-classes` and `class-sufficiency` |

### `stratification`

How far each part's class shares stray from the whole's: for each class and part, the gap between the class's share
of the part's labels and of the whole's, in percentage points. The largest, rounded to one place, is judged; the
table of counts across the parts is its evidence. Run once per fold over `kfold`'s lists. Configured by
{py:class}`~dataeval_flow.steps.checks.StratificationConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-health` Output over the whole Dataset the parts were split from |
| `parts` | one address or several | required | The parts' `label-health` Outputs, each judged |
| `shown` | one address or several, or `null` | `null` | `label-health` Outputs shown in the table but not judged, such as a rebalanced train |
| `info` | percentage points, or `null` | `2.0` | The largest deviation above which the finding is `info`; `null` has no `info` band |
| `warning` | percentage points, or `null` | `10.0` | The largest deviation above which the finding warns; `null` never warns |

### `uncovered-rate`

How much of a Dataset coverage left uncovered, as a share of its items. Judge only `naive` coverage: adaptive
coverage, DataEval's default, marks the sparsest `percent` of the items uncovered by construction, so its share says
nothing about the data. DataEval's naive radius overflows past about 340 embedding dimensions, so `naive` suits
low-dimensional embeddings: with a wide CNN or ONNX extractor, the coverage steps are skipped with "failed:
OverflowError". Configured by {py:class}`~dataeval_flow.steps.checks.UncoveredRateConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `coverage` Output |
| `rate` | a percentage, or `null` | `10.0` | The percent of items uncovered past which the finding warns |

### `completeness-score`

How much of the embedding space's dimensions the data fills, judged against two bands: the finding warns under
`warning`, informs under `info`, and is `ok` above. The score is rounded to three places first. With both bands `null`
nothing is judged and the finding informs. Configured by
{py:class}`~dataeval_flow.steps.checks.CompletenessScoreConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `completeness` Output |
| `warning` | a score from 0 to 1, or `null` | `0.5` | The score under which the finding warns; must not exceed `info` |
| `info` | a score from 0 to 1, or `null` | `0.8` | The score under which the finding informs |

### `class-coverage`

Which assessable classes `coverage` found clustered, one-dimensional or padded with near-duplicates, and how many items
it left uncovered. Warns on any flagged class; informs while any item is uncovered; ok otherwise. On detection crops it
notes the crops counted and the detections dropped. Configured by
{py:class}`~dataeval_flow.steps.checks.ClassCoverageConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `coverage` Output |
| `dispersion` | a number, or `null` | `0.5` | The dispersion under which a class is clustered; `null` turns it off |
| `isotropy` | a number, or `null` | `0.5` | The isotropy under which a class is one-dimensional; `null` turns it off |
| `near_duplicates` | a fraction, or `null` | `0.1` | The near-duplicate share over which a class is padded; `null` turns it off |

### `class-shortfall`

The classes short of an even spread over the classes the Dataset declares, and what each lacks. Warns on an unmet
minimum share (`expected`); informs while any class is short; ok otherwise. Configured by
{py:class}`~dataeval_flow.steps.checks.ClassShortfallConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `representation` Output, computed with no ontology |

### `leaf-coverage`

How much of an ontology's sanctioned leaves the Dataset has examples of, what to acquire for an even spread, the
wholly empty branches, and the asserted minimum shares (`expected`) not met. Warns on an unmet share, on leaf
coverage under `coverage`, or on more empty branches than `empty_branches`; informs while anything remains to
acquire; ok otherwise. Configured by {py:class}`~dataeval_flow.steps.checks.LeafCoverageConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `representation` Output, against a declared ontology |
| `coverage` | a fraction, or `null` | `0.9` | The least share of leaves with examples; `null` turns it off |
| `empty_branches` | a count, or `null` | `0` | Wholly empty branches tolerated; `null` turns it off |

### `label-conformance`

Which class names resolve to exactly one ontology concept. Warns on more unmatched names than `unmatched`, or on any
ambiguous name; ok otherwise. Configured by {py:class}`~dataeval_flow.steps.checks.LabelConformanceConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-reconciliation` Output |
| `unmatched` | a count, or `null` | `0` | Unmatched names tolerated; `null` turns it off |

### `mergeability`

Whether a Dataset's classes carry over to an ontology's vocabulary, with the `Relabel` stanza to paste into a view
that conforms it. Lossless is ok; lossy, where two classes collapse into one concept, informs; partial, where
`Relabel` would drop a class, warns. A target label several concepts share always warns: the stanza cannot be used
until the ontology is fixed. Configured by {py:class}`~dataeval_flow.steps.checks.MergeabilityConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-alignment` Output |

### `ontology-structure`

An ontology's size, depth and structural observations. Only a label several concepts share warns: it is what makes
reconciliation ambiguous. The rest are facts, so the finding informs. Configured by
{py:class}`~dataeval_flow.steps.checks.OntologyStructureConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An `ontology-validation` Output |

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

### `distribution-shift`

How far apart two sources' embeddings sit. Configured by
{py:class}`~dataeval_flow.steps.checks.DistributionShiftConfig`. The finding warns above `warning`, is `info` above
`info`, and is `ok` at or below both, as data-analysis banded it: high, moderate or low divergence. A `null` limit
judges nothing at its level, and with both `null` the finding is `info`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `divergence` Output |
| `warning` | a number from 0 to 1, or `null` | `0.5` | The divergence above which the finding warns; `null` never warns |
| `info` | a number from 0 to 1, or `null` | 0.4 times `warning` | The divergence above which the finding is `info`, at or below which it is `ok`; `null` has no `info` band; must not exceed `warning` |

### `leakage`

Whether items or group values sit in two splits at once. Configured by
{py:class}`~dataeval_flow.steps.checks.LeakageConfig`. It counts the items in duplicate groups that have members in two
splits, exact and near apart, and the values of a `factor-leakage` factor that both splits of a pair hold. It makes one
finding, which warns where a count passes its limit and lists each pair's groups and values. A `null` limit judges
nothing, and with all three `null` the finding is `info`. It is not assessed where no `duplicates` list holds an
element, as there is then no pair of splits; an empty or failed `factors` list leaves the duplicates judged.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `duplicates` | an address, or a list | required | `duplicates` Outputs over two sources: train with each evaluation split, and evaluation pairs |
| `factors` | an address, a list, or `null` | `null` | `factor-leakage` Outputs over the same pairs; unset judges duplicates alone |
| `exact` | an integer of at least 0, or `null` | `0` | Most items in exact-duplicate groups spanning two splits before the finding warns |
| `near` | an integer of at least 0, or `null` | `0` | The same for near-duplicate groups |
| `groups` | an integer of at least 0, or `null` | `0` | Most group values held by both splits of a pair before the finding warns |

### `class-sufficiency`

Whether each class has enough labels to learn and to evaluate. Configured by
{py:class}`~dataeval_flow.steps.checks.ClassSufficiencyConfig`. It judges the classes train holds: each needs `train`
labels in train and `eval` in every evaluation split, a class the split lacks included. It makes one finding, which
warns where a class falls short and tabulates each class's count in train and in each split. A `null` limit judges
nothing, and with both `null` the finding is `info`. It is not assessed where train holds no labelled class. An empty
`evals` list leaves train judged alone.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-health` Output over train |
| `evals` | an address, a list, or `null` | `null` | The evaluation splits' `label-health` Outputs, a list that may be empty; unset judges train alone |
| `train` | an integer of at least 0, or `null` | `20` | The fewest labels each class train holds needs in train |
| `eval` | an integer of at least 0, or `null` | `30` | The fewest labels each class train holds needs in each evaluation split; at 30, a per-class metric's 95% interval is about ±18 points |

### `untrained-classes`

Whether an evaluation split holds a class train lacks. Configured by
{py:class}`~dataeval_flow.steps.checks.UntrainedClassesConfig`. It makes one finding, which warns where a class has
labels in an evaluation split and none in train, and lists each such class with the splits that hold it. A declared
class with labels in no split is listed, and warns only with `declared: true`. With no evaluation split there is
nothing to compare, and the finding is `info` unless `declared` is true.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-health` Output over train |
| `evals` | an address, a list, or `null` | `null` | The evaluation splits' `label-health` Outputs, a list that may be empty |
| `declared` | `true` or `false` | `false` | Whether a declared class with no labels in train also warns |

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

### `ood`

How much of a test source an OOD detector flagged, as a percent of the images it assessed. On a detector's
`uncertainty` rows, an image with no detection at the confidence is not assessed, and the brief also counts the
detections flagged. Configured by {py:class}`~dataeval_flow.steps.checks.OODCheckConfig`. The finding warns from
`warning` percent, is `info` from `info` percent, and is `ok` below both; a `null` threshold judges nothing at its
level, and with both `null` the finding is `info`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An OOD evaluator's Output |
| `subject` | text, or `null` | `null` | The finding's title; unset, the evaluator's title, followed by its entry's name where that differs from its type |
| `warning` | a percentage, or `null` | `10.0` | The percent of assessed test images flagged at which the finding warns |
| `info` | a percentage, or `null` | `1.0` | The percent at which the finding is `info`, below which it is `ok` |

### `ood-agreement`

Whether OOD detectors agree. The aggregate finding judges the percent of assessed test images every detector flagged,
as `ood` judges its percent, and a second, `info` finding counts the images one detector alone flagged, where any
did. Configured by {py:class}`~dataeval_flow.steps.checks.OODAgreementConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An `ood-union` Output |
| `warning` | a percentage, or `null` | `10.0` | The percent of assessed test images every detector flagged at which the finding warns |
| `info` | a percentage, or `null` | `1.0` | The percent at which the finding is `info`, below which it is `ok` |

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

### `ood-union`

The OOD Outputs of one test source's comparison with one reference, combined. Each flagged image falls in one group:
flagged by every detector, by more than one but not every one (partial), or by one alone. Its agreement score is the
mean, over the detectors that scored it, of its score over the detector's threshold, which is derived from the
detector's flags. A detector whose derived threshold is not positive is left out, and the section names it. The
section pictures each flagged image once, most out of distribution first. Load refuses Outputs computed on different
Datasets. Configured by {py:class}`~dataeval_flow.steps.combines.OODUnionConfig`; makes an
{py:class}`~dataeval_flow.steps.combines.OODUnion`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address, or a list of them | required | Each detector's OOD Output, every one computed on the same reference and test source |

### `factor-gaps`

The class-factor-value combinations under-represented among the factors Balance ties to the class. It reads the mutual
information of a `balance` Output, runs no Balance of its own, and searches the factors at or over `mi_threshold`. A
combination is a gap where its count is under `min_representation` while its expected count, from the factor's overall
spread, is over it. The section ranks each factor's mutual information with the class. Configured by
{py:class}`~dataeval_flow.steps.combines.FactorGapsConfig`; makes a
{py:class}`~dataeval_flow.steps.combines.FactorGapsOutput`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset whose Metadata the gaps are counted in |
| `balance` | an address | required | A `balance` Output computed on exactly `input` |
| `mi_threshold` | a number | `0.1` | The least mutual information with the class a factor needs to be searched |
| `min_representation` | a count | `5` | A combination is a gap where its count is under this while its expected count is over it |
| `metadata` | a policy name, or `null` | `null` | The metadata policy the factors are read under; it should be the one `balance` read under |

The config refuses a `balance` computed on another Dataset when it loads, as `classwise-outliers` does.

### `factor-predictors`

How strongly each metadata factor goes with being flagged: DataEval's `factor_predictors`, normalized mutual
information from 0 to 1, strongest first, over the test images the detectors assessed. Factors are the item-level
metadata factors, without `id`, with `class_label` where there is one label per item, and the per-image statistics
named `f_<statistic>`; a factor counts where both Datasets have it, numeric, one-dimensional, finite in both, and not
constant in the test. Where a Dataset's metadata or statistics cannot be read, the rest is read without it, and the
section says so. Load refuses an `ood` Output computed on other Datasets than `reference` and `input`. Configured by
{py:class}`~dataeval_flow.steps.combines.FactorPredictorsConfig`; makes a
{py:class}`~dataeval_flow.steps.combines.FactorPredictors`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `ood` | an address | required | An `ood-union` Output, or one OOD evaluator's Output, computed on `reference` and `input` |
| `reference` | an address | required | The reference Dataset the detectors fitted on |
| `input` | an address | required | The test Dataset whose images were flagged |
| `metadata` | a policy name, or `null` | `null` | The metadata policy the factors are read under |
| `stats` | a policy name, or `null` | `null` | The stats policy the statistics are measured under; unset, every statistic |

### `factor-deviation`

The factors that set each of the most out-of-distribution agreed images apart from the reference: DataEval's
`factor_deviation`, each factor's scaled distance from the reference's median, most deviating first. It reads the
factors `factor-predictors` reads. Configured by {py:class}`~dataeval_flow.steps.combines.FactorDeviationConfig`; makes
a {py:class}`~dataeval_flow.steps.combines.FactorDeviations`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `ood` | an address | required | An `ood-union` Output, or one OOD evaluator's Output, computed on `reference` and `input` |
| `reference` | an address | required | The reference Dataset the detectors fitted on |
| `input` | an address | required | The test Dataset whose images were flagged |
| `max_items` | an integer of at least 1 | `50` | The most out-of-distribution agreed images explained, at most |
| `metadata` | a policy name, or `null` | `null` | The metadata policy the factors are read under |
| `stats` | a policy name, or `null` | `null` | The stats policy the statistics are measured under; unset, every statistic |

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
