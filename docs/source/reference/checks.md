# Check and Combine Catalog

A **check** is a step of a custom workflow that judges what evaluators found. It reads their Outputs, and makes
findings: each `ok`, `info` or `warning`, rolled up into the task's health, where a warning counts toward
`--fail-on-warning`. A **combine**, which the [Combine Catalog](combines.md) lists, reads Outputs, and the Datasets
they were computed on, and makes an Output a check reads. See
[Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) for how steps chain, and the
[Transform Catalog](transforms.md) for the steps that make Datasets.

The built-in checks are the ones `data-cleaning` runs, whose findings are theirs; `metadata-issues`, which makes
`metadata-triage`'s; `drift`, which judges `drift-monitoring`'s detectors; `ood`, which judges `ood-detection`'s
detectors; `stratification` and `uncovered-items`, which judge `data-splitting`'s split and coverage;
`leaf-coverage`, `label-conformance`, `ontology-structure` and `mergeability`, which make `label-space`'s; and
`class-coverage`, `dimensional-completeness`, `factor-coverage-gaps` and `class-shortfall`, which make
`data-coverage`'s with `class-imbalance` and `uncovered-items`; and `class-sufficiency`, `untrained-classes`,
`shortcut-risk`, `leakage`, `eval-coverage` and `distribution-shift`, which judge a set of splits as a pre-training
audit. See the [Preset Catalog](presets.md) for each preset's chain.

## At a glance

| Type | Kind | Reads | Makes |
| --- | --- | --- | --- |
| `image-outliers` | check | `input`: an `outliers` Output | Image Outliers |
| `target-outliers` | check | `input`: an `outliers` Output run with `per_target: true`; `labels`: a `label-health` Output | Target Outliers |
| `classwise-outliers` | check | `input`: a `outliers-by-class` Output | Classwise Outliers |
| `image-duplicates` | check | `input`: a `duplicates` Output | Image Duplicates |
| `class-imbalance` | check | `input`: a `label-health` Output | Class Imbalance |
| `class-sufficiency` | check | `input`: a `label-health` Output over train; `evals`: the evaluation splits' | Class Sufficiency |
| `untrained-classes` | check | `input`: a `label-health` Output over train; `evals`: the evaluation splits' | Untrained Classes |
| `stratification` | check | `input`: a `label-health` Output over the whole; `parts`: the parts'; `shown`: more, not judged | Stratification |
| `uncovered-items` | check | `input`: a `coverage` Output | Uncovered Items |
| `factor-coverage-gaps` | check | `input`: a `factor-gaps` Output | Factor Coverage Gaps |
| `dimensional-completeness` | check | `input`: a `completeness` Output | Dimensional Completeness |
| `class-coverage` | check | `input`: a `coverage` Output | Class Coverage |
| `class-shortfall` | check | `input`: a `representation` Output with no ontology | Class Shortfall |
| `leaf-coverage` | check | `input`: a `representation` Output against a declared ontology | Leaf Coverage |
| `label-conformance` | check | `input`: a `label-reconciliation` Output | Label Conformance |
| `mergeability` | check | `input`: a `label-alignment` Output | Mergeability |
| `ontology-structure` | check | `input`: an `ontology-validation` Output | Ontology Structure |
| `distribution-shift` | check | `input`: a `divergence` Output | Distribution Shift |
| `shortcut-risk` | check | `input`: a `balance` Output | Shortcut Risk |
| `eval-coverage` | check | `input`: an `ood-kneighbors` Output | Eval Coverage |
| `leakage` | check | `duplicates`: `duplicates` Outputs over two splits; `factors`: `factor-leakage` Outputs | Leakage |
| `drift` | check | `input`: a drift evaluator's Output | one finding: the verdict, or the chunks' verdicts |
| `ood-agreement` | check | `input`: an `ood-union` Output | OOD Agreement: the share every detector flagged, and the images one alone flagged |
| `ood` | check | `input`: an OOD evaluator's Output | one finding: the images flagged of those assessed |
| `metadata-issues` | check | `input`: a `factor-triage` Output | one finding per kind of issue, Suggested policy, Verified, Recommended policy |

## How thresholds work

A check's thresholds are written beside it, in the step entry, like a transform's settings. Each is a percentage or a
ratio, and a finding warns where the measured value passes it. A value equal to a bound does not warn: the bound is the
last value that does not. `null` switches a threshold off: the finding is still made, as `info`. The defaults are
`data-cleaning`'s `checks`.
A check with a criterion that has no threshold, such as an unmet share, an ambiguous name or an empty class, keeps
judging it, so its finding can still be `ok` or `warning` when its thresholds are `null`.

A check is never skipped because an input produced nothing. Where a step it reads failed or was skipped, it makes one
`info` finding briefed `not assessed`, titled with its `subject` where it takes one and with its own title otherwise,
followed by " by class" where it has `by: class`. Its description names that input and why it holds nothing: "Not
assessed: `count` failed: RuntimeError: …". A check that reads a list on a port that takes one Output runs once
per element, and each finding names its element under `step`, as `class-imbalance[train]`. The report groups those findings
by the element's key, `train`, in its summary and below it.

A check that has its inputs but cannot assess them raises `StepSkipped(reason)`. The engine records it as not assessed,
with that reason in the step's `not_assessed`, and never as skipped. `class-sufficiency`, `untrained-classes` and
`shortcut-risk` do so, as their sections say.

A check with `by: class` runs once per class, or per group of classes, of an Output made with the same `by:`, and
rolls the findings up into one, titled with the first's title and " by class". Its brief counts the classes that warn,
as `1/3 classes warn`, and its description names those that warned and those not assessed. See
[Write a custom workflow](../how_to/write_a_custom_workflow.md) for how to set it up.

## Checks

### `image-outliers`

The share of a Dataset's images with at least one image-level outlier flag. Configured by
{py:class}`~dataeval_flow.steps.checks.ImageOutliersConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An `outliers` Output |
| `warning` | a percentage, or `null` | `3.0` | Most images, as a percentage of the Dataset, that may be flagged before the finding warns |

With nothing flagged, the finding is `ok`.

### `target-outliers`

The share of a Dataset's boxes with at least one outlier flag. Configured by
{py:class}`~dataeval_flow.steps.checks.TargetOutliersConfig`. It makes no finding where nothing was flagged per
box, as on a classification Dataset.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An `outliers` Output run with `per_target: true` |
| `labels` | an address | required | A `label-health` Output on the same Dataset: its label count is the number of boxes |
| `warning` | a percentage, or `null` | `3.0` | Most boxes, as a percentage of all, that may be flagged before the finding warns |

### `classwise-outliers`

The worst class's share of outliers, how many classes pass the limit, and whether all classes together do.
Configured by {py:class}`~dataeval_flow.steps.checks.ClasswiseOutliersConfig`. Its evidence is the per-class table.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `outliers-by-class` Output |
| `warning` | a percentage, or `null` | `3.0` | Most items or boxes, as a percentage of all, the outliers may take up before the finding warns; each class is counted against it too |

### `factor-coverage-gaps`

Whether class-factor-value combinations are under-represented: a warning past `warning` gaps, `info` up to that many,
`ok` with none, and the gaps as a table, largest deficit first. Configured by
{py:class}`~dataeval_flow.steps.checks.FactorCoverageGapsConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `factor-gaps` Output |
| `warning` | a count, or `null` | `2` | The most under-represented class-factor-value combinations before the finding warns; `null` never warns |

### `image-duplicates`

The shares of a Dataset's images in exact and in near duplicate groups. Configured by
{py:class}`~dataeval_flow.steps.checks.ImageDuplicatesConfig`. It makes no finding where there are no duplicate
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
declared or observed, and lists the images with no labels. Its title reads "Class Imbalance" where
the labels come from file paths.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-health` Output |
| `warning` | a ratio of at least 1, or `null` | `5.0` | Largest class count over smallest that may hold before the finding warns; an empty class warns unless `empty` is `false` |
| `info` | a ratio, or `null` | `null` | A ratio at or under which the finding is ok; must not exceed `warning` |
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

### `uncovered-items`

How much of a Dataset coverage left uncovered, as a share of its items. Judge only `naive` coverage: adaptive
coverage, DataEval's default, marks the sparsest `percent` of the items uncovered by construction, so its share says
nothing about the data. DataEval's naive radius overflows past about 340 embedding dimensions, so `naive` suits
low-dimensional embeddings: with a wide CNN or ONNX extractor, the coverage steps are skipped with "failed:
OverflowError". Configured by {py:class}`~dataeval_flow.steps.checks.UncoveredItemsConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `coverage` Output |
| `warning` | a percentage, or `null` | `10.0` | The percent of items uncovered past which the finding warns |

### `dimensional-completeness`

How much of the embedding space's dimensions the data fills, judged against two bands: the finding warns under
`warning`, informs under `info`, and is `ok` above. The score is rounded to three places first. With both bands `null`
nothing is judged and the finding informs. Configured by
{py:class}`~dataeval_flow.steps.checks.DimensionalCompletenessConfig`.

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

Which class names resolve to exactly one ontology concept. Warns on more unmatched names than `warning`, or on any
ambiguous name; ok otherwise. Configured by {py:class}`~dataeval_flow.steps.checks.LabelConformanceConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-reconciliation` Output |
| `warning` | a count, or `null` | `0` | Unmatched names tolerated; `null` turns it off |

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
nothing, and with both `null` the finding is `info`. It is not assessed (`train holds no labelled class`) where train
holds no labelled class. An empty
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
nothing to compare, and the finding is `info` unless `declared` is true. Where evaluation splits are given but none
holds a labelled class, it is not assessed (`no evaluation split holds a labelled class`).

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-health` Output over train |
| `evals` | an address, a list, or `null` | `null` | The evaluation splits' `label-health` Outputs, a list that may be empty |
| `declared` | `true` or `false` | `false` | Whether a declared class with no labels in train also warns |

### `shortcut-risk`

Whether a metadata factor tells much about the class, which a model could learn instead of the task. Configured by
{py:class}`~dataeval_flow.steps.checks.ShortcutRiskConfig`. It makes one finding, which warns where a factor's mutual
information with the class is past `warning`, and lists the three most informative; a table ranks every
factor. `balance`'s own `class_label` row is not a factor, and where no factor is left it is not assessed
(`no factor to score`). With `warning: null` the finding is `info`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `balance` Output |
| `warning` | 0 to 1, or `null` | `0.1` | The mutual information with the class past which a factor warns; `null` judges nothing |

### `eval-coverage`

How much of an evaluation split lies farther from train than most of train lies from itself. Configured by
{py:class}`~dataeval_flow.steps.checks.EvalCoverageConfig`. The percent flagged is judged as `ood` judges its percent.
The percentile is the `ood-kneighbors` entry's `threshold_perc`, or DataEval's 95 where unset; a split drawn like train
has about 100 minus that percent flagged by construction, so `info: 2.0` suits `threshold_perc: 99`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An `ood-kneighbors` Output fitted on train and run on one evaluation split |
| `warning` | a percentage, or `null` | `10.0` | The percent flagged past which the finding warns |
| `info` | a percentage, or `null` | `2.0` | The percent past which the finding is `info`, at or below which it is `ok` |

### `drift`

Whether a drift detector found drift. Configured by {py:class}`~dataeval_flow.steps.checks.DriftConfig`. Without
chunking, drift is a warning, or `info` where `warn_on_drift` is false. With chunking, the finding warns when the share
of drifted chunks or the longest run of drifted chunks passes its limit, is `info` when some chunks drifted but
neither does, and is `ok` when no chunk drifted. With both limits `null` it judges nothing, and is `info` whether or
not a chunk drifted.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A drift evaluator's Output |
| `subject` | text, or `null` | `null` | The finding's title, a `not assessed` one's too; unset, the evaluator's title, followed by its entry's name where that differs from its type |
| `warn_on_drift` | true or false | `true` | Unchunked, and per class: whether drift warns, or is `info` |
| `chunk_percent` | a percentage, or `null` | `10.0` | Chunked: the share of drifted chunks past which the finding warns |
| `consecutive_chunks` | an integer of at least 1, or `null` | `2` | Chunked: the longest run of drifted chunks past which the finding warns, so 2 warns on three in a row |

### `ood`

How much of a test source an OOD detector flagged, as a percent of the images it assessed. On a detector's
`uncertainty` rows, an image with no detection at the confidence is not assessed, and the brief also counts the
detections flagged. Configured by {py:class}`~dataeval_flow.steps.checks.OODConfig`. The finding warns past
`warning` percent, is `info` past `info` percent, and is `ok` at or below it; a `null` threshold judges nothing at its
level, and with both `null` the finding is `info`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An OOD evaluator's Output |
| `subject` | text, or `null` | `null` | The finding's title; unset, the evaluator's title, followed by its entry's name where that differs from its type |
| `warning` | a percentage, or `null` | `10.0` | The percent of assessed test images flagged past which the finding warns |
| `info` | a percentage, or `null` | `1.0` | The percent past which the finding is `info`, at or below which it is `ok` |

### `ood-agreement`

Whether OOD detectors agree. The aggregate finding judges the percent of assessed test images every detector flagged,
as `ood` judges its percent, and a second, `info` finding counts the images one detector alone flagged, where any
did. Both are titled OOD Agreement, the check's title, and their briefs tell them apart. Configured by {py:class}`~dataeval_flow.steps.checks.OODAgreementConfig`.

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An `ood-union` Output |
| `warning` | a percentage, or `null` | `10.0` | The percent of assessed test images every detector flagged past which the finding warns |
| `info` | a percentage, or `null` | `1.0` | The percent past which the finding is `info`, at or below which it is `ok` |
