# Check Catalog

A **check** is a step, in a preset's chain or a custom workflow, that judges what evaluators found. It reads their
Outputs, and makes findings: each `ok`, `info` or `warning`, rolled up into the task's health, where a warning counts
toward `--fail-on-warning`. A **combine**, which the [Combine Catalog](combines.md) lists, reads Outputs, and the
Datasets they were computed on, and makes an Output a check reads. See [Workflows as Chains of
Steps](../concepts/WorkflowsAsChains.md) for how steps chain, and the [Transform Catalog](transforms.md) for the steps
that make Datasets.

Each entry's **Used in** names the presets that run the check; where it names none, chain the check in a [workflow of
your own](../how_to/write_a_custom_workflow.md). See the [Preset Catalog](presets.md) for each preset's chain.
[`audit`](presets.md#audit) runs up to sixteen of these checks to audit a set of splits before training, among them
`class-sufficiency`, `untrained-classes`, `shortcut-risk`, `leakage`, `eval-coverage` and `distribution-shift`; its
chain table lists them all. [Check a set of
splits](../how_to/write_a_custom_workflow.md#11-check-a-set-of-splits) runs `audit` as a step after `data-splitting`
and gives every one of these checks to the splits; chain the checks yourself only to audit one fold or a subset. Each
example assumes the pipeline defines
`datasets:`, the sources `train`, `test`, `validation`, `operational`, `labeled` and `unlabeled`, and the extractor
`bovw_ext`, as [Evaluator recipes](../how_to/evaluator_recipes.md) does.

## At a glance

| Type | Reads | Makes |
| --- | --- | --- |
| `image-outliers` | `input`: an `outliers` Output | Image Outliers |
| `target-outliers` | `input`: an `outliers` Output run with `per_target: true`; `labels`: a `label-health` Output | Target Outliers |
| `classwise-outliers` | `input`: a `outliers-by-class` Output | Classwise Outliers |
| `image-duplicates` | `input`: a `duplicates` Output | Image Duplicates |
| `class-imbalance` | `input`: a `label-health` Output | Class Imbalance |
| `class-sufficiency` | `input`: a `label-health` Output over train; `evals`: the evaluation splits' | Class Sufficiency |
| `untrained-classes` | `input`: a `label-health` Output over train; `evals`: the evaluation splits' | Untrained Classes |
| `leaf-coverage` | `input`: a `representation` Output against a declared ontology | Leaf Coverage |
| `label-conformance` | `input`: a `label-reconciliation` Output | Label Conformance |
| `mergeability` | `input`: a `label-alignment` Output | Mergeability |
| `ontology-structure` | `input`: an `ontology-validation` Output | Ontology Structure |
| `class-coverage` | `input`: a `coverage` Output | Class Coverage |
| `uncovered-items` | `input`: a `coverage` Output | Uncovered Items |
| `dimensional-completeness` | `input`: a `completeness` Output | Dimensional Completeness |
| `factor-coverage-gaps` | `input`: a `factor-gaps` Output | Factor Coverage Gaps |
| `class-shortfall` | `input`: a `representation` Output with no ontology | Class Shortfall |
| `shortcut-risk` | `input`: a `balance` Output | Shortcut Risk |
| `factor-parity` | `input`: a `parity` Output | Factor Parity |
| `stratification` | `input`: a `label-health` Output over the whole; `parts`: the parts'; `shown`: more, not judged | Stratification |
| `leakage` | `duplicates`: `duplicates` Outputs over two splits; `factors`: `factor-leakage` Outputs | Leakage |
| `distribution-shift` | `input`: a `divergence` Output | Distribution Shift |
| `eval-coverage` | `input`: an OOD evaluator's Output | Eval Coverage |
| `drift` | `input`: a drift evaluator's Output | one finding: the verdict, or the chunks' verdicts |
| `ood` | `input`: an OOD evaluator's Output | one finding: the images flagged of those assessed |
| `ood-agreement` | `input`: an `ood-union` Output | OOD Agreement: the share every detector flagged, and the images one alone flagged |
| `metadata-issues` | `input`: a `factor-triage` Output | one finding per kind of issue, Suggested policy, Verified, Recommended policy |

## How thresholds work

A check's thresholds are written beside it, in the step entry, like a transform's settings. Each is in the unit its
row names: a percentage, a ratio, a count, a fraction, a score, percentage points or mutual information. A finding
warns where the measured value passes it, and the glossary's {term}`Severity` entry defines `ok`, `info` and
`warning`. A value equal to a bound does not warn: the bound is the last value that does not. `null` switches a
threshold off: the finding is still made, as `info`. Each preset's `checks:` defaults are in the
[Preset Catalog](presets.md). A check with a criterion that has no threshold, such as an unmet share, an ambiguous name
or an empty class, keeps judging it, so its finding can still be `ok` or `warning` when its thresholds are `null`.

A check is never skipped because an input produced nothing. Where a step it reads failed or was skipped, it makes one
`info` finding briefed `not assessed`, titled with its `subject` where it takes one and with its own title otherwise,
followed by " by class" where it has `by: class`. Its description names that input and why it holds nothing: "Not
assessed: `count` failed: RuntimeError: …". A check that reads a list on a port that takes one Output runs once
per element, and each finding names its element under `step`, as `class-imbalance[train]`. The report groups those findings
by the element's key, `train`, in its summary and below it.

A check that has its inputs but cannot assess them raises `StepSkipped(reason)`. The engine records it as not assessed,
with that reason in the step's `not_assessed`, and never as skipped. `class-sufficiency`, `untrained-classes` and
`shortcut-risk` do so, as their sections say.

A check with `by: class` runs once per class, or per group of classes, of an Output made with the same `by:`, and rolls
the findings up into one, titled with the first's title and " by class". Its brief counts the classes that warn, as
`1/3 classes warn`, and its description names those that warned and those not assessed. See [Write a custom
workflow](../how_to/write_a_custom_workflow.md) for how to set it up, and [Drift in a model's
uncertainty](../how_to/monitor_drift.md#6-drift-in-a-models-uncertainty) for `by: predicted`, which keys by the class a
model predicts.

## Is the data clean?

### `image-outliers`

Warns when more than `warning` percent of a Dataset's images are outliers.

An image counts when it has at least one image-level outlier flag.

With nothing flagged, the finding is `ok`.

- **Reads:** `input`, an `outliers` Output.
- **Makes:** one finding, titled Image Outliers.

**Settings** ({py:class}`~dataeval_flow.steps.checks.ImageOutliersConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An `outliers` Output |
| `warning` | a percentage, or `null` | `3.0` | Most images, as a percentage of the Dataset, that may be flagged before the finding warns |

- **Judges:** [`outliers`](evaluators.md#outliers)
- **Used in:** [`audit`](presets.md#audit), [`data-cleaning`](presets.md#data-cleaning)

```yaml
evaluators:
  - {name: outliers, type: outliers, flags: [pixel, visual]}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: outliers, evaluator: outliers, input: data}
      - {name: image-outliers, check: image-outliers, input: outliers, warning: 5.0}
```

### `target-outliers`

Warns when more than `warning` percent of the boxes are outliers.

A box counts when it has at least one outlier flag.

- **Reads:** `input`, an `outliers` Output run with `per_target: true`; `labels`, a `label-health` Output on the same
  Dataset.
- **Makes:** one finding, titled Target Outliers; none where nothing was flagged per box, as on a classification
  Dataset.

**Settings** ({py:class}`~dataeval_flow.steps.checks.TargetOutliersConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An `outliers` Output run with `per_target: true` |
| `labels` | an address | required | A `label-health` Output on the same Dataset: its label count is the number of boxes |
| `warning` | a percentage, or `null` | `3.0` | Most boxes, as a percentage of all, that may be flagged before the finding warns |

- **Judges:** [`label-health`](evaluators.md#label-health), [`outliers`](evaluators.md#outliers)
- **Used in:** [`data-cleaning`](presets.md#data-cleaning)

```yaml
evaluators:
  - {name: outliers, type: outliers, per_target: true}
  - {name: label-health, type: label-health}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: outliers, evaluator: outliers, input: data}
      - {name: labels, evaluator: label-health, input: data}
      - {name: target-outliers, check: target-outliers, input: outliers, labels: labels, warning: 5.0}
```

### `classwise-outliers`

Warns when outliers pass `warning` percent across classes; names the worst class.

It judges the worst class's share, how many classes pass the limit, and whether all classes together do.

- **Reads:** `input`, an `outliers-by-class` Output.
- **Makes:** one finding, titled Classwise Outliers, whose evidence is the per-class table.

**Settings** ({py:class}`~dataeval_flow.steps.checks.ClasswiseOutliersConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `outliers-by-class` Output |
| `warning` | a percentage, or `null` | `3.0` | Most items or boxes, as a percentage of all, the outliers may take up before the finding warns; each class is counted against it too |

- **Judges:** [`outliers-by-class`](combines.md#outliers-by-class)
- **Used in:** [`data-cleaning`](presets.md#data-cleaning)

```yaml
evaluators:
  - {name: outliers, type: outliers, flags: [pixel, visual]}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: outliers, evaluator: outliers, input: data}
      - {name: by-class, combine: outliers-by-class, input: data, outliers: outliers}
      - {name: classwise-outliers, check: classwise-outliers, input: by-class, warning: 5.0}
```

### `image-duplicates`

Warns when more than `exact` or `near` percent of the images are duplicates.

The shares of images in exact and in near duplicate groups are judged apart.

- **Reads:** `input`, a `duplicates` Output.
- **Makes:** one finding, titled Image Duplicates; none where there are no duplicate images.

**Settings** ({py:class}`~dataeval_flow.steps.checks.ImageDuplicatesConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `duplicates` Output |
| `exact` | a percentage, or `null` | `0.0` | Most images that may sit in exact-duplicate groups before the finding warns |
| `near` | a percentage, or `null` | `5.0` | Most images that may sit in near-duplicate groups before the finding warns |

- **Judges:** [`duplicates`](evaluators.md#duplicates)
- **Used in:** [`audit`](presets.md#audit), [`data-cleaning`](presets.md#data-cleaning)

```yaml
evaluators:
  - {name: dupes, type: duplicates}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: dupes, evaluator: dupes, input: data}
      - {name: image-duplicates, check: image-duplicates, input: dupes, near: 10.0}
```

## Are the labels sound?

### `class-imbalance`

Warns when the largest class outnumbers the smallest by more than `warning`.

The ratio is taken over the classes with labels. A class with none is named and warns unless `empty` is `false`.

- **Reads:** `input`, a `label-health` Output.
- **Makes:** one finding, titled Class Imbalance, which lists the images with no labels. It is made whenever the Dataset
  has classes, declared or observed.

**Settings** ({py:class}`~dataeval_flow.steps.checks.ClassImbalanceConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-health` Output |
| `warning` | a ratio of at least 1, or `null` | `5.0` | Largest class count over smallest that may hold before the finding warns; an empty class warns unless `empty` is `false` |
| `info` | a ratio, or `null` | `null` | A ratio at or under which the finding is ok; must not exceed `warning`. `null`, which data-cleaning and data-splitting keep, makes every ratio that does not warn `info` |
| `empty` | `true` or `false` | `true` | Whether a declared class with no labels warns; `false` leaves it to `untrained-classes` and `class-sufficiency` |

- **Judges:** [`label-health`](evaluators.md#label-health)
- **Used in:** [`audit`](presets.md#audit), [`data-bias`](presets.md#data-bias)

```yaml
evaluators:
  - {name: label-health, type: label-health}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: labels, evaluator: label-health, input: data}
      - {name: class-imbalance, check: class-imbalance, input: labels, warning: 10.0}
```

### `class-sufficiency`

Warns when a class has too few labels to learn or to evaluate.

It judges the classes train holds: each needs `train` labels in train and `eval` in every evaluation split, a class the
split lacks included. A `null` limit judges nothing, and with both `null` the finding is `info`. It is not assessed
(`train holds no labelled class`) where train holds no labelled class. An empty `evals` list leaves train judged alone.

- **Reads:** `input`, a `label-health` Output over train; `evals`, the evaluation splits' `label-health` Outputs, a list
  that may be empty.
- **Makes:** one finding, titled Class Sufficiency, which warns where a class falls short and tabulates each class's
  count in train and in each split.

**Settings** ({py:class}`~dataeval_flow.steps.checks.ClassSufficiencyConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-health` Output over train |
| `evals` | an address, or `null` | `null` | The evaluation splits' `label-health` Outputs, a list that may be empty; unset judges train alone |
| `train` | an integer of at least 0, or `null` | `20` | The fewest labels each class train holds needs in train |
| `eval` | an integer of at least 0, or `null` | `30` | The fewest labels each class train holds needs in each evaluation split; at 30, a per-class metric's 95% interval is about ±18 points |

- **Judges:** [`label-health`](evaluators.md#label-health)
- **Used in:** [`audit`](presets.md#audit)

```yaml
evaluators:
  - {name: label-health, type: label-health}

workflows:
  - name: example
    inputs: [train, {name: evals, list: true}]
    steps:
      - {name: health-train, evaluator: label-health, input: train}
      - {name: health-evals, evaluator: label-health, input: evals}
      - {name: class-sufficiency, check: class-sufficiency, input: health-train, evals: health-evals, train: 30}
```

### `untrained-classes`

Warns when an evaluation split holds a class train lacks.

A declared class with labels in no split is listed, and warns only with `declared: true`. With no evaluation split there
is nothing to compare, and the finding is `info` unless `declared` is true. Where evaluation splits are given but none
holds a labelled class, it is not assessed (`no evaluation split holds a labelled class`).

- **Reads:** `input`, a `label-health` Output over train; `evals`, the evaluation splits' `label-health` Outputs, a list
  that may be empty.
- **Makes:** one finding, titled Untrained Classes, which lists each class with labels in an evaluation split and none
  in train, with the splits that hold it.

**Settings** ({py:class}`~dataeval_flow.steps.checks.UntrainedClassesConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-health` Output over train |
| `evals` | an address, or `null` | `null` | The evaluation splits' `label-health` Outputs, a list that may be empty |
| `declared` | `true` or `false` | `false` | Whether a declared class with no labels in train also warns |

- **Judges:** [`label-health`](evaluators.md#label-health)
- **Used in:** [`audit`](presets.md#audit)

```yaml
evaluators:
  - {name: label-health, type: label-health}

workflows:
  - name: example
    inputs: [train, {name: evals, list: true}]
    steps:
      - {name: health-train, evaluator: label-health, input: train}
      - {name: health-evals, evaluator: label-health, input: evals}
      - {name: untrained-classes, check: untrained-classes, input: health-train, evals: health-evals, declared: true}
```

## Do the labels match an ontology?

### `leaf-coverage`

Warns when too few of an ontology's leaves have examples, or a branch is empty.

It reports how much of the ontology's sanctioned leaves the Dataset has examples of, what to acquire for an even spread,
the wholly empty branches, and the asserted minimum shares (`expected`) not met. It warns on an unmet share, on leaf
coverage under `coverage`, or on more empty branches than `empty_branches`; it informs while anything remains to
acquire, and is `ok` otherwise.

- **Reads:** `input`, a `representation` Output against a declared ontology.
- **Makes:** one finding, titled Leaf Coverage.

**Settings** ({py:class}`~dataeval_flow.steps.checks.LeafCoverageConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `representation` Output, against a declared ontology |
| `coverage` | a fraction, or `null` | `0.9` | The least share of leaves with examples; `null` turns it off |
| `empty_branches` | a count, or `null` | `0` | Wholly empty branches tolerated; `null` turns it off |

- **Judges:** [`representation`](evaluators.md#representation)
- **Used in:** [`label-space`](presets.md#label-space)

```yaml
ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}
      - {id: Person, label: Person, synonyms: [person, pedestrian]}

evaluators:
  - {name: representation, type: representation, ontology: vehicles}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: representation, evaluator: representation, input: data}
      - {name: leaf-coverage, check: leaf-coverage, input: representation, coverage: 0.8}
```

### `label-conformance`

Warns when class names resolve to no ontology concept, or to several.

It warns on more unmatched names than `warning`, or on any ambiguous name, and is `ok` otherwise.

- **Reads:** `input`, a `label-reconciliation` Output.
- **Makes:** one finding, titled Label Conformance.

**Settings** ({py:class}`~dataeval_flow.steps.checks.LabelConformanceConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-reconciliation` Output |
| `warning` | a count, or `null` | `0` | Unmatched names tolerated; `null` turns it off |

- **Judges:** [`label-reconciliation`](evaluators.md#label-reconciliation)
- **Used in:** [`audit`](presets.md#audit), [`label-space`](presets.md#label-space)

```yaml
ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}
      - {id: Person, label: Person, synonyms: [person, pedestrian]}

evaluators:
  - {name: reconcile, type: label-reconciliation, ontology: vehicles}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: reconcile, evaluator: reconcile, input: data}
      - {name: label-conformance, check: label-conformance, input: reconcile, warning: 2}
```

### `mergeability`

Whether a Dataset's classes carry over to an ontology's vocabulary, with the stanza.

The stanza is the `Relabel` stanza to paste into a view that conforms the Dataset. Lossless is ok; lossy, where two
classes collapse into one concept, informs; partial, where `Relabel` would drop a class, warns. A target label several
concepts share always warns: the stanza cannot be used until the ontology is fixed.

- **Reads:** `input`, a `label-alignment` Output.
- **Makes:** one finding, titled Mergeability.

**Settings** ({py:class}`~dataeval_flow.steps.checks.MergeabilityConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-alignment` Output |

- **Judges:** [`label-alignment`](evaluators.md#label-alignment)
- **Used in:** [`label-space`](presets.md#label-space)

```yaml
ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}
      - {id: Person, label: Person, synonyms: [person, pedestrian]}

evaluators:
  - {name: align, type: label-alignment, ontology: vehicles}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: align, evaluator: align, input: data}
      - {name: mergeability, check: mergeability, input: align}
```

### `ontology-structure`

Reports an ontology's structure, and warns on a label several concepts share.

The finding informs on the ontology's size, depth and structural observations. Only a label several concepts share
warns, because it is what makes reconciliation ambiguous.

- **Reads:** `input`, an `ontology-validation` Output.
- **Makes:** one finding, titled Ontology Structure.

**Settings** ({py:class}`~dataeval_flow.steps.checks.OntologyStructureConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An `ontology-validation` Output |

- **Judges:** [`ontology-validation`](evaluators.md#ontology-validation)
- **Used in:** [`label-space`](presets.md#label-space)

```yaml
ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}
      - {id: Person, label: Person, synonyms: [person, pedestrian]}

evaluators:
  - {name: validate, type: ontology-validation, ontology: vehicles}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: validate, evaluator: validate, input: data}
      - {name: ontology-structure, check: ontology-structure, input: validate}
```

## Does the data cover what the model must handle?

### `class-coverage`

Warns when a class is clustered, one-dimensional or padded with near-duplicates.

It names the assessable classes `coverage` flagged and counts the items it left uncovered. It warns on any flagged
class, informs while any item is uncovered, and is `ok` otherwise. On detection crops it notes the crops counted and the
detections dropped.

- **Reads:** `input`, a `coverage` Output.
- **Makes:** one finding, titled Class Coverage.

**Settings** ({py:class}`~dataeval_flow.steps.checks.ClassCoverageConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `coverage` Output |
| `dispersion` | a number, or `null` | `0.5` | The dispersion under which a class is clustered; `null` turns it off |
| `isotropy` | a number, or `null` | `0.5` | The isotropy under which a class is one-dimensional; `null` turns it off |
| `near_duplicates` | a fraction, or `null` | `0.1` | The near-duplicate share over which a class is padded; `null` turns it off |

- **Judges:** [`coverage`](evaluators.md#coverage)
- **Used in:** [`audit`](presets.md#audit), [`data-coverage`](presets.md#data-coverage)

```yaml
evaluators:
  - {name: coverage, type: coverage}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: coverage, evaluator: coverage, input: data}
      - {name: class-coverage, check: class-coverage, input: coverage, dispersion: 0.3}
```

### `uncovered-items`

Warns when more than `warning` percent of a Dataset's items are uncovered.

Judge only `naive` coverage: adaptive coverage, DataEval's default, marks the sparsest `percent` of the items uncovered
by construction, so its share says nothing about the data. DataEval's naive radius overflows past about 340 embedding
dimensions, so `naive` suits low-dimensional embeddings: with a wide CNN or ONNX extractor, the coverage steps are
skipped with "failed: OverflowError".

- **Reads:** `input`, a `coverage` Output.
- **Makes:** one finding, titled Uncovered Items, the share of the Dataset's items coverage left uncovered.

**Settings** ({py:class}`~dataeval_flow.steps.checks.UncoveredItemsConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `coverage` Output |
| `warning` | a percentage, or `null` | `10.0` | The percent of items uncovered past which the finding warns |

- **Judges:** [`coverage`](evaluators.md#coverage)
- **Used in:** [`audit`](presets.md#audit), [`data-coverage`](presets.md#data-coverage)

```yaml
evaluators:
  - {name: coverage, type: coverage, method: naive}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: coverage, evaluator: coverage, input: data}
      - {name: uncovered-items, check: uncovered-items, input: coverage, warning: 5.0}
```

### `dimensional-completeness`

Warns when the embeddings fill too little of their space's dimensions.

The score is rounded to three places, then judged against two bands: under `warning` the finding warns, under `info` it
informs, and above both it is `ok`. With both bands `null` nothing is judged and the finding informs.

- **Reads:** `input`, a `completeness` Output.
- **Makes:** one finding, titled Dimensional Completeness.

**Settings** ({py:class}`~dataeval_flow.steps.checks.DimensionalCompletenessConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `completeness` Output |
| `warning` | a score from 0 to 1, or `null` | `0.5` | The score under which the finding warns; must not exceed `info` |
| `info` | a score from 0 to 1, or `null` | `0.8` | The score under which the finding informs |

- **Judges:** [`completeness`](evaluators.md#completeness)
- **Used in:** [`audit`](presets.md#audit), [`data-coverage`](presets.md#data-coverage)

```yaml
evaluators:
  - {name: completeness, type: completeness}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: completeness, evaluator: completeness, input: data}
      - {name: dimensional-completeness, check: dimensional-completeness, input: completeness, warning: 0.4}
```

### `factor-coverage-gaps`

Warns when enough class-factor-value combinations are under-represented.

It warns past `warning` gaps, is `info` up to that many, and is `ok` with none.

- **Reads:** `input`, a `factor-gaps` Output.
- **Makes:** one finding, titled Factor Coverage Gaps, with the gaps as a table, largest deficit first.

**Settings** ({py:class}`~dataeval_flow.steps.checks.FactorCoverageGapsConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `factor-gaps` Output |
| `warning` | a count, or `null` | `2` | The most under-represented class-factor-value combinations before the finding warns; `null` never warns |

- **Judges:** [`factor-gaps`](combines.md#factor-gaps)
- **Used in:** [`audit`](presets.md#audit), [`data-bias`](presets.md#data-bias)

```yaml
evaluators:
  - {name: balance, type: balance}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: balance, evaluator: balance, input: data}
      - {name: gaps, combine: factor-gaps, input: data, balance: balance}
      - {name: factor-coverage-gaps, check: factor-coverage-gaps, input: gaps, warning: 5}
```

### `class-shortfall`

Lists the classes short of an even spread, and warns on an unmet minimum share.

The spread is over the classes the Dataset declares. The finding warns on an unmet minimum share (`expected`), informs
while any class is short, and is `ok` otherwise.

- **Reads:** `input`, a `representation` Output with no ontology.
- **Makes:** one finding, titled Class Shortfall, giving what each short class lacks.

**Settings** ({py:class}`~dataeval_flow.steps.checks.ClassShortfallConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `representation` Output, computed with no ontology |

- **Judges:** [`representation`](evaluators.md#representation)
- **Used in:** [`data-coverage`](presets.md#data-coverage)

```yaml
evaluators:
  - {name: representation, type: representation}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: representation, evaluator: representation, input: data}
      - {name: class-shortfall, check: class-shortfall, input: representation}
```

## Could the model learn a shortcut?

### `shortcut-risk`

Warns when a metadata factor tells much about the class.

A model could learn such a factor instead of the task. The finding warns where a factor's mutual information with the
class is past `warning`. `balance`'s own `class_label` row is not a factor, and where no factor is left it is not
assessed (`no factor to score`). With `warning: null` the finding is `info`.

- **Reads:** `input`, a `balance` Output.
- **Makes:** one finding, titled Shortcut Risk, which lists the three most informative factors; a table ranks every
  factor.

**Settings** ({py:class}`~dataeval_flow.steps.checks.ShortcutRiskConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `balance` Output |
| `warning` | 0 to 1, or `null` | `0.1` | The mutual information with the class past which a factor warns; `null` judges nothing |

- **Judges:** [`balance`](evaluators.md#balance)
- **Used in:** [`audit`](presets.md#audit), [`data-bias`](presets.md#data-bias)

```yaml
evaluators:
  - {name: balance, type: balance}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: balance, evaluator: balance, input: data}
      - {name: shortcut-risk, check: shortcut-risk, input: balance, warning: 0.2}
```

### `factor-parity`

Warns when a metadata factor is significantly associated with the class.

Where `shortcut-risk` measures how much a factor tells about the class, this tests whether the association is real. A
factor warns where its bias-corrected Cramér's V with the class is past `warning` and its chi-square p-value is at or
under `p_value`. A factor whose contingency table has a cell expected to hold fewer than 5 items is named in the
description, since its p-value is unreliable. Where the `parity` Output scores no factor it is not assessed (`no factor
to score`). With `warning: null` the finding is `info`.

- **Reads:** `input`, a `parity` Output.
- **Makes:** one finding, titled Factor Parity, which lists the three most associated factors; a table gives every
  factor's Cramér's V and p-value, and marks the sparse ones.

**Settings** ({py:class}`~dataeval_flow.steps.checks.FactorParityConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `parity` Output |
| `warning` | 0 to 1, or `null` | `0.3` | The Cramér's V with the class past which a significant factor warns; `null` judges nothing |
| `p_value` | a number in (0, 1] | `0.05` | The p-value at or under which a factor's association counts as significant |

- **Judges:** [`parity`](evaluators.md#parity)
- **Used in:** [`data-bias`](presets.md#data-bias)

```yaml
evaluators:
  - {name: parity, type: parity}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: parity, evaluator: parity, input: data}
      - {name: factor-parity, check: factor-parity, input: parity, warning: 0.2}
```

## Are the splits fit to evaluate on?

### `stratification`

Judges how far each part's class shares stray from the whole's.

For each class and part, the gap between the class's share of the part's labels and of the whole's is taken in
percentage points. The largest, rounded to one place, is judged. Run once per fold over `kfold`'s lists. `audit`
passes train as `input` and the evaluation splits as `parts`, so each evaluation split is judged against train.

- **Reads:** `input`, a `label-health` Output over the whole Dataset the parts were split from; `parts`, the parts'
  `label-health` Outputs, each judged; `shown`, more `label-health` Outputs shown but not judged.
- **Makes:** one finding, titled Stratification, with the table of counts across the parts as its evidence.

**Settings** ({py:class}`~dataeval_flow.steps.checks.StratificationConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `label-health` Output over the whole Dataset the parts were split from |
| `parts` | one address or several | required | The parts' `label-health` Outputs, each judged |
| `shown` | one address or several, or `null` | `null` | `label-health` Outputs shown in the table but not judged, such as a rebalanced train |
| `info` | percentage points, or `null` | `2.0` | The largest deviation above which the finding is `info`; `null` has no `info` band |
| `warning` | percentage points, or `null` | `10.0` | The largest deviation above which the finding warns; `null` never warns |

- **Judges:** [`label-health`](evaluators.md#label-health)
- **Used in:** [`audit`](presets.md#audit), [`data-splitting`](presets.md#data-splitting)

```yaml
evaluators:
  - {name: label-health, type: label-health}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: whole, evaluator: label-health, input: data}
      - {name: split, transform: split, input: data, val_frac: 0.1, test_frac: 0.2}
      - {name: health-train, evaluator: label-health, input: split.train}
      - {name: health-val, evaluator: label-health, input: split.val}
      - {name: stratification, check: stratification, input: whole, parts: [health-train, health-val], warning: 5.0}
```

### `leakage`

Warns when items or group values sit in two splits at once.

It counts the items in duplicate groups that have members in two splits, exact and near apart, and the values of a
`factor-leakage` factor that both splits of a pair hold. The finding warns where a count passes its limit. A `null`
limit judges nothing, and with all three `null` the finding is `info`. It is not assessed where no `duplicates` list
holds an element, as there is then no pair of splits; an empty or failed `factors` list leaves the duplicates judged.

- **Reads:** `duplicates`, `duplicates` Outputs over two splits; `factors`, `factor-leakage` Outputs over the same
  pairs.
- **Makes:** one finding, titled Leakage, which lists each pair's groups and values.

**Settings** ({py:class}`~dataeval_flow.steps.checks.LeakageConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `duplicates` | an address of a list of Outputs | required | `duplicates` Outputs over two sources: train with each evaluation split, and evaluation pairs |
| `factors` | an address of a list of Outputs, or `null` | `null` | `factor-leakage` Outputs over the same pairs; unset judges duplicates alone |
| `exact` | an integer of at least 0, or `null` | `0` | Most items in exact-duplicate groups spanning two splits before the finding warns |
| `near` | an integer of at least 0, or `null` | `0` | The same for near-duplicate groups |
| `groups` | an integer of at least 0, or `null` | `0` | Most group values held by both splits of a pair before the finding warns |

- **Judges:** [`duplicates`](evaluators.md#duplicates), [`factor-leakage`](evaluators.md#factor-leakage)
- **Used in:** [`audit`](presets.md#audit)

```yaml
evaluators:
  - {name: dupes, type: duplicates}

workflows:
  - name: example
    inputs: [{name: splits, list: true}]
    steps:
      - {name: dupes, evaluator: dupes, input: splits, pairs: true}
      - {name: leakage, check: leakage, duplicates: dupes, exact: 2}
```

### `distribution-shift`

Warns when two sources' embeddings sit too far apart.

The finding warns above `warning`, is `info` above `info`, and is `ok` at or below both: high, moderate or low
divergence. A `null` limit judges nothing at its level, and with both `null` the finding is `info`.

- **Reads:** `input`, a `divergence` Output.
- **Makes:** one finding, titled Distribution Shift.

**Settings** ({py:class}`~dataeval_flow.steps.checks.DistributionShiftConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `divergence` Output |
| `warning` | a number from 0 to 1, or `null` | `0.5` | The divergence above which the finding warns; `null` never warns |
| `info` | a number from 0 to 1, or `null` | 0.4 times `warning` | The divergence above which the finding is `info`, at or below which it is `ok`; `null` has no `info` band; must not exceed `warning` |

- **Judges:** [`divergence`](evaluators.md#divergence)
- **Used in:** [`audit`](presets.md#audit)

```yaml
evaluators:
  - {name: divergence, type: divergence}

workflows:
  - name: example
    inputs: [train, val]
    steps:
      - {name: divergence, evaluator: divergence, input: [train, val]}
      - {name: distribution-shift, check: distribution-shift, input: divergence, warning: 0.4}
```

### `eval-coverage`

Warns when much of an evaluation split lies beyond what train covers.

The percent of the split flagged is judged as `ood` judges its percent. Any OOD evaluator's Output can be judged, but
only an `ood-kneighbors` Output relates the percent to a percentile of train: how much of the split lies farther from
train than that percent of train lies from itself. The percentile is the `ood-kneighbors` entry's `threshold_perc`, or
DataEval's 95 where unset; a split drawn like train has about 100 minus that percent flagged by construction, so
`info: 2.0` suits `threshold_perc: 99`.

- **Reads:** `input`, an OOD evaluator's Output fitted on train and run on one evaluation split.
- **Makes:** one finding, titled Eval Coverage.

**Settings** ({py:class}`~dataeval_flow.steps.checks.EvalCoverageConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An OOD evaluator's Output fitted on train and run on one evaluation split, best an `ood-kneighbors` one |
| `warning` | a percentage, or `null` | `10.0` | The percent flagged past which the finding warns |
| `info` | a percentage, or `null` | `2.0` | The percent past which the finding is `info`, at or below which it is `ok` |

- **Judges:** [`ood-domain-classifier`](evaluators.md#ood-domain-classifier),
  [`ood-kneighbors`](evaluators.md#ood-kneighbors)
- **Used in:** [`audit`](presets.md#audit)

```yaml
evaluators:
  - {name: knn, type: ood-kneighbors, threshold_perc: 99}

workflows:
  - name: example
    inputs: [train, val]
    steps:
      - {name: knn, evaluator: knn, input: [train, val]}
      - {name: eval-coverage, check: eval-coverage, input: knn, warning: 5.0, info: 2.0}
```

## Has new data drifted?

### `drift`

Warns when a drift detector finds drift, whole or chunk by chunk.

Without chunking, drift is a warning, or `info` where `warn_on_drift` is false. With chunking, the finding warns when
the share of drifted chunks or the longest run of drifted chunks passes its limit, is `info` when some chunks drifted
but neither does, and is `ok` when no chunk drifted. With both limits `null` it judges nothing, and is `info` whether or
not a chunk drifted.

- **Reads:** `input`, a drift evaluator's Output.
- **Makes:** one finding, titled with the detector's subject (see `subject`): the verdict, or the chunks' verdicts.

**Settings** ({py:class}`~dataeval_flow.steps.checks.DriftConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A drift evaluator's Output |
| `subject` | text, or `null` | `null` | The finding's title, a `not assessed` one's too; unset, the evaluator's title, followed by its entry's name where that differs from its type |
| `warn_on_drift` | true or false | `true` | Unchunked, and per class: whether drift warns, or is `info` |
| `chunk_percent` | a percentage, or `null` | `10.0` | Chunked: the share of drifted chunks past which the finding warns |
| `consecutive_chunks` | an integer of at least 1, or `null` | `2` | Chunked: the longest run of drifted chunks past which the finding warns, so 2 warns on three in a row |

- **Judges:** [`drift-domain-classifier`](evaluators.md#drift-domain-classifier),
  [`drift-kneighbors`](evaluators.md#drift-kneighbors), [`drift-mmd`](evaluators.md#drift-mmd),
  [`drift-univariate`](evaluators.md#drift-univariate), [`drift-wasserstein`](evaluators.md#drift-wasserstein)
- **Used in:** [`drift-monitoring`](presets.md#drift-monitoring)

```yaml
evaluators:
  - {name: mmd, type: drift-mmd}

workflows:
  - name: example
    inputs: [reference, tests]
    steps:
      - {name: mmd, evaluator: mmd, input: [reference, tests]}
      - {name: drift, check: drift, input: mmd, warn_on_drift: false}
```

## Which items are out of distribution?

### `ood`

Judges the share of a test source's images an OOD detector flagged.

The share is a percent of the images the detector assessed. On a detector's `uncertainty` rows ([Drift in a model's
uncertainty](../how_to/monitor_drift.md#6-drift-in-a-models-uncertainty)), which `ood-kneighbors` reads only with
`distance_metric: euclidean`, an image with no detection at the confidence is not assessed, and the brief also counts
the detections flagged. The finding warns past `warning` percent, is `info` past `info` percent, and is `ok` at or below
it; a `null` threshold judges nothing at its level, and with both `null` the finding is `info`.

- **Reads:** `input`, an OOD evaluator's Output.
- **Makes:** one finding, titled with the detector's subject (see `subject`): the images flagged of those assessed.

**Settings** ({py:class}`~dataeval_flow.steps.checks.OODConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An OOD evaluator's Output |
| `subject` | text, or `null` | `null` | The finding's title; unset, the evaluator's title, followed by its entry's name where that differs from its type |
| `warning` | a percentage, or `null` | `10.0` | The percent of assessed test images flagged past which the finding warns |
| `info` | a percentage, or `null` | `1.0` | The percent past which the finding is `info`, at or below which it is `ok` |

- **Judges:** [`ood-domain-classifier`](evaluators.md#ood-domain-classifier),
  [`ood-kneighbors`](evaluators.md#ood-kneighbors)
- **Used in:** [`ood-detection`](presets.md#ood-detection)

```yaml
evaluators:
  - {name: knn, type: ood-kneighbors}

workflows:
  - name: example
    inputs: [reference, tests]
    steps:
      - {name: knn, evaluator: knn, input: [reference, tests]}
      - {name: ood, check: ood, input: knn, warning: 5.0}
```

### `ood-agreement`

Judges the share of a test source's images every OOD detector flagged, and counts those one alone flagged.

The aggregate finding judges the percent of assessed test images every detector flagged, as `ood` judges its percent.

- **Reads:** `input`, an `ood-union` Output.
- **Makes:** an aggregate finding and, where any image was flagged by one detector alone, a second `info` finding
  counting those images. Both are titled OOD Agreement, the check's title, and their briefs tell them apart.

**Settings** ({py:class}`~dataeval_flow.steps.checks.OODAgreementConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | An `ood-union` Output |
| `warning` | a percentage, or `null` | `10.0` | The percent of assessed test images every detector flagged past which the finding warns |
| `info` | a percentage, or `null` | `1.0` | The percent past which the finding is `info`, at or below which it is `ok` |

- **Judges:** [`ood-union`](combines.md#ood-union)
- **Used in:** [`ood-detection`](presets.md#ood-detection)

```yaml
evaluators:
  - {name: knn, type: ood-kneighbors, k: 5}
  - {name: dc, type: ood-domain-classifier}

workflows:
  - name: example
    inputs: [reference, tests]
    steps:
      - {name: knn, evaluator: knn, input: [reference, tests]}
      - {name: dc, evaluator: dc, input: [reference, tests]}
      - {name: union, combine: ood-union, input: [knn, dc]}
      - {name: ood-agreement, check: ood-agreement, input: union, warning: 5.0}
```

## Is the metadata readable?

### `metadata-issues`

Warns where metadata triage found a factor the run could not read as configured.

It makes its findings in this order:

1. One finding per kind of issue `factor-triage` found. It is a warning where any issue of that kind is blocking,
   meaning the run did less than its configuration asked, and info otherwise.
2. "Suggested policy", holding the stanza to paste under `metadata:`.
3. "Verified", saying what each suggestion recovered, or "Verification failed", a warning, when verification raised.
4. "Recommended policy", opening with the caveat that a policy read from unrepresentative data can mislead, then the
   stanza that pins every factor the policy left unpinned; or "Recommendation failed", a warning, when reading the
   data back raised. Neither where the policy already pins everything.

It has no thresholds.

- **Reads:** `input`, a `factor-triage` Output.
- **Makes:** one finding per kind of issue, then Suggested policy, Verified and Recommended policy.

**Settings** ({py:class}`~dataeval_flow.steps.checks.MetadataIssuesConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | A `factor-triage` Output |
| `max_examples` | an integer of at least 1 | `20` | Distinct values shown per kind per factor; display only |

- **Judges:** [`factor-triage`](evaluators.md#factor-triage)
- **Used in:** [`audit`](presets.md#audit), [`metadata-triage`](presets.md#metadata-triage)

```yaml
evaluators:
  - {name: triage, type: factor-triage}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: triage, evaluator: triage, input: data}
      - {name: metadata-issues, check: metadata-issues, input: triage, max_examples: 10}
```
