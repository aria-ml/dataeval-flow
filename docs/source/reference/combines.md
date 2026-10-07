# Combine Catalog

A **combine** is a step, in a preset's chain or a custom workflow, that reads Outputs, and the Datasets they were
computed on, and makes an Output a check reads. See [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) for
how steps chain, the [Check Catalog](checks.md) for the checks that judge a combine's Output, and the [Transform
Catalog](transforms.md) for the steps that make Datasets. Each example assumes the pipeline defines `datasets:`, the
sources `train`, `test`, `validation`, `operational`, `labeled` and `unlabeled`, and the extractor `bovw_ext`, as
[Evaluator recipes](../how_to/evaluator_recipes.md) does.

## At a glance

| Type | Reads | Makes |
| --- | --- | --- |
| `outliers-by-class` | `input`: a Dataset; `outliers`: an `outliers` Output computed on it | outliers per class |
| `factor-gaps` | `input`: a Dataset; `balance`: a `balance` Output computed on it | each factor's MI with the class, and the under-represented combinations |
| `ood-union` | `input`: the OOD Outputs of one comparison of a test source with a reference | each flagged image as mutual, partial or unique, with its agreement score |
| `factor-predictors` | `ood`: an `ood-union` or OOD Output; `reference`, `input`: the Datasets it was computed on | each factor's association with being flagged |
| `factor-deviation` | the same | the factors setting each of the most out-of-distribution agreed images apart |

## Is the data clean?

### `outliers-by-class`

Pivots an Outliers Output by class: each class's flagged items or boxes.

Each class shows a count and a share of its items, or boxes for detection, most flagged first, with the total. On a
detection Dataset it refuses outliers not computed per box (`per_target: true`), rather than report none.

- **Reads:** `input`, the Dataset the outliers were found in, whose labels name each item's class; `outliers`, an
  `outliers` Output computed on exactly `input`.
- **Makes:** a {py:class}`~dataeval_flow.steps.combines.OutliersByClassOutput`: the outliers per class.

**Settings** ({py:class}`~dataeval_flow.steps.combines.OutliersByClassConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset the outliers were found in |
| `outliers` | an address | required | An `outliers` Output computed on exactly `input`; for a detection Dataset, with `per_target: true` |

The config refuses an `outliers` computed on another Dataset when it loads, as `remove` does.

- **Judged by:** [`class-outliers`](checks.md#class-outliers)
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
      - {name: class-outliers, check: class-outliers, input: by-class, warning: 5.0}
```

## Does the data cover what the model must handle?

### `factor-gaps`

Class-factor-value combinations under-represented, among the factors tied to the class.

It reads the mutual information of a `balance` Output, runs no Balance of its own, and searches the factors at or over
`mi_threshold`. A combination is a gap where its count is under `min_representation` while its expected count, from the
factor's overall spread, is over it.

- **Reads:** `input`, the Dataset whose Metadata the gaps are counted in; `balance`, a `balance` Output computed on
  exactly `input`.
- **Makes:** a {py:class}`~dataeval_flow.steps.combines.FactorGapsOutput`: each factor's mutual information with the
  class, and the under-represented combinations.

**Settings** ({py:class}`~dataeval_flow.steps.combines.FactorGapsConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address | required | The Dataset whose Metadata the gaps are counted in |
| `balance` | an address | required | A `balance` Output computed on exactly `input` |
| `mi_threshold` | a number | `0.1` | The least mutual information with the class a factor needs to be searched |
| `min_representation` | a count | `5` | A combination is a gap where its count is under this while its expected count is over it |
| `metadata` | a policy name, or `null` | `null` | The metadata policy the factors are read under; it should be the one `balance` read under |

The config refuses a `balance` computed on another Dataset when it loads, as `outliers-by-class` does.

- **Judged by:** [`factor-coverage-gaps`](checks.md#factor-coverage-gaps)
- **Used in:** [`audit`](presets.md#audit), [`data-bias`](presets.md#data-bias)

```yaml
evaluators:
  - {name: balance, type: balance}

workflows:
  - name: example
    inputs: [data]
    steps:
      - {name: balance, evaluator: balance, input: data}
      - {name: gaps, combine: factor-gaps, input: data, balance: balance, mi_threshold: 0.2, min_representation: 10}
      - {name: factor-coverage-gaps, check: factor-coverage-gaps, input: gaps}
```

## Which items are out of distribution?

### `ood-union`

Groups each flagged image as flagged by every OOD detector, by some, or by one alone.

Each flagged image falls in one group: flagged by every detector (mutual), by more than one but not every one (partial),
or by one alone (unique). Its agreement score is the mean, over the detectors that scored it, of its score over the
detector's threshold, which is derived from the detector's flags. A detector whose derived threshold is not positive is
left out, and the section names it. The section pictures each flagged image once, most out of distribution first. Load
refuses Outputs computed on different Datasets.

- **Reads:** `input`, the OOD Outputs of one comparison of a test source with a reference, as one address or a list.
- **Makes:** an {py:class}`~dataeval_flow.steps.combines.OODUnionOutput`: each flagged image as mutual, partial or
  unique, with its agreement score.

**Settings** ({py:class}`~dataeval_flow.steps.combines.OODUnionConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `input` | an address, or a list of them | required | Each detector's OOD Output, every one computed on the same reference and test source |

- **Judged by:** [`ood-agreement`](checks.md#ood-agreement)
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

### `factor-predictors`

Ranks the metadata factors that go with the images OOD detectors flagged.

DataEval's `factor_predictors`: normalized mutual information from 0 to 1, strongest first, over the test images the
detectors assessed. Factors are the item-level metadata factors, without `id`, with `class_label` where there is one
label per item, and the per-image statistics named `f_<statistic>`; a factor counts where both Datasets have it,
numeric, one-dimensional, finite in both, and not constant in the test. Where a Dataset's metadata or statistics cannot
be read, the rest is read without it, and the section says so. Load refuses an `ood` Output computed on other Datasets
than `reference` and `input`.

- **Reads:** `ood`, an `ood-union` or OOD Output; `reference`, the reference Dataset the detectors fitted on; `input`,
  the test Dataset whose images were flagged.
- **Makes:** a {py:class}`~dataeval_flow.steps.combines.FactorPredictorsOutput`: each factor's association with being
  flagged.

**Settings** ({py:class}`~dataeval_flow.steps.combines.FactorPredictorsConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `ood` | an address | required | An `ood-union` Output, or one OOD evaluator's Output, computed on `reference` and `input` |
| `reference` | an address | required | The reference Dataset the detectors fitted on |
| `input` | an address | required | The test Dataset whose images were flagged |
| `metadata` | a policy name, or `null` | `null` | The metadata policy the factors are read under |
| `stats` | a policy name, or `null` | `null` | The stats policy the statistics are measured under; unset, every statistic |

- **Judged by:** none
- **Used in:** [`ood-detection`](presets.md#ood-detection)

```yaml
evaluators:
  - {name: knn, type: ood-kneighbors}
  - {name: dc, type: ood-domain-classifier}

workflows:
  - name: example
    inputs: [reference, tests]
    steps:
      - {name: knn, evaluator: knn, input: [reference, tests]}
      - {name: dc, evaluator: dc, input: [reference, tests]}
      - {name: union, combine: ood-union, input: [knn, dc]}
      - {name: predictors, combine: factor-predictors, ood: union, reference: reference, input: tests}
```

### `factor-deviation`

Names the factors that set the most out-of-distribution agreed images apart.

DataEval's `factor_deviation`: each factor's scaled distance from the reference's median, most deviating first. It
reads the factors `factor-predictors` reads.

- **Reads:** `ood`, an `ood-union` or OOD Output; `reference`, the reference Dataset the detectors fitted on; `input`,
  the test Dataset whose images were flagged.
- **Makes:** a {py:class}`~dataeval_flow.steps.combines.FactorDeviationOutput`: the factors setting each of the most
  out-of-distribution agreed images apart.

**Settings** ({py:class}`~dataeval_flow.steps.combines.FactorDeviationConfig`):

| Field | Takes | Default | Description |
| --- | --- | --- | --- |
| `ood` | an address | required | An `ood-union` Output, or one OOD evaluator's Output, computed on `reference` and `input` |
| `reference` | an address | required | The reference Dataset the detectors fitted on |
| `input` | an address | required | The test Dataset whose images were flagged |
| `max_items` | an integer of at least 1 | `50` | The most out-of-distribution agreed images explained, at most |
| `metadata` | a policy name, or `null` | `null` | The metadata policy the factors are read under |
| `stats` | a policy name, or `null` | `null` | The stats policy the statistics are measured under; unset, every statistic |

- **Judged by:** none
- **Used in:** [`ood-detection`](presets.md#ood-detection)

```yaml
evaluators:
  - {name: knn, type: ood-kneighbors}
  - {name: dc, type: ood-domain-classifier}

workflows:
  - name: example
    inputs: [reference, tests]
    steps:
      - {name: knn, evaluator: knn, input: [reference, tests]}
      - {name: dc, evaluator: dc, input: [reference, tests]}
      - {name: union, combine: ood-union, input: [knn, dc]}
      - {name: deviation, combine: factor-deviation, ood: union, reference: reference, input: tests, max_items: 20}
```
