# Evaluator Catalog

Each evaluator runs one DataEval evaluator and reports its determinations, with no health status (see [Workflows and
Evaluators](../concepts/WorkflowsAndEvaluators.md)). An evaluator's `type` names what it computes, in kebab case, after
DataEval's class where there is one (`drift-mmd` runs `DriftMMD`). To run one, see [Run a single
evaluator](../how_to/run_a_single_evaluator.md); for worked examples, see [Evaluator
recipes](../how_to/evaluator_recipes.md). Each entry's **Combined by**, where it has one, names the combines that read
its Output, and its **Used in** names the presets that run the evaluator; where it names none, run it as a task, or
chain it in a [workflow of your own](../how_to/write_a_custom_workflow.md). Each example assumes the pipeline defines
`datasets:`, the sources `train`, `test`, `validation`, `operational`, `labeled` and `unlabeled`, and the extractor
`bovw_ext`, as [Evaluator recipes](../how_to/evaluator_recipes.md) does.

## At a glance

| Type | DataEval class | Consumes | Sources | Extractor |
| --- | --- | --- | --- | --- |
| `outliers` | `dataeval.quality.Outliers` | stats; clusters in cluster mode | 1 or more; 1 in cluster mode | needed in cluster mode; accepted but unused otherwise |
| `duplicates` | `dataeval.quality.Duplicates` | stats; clusters in cluster mode | 1 or more; 1 in cluster mode | needed in cluster mode; accepted but unused otherwise |
| `label-health` | `dataeval.core.label_stats` | metadata | 1 | refused |
| `representation` | `dataeval.scope.Representation` | labels | 1 | refused |
| `label-reconciliation` | `dataeval.core.label_reconciliation` | labels | 1 | refused |
| `label-alignment` | `dataeval.core.label_alignment` | labels | 1 | refused |
| `ontology-validation` | `dataeval.core.ontology_validation` | labels, read only to place the report | 1 | refused |
| `coverage` | `dataeval.scope.Coverage` | embeddings; labels where there is one per item | 1 | required |
| `completeness` | `dataeval.core.completeness` | embeddings | 1 | required |
| `balance` | `dataeval.bias.Balance` | metadata | 1 | refused |
| `parity` | `dataeval.bias.Parity` | metadata | 1 | refused |
| `diversity` | `dataeval.bias.Diversity` | metadata | 1 | refused |
| `factor-summary` | `dataeval.Metadata` | metadata | 1 | refused |
| `factor-leakage` | `dataeval.Metadata` | metadata | 2 | refused |
| `divergence` | `dataeval.core.divergence_mst`, `dataeval.core.divergence_fnn` | embeddings | 2: the first source, then the second | required |
| `drift-univariate` | `dataeval.shift.DriftUnivariate` | embeddings | 2: the reference, then the data to test | required |
| `drift-mmd` | `dataeval.shift.DriftMMD` | embeddings | 2: the reference, then the data to test | required |
| `drift-kneighbors` | `dataeval.shift.DriftKNeighbors` | embeddings | 2: the reference, then the data to test | required |
| `drift-wasserstein` | `dataeval.shift.DriftWasserstein` | embeddings | 3: the reference, a validation set, then the data to test | required |
| `drift-domain-classifier` | `dataeval.shift.DriftDomainClassifier` | embeddings | 2: the reference, then the data to test | required |
| `ood-kneighbors` | `dataeval.shift.OODKNeighbors` | embeddings | 2: the reference, then the data to test | required |
| `ood-domain-classifier` | `dataeval.shift.OODDomainClassifier` | embeddings | 2: the reference, then the data to test | required |
| `prioritization` | `dataeval.scope.Prioritize` | embeddings; labels where there is one per item | 1, or 2: the data, then a reference | required |
| `factor-triage` | `dataeval.Metadata` | metadata | 1 | refused |
| `content-digest` | `dataeval_flow.dataset_digest` | the Dataset itself, every item | 1 | refused |

A task's `extractor:` still lands in the result envelope's `model_id`, whether or not that run's mode actually reads it.

## How parameters work

Parameter names are DataEval's argument names, unchanged. A parameter you leave out is not passed, so DataEval's own
default applies. An unknown parameter fails the config load, as does any value DataEval itself refuses, such as an
unknown threshold method. Print any evaluator's JSON Schema with `dataeval-flow evaluators <type>`.

## Is the data clean?

Both evaluators are explained in DataEval's [Data Integrity
explanation](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html), and their classes are documented in
the [DataEval `dataeval.quality`
reference](https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/quality/index.html).

### `outliers`

Images whose statistics are outliers (DataEval Outliers).

It runs `dataeval.quality.Outliers`, flagging the statistics that sit outside the threshold.

- **Reads:** `input`: one Dataset or more, all measured together; Flow derives their stats under the `stats:` policy the
  evaluator names. In cluster mode, one Dataset, with embeddings through the task's extractor.
- **Makes:** an `outliers` Output: a table with one row per flagged statistic (`item_index`, `metric_name`,
  `metric_value`, and `target_index` for per-target results).

**Settings** ({py:class}`~dataeval_flow.evaluators.quality.OutliersConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `stats` | (DataEval Flow) the name of a `stats:` policy; its `outliers_from` views apply | the whole image is measured |
| `flags` | `flags`: `dimension`, `pixel`, `visual` | DataEval's default families |
| `outlier_threshold` | `outlier_threshold`: a method, `[method, bounds]`, or a mapping of metric name to either | DataEval's default |
| `cluster_threshold` | `cluster_threshold`; setting it turns on cluster mode | cluster mode is off |
| `cluster_algorithm` | `cluster_algorithm`: `kmeans` or `hdbscan` | DataEval's default |
| `n_clusters` | `n_clusters` | DataEval chooses |
| `per_image` | `from_stats(per_image=...)` | DataEval's default |
| `per_target` | `from_stats(per_target=...)` | DataEval's default |

- **Judged by:** [`image-outliers`](checks.md#image-outliers), [`target-outliers`](checks.md#target-outliers)
- **Combined by:** [`outliers-by-class`](combines.md#outliers-by-class)
- **Used in:** [`audit`](presets.md#audit), [`data-cleaning`](presets.md#data-cleaning),
  [`data-prioritization`](presets.md#data-prioritization)

```yaml
evaluators:
  - {name: outliers, type: outliers, flags: [pixel, visual]}

tasks:
  - {name: outliers, evaluator: outliers, sources: [train]}
```

### `duplicates`

Exact and near duplicate groups (DataEval Duplicates).

It runs `dataeval.quality.Duplicates` across every source the task names.

- **Reads:** `input`: one Dataset or more, all searched together; Flow derives their stats under the `stats:` policy the
  evaluator names. In cluster mode, one Dataset, with embeddings through the task's extractor.
- **Makes:** a `duplicates` Output: a table with one row per duplicate group (`group_id`, `level`, `dup_type`,
  `item_indices`, `methods`, and `dataset_indices` when the task names several sources). `extras` holds
  `annotation_divergences` and `factor_cardinality`, `null` unless the annotation or factor axis ran.

**Settings** ({py:class}`~dataeval_flow.evaluators.quality.DuplicatesConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `stats` | (DataEval Flow) the name of a `stats:` policy | the whole image is measured |
| `flags` | `flags`: `hash_basic`, `hash_d4` | DataEval's default hash families |
| `merge_near_duplicates` | `merge_near_duplicates` | DataEval's default |
| `hash_radius` | `hash_radius` | DataEval's default (`0`, documented to change in a future major release) |
| `cluster_sensitivity` | `cluster_sensitivity`; setting it turns on cluster mode | cluster mode is off |
| `redundancy_radius` | `redundancy_radius` (video) | DataEval's default (`4`) |
| `min_segment_frames` | `min_segment_frames` (video) | DataEval's default (`30`) |
| `max_segment_gap` | `max_segment_gap` (video) | DataEval's default (`5`) |
| `segment_offset_tolerance` | `segment_offset_tolerance` (video) | DataEval's default (`0`) |
| `verify_alignment` | `verify_alignment` (video) | warped matching is off |
| `min_track_frames` | `min_track_frames` (video, with `per_target`) | DataEval's default (`5`) |
| `frame_sample` | `frame_sample` (video): a stride in frames, or a rate in frames per second | every frame is read |
| `cluster_algorithm` | `cluster_algorithm`: `kmeans` or `hdbscan` | DataEval's default |
| `n_clusters` | `n_clusters` | DataEval chooses |
| `per_image` | `from_stats(per_image=...)` | DataEval's default |
| `per_target` | `from_stats(per_target=...)` | DataEval's default |

- **Judged by:** [`image-duplicates`](checks.md#image-duplicates), [`leakage`](checks.md#leakage)
- **Used in:** [`audit`](presets.md#audit), [`data-cleaning`](presets.md#data-cleaning),
  [`data-prioritization`](presets.md#data-prioritization)

```yaml
evaluators:
  - {name: duplicates, type: duplicates, flags: [hash_basic]}

tasks:
  - {name: duplicates, evaluator: duplicates, sources: [train]}
```

## Are the labels sound?

### `label-health`

How a Dataset's labels spread over its classes (DataEval label_stats).

It runs `dataeval.core.label_stats` on the Dataset's metadata, counting each class's labels, the items that carry each,
and the items that carry none.

- **Reads:** `input`: one Dataset; Flow derives its metadata under the `metadata:` policy the evaluator names.
- **Makes:** a `label-health` Output: a mapping of `item_count`, `class_count` (the classes the Dataset declares, used
  or not), `label_count`, `label_counts_per_class` and `image_counts_per_class` (by class name, for every declared
  class, at 0 where unseen), `empty_image_count`, `empty_image_indices` (the items with no label), and `label_source`,
  where the labels came from.

**Settings** ({py:class}`~dataeval_flow.evaluators.quality.LabelHealthConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `metadata` | (DataEval Flow) a policy under `metadata:`, which the metadata is built under | DataEval's defaults |

- **Judged by:** [`class-imbalance`](checks.md#class-imbalance), [`class-sufficiency`](checks.md#class-sufficiency),
  [`stratification`](checks.md#stratification), [`target-outliers`](checks.md#target-outliers),
  [`untrained-classes`](checks.md#untrained-classes)
- **Used in:** [`audit`](presets.md#audit), [`data-cleaning`](presets.md#data-cleaning),
  [`data-coverage`](presets.md#data-coverage), [`data-splitting`](presets.md#data-splitting)

```yaml
metadata:
  - name: standard
    exclude: [id]

evaluators:
  - {name: label-health, type: label-health, metadata: standard}

tasks:
  - {name: label-health, evaluator: label-health, sources: [train]}
```

## Do the labels match an ontology?

These evaluators read one source's labels. `representation` is explained in DataEval's [Ontology
explanation](https://dataeval.readthedocs.io/en/latest/concepts/Ontology.html), and the classes are documented in the
[DataEval `dataeval.scope`
reference](https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/scope/index.html).

### `representation`

Class counts against an ontology's leaves (DataEval Representation).

It finds the classes that fall short of their share of the source, and what to acquire. It runs
`dataeval.scope.Representation`, and needs a dataset with labels. Without them it fails, naming the source.

- **Reads:** `input`: one Dataset; Flow derives its labels.
- **Makes:** a `representation` Output: a table, the worklist, with one row per concept short of its target (`concept`,
  `label`, `parent`, `action`, `count`, `target`, `deficit`). `extras` holds `leaf_coverage`, `total_deficit`, the
  `violations` and `dark_branches` tables, and `ignored_expected`, the `expected` names that resolve to no concept or to
  several.

**Settings** ({py:class}`~dataeval_flow.evaluators.scope.RepresentationConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `ontology` | (DataEval Flow) a name under `ontologies:`, an RDF file, or an inline hierarchy; becomes `Representation(ontology)` | a flat ontology synthesized from the dataset's `index2label` |
| `expected` | `expected`: class name to its minimum share, a fraction | a uniform share for every leaf |

- **Judged by:** [`class-shortfall`](checks.md#class-shortfall), [`leaf-coverage`](checks.md#leaf-coverage)
- **Used in:** [`data-coverage`](presets.md#data-coverage), [`label-space`](presets.md#label-space)

```yaml
ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}
      - {id: Person, label: Person, synonyms: [person, pedestrian]}

evaluators:
  - {name: representation, type: representation, ontology: vehicles, expected: {Vehicle: 0.2}}

tasks:
  - {name: representation, evaluator: representation, sources: [train]}
```

### `label-reconciliation`

Which of a Dataset's class names resolve to exactly one ontology concept.

An unmatched name is out of vocabulary: a typo, or a class the ontology does not sanction. An ambiguous name answers to
several concepts. The names are the Dataset's `index2label` values, in index order. It runs
`dataeval.core.label_reconciliation`.

- **Reads:** `input`: one Dataset; Flow derives its labels.
- **Makes:** a `label-reconciliation` Output: a mapping: `conforms`, `matched` (each name to its concept id),
  `unmatched`, and `ambiguous` (each name to the ids of the concepts it names).

**Settings** ({py:class}`~dataeval_flow.evaluators.scope.LabelReconciliationConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `ontology` | `ontology`: a name under `ontologies:`, a path, or an inline hierarchy | required |

- **Judged by:** [`label-conformance`](checks.md#label-conformance)
- **Used in:** [`audit`](presets.md#audit), [`label-space`](presets.md#label-space)

```yaml
ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}

evaluators:
  - {name: label-reconciliation, type: label-reconciliation, ontology: vehicles}

tasks:
  - {name: label-reconciliation, evaluator: label-reconciliation, sources: [train]}
```

### `label-alignment`

How a Dataset's class names align to an ontology: the remap, and whether it is lossless.

A `conform` step applies the remap, gated by how much loss it declares it will accept. It runs
`dataeval.core.label_alignment`.

- **Reads:** `input`: one Dataset; Flow derives its labels.
- **Makes:** a `label-alignment` Output: a mapping, the alignment (`mergeability`, `correspondences`,
  `unaligned_source`, `unaligned_target`, `class_remap`, `paste_remap`, `target_vocabulary`, `ambiguous_labels`,
  `label_space_digest`).

**Settings** ({py:class}`~dataeval_flow.evaluators.scope.LabelAlignmentConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `ontology` | (DataEval Flow) a name under `ontologies:`, a path, or an inline hierarchy; the alignment's target | required |
| `threshold` | `threshold`: the lowest confidence a fuzzy match keeps | DataEval's default (`0.0`) |

- **Judged by:** [`mergeability`](checks.md#mergeability)
- **Used in:** [`label-space`](presets.md#label-space)

```yaml
ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}

evaluators:
  - {name: label-alignment, type: label-alignment, ontology: vehicles, threshold: 0.8}

tasks:
  - {name: label-alignment, evaluator: label-alignment, sources: [train]}
```

### `ontology-validation`

An ontology's structural and naming facts: depth, roots, collisions, and more.

Only a shared label is a defect: it is what makes reconciliation ambiguous. It runs `dataeval.core.ontology_validation`.

- **Reads:** `input`: one Dataset; Flow derives its labels, read only to place the report.
- **Makes:** an `ontology-validation` Output: a mapping of `concept_count`, `leaf_count` and `max_depth`; `roots` and
  `isolated` (isolated concepts); `external_ancestors` (truncated ancestries); `redundant_edges`; `ancestor_siblings`
  (ancestor-sibling pairs); `unary_parents` (single-child links); `label_collisions` (labels several concepts share);
  and `nonconforming_labels` (labels that break `label_pattern`).

**Settings** ({py:class}`~dataeval_flow.evaluators.scope.OntologyValidationConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `ontology` | `ontology`: a name under `ontologies:`, a path, or an inline hierarchy | required |
| `label_pattern` | `label_pattern`: a regex every concept label should match | no naming check |

- **Judged by:** [`ontology-structure`](checks.md#ontology-structure)
- **Used in:** [`label-space`](presets.md#label-space)

```yaml
ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}

evaluators:
  - {name: ontology-validation, type: ontology-validation, ontology: vehicles, label_pattern: "^[A-Z]"}

tasks:
  - {name: ontology-validation, evaluator: ontology-validation, sources: [train]}
```

## Does the data cover what the model must handle?

These evaluators read one Dataset, and Flow derives its embeddings through the task's extractor, and for `coverage` its
labels too.

### `coverage`

Embedding-space coverage, broken down by class (DataEval Coverage).

It finds the items in sparse regions of the embedding space, uncovered by the rest of the data. With no label per item
(a dataset without labels, or a detection dataset's labels per target), it runs over every item as one class, `0`, and
logs a warning. Crop detections first with a `wrap` step, as `data-coverage` does. It runs `dataeval.scope.Coverage`.

- **Reads:** `input`: one Dataset; Flow derives its embeddings through the task's extractor, and its labels where there
  is one per item.
- **Makes:** a `coverage` Output: a table with one row per class (`class`, `count`, `uncovered`, `uncovered_fraction`,
  `dispersion`, `isotropy`, `near_duplicate_fraction`, `assessable`). `extras` holds `uncovered_indices`,
  `coverage_radius`, `critical_value_radii` and `uncovered_classes` (each uncovered item's class, `None` without a class
  breakdown).

**Settings** ({py:class}`~dataeval_flow.evaluators.scope.CoverageConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `method` | `method`: `naive` or `adaptive` | DataEval's default (`adaptive`) |
| `num_observations` | `num_observations`; must be fewer than the source's items | DataEval's default (`20`) |
| `percent` | `percent` (`adaptive` only) | DataEval's default (`0.01`) |
| `min_class_samples` | `min_class_samples` | DataEval's default (`20`) |
| `isotropy_min_samples` | `isotropy_min_samples` | one more than the embedding size |
| `near_duplicate_factor` | `near_duplicate_factor` | DataEval's default (`0.5`) |

- **Judged by:** [`class-coverage`](checks.md#class-coverage), [`uncovered-items`](checks.md#uncovered-items)
- **Used in:** [`audit`](presets.md#audit), [`data-coverage`](presets.md#data-coverage)

```yaml
evaluators:
  - {name: coverage, type: coverage, num_observations: 10}

tasks:
  - {name: coverage, evaluator: coverage, sources: [train], extractor: bovw_ext}
```

### `completeness`

How much of the embedding space's dimensions the data fills (DataEval completeness).

The embeddings are rescaled to the unit interval per dimension first, a constant dimension at 0. It runs
`dataeval.core.completeness`.

- **Reads:** `input`: one Dataset; Flow derives its embeddings through the task's extractor.
- **Makes:** a `completeness` Output: a mapping: `completeness` and `nearest_neighbor_pairs`.

**Settings** ({py:class}`~dataeval_flow.evaluators.scope.CompletenessConfig`):

It takes no parameters.

- **Judged by:** [`dimensional-completeness`](checks.md#dimensional-completeness)
- **Used in:** [`audit`](presets.md#audit), [`data-coverage`](presets.md#data-coverage)

```yaml
evaluators:
  - {name: completeness, type: completeness}

tasks:
  - {name: completeness, evaluator: completeness, sources: [train], extractor: bovw_ext}
```

## Could the model learn a shortcut?

The bias evaluators read one source's metadata, built under the metadata policy `metadata:` names, and need no
extractor. They are explained in DataEval's [Dataset Bias
explanation](https://dataeval.readthedocs.io/en/latest/concepts/DatasetBias.html), and their classes are documented in
the [DataEval `dataeval.bias`
reference](https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/bias/index.html).

### `balance`

Mutual information between metadata factors and class labels (DataEval Balance).

It runs `dataeval.bias.Balance` on the Dataset's metadata.

- **Reads:** `input`: one Dataset; Flow derives its metadata under the `metadata:` policy the evaluator names.
- **Makes:** a `balance` Output: a mapping of three tables. `balance` has each factor's mutual information with the
  class labels, `factors` each factor pair's, and `classwise` each class's against each factor.

**Settings** ({py:class}`~dataeval_flow.evaluators.bias.BalanceConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `metadata` | (DataEval Flow) the name of a `metadata:` policy | DataEval's default encoding |
| `num_neighbors` | `num_neighbors` | DataEval's default (`5`) |
| `class_imbalance_threshold` | `class_imbalance_threshold` | DataEval's default (`0.3`) |
| `factor_correlation_threshold` | `factor_correlation_threshold` | DataEval's default (`0.5`) |
| `label` | `label`: a factor, or a list of factors, to condition on | the class labels |
| `factor_source` | `factor_source`: `coded`, `values` or `auto` | the policy's `factor_source`, else DataEval's default (`auto`) |

- **Judged by:** [`shortcut-risk`](checks.md#shortcut-risk)
- **Combined by:** [`factor-gaps`](combines.md#factor-gaps)
- **Used in:** [`audit`](presets.md#audit), [`data-coverage`](presets.md#data-coverage),
  [`data-splitting`](presets.md#data-splitting)

```yaml
evaluators:
  - {name: balance, type: balance, num_neighbors: 10}

tasks:
  - {name: balance, evaluator: balance, sources: [train]}
```

### `parity`

Association between metadata factors and class labels (DataEval Parity).

It runs `dataeval.bias.Parity` on the Dataset's metadata.

- **Reads:** `input`: one Dataset; Flow derives its metadata under the `metadata:` policy the evaluator names.
- **Makes:** a `parity` Output: a mapping. `factors` is a table of each factor's Cramér's V, p-value and significance,
  and `insufficient_data` lists the factor values with too few samples per class.

**Settings** ({py:class}`~dataeval_flow.evaluators.bias.ParityConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `metadata` | (DataEval Flow) the name of a `metadata:` policy | DataEval's default encoding |
| `score_threshold` | `score_threshold` | DataEval's default (`0.3`) |
| `p_value_threshold` | `p_value_threshold` | DataEval's default (`0.05`) |
| `label` | `label`: a factor, or a list of factors, to condition on | the class labels |

- **Judged by:** none
- **Used in:** none; chain it in a [workflow of your own](../how_to/write_a_custom_workflow.md)

```yaml
evaluators:
  - {name: parity, type: parity, score_threshold: 0.2}

tasks:
  - {name: parity, evaluator: parity, sources: [train]}
```

### `diversity`

How evenly metadata factor values are spread (DataEval Diversity).

It runs `dataeval.bias.Diversity` on the Dataset's metadata, overall and within each class.

- **Reads:** `input`: one Dataset; Flow derives its metadata under the `metadata:` policy the evaluator names.
- **Makes:** a `diversity` Output: a mapping of two tables. `factors` has each factor's diversity and whether it is low,
  and `classwise` each class's.

**Settings** ({py:class}`~dataeval_flow.evaluators.bias.DiversityConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `metadata` | (DataEval Flow) the name of a `metadata:` policy | DataEval's default encoding |
| `method` | `method`: `simpson` or `shannon` | DataEval's default (`simpson`) |
| `threshold` | `threshold` | DataEval's default (`0.5`) |
| `label` | `label`: a factor, or a list of factors, to condition on | the class labels |

- **Judged by:** none
- **Used in:** [`audit`](presets.md#audit), [`data-coverage`](presets.md#data-coverage),
  [`data-splitting`](presets.md#data-splitting)

```yaml
evaluators:
  - {name: diversity, type: diversity, method: shannon}

tasks:
  - {name: diversity, evaluator: diversity, sources: [train]}
```

### `factor-summary`

Each metadata factor's type, binning, nulls, and range or top values.

It reads the Dataset's metadata through DataEval's `Metadata`.

- **Reads:** `input`: one Dataset; Flow derives its metadata under the `metadata:` policy the evaluator names.
- **Makes:** a `factor-summary` Output: a mapping of `factors`, the kept factor names, and `summary`, each factor's
  type, level, binning, nulls, and its range or top values, and each dropped factor with its reasons.

**Settings** ({py:class}`~dataeval_flow.evaluators.bias.FactorSummaryConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `metadata` | (DataEval Flow) the name of a `metadata:` policy | DataEval's default encoding |

- **Judged by:** none
- **Used in:** [`audit`](presets.md#audit), [`data-coverage`](presets.md#data-coverage)

```yaml
metadata:
  - name: standard
    exclude: [id]

evaluators:
  - {name: factor-summary, type: factor-summary, metadata: standard}

tasks:
  - {name: factor-summary, evaluator: factor-summary, sources: [train]}
```

## Are the splits fit to evaluate on?

### `factor-leakage`

The raw values of named metadata factors each of two sources holds.

It counts the items each source holds each value in. It reads each factor as the dataset recorded it, whatever the
policy's `exclude` says, because group factors such as scene, sequence or site IDs are the high-cardinality columns a
policy usually excludes. A level-prefixed name (`unit_scene`) matches its bare form. A source that lacks a named factor
fails the run, naming the source. The `leakage` check judges the values both sources share.

- **Reads:** `input`: two Datasets, the first source then the second; Flow derives each one's metadata under the
  `metadata:` policy the evaluator names.
- **Makes:** a `factor-leakage` Output: a mapping of `sources`, `items` and `factors`, each factor's values with their
  item counts in the first and the second source.

**Settings** ({py:class}`~dataeval_flow.evaluators.quality.FactorLeakageConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `metadata` | (DataEval Flow) the name of a `metadata:` policy | DataEval's default encoding |
| `factors` | (DataEval Flow) the factors to compare, by name | required |

- **Judged by:** [`leakage`](checks.md#leakage)
- **Used in:** [`audit`](presets.md#audit)

```yaml
evaluators:
  - {name: factor-leakage, type: factor-leakage, factors: [scene]}

tasks:
  - {name: factor-leakage, evaluator: factor-leakage, sources: [train, test]}
```

### `divergence`

How far apart two sources' embeddings sit (DataEval divergence_mst, divergence_fnn).

The distance runs from 0 (they overlap) to 1 (they are wholly apart), counted over the minimum spanning tree of both
sources together (`mst`) or by nearest-neighbour disagreement (`fnn`). It is a distance, not a test, and is explained in
DataEval's [Distribution Shift explanation](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html).
It runs `dataeval.core.divergence_mst` or `divergence_fnn`.

- **Reads:** `input`: two Datasets, the first source then the second; Flow derives their embeddings through the task's
  extractor.
- **Makes:** a `divergence` Output: a mapping of `divergence`, `errors` and `method`.

**Settings** ({py:class}`~dataeval_flow.evaluators.shift.DivergenceConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `method` | (DataEval Flow) `mst` or `fnn` | `mst` |

- **Judged by:** [`distribution-shift`](checks.md#distribution-shift)
- **Used in:** [`audit`](presets.md#audit)

```yaml
evaluators:
  - {name: divergence, type: divergence, method: fnn}

tasks:
  - {name: divergence, evaluator: divergence, sources: [train, test], extractor: bovw_ext}
```

## Has new data drifted?

The drift evaluators compare a task's sources through its extractor, by position: the first source is the reference, and
the last is the data to test. They are explained in DataEval's [Distribution Shift
explanation](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html), and their classes are
documented in the [DataEval `dataeval.shift`
reference](https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/shift/index.html).

Every drift type takes `chunking:`, a {py:class}`~dataeval_flow.evaluators.shift.ChunkedDriftConfig`. It tests each
chunk of the data against the spread of the reference's chunks, rather than the data as a whole.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `chunk_size` | `chunked(chunk_size=...)` | set this or `chunk_count` |
| `chunk_count` | `chunked(chunk_count=...)` | set this or `chunk_size` |
| `threshold` | `chunked(threshold=...)`: a method, bounds, or `[method, bounds]` | the detector's default |
| `incomplete` | `chunked(incomplete=...)`: `keep`, `drop` or `append` the reference's short final chunk; with `chunk_size` only | DataEval's default (`keep`) |

Each drift type's output is a mapping: `drifted`, `distance`, `threshold`, `metric_name`, `feature_names`, and
`details`. `details` holds the test's statistics, or, with `chunking`, a table with one row per chunk.

### `drift-univariate`

Per-dimension statistical tests for drift (DataEval DriftUnivariate).

Each embedding dimension is tested on its own, and drift is declared when any drifts after a multiple-testing
correction. It runs `dataeval.shift.DriftUnivariate`.

- **Reads:** `input`: two Datasets, the reference then the data to test; Flow derives their embeddings through the
  task's extractor.
- **Makes:** a drift Output, the mapping [described above](#has-new-data-drifted).

**Settings** ({py:class}`~dataeval_flow.evaluators.shift.DriftUnivariateConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `method` | `method`: `ks`, `cvm`, `mwu`, `anderson` or `bws` | DataEval's default (`ks`) |
| `p_val` | `p_val` | DataEval's default (`0.05`) |
| `correction` | `correction`: `bonferroni` or `fdr` | DataEval's default (`bonferroni`) |
| `alternative` | `alternative`: `two-sided`, `less` or `greater` | DataEval's default (`two-sided`) |
| `n_features` | `n_features` | inferred from the embeddings |
| `chunking` | (DataEval Flow) `chunked(...)`, above | the data is tested whole |

- **Judged by:** [`drift`](checks.md#drift)
- **Used in:** [`drift-monitoring`](presets.md#drift-monitoring)

```yaml
evaluators:
  - {name: drift-univariate, type: drift-univariate, method: cvm}

tasks:
  - {name: drift-univariate, evaluator: drift-univariate, sources: [train, operational], extractor: bovw_ext}
```

### `drift-mmd`

Maximum mean discrepancy between reference and test (DataEval DriftMMD).

The discrepancy is tested against a permutation estimate of its no-drift distribution. It runs
`dataeval.shift.DriftMMD`.

- **Reads:** `input`: two Datasets, the reference then the data to test; Flow derives their embeddings through the
  task's extractor.
- **Makes:** a drift Output, the mapping [described above](#has-new-data-drifted).

**Settings** ({py:class}`~dataeval_flow.evaluators.shift.DriftMMDConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `p_val` | `p_val` | DataEval's default (`0.05`) |
| `n_permutations` | `n_permutations` | DataEval's default (`100`) |
| `permutation_batch_size` | `permutation_batch_size`: a count, or `auto` | DataEval's default (`auto`) |
| `chunking` | (DataEval Flow) `chunked(...)`, above | the data is tested whole |

- **Judged by:** [`drift`](checks.md#drift)
- **Used in:** [`drift-monitoring`](presets.md#drift-monitoring)

```yaml
evaluators:
  - {name: drift-mmd, type: drift-mmd, n_permutations: 200}

tasks:
  - {name: drift-mmd, evaluator: drift-mmd, sources: [train, operational], extractor: bovw_ext}
```

### `drift-kneighbors`

Nearest-neighbor distances to the reference, tested for drift (DataEval DriftKNeighbors).

The data's distances to their nearest reference neighbors are compared with the reference's own. It runs
`dataeval.shift.DriftKNeighbors`.

- **Reads:** `input`: two Datasets, the reference then the data to test; Flow derives their embeddings through the
  task's extractor.
- **Makes:** a drift Output, the mapping [described above](#has-new-data-drifted).

**Settings** ({py:class}`~dataeval_flow.evaluators.shift.DriftKNeighborsConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `k` | `k` | DataEval's default (`10`) |
| `distance_metric` | `distance_metric`: `cosine` or `euclidean` | DataEval's default (`euclidean`) |
| `p_val` | `p_val` (without chunking) | DataEval's default (`0.05`) |
| `chunking` | (DataEval Flow) `chunked(...)`, above | the data is tested whole |

- **Judged by:** [`drift`](checks.md#drift)
- **Used in:** [`drift-monitoring`](presets.md#drift-monitoring)

```yaml
evaluators:
  - {name: drift-kneighbors, type: drift-kneighbors, k: 5}

tasks:
  - {name: drift-kneighbors, evaluator: drift-kneighbors, sources: [train, operational], extractor: bovw_ext}
```

### `drift-wasserstein`

Per-dimension Wasserstein distance against a validation baseline (DataEval DriftWasserstein).

Each dimension's Wasserstein distance from the reference to the data is set against its distance to the validation set.
It runs `dataeval.shift.DriftWasserstein`. `drift-monitoring` takes no validation source and refuses it, so chain it
as [Drift in a model's uncertainty](../how_to/monitor_drift.md#6-drift-in-a-models-uncertainty) does.

- **Reads:** `input`: three Datasets, the reference, an in-distribution validation set (the task's middle source, which
  DataEval requires), then the data to test; Flow derives their embeddings through the task's extractor.
- **Makes:** a drift Output, the mapping [described above](#has-new-data-drifted).

**Settings** ({py:class}`~dataeval_flow.evaluators.shift.DriftWassersteinConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `ratio_threshold` | `ratio_threshold` | DataEval's default (`1.4`) |
| `n_features` | `n_features` | inferred from the embeddings |
| `chunking` | (DataEval Flow) `chunked(...)`, above | the data is tested whole |

- **Judged by:** [`drift`](checks.md#drift)
- **Used in:** none; chain it in a [workflow of your own](../how_to/write_a_custom_workflow.md)

```yaml
evaluators:
  - {name: drift-wasserstein, type: drift-wasserstein, ratio_threshold: 1.5}

tasks:
  - {name: drift-wasserstein, evaluator: drift-wasserstein, sources: [train, validation, operational], extractor: bovw_ext}
```

### `drift-domain-classifier`

A classifier's ability to tell reference from test (DataEval DriftDomainClassifier).

The classifier is trained to tell the reference from the data under cross-validation. Drift is declared when it does
better than `threshold` (AUROC). It runs `dataeval.shift.DriftDomainClassifier`.

- **Reads:** `input`: two Datasets, the reference then the data to test; Flow derives their embeddings through the
  task's extractor.
- **Makes:** a drift Output, the mapping [described above](#has-new-data-drifted).

**Settings** ({py:class}`~dataeval_flow.evaluators.shift.DriftDomainClassifierConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `n_folds` | `n_folds` | DataEval's default (`5`) |
| `threshold` | `threshold`: an AUROC, or with `chunking` a `[lower, upper]` pair | DataEval's default (`0.55`) |
| `chunking` | (DataEval Flow) `chunked(...)`, above | the data is tested whole |

- **Judged by:** [`drift`](checks.md#drift)
- **Used in:** [`drift-monitoring`](presets.md#drift-monitoring)

```yaml
evaluators:
  - {name: drift-domain-classifier, type: drift-domain-classifier, n_folds: 3}

tasks:
  - {name: drift-domain-classifier, evaluator: drift-domain-classifier, sources: [train, operational], extractor: bovw_ext}
```

## Which items are out of distribution?

The OOD evaluators read their sources by the same rule as the [drift evaluators](#has-new-data-drifted): the first
source is the reference, and the last is the data to test.

### `ood-kneighbors`

Test items far from their nearest reference neighbors (DataEval OODKNeighbors).

Each test item is scored by its distance to its nearest reference neighbors, and flagged beyond the distance
`threshold_perc` percent of the reference stays within. It runs `dataeval.shift.OODKNeighbors`. On an `uncertainty`
extractor's rows it needs `distance_metric: euclidean`, since cosine distance cannot rank one number. `eval-coverage`
relates the share of its Output flagged to that percentile of train.

- **Reads:** `input`: two Datasets, the reference then the data to test; Flow derives their embeddings through the
  task's extractor.
- **Makes:** an `ood` Output: a mapping. `is_ood` and `instance_score` hold one value per test item, and `feature_score`
  is `null`.

**Settings** ({py:class}`~dataeval_flow.evaluators.shift.OODKNeighborsConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `k` | `k` | DataEval's default (`10`) |
| `distance_metric` | `distance_metric`: `cosine` or `euclidean` | DataEval's default (`cosine`) |
| `threshold_perc` | `threshold_perc`, 0 to 100 | DataEval's default (`95`) |

- **Judged by:** [`eval-coverage`](checks.md#eval-coverage), [`ood`](checks.md#ood)
- **Combined by:** [`factor-deviation`](combines.md#factor-deviation),
  [`factor-predictors`](combines.md#factor-predictors), [`ood-union`](combines.md#ood-union)
- **Used in:** [`audit`](presets.md#audit), [`ood-detection`](presets.md#ood-detection)

```yaml
evaluators:
  - {name: ood-kneighbors, type: ood-kneighbors, distance_metric: euclidean}

tasks:
  - {name: ood-kneighbors, evaluator: ood-kneighbors, sources: [train, operational], extractor: bovw_ext}
```

### `ood-domain-classifier`

Test items a classifier tells apart from the reference (DataEval OODDomainClassifier).

The classifier is trained to tell each test item from the reference under repeated cross-validation, and the items it
separates well are flagged. It runs `dataeval.shift.OODDomainClassifier`. `eval-coverage` judges the share of its
Output flagged; it relates the share to a percentile of train only for `ood-kneighbors`.

- **Reads:** `input`: two Datasets, the reference then the data to test; Flow derives their embeddings through the
  task's extractor.
- **Makes:** an `ood` Output: a mapping. `is_ood` and `instance_score` hold one value per test item, and `feature_score`
  is `null`.

**Settings** ({py:class}`~dataeval_flow.evaluators.shift.OODDomainClassifierConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `n_folds` | `n_folds` | DataEval's default (`5`) |
| `n_repeats` | `n_repeats` | DataEval's default (`5`) |
| `n_std` | `n_std` (without `threshold_perc`) | DataEval's default (`2.0`) |
| `hyperparameters` | `hyperparameters`: LightGBM's | DataEval's |
| `threshold_perc` | `threshold_perc`, 0 to 100; overrides `n_std` | `n_std` sets the threshold |

- **Judged by:** [`eval-coverage`](checks.md#eval-coverage), [`ood`](checks.md#ood)
- **Combined by:** [`factor-deviation`](combines.md#factor-deviation),
  [`factor-predictors`](combines.md#factor-predictors), [`ood-union`](combines.md#ood-union)
- **Used in:** [`ood-detection`](presets.md#ood-detection)

```yaml
evaluators:
  - {name: ood-domain-classifier, type: ood-domain-classifier, n_folds: 3}

tasks:
  - {name: ood-domain-classifier, evaluator: ood-domain-classifier, sources: [train, operational], extractor: bovw_ext}
```

## Which items should be labeled next?

`prioritization` reads its first source's embeddings through the task's extractor, and its labels where there is one per
item. Its class is documented in the [DataEval `dataeval.scope`
reference](https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/scope/index.html).

### `prioritization`

Items ranked by difficulty, optionally against a reference (DataEval Prioritize).

Items are ranked from easiest to hardest, or the reverse. A second source is the reference, and the ranking is then
relative to it, as when choosing what to label next beside data already labeled. It runs `dataeval.scope.Prioritize`.
The [`data-prioritization`](presets.md#data-prioritization) preset reads its sources the other way round, the
reference first and then the pools, and hands each pool to this evaluator first.

- **Reads:** `input`: one Dataset, or two: the data to rank, then a reference; Flow derives their embeddings through the
  task's extractor, and the labels where there is one per item.
- **Makes:** a `prioritization` Output: an array of the first source's item indices in ranked order. `extras` holds each
  item's `scores`.

**Settings** ({py:class}`~dataeval_flow.evaluators.scope.PrioritizationConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `method` | `method`: `knn`, `kmeans_distance`, `kmeans_complexity`, `hdbscan_distance` or `hdbscan_complexity` | DataEval's default (`knn`) |
| `k` | `k` (`knn`) | the square root of the item count |
| `c` | `c` (clustering methods) | the square root of the item count |
| `n_init` | `n_init` (`kmeans_*`): a count, or `auto` | DataEval's default (`auto`) |
| `max_cluster_size` | `max_cluster_size` (`hdbscan_*`) | unbounded |
| `order` | `order`: `easy_first` or `hard_first` | DataEval's default (`easy_first`) |
| `policy` | `policy`: `difficulty`, `stratified` or `class_balanced` | DataEval's default (`difficulty`) |
| `num_bins` | `num_bins` (`stratified`) | DataEval's default (`50`) |

- **Judged by:** none
- **Used in:** [`data-prioritization`](presets.md#data-prioritization)

```yaml
evaluators:
  - {name: prioritization, type: prioritization, order: hard_first}

tasks:
  - {name: prioritization, evaluator: prioritization, sources: [unlabeled, labeled], extractor: bovw_ext}
```

## Is the metadata readable?

### `factor-triage`

What a Dataset's metadata failed to read, and a policy that repairs it.

It reports each factor the run could not read as configured, in six categories, with a correction or bin count where one
repairs it. It merges every suggestion into one `metadata:` stanza, and, with `verify`, reads the metadata back under
the suggestions to say what each recovered. It also recommends a policy: the suggestions, each value triage could not
read dropped to missing, plus explicit edges or levels for every factor the policy left unpinned, read back from this
data. A policy read from data that does not represent what you expect can give invalid or misleading results, and the
recommendation says so first. It reads the Dataset's metadata through DataEval's `Metadata`, whose `repair` verifies
without a second walk.

- **Reads:** `input`: one Dataset; Flow derives its metadata under the `metadata:` policy the evaluator names.
- **Makes:** a `factor-triage` Output: a mapping of:

  - `findings`, each issue with its `factor`, `category`, `severity`, `reasons`, `remedy` and `suggestion`;
  - `suggested_policy` and `suggested_policy_yaml`;
  - `verification` and `verification_error`;
  - `recommended_policy`, `recommended_policy_yaml` and `recommendation_error`;
  - `counts`, by category and severity;
  - `factor_count`;
  - `places`, where each mixed column's problem values sit.

**Settings** ({py:class}`~dataeval_flow.evaluators.quality.FactorTriageConfig`):

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `metadata` | (DataEval Flow) a policy under `metadata:`, which the metadata is built under | DataEval's defaults |
| `verify` | (DataEval Flow) read the metadata back under the complete suggestions, and say what each recovered | `true` |
| `default_bins` | (DataEval Flow) the bin count a suggestion falls back to where the run left no cut to read | `10` |
| `min_missing_fraction` | (DataEval Flow) the share of rows recording no value above which a factor is degenerate | `0.2` |

- **Judged by:** [`metadata-issues`](checks.md#metadata-issues)
- **Used in:** [`audit`](presets.md#audit), [`metadata-triage`](presets.md#metadata-triage)

```yaml
evaluators:
  - {name: factor-triage, type: factor-triage, min_missing_fraction: 0.3}

tasks:
  - {name: factor-triage, evaluator: factor-triage, sources: [train]}
```

## What exactly was evaluated?

### `content-digest`

SHA-256 digests of every item's image and labels, and of its metadata.

Every item is read from the Dataset itself and never through a cache. A content digest covers each item's image and
labels and the class names, and a metadata digest covers each item's metadata, as attached to that item. Neither depends
on the items' order. {py:func}`~dataeval_flow.dataset_digest` computes the same values in Python.

The Output also keeps each item's hash, as a manifest, `output.manifest()`, in memory only. With `--output`, the
command writes it under `results/manifests/<task>/`: as `<step>.json` for a step of a chain, `<step>/<key>.json` for
each run of a step run once per element, and `content-digest.json` for a task that runs the evaluator alone.
`dataeval-flow verify` checks a source against it; see
[Gate training on an audit](../how_to/gate_training_on_an_audit.md#find-what-changed).

- **Reads:** `input`: one Dataset; Flow reads every item from it directly, and derives nothing.
- **Makes:** a `content-digest` Output: a mapping of `content` and `metadata`, each 64 hex characters, `items`, how many
  items were read, and `scheme`, the version of the digest scheme.

**Settings** ({py:class}`~dataeval_flow.evaluators.quality.ContentDigestConfig`):

It takes no parameters.

- **Judged by:** none
- **Used in:** [`audit`](presets.md#audit)

```yaml
evaluators:
  - {name: content-digest, type: content-digest}

tasks:
  - {name: content-digest, evaluator: content-digest, sources: [train]}
```

## Not in the catalog yet

DataEval's `DriftReconstruction`, `OODReconstruction` and `Sufficiency` train a PyTorch model, which a config file
cannot describe yet.
