# Evaluator Catalog

Each evaluator runs one DataEval evaluator and reports its determinations, with no
health status (see [Workflows and Evaluators](../concepts/WorkflowsAndEvaluators.md)).
An evaluator's `type` is DataEval's module and class, in kebab case. To run one, see
[Run a single evaluator](../how_to/run_a_single_evaluator.md).

## At a glance

| Type | DataEval class | Consumes | Sources | Extractor |
| --- | --- | --- | --- | --- |
| `bias.balance` | `dataeval.bias.Balance` | metadata | 1 | refused |
| `bias.diversity` | `dataeval.bias.Diversity` | metadata | 1 | refused |
| `bias.parity` | `dataeval.bias.Parity` | metadata | 1 | refused |
| `quality.duplicates` | `dataeval.quality.Duplicates` | stats; clusters in cluster mode | 1 or more; 1 in cluster mode | needed in cluster mode; accepted but unused otherwise |
| `quality.outliers` | `dataeval.quality.Outliers` | stats; clusters in cluster mode | 1 or more; 1 in cluster mode | needed in cluster mode; accepted but unused otherwise |
| `scope.representation` | `dataeval.scope.Representation` | labels | 1 | refused |
| `scope.coverage` | `dataeval.scope.Coverage` | embeddings; labels where there is one per item | 1 | required |
| `scope.prioritize` | `dataeval.scope.Prioritize` | embeddings; labels where there is one per item | 1, or 2: the data, then a reference | required |
| `shift.drift-domain-classifier` | `dataeval.shift.DriftDomainClassifier` | embeddings | 2: the reference, then the data to test | required |
| `shift.drift-kneighbors` | `dataeval.shift.DriftKNeighbors` | embeddings | 2: the reference, then the data to test | required |
| `shift.drift-mmd` | `dataeval.shift.DriftMMD` | embeddings | 2: the reference, then the data to test | required |
| `shift.drift-univariate` | `dataeval.shift.DriftUnivariate` | embeddings | 2: the reference, then the data to test | required |
| `shift.drift-wasserstein` | `dataeval.shift.DriftWasserstein` | embeddings | 3: the reference, a validation set, then the data to test | required |
| `shift.ood-domain-classifier` | `dataeval.shift.OODDomainClassifier` | embeddings | 2: the reference, then the data to test | required |
| `shift.ood-kneighbors` | `dataeval.shift.OODKNeighbors` | embeddings | 2: the reference, then the data to test | required |

A task's `extractor:` still lands in the result envelope's `model_id`, whether or not that
run's mode actually reads it.

## How parameters work

Parameter names are DataEval's argument names, unchanged. A parameter you leave out
is not passed, so DataEval's own default applies. An unknown parameter fails the
config load, as does any value DataEval itself refuses, such as an unknown threshold
method. Print any evaluator's JSON Schema with `dataeval-flow evaluators <type>`.

## Bias

The bias evaluators read one source's metadata, built under the metadata policy `metadata:` names, and need no
extractor. They are explained in DataEval's
[Dataset Bias explanation](https://dataeval.readthedocs.io/en/latest/concepts/DatasetBias.html), and their classes
are documented in the
[DataEval `dataeval.bias` reference](https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/bias/index.html).

### `bias.balance`

How much each metadata factor tells about the class label, and each pair of factors about each other.
Configured by {py:class}`~dataeval_flow.evaluators.bias.BalanceConfig`; runs `dataeval.bias.Balance`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `metadata` | (DataEval Flow) the name of a `metadata:` policy | DataEval's default encoding |
| `num_neighbors` | `num_neighbors` | DataEval's default (`5`) |
| `class_imbalance_threshold` | `class_imbalance_threshold` | DataEval's default (`0.3`) |
| `factor_correlation_threshold` | `factor_correlation_threshold` | DataEval's default (`0.5`) |
| `label` | `label`: a factor, or a list of factors, to condition on | the class labels |
| `factor_source` | `factor_source`: `coded`, `values` or `auto` | the policy's `factor_source`, else DataEval's default (`auto`) |

Output: a mapping of three tables. `balance` has each factor's mutual information with the class labels, `factors`
each factor pair's, and `classwise` each class's against each factor.

### `bias.diversity`

How evenly each metadata factor's values are spread, overall and within each class. Configured by
{py:class}`~dataeval_flow.evaluators.bias.DiversityConfig`; runs `dataeval.bias.Diversity`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `metadata` | (DataEval Flow) the name of a `metadata:` policy | DataEval's default encoding |
| `method` | `method`: `simpson` or `shannon` | DataEval's default (`simpson`) |
| `threshold` | `threshold` | DataEval's default (`0.5`) |
| `label` | `label`: a factor, or a list of factors, to condition on | the class labels |

Output: a mapping of two tables. `factors` has each factor's diversity and whether it is low, and `classwise` each
class's.

### `bias.parity`

Which metadata factors are associated with the class label. Configured by
{py:class}`~dataeval_flow.evaluators.bias.ParityConfig`; runs `dataeval.bias.Parity`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `metadata` | (DataEval Flow) the name of a `metadata:` policy | DataEval's default encoding |
| `score_threshold` | `score_threshold` | DataEval's default (`0.3`) |
| `p_value_threshold` | `p_value_threshold` | DataEval's default (`0.05`) |
| `label` | `label`: a factor, or a list of factors, to condition on | the class labels |

Output: a mapping. `factors` is a table of each factor's Cramér's V, p-value and significance, and
`insufficient_data` lists the factor values with too few samples per class.

## Quality

Both quality evaluators are explained in DataEval's
[Data Integrity explanation](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html),
and their classes are documented in the
[DataEval `dataeval.quality` reference](https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/quality/index.html).

### `quality.duplicates`

Which images are exact or near duplicates of each other, across every source the
task names. Configured by {py:class}`~dataeval_flow.evaluators.quality.DuplicatesConfig`;
runs `dataeval.quality.Duplicates`.

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

Output: a table with one row per duplicate group (`group_id`, `level`, `dup_type`,
`item_indices`, `methods`, and `dataset_indices` when the task names several sources).
`extras` holds `annotation_divergences` and `factor_cardinality`, `null` unless the annotation or factor axis
ran.

### `quality.outliers`

Which images' statistics sit outside the threshold, across every source the task
names. Configured by {py:class}`~dataeval_flow.evaluators.quality.OutliersConfig`; runs
`dataeval.quality.Outliers`.

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

Output: a table with one row per flagged statistic (`item_index`, `metric_name`,
`metric_value`, and `target_index` for per-target results).

## Scope

The scope evaluators read one source's labels, and, for coverage and prioritization, its embeddings through the
task's extractor. `scope.representation` is explained in DataEval's
[Ontology explanation](https://dataeval.readthedocs.io/en/latest/concepts/Ontology.html), and the classes are
documented in the
[DataEval `dataeval.scope` reference](https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/scope/index.html).

### `scope.representation`

Which of an ontology's classes fall short of their share of the source, and what to acquire. It needs a dataset
with labels, and fails, naming the source, without them. Configured by
{py:class}`~dataeval_flow.evaluators.scope.RepresentationConfig`; runs `dataeval.scope.Representation`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `ontology` | (DataEval Flow) a name under `ontologies:`, an RDF file, or an inline hierarchy; becomes `Representation(ontology)` | a flat ontology synthesized from the dataset's `index2label` |
| `expected` | `expected`: class name to its minimum share, a fraction | a uniform share for every leaf |

Output: a table, the worklist, with one row per concept short of its target (`concept`, `label`, `parent`,
`action`, `count`, `target`, `deficit`). `extras` holds `leaf_coverage`, `total_deficit`, and the `violations` and
`dark_branches` tables.

### `scope.coverage`

Which items sit in sparse regions of the embedding space, uncovered by the rest of the data, broken down by class.
With no label per item (a dataset without labels, or a detection dataset's labels per target), it runs over every
item as one class, `0`, and logs a warning. `data-coverage` crops detections first; this evaluator does not.
Configured by {py:class}`~dataeval_flow.evaluators.scope.CoverageConfig`; runs `dataeval.scope.Coverage`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `method` | `method`: `naive` or `adaptive` | DataEval's default (`adaptive`) |
| `num_observations` | `num_observations`; must be fewer than the source's items | DataEval's default (`20`) |
| `percent` | `percent` (`adaptive` only) | DataEval's default (`0.01`) |
| `min_class_samples` | `min_class_samples` | DataEval's default (`20`) |
| `isotropy_min_samples` | `isotropy_min_samples` | one more than the embedding size |
| `near_duplicate_factor` | `near_duplicate_factor` | DataEval's default (`0.5`) |

Output: a table with one row per class (`class`, `count`, `uncovered`, `uncovered_fraction`, `dispersion`,
`isotropy`, `near_duplicate_fraction`, `assessable`). `extras` holds `uncovered_indices`, `coverage_radius` and
`critical_value_radii`.

### `scope.prioritize`

The first source's items ranked from easiest to hardest, or the reverse. A second source is the reference: the
ranking is then relative to it, as when choosing what to label next beside data already labeled. Configured by
{py:class}`~dataeval_flow.evaluators.scope.PrioritizeConfig`; runs `dataeval.scope.Prioritize`.

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

Output: an array of the first source's item indices in ranked order. `extras` holds each item's `scores`.

## Shift

The shift evaluators compare a task's sources through its extractor, by position: the first source is the
reference, and the last is the data to test. They are explained in DataEval's
[Distribution Shift explanation](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html), and
their classes are documented in the
[DataEval `dataeval.shift` reference](https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/shift/index.html).

Every drift type takes `chunking:`, a {py:class}`~dataeval_flow.evaluators.shift.ChunkedDriftConfig`. It tests
each chunk of the data against the spread of the reference's chunks, rather than the data as a whole.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `chunk_size` | `chunked(chunk_size=...)` | set this or `chunk_count` |
| `chunk_count` | `chunked(chunk_count=...)` | set this or `chunk_size` |
| `threshold` | `chunked(threshold=...)`: a method, bounds, or `[method, bounds]` | the detector's default |
| `incomplete` | `chunked(incomplete=...)`: `keep`, `drop` or `append` the reference's short final chunk; with `chunk_size` only | DataEval's default (`keep`) |

Each drift type's output is a mapping: `drifted`, `distance`, `threshold`, `metric_name`, `feature_names`, and
`details`. `details` holds the test's statistics, or, with `chunking`, a table with one row per chunk.

### `shift.drift-univariate`

Each embedding dimension tested on its own, drift declared when any drifts after a multiple-testing correction.
Configured by {py:class}`~dataeval_flow.evaluators.shift.DriftUnivariateConfig`; runs
`dataeval.shift.DriftUnivariate`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `method` | `method`: `ks`, `cvm`, `mwu`, `anderson` or `bws` | DataEval's default (`ks`) |
| `p_val` | `p_val` | DataEval's default (`0.05`) |
| `correction` | `correction`: `bonferroni` or `fdr` | DataEval's default (`bonferroni`) |
| `alternative` | `alternative`: `two-sided`, `less` or `greater` | DataEval's default (`two-sided`) |
| `n_features` | `n_features` | inferred from the embeddings |
| `chunking` | (DataEval Flow) `chunked(...)`, above | the data is tested whole |

### `shift.drift-mmd`

The maximum mean discrepancy between the two sources, tested against a permutation estimate of its no-drift
distribution. Configured by {py:class}`~dataeval_flow.evaluators.shift.DriftMMDConfig`; runs
`dataeval.shift.DriftMMD`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `p_val` | `p_val` | DataEval's default (`0.05`) |
| `n_permutations` | `n_permutations` | DataEval's default (`100`) |
| `permutation_batch_size` | `permutation_batch_size`: a count, or `auto` | DataEval's default (`auto`) |
| `chunking` | (DataEval Flow) `chunked(...)`, above | the data is tested whole |

### `shift.drift-kneighbors`

The data's distances to their nearest reference neighbors, compared with the reference's own. Configured by
{py:class}`~dataeval_flow.evaluators.shift.DriftKNeighborsConfig`; runs `dataeval.shift.DriftKNeighbors`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `k` | `k` | DataEval's default (`10`) |
| `distance_metric` | `distance_metric`: `cosine` or `euclidean` | DataEval's default (`euclidean`) |
| `p_val` | `p_val` (without chunking) | DataEval's default (`0.05`) |
| `chunking` | (DataEval Flow) `chunked(...)`, above | the data is tested whole |

### `shift.drift-wasserstein`

Each dimension's Wasserstein distance from the reference to the data, against its distance to an in-distribution
validation set: the task's middle source, which DataEval requires. Configured by
{py:class}`~dataeval_flow.evaluators.shift.DriftWassersteinConfig`; runs `dataeval.shift.DriftWasserstein`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `ratio_threshold` | `ratio_threshold` | DataEval's default (`1.4`) |
| `n_features` | `n_features` | inferred from the embeddings |
| `chunking` | (DataEval Flow) `chunked(...)`, above | the data is tested whole |

### `shift.drift-domain-classifier`

A classifier trained to tell the reference from the data under cross-validation. Drift is declared when it does
better than `threshold` (AUROC). Configured by
{py:class}`~dataeval_flow.evaluators.shift.DriftDomainClassifierConfig`; runs
`dataeval.shift.DriftDomainClassifier`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `n_folds` | `n_folds` | DataEval's default (`5`) |
| `threshold` | `threshold`: an AUROC, or with `chunking` a `[lower, upper]` pair | DataEval's default (`0.55`) |
| `chunking` | (DataEval Flow) `chunked(...)`, above | the data is tested whole |

### `shift.ood-kneighbors`

Each test item scored by its distance to its nearest reference neighbors, flagged beyond the distance
`threshold_perc` percent of the reference stays within. Configured by
{py:class}`~dataeval_flow.evaluators.shift.OODKNeighborsConfig`; runs `dataeval.shift.OODKNeighbors`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `k` | `k` | DataEval's default (`10`) |
| `distance_metric` | `distance_metric`: `cosine` or `euclidean` | DataEval's default (`cosine`) |
| `threshold_perc` | `threshold_perc`, 0 to 100 | DataEval's default (`95`) |

Output: a mapping. `is_ood` and `instance_score` hold one value per test item, and `feature_score` is `null`.

### `shift.ood-domain-classifier`

A classifier trained to tell each test item from the reference under repeated cross-validation, flagging the items
it separates well. Configured by {py:class}`~dataeval_flow.evaluators.shift.OODDomainClassifierConfig`; runs
`dataeval.shift.OODDomainClassifier`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `n_folds` | `n_folds` | DataEval's default (`5`) |
| `n_repeats` | `n_repeats` | DataEval's default (`5`) |
| `n_std` | `n_std` (without `threshold_perc`) | DataEval's default (`2.0`) |
| `hyperparameters` | `hyperparameters`: LightGBM's | DataEval's |
| `threshold_perc` | `threshold_perc`, 0 to 100; overrides `n_std` | `n_std` sets the threshold |

Output: a mapping. `is_ood` and `instance_score` hold one value per test item, and `feature_score` is `null`.

## Not in the catalog yet

`shift.drift-reconstruction`, `shift.ood-reconstruction` and `performance.sufficiency` train a PyTorch model,
which a config file cannot describe yet.
