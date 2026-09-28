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

## Planned

These follow in later releases, under the same rules:

- `scope.representation`: reads labels against an ontology
- `scope.coverage`, `scope.prioritize`: read embeddings
- `shift.drift-univariate`, `shift.drift-mmd`, `shift.drift-kneighbors`,
  `shift.drift-wasserstein`, `shift.drift-domain-classifier`: read reference and
  test embeddings, with an optional `chunking:` block
- `shift.ood-kneighbors`, `shift.ood-domain-classifier`: read reference and test
  embeddings
