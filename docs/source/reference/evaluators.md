# Evaluator Catalog

Each evaluator runs one DataEval evaluator and reports its determinations, with no
health status (see [Workflows and Evaluators](../concepts/WorkflowsAndEvaluators.md)).
An evaluator's `type` is DataEval's module and class, in kebab case. To run one, see
[Run a single evaluator](../how_to/run_a_single_evaluator.md).

## At a glance

| Type | DataEval class | Consumes | Sources | Extractor |
| --- | --- | --- | --- | --- |
| `quality.duplicates` | `dataeval.quality.Duplicates` | stats; clusters in cluster mode | 1 or more; 1 in cluster mode | needed in cluster mode; accepted but unused otherwise |
| `quality.outliers` | `dataeval.quality.Outliers` | stats; clusters in cluster mode | 1 or more; 1 in cluster mode | needed in cluster mode; accepted but unused otherwise |

A task's `extractor:` still lands in the result envelope's `model_id`, whether or not that
run's mode actually reads it.

## How parameters work

Parameter names are DataEval's argument names, unchanged. A parameter you leave out
is not passed, so DataEval's own default applies. An unknown parameter fails the
config load, as does any value DataEval itself refuses, such as an unknown threshold
method. Print any evaluator's JSON Schema with `dataeval-flow evaluators <type>`.

## Quality

Both quality evaluators are explained in DataEval's
[Data Integrity explanation](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html),
and their classes are documented in the
[DataEval `dataeval.quality` reference](https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/quality/index.html).

### `quality.duplicates`

Which images are exact or near duplicates of each other, across every source the
task names. Configured by {py:class}`~dataeval_flow.config.DuplicatesEvaluatorConfig`;
runs `dataeval.quality.Duplicates`.

| Parameter | DataEval argument | Left unset |
| --- | --- | --- |
| `stats` | (DataEval Flow) the name of a `stats:` policy | the whole image is measured |
| `flags` | `flags`: `hash_basic`, `hash_d4` | DataEval's default hash families |
| `merge_near_duplicates` | `merge_near_duplicates` | DataEval's default |
| `hash_radius` | `hash_radius` | DataEval's default (`0`, documented to change in a future major release) |
| `cluster_sensitivity` | `cluster_sensitivity`; setting it turns on cluster mode | cluster mode is off |
| `cluster_algorithm` | `cluster_algorithm`: `kmeans` or `hdbscan` | DataEval's default |
| `n_clusters` | `n_clusters` | DataEval chooses |
| `per_image` | `from_stats(per_image=...)` | DataEval's default |
| `per_target` | `from_stats(per_target=...)` | DataEval's default |

Output: a table with one row per duplicate group (`group_id`, `level`, `dup_type`,
`item_indices`, `methods`, and `dataset_indices` when the task names several sources).

### `quality.outliers`

Which images' statistics sit outside the threshold, across every source the task
names. Configured by {py:class}`~dataeval_flow.config.OutliersEvaluatorConfig`; runs
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

- `bias.balance`, `bias.diversity`, `bias.parity`: read the metadata policy
- `scope.representation`: reads labels against an ontology
- `scope.coverage`, `scope.prioritize`: read embeddings
- `shift.drift-univariate`, `shift.drift-mmd`, `shift.drift-kneighbors`,
  `shift.drift-wasserstein`, `shift.drift-domain-classifier`: read reference and
  test embeddings, with an optional `chunking:` block
- `shift.ood-kneighbors`, `shift.ood-domain-classifier`: read reference and test
  embeddings
