# Find regions the data does not cover

Which items sit in embedding-space regions the rest of the data does not cover? Run the
[`coverage`](../reference/evaluators.md#coverage) evaluator on a source to find out.
[Dataset Coverage](../concepts/Coverage.md) explains what coverage measures. An evaluator reports DataEval's answer to
its question and renders no health status. The [Evaluator Catalog](../reference/evaluators.md) lists every parameter it
takes. You need a pipeline that defines `datasets:`, a source named `train` and the extractor `bovw_ext` from
[Run a single evaluator](run_a_single_evaluator.md).

## Define the evaluator and task

```yaml
evaluators:
  - name: coverage
    type: coverage
    num_observations: 20

tasks:
  - name: coverage_check
    evaluator: coverage
    sources: train
    extractor: bovw_ext
```

Keep these points in mind:

- The task needs an extractor, because coverage measures in embedding space.
- `num_observations` must be fewer than the source's items.
- Without one label per item, as in an unlabeled or detection dataset, it logs a warning and reports one class, `0`,
  covering every item.
- `is_clustered`, `is_one_dimensional` and `has_near_duplicates` are DataEval's verdict on each class, null where it
  cannot judge. `dispersion_factor`, `isotropy_factor` and `near_duplicate_threshold` set them, and a `class-coverage`
  check reports them.
- `extras` holds `uncovered_indices`, `coverage_radius` and `critical_value_radii`.

## Read the output

`coverage_check`'s output (trimmed; from a small run):

```json
{
  "shape": "table",
  "columns": ["class", "count", "uncovered", "uncovered_fraction", "dispersion", "isotropy",
              "near_duplicate_fraction", "assessable", "is_clustered", "is_one_dimensional",
              "has_near_duplicates"],
  "rows": [
    {"class": "a", "count": 20, "uncovered": 0, "uncovered_fraction": 0.0, "dispersion": 0.9783207970130985,
     "isotropy": null, "near_duplicate_fraction": 0.0, "assessable": true, "is_clustered": false,
     "is_one_dimensional": null, "has_near_duplicates": false},
    {"class": "b", "count": 20, "uncovered": 1, "uncovered_fraction": 0.05, "dispersion": 1.0216792029869015,
     "isotropy": null, "near_duplicate_fraction": 0.0, "assessable": true, "is_clustered": false,
     "is_one_dimensional": null, "has_near_duplicates": false}
  ],
  "extras": {
    "uncovered_indices": [7],
    "coverage_radius": 13.830693244934082,
    "critical_value_radii": [11.495150566101074, 11.507107734680176, 11.457376480102539, 11.39704704284668]
  }
}
```

Class `b` has one item, index 7, that the rest of the data does not cover. `critical_value_radii` holds one radius per
item, the value its own neighborhood needed to reach `num_observations`.

## Run it from Python

```python
from dataeval_flow import run
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.evaluators.scope import CoverageConfig

bovw_ext = BoVWExtractorConfig(name="bovw_ext", vocab_size=512, batch_size=32)
result = run(CoverageConfig(num_observations=20), dataset, extractor=bovw_ext)
```

## Next steps

- [Assess dataset coverage](../notebooks/data_coverage.py) — Work through coverage on a real dataset and read the
  report.
- [Dataset Coverage](../concepts/Coverage.md) — Learn why coverage is asked of both the labels and the embeddings.
- [Read the result envelope](read_the_result_envelope.md) — Read the envelope that evaluators and workflows share.
