# Flag out-of-distribution items

Which items of the operational data sit outside the distribution the reference established? Run an out-of-distribution
evaluator, such as [`ood-kneighbors`](../reference/evaluators.md#ood-kneighbors), on the reference and the data to test.
[Distribution Shift](../concepts/DistributionShift.md) explains how this differs from drift. An evaluator reports
DataEval's answer to its question and renders no health status. The [Evaluator Catalog](../reference/evaluators.md)
lists every parameter each evaluator takes. You need a pipeline that defines `datasets:`, the sources `train` and
`operational`, and the extractor `bovw_ext` from [Run a single evaluator](run_a_single_evaluator.md).

## Define the evaluator and task

```yaml
evaluators:
  - name: ood
    type: ood-kneighbors
    distance_metric: euclidean

tasks:
  - name: ood_check
    evaluator: ood
    sources: [train, operational]
    extractor: bovw_ext
```

Keep three points in mind:

- `is_ood` and `instance_score` hold one value per item of the data to test.
- The console report cuts them to ten plus a count, while `-v` and `result.txt` show them all.
- Cosine distance, the default, barely moves under a brightness change, where euclidean does. This is why the example
  sets `distance_metric: euclidean`.

## Read the output

`ood_check`'s output (trimmed; from a small run):

```json
{
  "shape": "mapping",
  "data": {
    "is_ood": [true, true, true, true, true],
    "instance_score": [3128.990966796875, 3156.11474609375, 3145.77978515625, 3122.916748046875, 3138.1005859375],
    "feature_score": null
  }
}
```

Every item of this toy's brightened operational set reads as out of distribution under euclidean distance. Brightening
every pixel moves the whole embedding, which euclidean distance is sensitive to.

## Run it from Python

```python
from dataeval_flow import run
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.evaluators.shift import OODKNeighborsConfig

bovw_ext = BoVWExtractorConfig(name="bovw_ext", vocab_size=512, batch_size=32)
sources = {"reference": reference_dataset, "test": operational_dataset}
result = run(OODKNeighborsConfig(distance_metric="euclidean"), sources, extractor=bovw_ext)
```

## Next steps

- [Detect out-of-distribution images](../notebooks/ood_detection.py) — Run the `shift` preset with two OOD detectors and
  benchmark them against known anomalies.
- [Distribution Shift](../concepts/DistributionShift.md) — Learn how drift and out-of-distribution detection differ.
- [Monitor a model's uncertainty](monitor_model_uncertainty.md) — Flag images by the uncertainty of a deployed model.
