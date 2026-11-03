# Detect drift with an evaluator

Has the operational data drifted from the reference it was validated against? Run a drift evaluator, such as
[`drift-mmd`](../reference/evaluators.md#drift-mmd) or
[`drift-wasserstein`](../reference/evaluators.md#drift-wasserstein), on the reference and the data to test.
[Distribution Shift](../concepts/DistributionShift.md) explains what drift asks. An evaluator reports DataEval's answer
to its question and renders no health status. The [Evaluator Catalog](../reference/evaluators.md) lists every parameter
each evaluator takes. You need a pipeline that defines `datasets:`, the sources `train`, `validation` and `operational`,
and the extractor `bovw_ext` from [Run a single evaluator](run_a_single_evaluator.md).

## Define the evaluators and tasks

```yaml
evaluators:
  - name: mmd
    type: drift-mmd
  - name: mmd_chunked
    type: drift-mmd
    chunking:
      chunk_size: 100
      threshold: [zscore, 3.0]
  - name: wasserstein
    type: drift-wasserstein

tasks:
  - name: drift
    evaluator: mmd
    sources: [train, operational]
    extractor: bovw_ext
  - name: drift_by_chunk
    evaluator: mmd_chunked
    sources: [train, operational]
    extractor: bovw_ext
  - name: drift_against_validation
    evaluator: wasserstein
    sources: [train, validation, operational]
    extractor: bovw_ext
```

Keep three points in mind:

- Sources run the reference first, then the data to test.
- `chunking:` tests each chunk of the data against the spread of the reference's chunks. DataEval needs at least three
  reference chunks, so `chunk_size` must be small enough to give it that many.
- `drift-wasserstein` takes a validation source between the two.

## Read the output

`drift`'s output (trimmed; from a small run):

```json
{
  "shape": "mapping",
  "data": {
    "drifted": true,
    "threshold": 0.05,
    "distance": 0.41380244493484497,
    "metric_name": "mmd2",
    "details": {"p_val": 0.0, "distance_threshold": 0.01657271385192871},
    "feature_names": []
  }
}
```

`drift_by_chunk`'s output (trimmed; from a small run). The toy sources hold 40 items each, too few for
`chunk_size: 100`, so this excerpt used `chunk_count: 4` instead:

```json
{
  "shape": "mapping",
  "data": {
    "drifted": true,
    "threshold": 0.030687406659126282,
    "distance": 0.4389404356479645,
    "metric_name": "mmd2",
    "details": {
      "shape": "table",
      "columns": ["key", "index", "start_index", "end_index", "value", "upper_threshold", "lower_threshold",
                  "drifted"],
      "rows": [
        {"key": "[0:9]", "index": 0, "start_index": 0, "end_index": 9, "value": 0.46045851707458496,
         "upper_threshold": 0.030687406659126282, "lower_threshold": -0.0239151269197464, "drifted": true},
        {"key": "[10:19]", "index": 1, "start_index": 10, "end_index": 19, "value": 0.43915778398513794,
         "upper_threshold": 0.030687406659126282, "lower_threshold": -0.0239151269197464, "drifted": true}
      ]
    },
    "feature_names": []
  }
}
```

`drift_against_validation`'s output (trimmed; from a small run):

```json
{
  "shape": "mapping",
  "data": {
    "drifted": true,
    "threshold": 1.4,
    "distance": 5.285390853881836,
    "metric_name": "wasserstein_ratio",
    "details": {
      "ratio": 5.285390853881836,
      "feature_drift": [true, true, true, true, true, true, true, false]
    }
  }
}
```

All three declare drift on this toy's brightened operational set. The `details` of `drift_by_chunk` break that call down
by chunk instead of reporting it whole, and the `feature_drift` of `drift_against_validation` has one entry per
embedding dimension.

## Run it from Python

```python
from dataeval_flow import run
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.evaluators.shift import ChunkedDriftConfig, DriftMMDConfig, DriftWassersteinConfig

bovw_ext = BoVWExtractorConfig(name="bovw_ext", vocab_size=512, batch_size=32)
sources = {"reference": reference_dataset, "test": operational_dataset}

drift = run(DriftMMDConfig(), sources, extractor=bovw_ext)
drift_by_chunk = run(
    DriftMMDConfig(chunking=ChunkedDriftConfig(chunk_size=100, threshold=("zscore", 3.0))),
    sources,
    extractor=bovw_ext,
)

validation_sources = {"reference": reference_dataset, "validation": validation_dataset, "test": operational_dataset}
drift_against_validation = run(DriftWassersteinConfig(), validation_sources, extractor=bovw_ext)
```

## Next steps

- [Monitor incoming data for drift](../notebooks/drift_monitoring.py) — Run the `shift` preset on real imagery and read
  its chunked report.
- [Distribution Shift](../concepts/DistributionShift.md) — Learn how drift and out-of-distribution detection differ.
- [Monitor drift with steps](monitor_drift.md) — Judge each drift detector with checks, for each test source.
