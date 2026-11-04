# Distribution Shift

A model is trained on one distribution and deployed into a changing one. Sensors degrade, seasons change, and target
populations evolve. As the data a model sees in operation diverges from the data it was trained on, its performance
degrades. **Distribution shift** is the gap between the training distribution and the operational one, and detecting it
is a core part of keeping a fielded model trustworthy.

DataEval Flow frames shift detection as workflows over a reference and the data under test, and addresses it at two
complementary levels:

- Drift is a population-level question. Has a batch of incoming data, as a whole, moved away from the training
  reference? The [`shift`](../reference/presets.md#shift) preset compares each operational source against a reference
  source using one or more drift detectors. Its default,
  [`drift-univariate`](../reference/evaluators.md#drift-univariate), can miss correlated shifts across embedding
  dimensions. Add [`drift-mmd`](../reference/evaluators.md#drift-mmd) or
  [`drift-domain-classifier`](../reference/evaluators.md#drift-domain-classifier) to catch those. The preset is made of
  chained steps, so a custom workflow can
  [merge test sources](../how_to/test_merged_sources_for_drift.md),
  [compare groups of classes](../how_to/compare_class_groups.md), or
  [drift on the crops of detection data](../how_to/detect_drift_on_detection_crops.md).
- Out-of-distribution (OOD) detection is an instance-level question. Is this specific sample anomalous relative to
  training? The same `shift` preset, with OOD detectors in its `detectors:` list, scores individual images, on
  embeddings or on a model's uncertainty. This catches real anomalies that a batch-level test might dilute below its
  threshold.
- Classwise drift narrows the population question to each class. It shows which classes a shift most affects, and not
  only that the batch as a whole moved.

These are workflows in DataEval Flow. The detectors behind them are DataEval's: univariate and multivariate drift tests,
reconstruction- and distance-based OOD scorers, and uncertainty-based and classwise monitoring. So is the theory of
covariate, label, and concept shift. DataEval's
[Distribution Shift explanation](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html) is the
authoritative account of both.

## When to use it

Begin drift monitoring as soon as a model enters operation. Every batch it processes is a candidate for monitoring. Use
OOD detection during data ingestion, to flag anomalous samples before they reach a model, and in operation, to flag
individual predictions whose inputs fall outside the training distribution. Use classwise drift when aggregate
monitoring hides a problem concentrated in a few classes.

## Related concept pages

- [Preprocessing and Feature Extraction](PreprocessingAndExtraction.md): Drift and OOD operate in the embedding space
  the extractor defines.
- [Data Quality and Cleaning](DataQualityAndCleaning.md): Distinguishing an operational shift from a collection
  artifact.
- [Data Prioritization](Prioritization.md): OOD scores can rank which samples to review first.

## See this in practice

### Tutorials

- [Monitor incoming data for drift](../notebooks/drift_monitoring.py): Run population-level drift detection against a
  reference.
- [Detect classwise drift: Which classes are changing?](../notebooks/classwise_drift.py): Track drift per class.
- [Detect out-of-distribution images](../notebooks/ood_detection.py): Run instance-level anomaly detection.

### How-to guides

- [Monitor drift with steps](../how_to/monitor_drift.md): Read what the `shift` preset makes for each test source and
  for each class, and read many verdicts with care.
- [Test merged sources for drift](../how_to/test_merged_sources_for_drift.md): Merge the test sources and run one
  detector on the pool.
- [Compare one group of classes with another](../how_to/compare_class_groups.md): Narrow each side with a `view` step
  and run a detector on two groups of classes.
- [Detect drift on detection crops](../how_to/detect_drift_on_detection_crops.md): Wrap detection data with
  `DetectionCrops` and run the `shift` preset on the crops.
- [Detect drift with an evaluator](../how_to/detect_drift_with_an_evaluator.md): Run `drift-mmd`, chunked drift and
  `drift-wasserstein` as evaluators.
- [Flag out-of-distribution items](../how_to/flag_out_of_distribution_items.md): Run `ood-kneighbors` on a reference
  and on the data to test.
- [Monitor a model's uncertainty](../how_to/monitor_model_uncertainty.md): Test the `uncertainty` extractor with drift
  detectors, and flag images with OOD detectors.

### Reference

- [`shift`](../reference/presets.md#shift): The Preset Catalog entry with the preset's chain, settings and `checks:`
  defaults.

## References

1. DataEval, [Distribution Shift](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html). Drift
   detection, OOD detection, and the taxonomy of shift.
