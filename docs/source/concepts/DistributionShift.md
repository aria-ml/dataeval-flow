# Distribution Shift

A model is trained on one distribution and deployed into a world that does not
hold still. Sensors degrade, seasons change, target populations evolve — and as
the data a model sees in operation diverges from the data it was trained on, its
performance degrades. **Distribution shift** is the gap between the training
distribution and the operational one, and detecting it is a core part of keeping a
fielded model trustworthy.

DataEval Flow frames shift detection as workflows over a reference and the data
under test, and addresses it at two complementary levels:

- **Drift** is a *population-level* question: has a batch of incoming data, as a
  whole, moved away from the training reference? The `drift-monitoring` workflow
  compares an operational source against a reference source using one or more
  drift detectors.
- **Out-of-distribution (OOD) detection** is an *instance-level* question: is this
  *specific* sample anomalous relative to training? The `ood-detection` workflow
  scores individual samples, catching genuine anomalies that a batch-level test
  might dilute below its threshold.
- **Classwise drift** narrows the population question per class, revealing *which*
  classes a shift most affects rather than only that the batch as a whole moved.

These are workflows in DataEval Flow; the detectors behind them — univariate and
multivariate drift tests, reconstruction- and distance-based OOD scorers,
uncertainty-based and classwise monitoring — and the theory of covariate, label,
and concept shift are DataEval's. They are explained authoritatively in DataEval's
[Distribution Shift explanation](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html).

## When to use it

Begin drift monitoring as soon as a model enters operation; every batch it
processes is a candidate for monitoring. Use OOD detection during data ingestion,
to flag anomalous samples before they reach a model, and in operation, to flag
individual predictions whose inputs fall outside the training distribution. Reach
for classwise drift when aggregate monitoring hides a problem concentrated in a
few classes.

## Related concept pages

- [Preprocessing and Feature Extraction](PreprocessingAndExtraction.md) — drift
  and OOD operate in the embedding space the extractor defines
- [Data Quality and Cleaning](DataQualityAndCleaning.md) — distinguishing an
  operational shift from a collection artifact
- [Data Prioritization](Prioritization.md) — OOD scores can rank which samples to
  review first

## See this in practice

### Tutorials

- [Monitoring drift](../notebooks/drift_monitoring) — population-level drift
  detection against a reference
- [Classwise drift](../notebooks/classwise_drift) — drift tracked per class
- [Detecting OOD samples](../notebooks/ood_detection) — instance-level anomaly
  detection

### Authoritative reference

- DataEval —
  [Distribution Shift](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html)
  (drift detection, OOD detection, and the taxonomy of shift)
