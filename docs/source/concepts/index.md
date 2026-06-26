# Explanations

These pages explain the concepts behind DataEval Flow's workflows — what each is
for, why it works the way it does, and how the pieces fit together.

DataEval Flow is an orchestration layer over the
[DataEval](https://dataeval.readthedocs.io/) library. Accordingly, these
explanations divide into two kinds. The **cross-cutting concepts** — what makes a
DataEval Flow evaluation trustworthy: reproducibility, provenance, and the
representation it measures in — are unique to DataEval Flow and are explained here
in full. The **task-framing concepts** — data cleaning, distribution shift,
splitting, prioritization — frame each evaluation at the orchestration level and
then defer to DataEval for the underlying science, which is covered authoritatively
in the
[DataEval explanation library](https://dataeval.readthedocs.io/en/latest/concepts/index.html)
and linked from each page. We reference those authoritative sources rather than
duplicate them.

## Cross-cutting concepts

- [Reproducibility](Reproducibility.md) — why the same evaluation on the same data
  must yield the same result, and how declarative configuration, validation, and
  config-keyed caching deliver it
- [Provenance](Provenance.md) — why a result must carry its lineage, and how the
  result envelope records the resolved configuration, identities, and versions that
  make a finding auditable and interoperable
- [Preprocessing and Feature Extraction](PreprocessingAndExtraction.md) — transform
  pipelines, extractor types, and why the representation a workflow measures in
  matters

## Task-framing concepts

- [Data Quality and Cleaning](DataQualityAndCleaning.md) — outliers, duplicates,
  and label issues
- [Distribution Shift](DistributionShift.md) — population-level drift,
  instance-level OOD, and classwise drift
- [Dataset Splitting](DatasetSplitting.md) — stratified versus random splits and
  leakage avoidance
- [Data Prioritization](Prioritization.md) — ranking abundant data for labeling and
  review

## Underlying evaluator science (DataEval)

The evaluators these workflows run are DataEval's, and DataEval documents their
science in depth. The most relevant explanation pages are:

- [Data Integrity](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html)
  — outliers, duplicates, and label issues
- [Distribution Shift](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html)
  — drift and out-of-distribution detection
- [Embeddings](https://dataeval.readthedocs.io/en/latest/concepts/Embeddings.html)
  — the representation space evaluators operate in
- [Data Leakage](https://dataeval.readthedocs.io/en/latest/concepts/Leakage.html)
  — leakage and dataset splitting
- [Performance Limits](https://dataeval.readthedocs.io/en/latest/concepts/PerformanceLimits.html)
  — achievable performance and hard samples
- [Acting on Results](https://dataeval.readthedocs.io/en/latest/concepts/ActingOnResults.html)
  — interpreting and responding to findings

:::{toctree}
:hidden:

Reproducibility
Provenance
PreprocessingAndExtraction
DataQualityAndCleaning
DistributionShift
DatasetSplitting
Prioritization
:::
