# Data Quality and Cleaning

Before a dataset is fit to train or evaluate a model, it has to be sound. Real
collections accumulate problems: **outliers** — samples that deviate sharply from
the rest, often from corruption or collection error; **duplicates** and
near-duplicates that inflate apparent dataset size and leak between splits; and
**label issues** — mislabeled or inconsistently labeled samples that teach a model
the wrong thing. Left in place, these defects degrade training, distort metrics,
and undermine every downstream conclusion. Data cleaning is the task of finding
and flagging them so they can be reviewed or removed.

In DataEval Flow, {term}`data cleaning <Data Cleaning>` is a workflow: you point
the `data-cleaning` workflow at a source, and it flags {term}`outliers <Outlier>`,
duplicates, and label problems and produces a report of clean versus flagged
samples. The orchestration layer's contribution is making this a declarative,
reproducible step in a pipeline; the detection methods themselves — the
statistical outlier tests, the duplicate-detection hashing and clustering, and the
label-quality checks — are DataEval's.

The underlying science is explained authoritatively in DataEval's
[Data Integrity explanation](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html),
which covers how outliers, duplicates, and label issues are detected and what each
signal means.

## When to use it

Run data cleaning early — on freshly collected or freshly ingested data, before it
feeds training, splitting, or analysis. It is also worth re-running whenever a
dataset is augmented, since new data brings new opportunities for duplication and
labeling error.

## Related concept pages

- [Distribution Shift](DistributionShift.md) — when anomalous samples reflect a
  genuine shift rather than a collection defect
- [Dataset Splitting](DatasetSplitting.md) — duplicates are a leakage risk that
  splitting must avoid

## See this in practice

### Tutorials

- [Cleaning a dataset](../notebooks/data_cleaning) — the `data-cleaning` workflow
  end to end

### Authoritative reference

- DataEval —
  [Data Integrity](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html)
  (outliers, duplicates, and label-issue detection)
