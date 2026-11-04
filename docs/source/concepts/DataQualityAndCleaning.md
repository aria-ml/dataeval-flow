# Data Quality and Cleaning

Before you train or evaluate a model on a dataset, the dataset has to be sound. Real collections accumulate problems.
**Outliers** are samples that deviate sharply from the rest, often from corruption or collection error. **Duplicates**
and near-duplicates inflate the apparent dataset size and leak between splits. **Label issues** are mislabeled or
inconsistently labeled samples that teach a model the wrong thing. Left in place, these defects degrade training,
distort metrics, and undermine every downstream conclusion. Data cleaning is the task of finding and flagging them so
they can be reviewed or removed.

In DataEval Flow, {term}`data cleaning <Data Cleaning>` is a workflow. You point the
[`quality`](../reference/presets.md#quality) workflow at a source. It flags {term}`outliers <Outlier>` and duplicates,
reports them with the source's label counts, and hands on the dataset without the outliers and duplicates. Class
imbalance is the [`bias`](../reference/presets.md#bias) preset's. The orchestration layer makes this a declarative,
reproducible step in a pipeline. The detection methods themselves are DataEval's: the statistical outlier tests, the
duplicate-detection hashing and clustering, and the label statistics.

DataEval's [Data Integrity explanation](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html) is the
authoritative account of the science. It covers how outliers, duplicates, and label issues are detected and what each
signal means.

To run only duplicate or outlier detection, without the rest of data cleaning or its checks, use the
[`duplicates`](../reference/evaluators.md#duplicates) or [`outliers`](../reference/evaluators.md#outliers) evaluator.
[Run a single evaluator](../how_to/run_a_single_evaluator.md) shows how.

## When to use it

Run data cleaning early, on freshly collected or freshly ingested data, before it feeds training, splitting, or
analysis. Run it again whenever you augment a dataset, because new data brings new opportunities for duplication and
labeling error.

## Related concept pages

- [Distribution Shift](DistributionShift.md): When anomalous samples reflect a real shift rather than a collection
  defect.
- [Dataset Splitting](DatasetSplitting.md): Duplicates are a leakage risk that splitting must avoid.

## See this in practice

### Tutorials

- [Clean a dataset](../notebooks/data_cleaning.py): Run the `quality` workflow end to end.

### How-to guides

- [Configure outlier detection](../how_to/configure_outlier_detection.md): Pick the method, the statistics and the
  thresholds that decide what `quality` flags and warns on.
- [Run a single evaluator](../how_to/run_a_single_evaluator.md): Run one DataEval evaluator, such as `duplicates`, with
  no health status.

### Reference

- [`quality`](../reference/presets.md#quality): The Preset Catalog entry with the preset's chain, settings and `checks:`
  defaults.

## References

1. DataEval, [Data Integrity](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html). Outlier,
   duplicate, and label-issue detection.
