# Dataset Splitting

Trustworthy evaluation depends on a clean separation between the data a model learns from and the data it is judged on.
**Dataset splitting** partitions a dataset into training, validation, and test sets. How the partition is drawn
determines whether the resulting metrics can be trusted. Two concerns dominate.

The first is **representativeness**. A purely **random** split can, by chance, leave the test set with too few examples
of a rare class to measure performance on it. A **stratified** split preserves the class distribution across every
partition, so each split reflects the dataset as a whole and per-class metrics remain meaningful.

The second concern, and the more dangerous one, is **leakage**. Information from the test set reaches training, which
inflates measured performance and hides the model's true generalization. The usual cause is duplicates or
near-duplicates landing on both sides of a split, but leakage can also arise from shared sources or correlated groups of
samples. A split that ignores these relationships produces optimistic, untrustworthy numbers.

In DataEval Flow, the [`splits`](../reference/presets.md#splits) workflow produces train/validation/test index sets
from a source, supporting stratification and configurable fractions. The same data gives the same split on every run.
The orchestration layer makes the split a declarative, repeatable pipeline step. The splitting logic and its
leakage-avoidance guarantees are DataEval's. DataEval's
[Data Leakage explanation](https://dataeval.readthedocs.io/en/latest/concepts/Leakage.html) is the authoritative account
of how leakage arises and how to prevent it.

## The preset

`splits` is a {term}`preset <Preset>`. Its settings expand to a chain of steps, which the
[Preset Catalog](../reference/presets.md#splits) lists with its settings and findings. The preset judges each part's
class shares, not coverage or leakage, and not the whole set's class balance, which is the
[`bias`](../reference/presets.md#bias) preset's. On classification data, its `split_on` keeps each value of a metadata
factor, such as a scene or site, in one part, so correlated groups of samples do not straddle a split. To judge
coverage, leakage and shift, run [`audit`](../reference/presets.md#audit) as a step after the split, as
[Check a set of splits](../how_to/check_a_set_of_splits.md) does. You can also give exported parts to an `audit` task as
sources, train first.

The result is a `ChainResult`. Each part's indices into the source are in `result.steps["split"].details["indices"]`, as
`train`, `val` and `test`. Under k-fold, `train` and `val` are keyed by fold, `"0"` to `"k-1"`. The rebalanced train's
indices are in `result.steps["rebalanced"].details["indices"]`. Under k-fold, each fold's are in
`result.steps["rebalanced"].elements["<k>"].details["indices"]`. Where rebalancing kept the train as it was, `details`
is `None` and the train's indices from `split` apply.

Run as a step of a custom workflow, the entry hands on its parts as `<step>.train`, `<step>.val` and `<step>.test`. Only
object-detection Datasets can be exported. See
[Export the parts of a split](../how_to/export_a_dataset.md#export-the-parts-of-a-split).

## When to use it

Split when you prepare a dataset for training and evaluation, after cleaning, so that duplicates that would cause
leakage have already been flagged. Prefer a stratified split whenever class balance matters or rare classes are present.
The partition draws no random numbers, so rerunning on the same data reproduces it.

## Related concept pages

- [Data Quality and Cleaning](DataQualityAndCleaning.md): Duplicate detection is the first defense against split
  leakage.
- [Distribution Shift](DistributionShift.md): Splits also need to be representative for drift baselines to be
  meaningful.

## See this in practice

### Tutorials

- [Split a dataset](../notebooks/dataset_splitting.py): Run stratified train/val/test splitting with the `splits`
  workflow.
- [Audit a set of splits before training](../notebooks/audit.py): Run `audit` on real splits and read its verdict.

### How-to guides

- [Check a set of splits](../how_to/check_a_set_of_splits.md): Collect the evaluation parts of a split and run `audit`
  as a step.
- [Export a dataset](../how_to/export_a_dataset.md): Write a source, or the parts of a split, out as a dataset on disk
  with the provenance that produced it.

### Reference

- [`splits`](../reference/presets.md#splits): The Preset Catalog entry with the preset's chain, settings and `checks:`
  defaults.

## References

1. DataEval, [Data Leakage](https://dataeval.readthedocs.io/en/latest/concepts/Leakage.html). How leakage arises and how
   splitting avoids it.
