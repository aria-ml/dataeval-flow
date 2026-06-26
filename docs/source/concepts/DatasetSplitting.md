# Dataset Splitting

Honest evaluation depends on a clean separation between the data a model learns
from and the data it is judged on. **Dataset splitting** partitions a dataset into
training, validation, and test sets — and how that partition is drawn determines
whether the resulting metrics can be trusted. Two concerns dominate.

The first is **representativeness**. A purely **random** split is simple but can,
by chance, leave the test set with too few examples of a rare class to measure
performance on it. A **stratified** split instead preserves the class distribution
across every partition, so each split reflects the dataset as a whole and per-class
metrics remain meaningful.

The second, and more dangerous, is **leakage**: information from the test set
bleeding into training, which inflates measured performance and hides the model's
true generalization. The classic culprit is duplicates or near-duplicates landing
on both sides of a split, but leakage can also arise from shared sources or
correlated groups of samples. A split that ignores these relationships produces
optimistic, untrustworthy numbers.

In DataEval Flow, the `data-splitting` workflow produces train/validation/test
index sets from a source, supporting stratification, configurable fractions, and a
fixed seed for reproducibility. The orchestration layer makes the split a
declarative, repeatable pipeline step; the splitting logic and its
leakage-avoidance guarantees are DataEval's. The science of leakage — how it
arises and how to prevent it — is explained authoritatively in DataEval's
[Data Leakage explanation](https://dataeval.readthedocs.io/en/latest/concepts/Leakage.html).

## When to use it

Split when preparing a dataset for training and evaluation — after cleaning, so
that duplicates that would cause leakage have already been flagged. Prefer a
stratified split whenever class balance matters or rare classes are present, and
fix the seed so the partition is reproducible across runs.

## Related concept pages

- [Data Quality and Cleaning](DataQualityAndCleaning.md) — duplicate detection is
  the first defense against split leakage
- [Distribution Shift](DistributionShift.md) — splits also need to be
  representative for drift baselines to be meaningful

## See this in practice

### Tutorials

- [Splitting a dataset](../notebooks/dataset_splitting) — stratified train/val/test
  splitting with the `data-splitting` workflow

### Authoritative reference

- DataEval —
  [Data Leakage](https://dataeval.readthedocs.io/en/latest/concepts/Leakage.html)
  (how leakage arises and how splitting avoids it)
