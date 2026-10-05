# Dataset Splitting

Honest evaluation depends on a clean separation between the data a model learns
from and the data it is judged on. **Dataset splitting** partitions a dataset into
training, validation, and test sets — and how that partition is drawn determines
whether the resulting metrics can be trusted. Two concerns dominate.

The first is **representativeness**. A purely **random** split can, by chance,
leave the test set with too few examples of a rare class to measure performance
on it. A **stratified** split preserves the class distribution across every
partition, so each split reflects the dataset as a whole and per-class metrics
remain meaningful.

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

## The preset

`data-splitting` is a {term}`preset <Preset>`: its settings expand to a chain of steps. The chain judges the whole
set's labels, balance and diversity, splits it (`folds: 1`) or cuts it into k folds (`folds` of 2 or more) with a
shared test part, optionally rebalances each train, and judges each part's labels, its stratification against the
whole, and, when the task names an extractor, its coverage. Its findings are Class Imbalance for the whole set,
Stratification for each fold, and Uncovered Items under `naive` coverage; balance and diversity are report sections.

The result is a `ChainResult`. Each part's indices into the source are in
`result.steps["split"].details["indices"]`, as `train`, `val` and `test`; under k-fold, `train` and `val` are keyed by
fold, `"0"` to `"k-1"`. The rebalanced train's indices are in `result.steps["rebalanced"].details["indices"]`; under
k-fold, each fold's are in `result.steps["rebalanced"].elements["<k>"].details["indices"]`. Where rebalancing kept the
train as it was, `details` is `None` and the train's indices from `split` apply.

Run as a step of a custom workflow, the entry hands on three Datasets: `<step>.train` (the rebalanced train, where the
entry sets `rebalance:`), `<step>.val` and `<step>.test`. Under `folds` of 2 or more, `train` and `val` are lists keyed
by fold. Only object-detection Datasets can be exported; see
[Export the parts of a split](../how_to/export_a_dataset.md#export-the-parts-of-a-split).

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

- [Splitting a dataset](../notebooks/dataset_splitting.py) — stratified train/val/test
  splitting with the `data-splitting` workflow

### Authoritative reference

- DataEval —
  [Data Leakage](https://dataeval.readthedocs.io/en/latest/concepts/Leakage.html)
  (how leakage arises and how splitting avoids it)
