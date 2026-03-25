# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dataeval-flow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Split a dataset
#
# Partition a HuggingFace dataset into stratified train/val/test splits
# using the config-driven `data-splitting` workflow.

# %% [markdown]
# ## What you'll do
#
# - Download the MNIST test split (10K images, 10 digit classes) from HuggingFace
# - Build a splitting workflow configuration with stratified partitioning
# - Run `run_task()` to produce train/val/test index sets
# - View the built-in splitting report for class distribution and split sizes
# - Inspect per-split label statistics and balance/diversity metrics
# - Export split indices for downstream use

# %% [markdown]
# ## What you'll learn
#
# - How to configure and run the `data-splitting` workflow via `run_task()`
# - What splitting parameters are available (`test_frac`, `val_frac`, `num_folds`, `stratify`)
# - How to read the splitting report for class distribution health
# - How to access raw split indices for downstream filtering or training
# - How pre-split balance and diversity metrics assess metadata factor influence

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datasets`, `maite-datasets`, `pydantic`)
# - Internet connection (to download MNIST from HuggingFace Hub on first run)

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load and prepare the dataset
#
# Download [MNIST](https://huggingface.co/datasets/ylecun/mnist) from HuggingFace and save it
# to disk. We use the test split (10K images, 10 classes) for fast execution.

# %% tags=["remove_output"]
from typing import cast

from datasets import Dataset
from datasets import load_dataset as hf_load

mnist_test = cast(Dataset, hf_load("ylecun/mnist", split="test"))

# %% tags=["remove_output"]
from pathlib import Path

# Save to disk in HuggingFace arrow format
data_path = Path("./data/mnist/test")
mnist_test.save_to_disk(str(data_path))

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# The `data-splitting` workflow requires explicit parameters (no hidden defaults).
# We'll configure stratified splitting with a 20% test holdout and 3-fold
# cross-validation.
#
# With `num_folds=3`, the validation fraction is automatically set to `1/num_folds`
# (i.e., 1/3 of the training portion). On 10K items with `test_frac=0.2`:
#
# - test = 20% of 10K = 2000 (shared across all folds)
# - val = 1/3 of remaining 8000 ≈ 2667 (per fold)
# - train ≈ 5333 (per fold)
#
# Each fold gets a distinct train/val partition while the test holdout stays fixed.
# Exact counts may vary slightly due to stratification rounding per class.
#
# No extractor is needed — splitting works on labels and metadata, not embeddings.

# %%
from dataeval_flow.config import (
    HuggingFaceDatasetConfig,
    PipelineConfig,
    SourceConfig,
)
from dataeval_flow.config.schemas import (
    DataSplittingTaskConfig,
    DataSplittingWorkflowConfig,
)
from dataeval_flow.workflow import run_task

workflow = DataSplittingWorkflowConfig(
    name="mnist_split",
    test_frac=0.2,  # 20% of full dataset held out for test
    val_frac=0.0,  # Must be 0 when num_folds > 1; validation is 1/num_folds
    num_folds=3,  # 3-fold cross-validation
    stratify=True,  # Preserve class distribution in each partition
)

task = DataSplittingTaskConfig(
    name="split_mnist",
    workflow="mnist_split",
    sources="mnist_src",
)

# Build the full pipeline config — datasets, sources, workflows, and tasks
config = PipelineConfig(
    datasets=[
        HuggingFaceDatasetConfig(name="mnist_test", path=str(data_path)),
    ],
    sources=[
        SourceConfig(name="mnist_src", dataset="mnist_test"),
    ],
    workflows=[workflow],
    tasks=[task],
)

# %% [markdown]
# ## Step 2: Run the splitting workflow

# %%
result = run_task(task, config, cache_dir=Path("./cache"))

# %% tags=["remove_cell"]
if not result.success:
    print(f"Workflow failed: {result.errors}")
assert result.success

# %% [markdown]
# ### Splitting report
#
# The workflow result has a built-in `report()` method that renders a formatted
# text summary — class distribution, split sizes, balance, and diversity
# metrics in one view.

# %%
print(result.report())

# %% [markdown]
# ### Understanding the report
#
# The report contains several findings:
#
# - **Class distribution** — per-class counts and max imbalance ratio.
#   With MNIST's approximately balanced classes, this should show `[ok]`.
# - **Split sizes** — train/val/test sample counts per fold. Expect ~5333 train,
#   ~2667 val, ~2000 test.
# - **Pre-split balance** — mutual information between metadata factors and
#   class labels. High MI means a factor is predictive of the label (potential
#   bias source).
# - **Pre-split diversity** — Shannon diversity of metadata factors.
#   Low diversity means a factor has limited variation in the dataset.
#
# For uniform-dimension datasets like MNIST (all 28×28 grayscale), these tables
# will be sparse. Richer datasets with additional metadata columns will produce
# more detailed factor analysis.

# %% [markdown]
# ### Split indices
#
# The raw output contains the actual index lists for each split. These can be
# used to build filtered datasets for downstream training or evaluation.

# %%
raw = result.data.raw

print(f"Dataset size: {raw.dataset_size}")
print(f"Test indices: {len(raw.test_indices)}")
print(f"Number of folds: {len(raw.folds)}")

# %%
import polars as pl

rows = []
for i, fold in enumerate(raw.folds):
    rows.append({"fold": i, "train": len(fold.train_indices), "val": len(fold.val_indices)})
print(pl.DataFrame(rows))
print(f"\nTest (shared across folds): {len(raw.test_indices)} samples")

# %%
# Verify no overlap between splits and full coverage per fold
test_set = set(raw.test_indices)

for i, fold in enumerate(raw.folds):
    train_set = set(fold.train_indices)
    val_set = set(fold.val_indices)

    assert train_set.isdisjoint(val_set), f"Fold {i}: train/val overlap!"
    assert train_set.isdisjoint(test_set), f"Fold {i}: train/test overlap!"
    assert val_set.isdisjoint(test_set), f"Fold {i}: val/test overlap!"

    total = len(train_set) + len(val_set) + len(test_set)
    assert total == raw.dataset_size, f"Fold {i}: missing indices: {total} != {raw.dataset_size}"

print(f"All {len(raw.folds)} folds verified: no overlap, full coverage.")

# %% [markdown]
# ### Label distribution per split
#
# With `stratify=True`, each split should have roughly proportional class counts.
# MNIST has ~1000 images per class in the test set. With 3-fold splitting, each
# class should appear proportionally in train (~533), val (~267), and test (~200).

# %%
# Full dataset label stats
if raw.label_stats_full:
    print("Full dataset:")
    print(f"  Classes: {raw.label_stats_full.get('class_count', '?')}")
    print(f"  Per-class counts: {raw.label_stats_full.get('label_counts_per_class', [])}")

# Per-fold and test label stats
for i, fold in enumerate(raw.folds):
    if fold.label_stats_train:
        print(f"\nFold {i} train: {fold.label_stats_train.get('label_counts_per_class', [])}")
    if fold.label_stats_val:
        print(f"Fold {i} val:   {fold.label_stats_val.get('label_counts_per_class', [])}")
if raw.label_stats_test:
    print(f"\nTest:  {raw.label_stats_test.get('label_counts_per_class', [])}")

# %% [markdown]
# ### Balance and diversity
#
# The workflow runs DataEval's `Balance` and `Diversity` evaluators on the full
# dataset before splitting. The results are serialized from Polars DataFrames
# into list-of-dicts format.
#
# The balance output shows mutual information between each metadata factor and
# the class labels — high values indicate a factor that predicts the label
# (a potential bias source). The diversity output shows Shannon diversity per
# factor.
#
# For uniform-dimension datasets like MNIST, these tables will be sparse since
# all images share the same 28×28 dimensions. Richer datasets with additional
# metadata columns will produce more detailed factor tables.

# %%
# Pre-split balance — mutual information between factors and labels
balance_rows = raw.pre_split_balance.get("balance")
if balance_rows:
    print("Pre-split balance (mutual information):")
    print(pl.DataFrame(balance_rows))
else:
    print("No balance data (dataset may lack metadata factors)")

# Pre-split diversity — Shannon diversity per factor
diversity_rows = raw.pre_split_diversity.get("factors")
if diversity_rows:
    print("\nPre-split diversity:")
    print(pl.DataFrame(diversity_rows))
else:
    print("No diversity data (dataset may lack metadata factors)")

# %% [markdown]
# ## Results Exploration: Export and metadata

# %%
meta = result.metadata
print(f"Stratified:  {meta.stratified}")
print(f"Num folds:   {meta.num_folds}")
print(f"Split sizes: {meta.split_sizes}")

# %%
import json

json_str = result.export(fmt="json")
exported = json.loads(json_str)

# Extract test indices (nested under "raw")
test_idx = exported["raw"]["test_indices"]
print(f"Test indices ({len(test_idx)} samples): {test_idx[:10]}...")

# Extract per-fold train/val indices
for i, fold in enumerate(exported["raw"]["folds"]):
    print(f"Fold {i}: train={len(fold['train_indices'])}, val={len(fold['val_indices'])}")

# %% [markdown]
# The exported JSON contains the split indices directly. You can parse them and use
# `Dataset.select()` to build filtered datasets for training, evaluation, or further
# analysis.

# %%
from datasets import load_from_disk

ds = load_from_disk(str(data_path))

test_ds = ds.select(test_idx)
train_ds = ds.select(exported["raw"]["folds"][0]["train_indices"])
val_ds = ds.select(exported["raw"]["folds"][0]["val_indices"])

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# %% [markdown]
# ## Conclusion
#
# In this tutorial you learned how to:
#
# - **Configure** the `data-splitting` workflow with test/val fractions, multi-fold, and stratification
# - **Run** the workflow via `run_task()` on MNIST
# - **Read the splitting report** for class distribution health and split sizes
# - **Access raw split indices** — train, val, and test index lists across multiple folds
# - **Verify split integrity** — no overlap, full coverage, proportional class distribution
# - **Inspect balance and diversity** metrics computed on the pre-split dataset
# - **Use split indices** with `Dataset.select()` to build filtered datasets for downstream use
# - **Export** results to JSON and extract indices for integration with other tools

# %% [markdown]
# ## What's next
#
# - **Data cleaning** — Use the `data-cleaning` workflow to flag outliers and duplicates
#   in each split before training
# - **Higher fold counts** — Increase `num_folds` for more robust cross-validation estimates
# - **Group-aware splits** — Use `split_on=["group_id"]` to keep related samples
#   together (e.g., same patient, same video sequence)
# - **Rebalancing** — Set `rebalance_method="global"` to address class imbalance
#   in the training split
