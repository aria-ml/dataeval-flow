# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: dataeval-flow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Split a dataset
#
# Partition a dataset into stratified train, validation, and test splits
# using the config-driven `data-splitting` workflow.

# %% [markdown]
# **Target audience**: You are a model developer or T&E engineer who needs
# defensible, reproducible partitions for model development and evaluation.
#
# **Workflow role**: You should split your dataset after cleaning and data
# analysis. Dataset splitting creates isolated partitions for training, validation
# tuning, and unbiased test evaluation. See [Dataset splitting](../concepts/DatasetSplitting.md)
# for background on stratification and leakage avoidance.

# %% [markdown]
# ## What you will do
#
# - Load the MilitaryVehicles dataset (7,823 training images across 24 classes).
# - Configure a stratified splitting workflow with a test holdout and 3-fold cross-validation.
# - Run `run_task()` to generate partition index sets.
# - Inspect the splitting report for class distribution and split sizes.
# - Review label distribution statistics and metadata balance metrics.
# - Export split indices and construct sliced dataset views.

# %% [markdown]
# ## What you will learn
#
# - How to configure and execute the `data-splitting` workflow.
# - How to set splitting parameters (`test_frac`, `val_frac`, `num_folds`, `stratify`).
# - How to evaluate class distribution balance across splits.
# - How to access split index lists for downstream training workflows.
# - How pre-split balance and diversity metrics evaluate metadata factor correlation.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`).
# - Install `maite-datasets[datamaite]` to download and export MilitaryVehicles.
# - Ensure network access for the initial download; subsequent runs use local cache.

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the dataset
#
# [MilitaryVehicles](https://huggingface.co/datasets/leibnitz-lab/military_vehicles)
# contains 9,444 images across 24 vehicle types. In this tutorial, you will split
# the `train` image set of 7,823 images. Per-class counts range from 119 to 424 images.
#
# Stratification maintains proportional representation for rare classes across all
# folds. Without stratification, small classes can be underrepresented in validation
# or test sets.
#
# Setting `as_datamaite=True` writes the dataset as a class-per-directory ImageFolder
# structure for the `huggingface` dataset loader.

# %% tags=["remove_output"]
from pathlib import Path

from maite_datasets.image_classification import MilitaryVehicles

data_root = Path("./data")

# One download shared by every export; a re-run reads what is already on disk.
MilitaryVehicles(root=data_root, image_set="base", download=True)
MilitaryVehicles(root=data_root, image_set="train", as_datamaite=True)

# The export nests its images one level down, under the split name.
data_path = data_root / "militaryvehicles_datamaite_train" / "train"
print(f"Reading from {data_path}")

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# You must specify splitting parameters explicitly. In this example, you will
# configure stratified splitting with a 20% test holdout and 3-fold cross-validation.
#
# When `num_folds=3`, the validation fraction is `1/num_folds` (one third of the
# non-test portion). For 7,823 items with `test_frac=0.2`:
#
# - Test set: 20% of 7,823 ≈ 1,565 samples (shared across folds).
# - Validation set: 1/3 of the remaining 6,258 ≈ 2,086 samples per fold.
# - Training set: remaining 2/3 ≈ 4,172 samples per fold.
#
# Each fold receives a distinct train and validation split while preserving the
# shared test set. Exact counts may vary slightly due to per-class rounding.
#
# You do not need an extractor because dataset splitting operates on labels and
# metadata rather than embeddings.

# %%
from dataeval_flow import PipelineConfig, run_task
from dataeval_flow.config import HuggingFaceDatasetConfig, SourceConfig, TaskConfig
from dataeval_flow.workflows.data_splitting import DataSplittingConfig

workflow = DataSplittingConfig(
    name="mv_split",
    test_frac=0.2,  # 20% of full dataset held out for test
    val_frac=0.0,  # Must be 0 when num_folds > 1; validation is 1/num_folds
    num_folds=3,  # 3-fold cross-validation
    stratify=True,  # Preserve class distribution in each partition
)

task = TaskConfig(
    name="split_military_vehicles",
    workflow="mv_split",
    sources="mv_src",
)

# Build the pipeline configuration
config = PipelineConfig(
    datasets=[
        HuggingFaceDatasetConfig(name="mv_train", path=str(data_path), task="image_classification"),
    ],
    sources=[
        SourceConfig(name="mv_src", dataset="mv_train"),
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
# Call `result.report()` to display class distributions, split sizes, and metadata
# balance metrics in a formatted summary.

# %%
print(result.report())

# %% [markdown]
# ### Understanding the report
#
# You should inspect these report sections:
#
# - **Class distribution**: Per-class counts and maximum imbalance ratio.
#   MilitaryVehicles has an imbalance ratio of approximately 3.6:1.
# - **Split sizes**: Train, validation, and test sample counts per fold.
# - **Pre-split balance**: Mutual information between metadata factors and
#   labels. High mutual information indicates potential label bias.
# - **Pre-split diversity**: Shannon diversity of metadata factors. Low values
#   indicate limited metadata variation.
#
# MilitaryVehicles includes `height` and `width` as metadata factors. If your
# dataset uses uniform image dimensions and no additional attributes, metadata
# factor tables will be empty.

# %% [markdown]
# ### Split indices
#
# You can retrieve raw split index lists from `result.output.raw` to build filtered
# datasets for training or evaluation.

# %%
raw = result.output.raw

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
# When you set `stratify=True`, each split preserves the overall class distribution.
#
# For example, a class with 119 images yields 63 training, 32 validation, and 24
# test samples under this 3-fold split. The report section **Stratification quality**
# computes deviations between split proportions and overall dataset proportions.

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
# The workflow evaluates `Balance` and `Diversity` on the full dataset before
# splitting.
#
# The balance output displays mutual information between metadata factors and class
# labels. High mutual information indicates that a metadata factor predicts the label.
# The diversity output displays Shannon diversity per factor.
#
# In MilitaryVehicles, `height` and `width` have low mutual information scores
# (around 0.01), showing vehicle labels do not depend on image resolution.

# %%
# Pre-split balance: mutual information between factors and labels
balance_rows = raw.pre_split_balance.get("balance")
if balance_rows:
    print("Pre-split balance (mutual information):")
    print(pl.DataFrame(balance_rows))
else:
    print("No balance data (dataset may lack metadata factors)")

# Pre-split diversity: Shannon diversity per factor
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
# You can extract split indices from exported JSON. You can apply them directly to
# `result.dataset` using `View` to create training and evaluation datasets.
#
# You should slice `result.dataset` directly so the indices align with the resolved
# dataset ordering.

# %%
from dataeval.data import Indices, View

ds = result.dataset
assert ds is not None

test_ds = View(ds, operations=[Indices(test_idx)])
train_ds = View(ds, operations=[Indices(exported["raw"]["folds"][0]["train_indices"])])
val_ds = View(ds, operations=[Indices(exported["raw"]["folds"][0]["val_indices"])])

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Configure the `data-splitting` workflow with test fractions, fold counts, and stratification.
# - Execute the workflow with `run_task()`.
# - Read the splitting report for class distributions and partition sizes.
# - Access raw partition index arrays for train, validation, and test sets.
# - Verify partition coverage and verify that partitions do not overlap.
# - Inspect pre-split balance and diversity metrics across metadata factors.
# - Apply split indices to `result.dataset` using `View`.
# - Export split definitions to JSON for downstream integration.

# %% [markdown]
# ## Next steps
#
# - **Data cleaning**: Use the `data-cleaning` workflow to detect outliers and duplicates
#   in each split before training.
# - **Cross-validation**: Increase `num_folds` to evaluate model stability across more folds.
# - **Group-aware splits**: Set `split_on=["group_id"]` to prevent leakage across related samples.
# - **Rebalancing**: Set `rebalance_method="global"` to rebalance class distributions in training splits.

# %% [markdown]
# ## Related guides
#
# - **Concept**: [Dataset splitting](../concepts/DatasetSplitting.md) covers
#   stratification, cross-validation, and data leakage avoidance.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md)
#   explains how to run splitting workflows from YAML configurations inside containers.
