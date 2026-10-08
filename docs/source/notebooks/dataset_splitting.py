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
# using the config-driven `splits` workflow.

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
# - Inspect the splitting report for class distribution, stratification and split sizes.
# - Run `bias` on the source to review its class balance, and its metadata balance and diversity.
# - Read the split indices from the result and export them to JSON.

# %% [markdown]
# ## What you will learn
#
# - How to configure and execute the `splits` workflow.
# - How to set splitting parameters (`test_frac`, `folds`, `stratify`, `rebalance`).
# - How to evaluate class distribution balance across splits.
# - How to read each part's indices from the split step's details.
# - How the `bias` preset's balance and diversity evaluate metadata factor correlation before a split.

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
# With `folds=3`, each fold's validation part is one third of the non-test items, so
# `val_frac` is left unset: setting it with 2 or more folds is refused. For 7,823 items
# with `test_frac=0.2`:
#
# - Test set: 20% of 7,823 ≈ 1,565 samples (shared across folds).
# - Validation set: 1/3 of the remaining 6,258 ≈ 2,086 samples per fold.
# - Training set: remaining 2/3 ≈ 4,172 samples per fold.
#
# Each fold receives a distinct train and validation split while preserving the
# shared test set. Exact counts may vary slightly due to per-class rounding.
#
# The split reads labels and metadata only, so it needs no extractor, and it judges
# class shares.
#
# The whole set's class balance and metadata factors are judged by the `bias` preset,
# not by the split. The pipeline holds a `bias` task on the same source, which a later
# step runs.

# %%
from dataeval_flow import run_task
from dataeval_flow.config import HuggingFaceDatasetConfig, PipelineConfig, SourceConfig, TaskConfig
from dataeval_flow.workflows.bias import BiasConfig
from dataeval_flow.workflows.splits import SplitsConfig

workflow = SplitsConfig(
    name="mv_split",
    test_frac=0.2,  # 20% of full dataset held out for test
    folds=3,  # 3-fold cross-validation; each fold's val is 1/3 of the non-test items
    stratify=True,  # Preserve class distribution in each partition
)

task = TaskConfig(
    name="split_military_vehicles",
    workflow="mv_split",
    sources="mv_src",
)

bias_task = TaskConfig(name="bias_military_vehicles", workflow="mv_bias", sources="mv_src")

# Build the pipeline configuration
config = PipelineConfig(
    max_processes=2,
    datasets=[
        HuggingFaceDatasetConfig(name="mv_train", path=str(data_path), task="image_classification"),
    ],
    sources=[
        SourceConfig(name="mv_src", dataset="mv_train"),
    ],
    workflows=[workflow, BiasConfig(name="mv_bias")],
    tasks=[task, bias_task],
)

# %% [markdown]
# ## Step 2: Run the splitting workflow

# %%
result = run_task(config, task, cache_dir=Path("./cache"))

# %% tags=["remove_cell"]
if not result.success:
    print(f"Workflow failed: {result.errors}")
assert result.success

# %% [markdown]
# ### Splitting report
#
# Call `result.report()` to display the findings, the class distributions and the split
# sizes in a formatted summary.

# %%
print(result.report())

# %% [markdown]
# ### Understanding the report
#
# The summary lists one finding per fold, **Stratification**: how far each part's class
# shares stray from the whole's, in percentage points. Every fold's largest deviation is
# 0.1 points (class `T-72` in the test part), so all three pass.
#
# The report's **K-Fold** section, not a finding, gives the sizes of each fold's train and
# val, and of the shared test.
#
# The Steps table lists each step and its status.

# %% [markdown]
# ### Split indices
#
# The split step holds each part's indices into the source's items, in
# `result.steps["split"].details["indices"]`. With `folds` of 2 or more, `train` and
# `val` are keyed by fold, `"0"` to `"2"` here, and `test` is one list shared by every
# fold. With `folds: 1`, `train`, `val` and `test` are each one list.

# %%
indices = result.steps["split"].details["indices"]
test = indices["test"]
print(f"Folds: {list(indices['train'])}")
print(f"Test indices: {len(test)} (first ten: {test[:10]})")

# %%
import polars as pl

rows = [
    {"fold": fold, "train": len(indices["train"][fold]), "val": len(indices["val"][fold])} for fold in indices["train"]
]
print(pl.DataFrame(rows))
print(f"\nTest (shared across folds): {len(test)} samples")

# %%
# Verify no overlap between parts and full coverage per fold
dataset_size = sum(len(indices[part]["0"]) for part in ("train", "val")) + len(test)
test_set = set(test)

for fold in indices["train"]:
    train_set = set(indices["train"][fold])
    val_set = set(indices["val"][fold])

    assert train_set.isdisjoint(val_set), f"Fold {fold}: train/val overlap!"
    assert train_set.isdisjoint(test_set), f"Fold {fold}: train/test overlap!"
    assert val_set.isdisjoint(test_set), f"Fold {fold}: val/test overlap!"
    assert len(train_set | val_set | test_set) == dataset_size, f"Fold {fold}: missing indices"

print(f"All {len(indices['train'])} folds verified: no overlap, full coverage of {dataset_size} items.")

# %% [markdown]
# ### Label distribution per part
#
# When you set `stratify=True`, each part keeps the whole set's class shares.
#
# For example, the class with 119 images, `30N6E`, has 63 training, 32 validation, and
# 24 test samples in fold 0. The **Stratification** section of the report tabulates every
# class's count in each part, and the **Label Health** block beneath it counts each
# part's labels.

# %%
for item in result.findings:
    print(f"{item.severity:8} {item.title}: {item.brief}")

# %% [markdown]
# ### Class balance, balance and diversity before the split
#
# The `bias` task judges the whole set before it is split. Its **Class Imbalance**
# finding gives the class counts and the imbalance ratio, the largest class count over the
# smallest: MilitaryVehicles has 24 classes and 7,823 items, with a ratio of 3.6:1, under the
# default warning limit of 5:1 and over the 2:1 at which it informs. Its `balance` and
# `diversity` steps are report sections, not findings, and Shortcut Risk and Factor Parity
# judge each factor against the class.
#
# Balance reports the mutual information between each metadata factor and the class
# label. In MilitaryVehicles, `height` and `width` score about 0.01, so the vehicle
# classes do not depend on image resolution. The two factors score 0.99 against each
# other, which is expected for image dimensions.
#
# Diversity reports a score per factor. `class_label` scores 0.96, close to even.
# `height` and `width` score 0.15 and 0.17 and are flagged as low diversity, because
# most images share a few sizes.

# %%
bias = run_task(config, bias_task, cache_dir=Path("./cache"))
for item in bias.findings:
    print(f"{item.severity:8} {item.title}: {item.brief}")
print(bias.steps["balance"].output.balance)
print(bias.steps["diversity"].output.factors)

# %% [markdown]
# ## Results Exploration: Export and lineage

# %%
print(f"Folds:      {len(indices['train'])}")
print(f"Source:     {result.metadata.lineage[0].name} ({result.metadata.lineage[0].items} items)")
for record in result.metadata.lineage[1:4]:
    print(f"{record.name:16} {record.type:6} {record.items} items")

# %%
import json

exported = json.loads(result.export(fmt="json"))

# The same indices, under the split step's details
exported_indices = exported["steps"]["split"]["details"]["indices"]
print(f"Test indices ({len(exported_indices['test'])} samples): {exported_indices['test'][:10]}...")

for fold in exported_indices["train"]:
    print(f"Fold {fold}: train={len(exported_indices['train'][fold])}, val={len(exported_indices['val'][fold])}")

# %% [markdown]
# You can save these indices and apply them to the dataset you loaded, in the source's
# order, to build training and evaluation datasets for each fold. A custom workflow that
# runs a `splits` entry as a step reads the parts directly as `<step>.train`,
# `<step>.val` and `<step>.test`. See
# [Export the parts of a split](../how_to/export_a_dataset.md).

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Configure the `splits` workflow with test fractions, fold counts, and stratification.
# - Execute the workflow with `run_task()`.
# - Read the splitting report for class distributions and partition sizes.
# - Read the partition indices for train, validation, and test sets.
# - Verify partition coverage and verify that partitions do not overlap.
# - Run `bias` to inspect class balance, and balance and diversity across metadata factors.
# - Export the split to JSON for downstream integration.

# %% [markdown]
# ## Next steps
#
# - **Data cleaning**: Use the `quality` workflow to detect outliers and duplicates
#   in each split before training.
# - **Cross-validation**: Increase `folds` to evaluate model stability across more folds.
# - **Group-aware splits**: Set `split_on=["group_id"]` to prevent leakage across related samples.
# - **Rebalancing**: Set `rebalance: global` or `interclass` to rebalance the class distribution of each train.

# %% [markdown]
# ## Related guides
#
# - **Concept**: [Dataset splitting](../concepts/DatasetSplitting.md) covers
#   stratification, cross-validation, and data leakage avoidance.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md)
#   explains how to run splitting workflows from YAML configurations inside containers.
