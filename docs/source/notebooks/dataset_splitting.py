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
# Partition a dataset into stratified train/val/test splits
# using the config-driven `data-splitting` workflow.

# %% [markdown]
# **Who this is for** — Model developers and T&E engineers who need defensible,
# reproducible train/val/test partitions for model development and evaluation.
#
# **Where this fits** — Splitting comes after a dataset has been cleaned and
# analyzed: you turn one trustworthy dataset into the partitions a higher-level
# T&E workflow relies on — training on the train fold, tuning on validation, and
# holding out a test set for unbiased evaluation. See the
# [Dataset splitting](../concepts/DatasetSplitting.md) concept page for stratification
# and leakage-avoidance background.

# %% [markdown]
# ## What you'll do
#
# - Load MilitaryVehicles (7,823 train images across 24 vehicle classes)
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
# - `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`)
# - `maite-datasets[datamaite]` (to download MilitaryVehicles and export it)
# - Internet connection on the first run; everything after that comes from disk

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the dataset
#
# [MilitaryVehicles](https://huggingface.co/datasets/leibnitz-lab/military_vehicles) is a
# classification dataset of 9,444 images across 24 vehicle types. We split its `train` image
# set — 7,823 images, between 119 and 424 per class.
#
# That imbalance is the point. A 3.6:1 spread means the rarest class contributes only about
# 24 images to a 20% test holdout, and a partition drawn without stratification can easily
# leave a fold with too few of it to evaluate on. MNIST-style balanced data never exercises
# this; real collections nearly always do.
#
# `as_datamaite=True` writes the dataset as a class-per-directory tree — the HuggingFace
# **ImageFolder** layout the `huggingface` dataset format reads.

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
# The `data-splitting` workflow requires explicit parameters (no hidden defaults).
# We'll configure stratified splitting with a 20% test holdout and 3-fold
# cross-validation.
#
# With `num_folds=3`, the validation fraction is automatically set to `1/num_folds`
# (i.e., 1/3 of the training portion). On 7,823 items with `test_frac=0.2`:
#
# - test = 20% of 7,823 ≈ 1,565 (shared across all folds)
# - val = 1/3 of the remaining 6,258 ≈ 2,086 (per fold)
# - train ≈ 4,172 (per fold)
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
    name="mv_split",
    test_frac=0.2,  # 20% of full dataset held out for test
    val_frac=0.0,  # Must be 0 when num_folds > 1; validation is 1/num_folds
    num_folds=3,  # 3-fold cross-validation
    stratify=True,  # Preserve class distribution in each partition
)

task = DataSplittingTaskConfig(
    name="split_military_vehicles",
    workflow="mv_split",
    sources="mv_src",
)

# Build the full pipeline config — datasets, sources, workflows, and tasks
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
# - **Class distribution** — per-class counts and max imbalance ratio. MilitaryVehicles
#   spans 119 to 424 images per class, so expect a ratio around 3.6:1 rather than the
#   flat distribution a benchmark dataset would report.
# - **Split sizes** — train/val/test sample counts per fold. Expect ~4,172 train,
#   ~2,086 val, ~1,565 test.
# - **Pre-split balance** — mutual information between metadata factors and
#   class labels. High MI means a factor is predictive of the label (potential
#   bias source).
# - **Pre-split diversity** — Shannon diversity of metadata factors.
#   Low diversity means a factor has limited variation in the dataset.
#
# These tables are only as informative as the metadata behind them. MilitaryVehicles carries
# no telemetry, so the factors available are the ones measured from the imagery — and because
# its images vary in size (roughly 100×100 to 224×224), `height` and `width` are real factors
# here rather than constants. A dataset of uniformly sized images would leave these tables
# nearly empty.

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
# With `stratify=True`, each split should hold roughly proportional class counts — and on an
# imbalanced dataset that is a claim worth checking rather than assuming.
#
# The rarest class here has 119 images, so proportional allocation gives it 63 in train, 32
# in val and 24 in test. The largest has 424, giving 227/113/84. Watch the ratio between a
# class's share of a split and its share of the whole dataset: that is what stratification
# holds fixed, not the raw counts. The report's **Stratification quality** finding does this
# arithmetic for you — here it comes back with a maximum deviation of 0.1 percentage points.

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
# The factors here are `height` and `width`, the only ones the export carries besides the
# label. Both score near zero (about 0.01), which is the answer you want: vehicle type is not
# predictable from image geometry, so a model cannot take that shortcut instead of learning
# the vehicle itself. Diversity flags both as *low*, which says something different — image
# sizes cluster tightly even though they are not all identical.

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
# The exported JSON contains the split indices directly. Pair them with
# `result.dataset` — the resolved dataset the workflow actually ran on — and
# `View` to build filtered datasets for training, evaluation, or further analysis.
#
# Slicing `result.dataset` rather than re-loading from disk is what keeps the
# indices meaningful: they refer to positions in *that* dataset.

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
# In this tutorial you learned how to:
#
# - **Configure** the `data-splitting` workflow with test/val fractions, multi-fold, and stratification
# - **Run** the workflow via `run_task()` on MilitaryVehicles
# - **Read the splitting report** for class distribution health and split sizes
# - **Access raw split indices** — train, val, and test index lists across multiple folds
# - **Verify split integrity** — no overlap, full coverage, proportional class distribution
# - **Inspect balance and diversity** metrics computed on the pre-split dataset
# - **Use split indices** with `View` on `result.dataset` to build filtered datasets for downstream use
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

# %% [markdown]
# ## Related guides
#
# - **Concept** — [Dataset splitting](../concepts/DatasetSplitting.md):
#   stratification, multi-fold cross-validation, and leakage avoidance.
# - **How-to: Run workflows in containers** — [Containerized workflows](../how_to/containerized_workflows.md)
#   to run the splitting workflow from a YAML config inside a container.
