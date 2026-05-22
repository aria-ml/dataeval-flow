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
# # Parameter Sweep for Data Cleaning
#
# Analyze the sensitivity of outlier and duplicate detection results across a range of statistical and clustering parameters using the `parameter-sweep` workflow.

# %% [markdown]
# ## What you'll do
#
# - Load a subset of the MNIST digit dataset
# - Configure a `parameter-sweep` workflow to test multiple outlier thresholds and clustering sensitivities
# - Run the sweep efficiently (computing embeddings and statistics only once)
# - View the **Sweep Results** pivot table to compare findings across parameter combinations
# - Learn how to interpret the results to select optimal parameters for your specific dataset

# %% [markdown]
# ## What you'll learn
#
# - How to use the `parameter-sweep` workflow to automate hyperparameter tuning for data cleaning
# - How to specify sequences of parameters in your configuration
# - How `dataeval-flow` optimizes multi-run execution by caching expensive intermediate data
# - Why datasets with high visual similarity (like MNIST) benefit from sweeping clustering sensitivity

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the dataset
#
# We'll use [MNIST](https://huggingface.co/datasets/ylecun/mnist), a classic dataset of handwritten digits.
# Because handwriting varies but often produces very similar shapes, MNIST is an excellent candidate
# for demonstrating how "near duplicate" detection sensitivity changes the number of groups found.

# %% tags=["remove_output"]
from pathlib import Path
from typing import cast

from dataeval.config import set_seed
from datasets import Dataset
from datasets import load_dataset as hf_load

# Seed the random state used by the BoVW vocabulary (MiniBatchKMeans) and the
# KMeans clusterer so the sweep produces identical numbers across machines and
# CI runs. Without this, results vary because random_state defaults to None.
set_seed(42)

# Load MNIST train split
mnist_train = cast(Dataset, hf_load("ylecun/mnist", split="train"))

# Save a subset of 1000 images to disk for the tutorial
# This size is large enough to contain natural similarities between digits
data_path = Path("./data/mnist_sweep")
mnist_train.select(range(1000)).save_to_disk(str(data_path))

# %% [markdown]
# ## Step 1: Build the Sweep Configuration
#
# Unlike the standard `data-cleaning` workflow which takes single values, the `parameter-sweep`
# workflow expects sequences (lists) for sweepable parameters.
#
# We will sweep:
# - `outlier_threshold`: From 2.5 (aggressive) to 4.5 (conservative)
# - `duplicate_cluster_sensitivity`: Different levels of sensitivity for finding near-duplicates
#
# We'll use the BoVW (Bag of Visual Words) extractor to provide the embeddings needed for
# cluster-based detection.

# %%
from dataeval_flow.config import (
    BoVWExtractorConfig,
    HuggingFaceDatasetConfig,
    ParameterSweepTaskConfig,
    ParameterSweepWorkflowConfig,
    PipelineConfig,
    SourceConfig,
)
from dataeval_flow.workflow import run_task

# Define the sweep workflow
sweep_workflow = ParameterSweepWorkflowConfig(
    name="mnist_sensitivity_sweep",
    # Outlier parameters (sweeping threshold)
    outlier_method=["adaptive"],
    outlier_threshold=[2.5, 3.5, 4.5],
    outlier_flags=["dimension", "pixel", "visual"],
    # Duplicate parameters (sweeping sensitivity)
    # High sensitivity (3.0) flags digits that are visually similar but not identical
    # Low sensitivity (0.5) only flags digits that are extremely similar
    duplicate_cluster_sensitivity=[0.5, 2.0, 3.0],
    duplicate_cluster_algorithm=["hdbscan"],
    duplicate_merge_near=True,
)

# Define the task referencing the sweep workflow
task = ParameterSweepTaskConfig(
    name="mnist_param_sweep", workflow="mnist_sensitivity_sweep", sources="mnist_src", extractor="bovw_ext"
)

# Build the full pipeline config
config = PipelineConfig(
    datasets=[
        HuggingFaceDatasetConfig(name="mnist_sweep", path=str(data_path)),
    ],
    sources=[
        SourceConfig(name="mnist_src", dataset="mnist_sweep"),
    ],
    extractors=[
        BoVWExtractorConfig(name="bovw_ext", vocab_size=256, batch_size=32),
    ],
    workflows=[sweep_workflow],
    tasks=[task],
)

# %% [markdown]
# ## Step 2: Run the Parameter Sweep
#
# The orchestrator identifies the `parameter-sweep` type and executes the Cartesian product
# of all provided parameter sequences.
#
# **Optimization Check**: Intermediate data like image statistics and BoVW embeddings are
# computed **once** and reused across all 9 combinations (3 thresholds * 3 sensitivities),
# making the sweep significantly faster than running 9 separate workflows.

# %%
result = run_task(task, config, cache_dir=Path("./cache"))

# %% [markdown]
# ## Step 3: Analyze the Results
#
# The report contains one pivot table per outcome — each table only shows the inputs that
# affect that outcome. With the sweep above you'll see:
#
# - **Outliers Sweep** — rows deduplicated by `outlier_threshold`
# - **Near Duplicates Sweep** — rows deduplicated by `duplicate_cluster_sensitivity`
#
# Exact duplicates depend on no swept input and are reported in `result.data.raw.results`
# rather than the pivot tables.

# %%
print(result.report())

# %% [markdown]
# ### Interpreting the Table for MNIST
#
# 1. **Duplicate Sensitivity**: In MNIST, you'll often see the `Near Duplicates` count increase
#    sharply as `duplicate_cluster_sensitivity` moves from 0.5 to 0.8. At 0.8, the workflow
#    is more likely to flag different people's handwriting of the same digit as "duplicates".
# 2. **Stability**: Look for a range where the `Outliers` count remains stable. If the count
#    is the same at threshold 3.5 and 4.5, it suggests those outliers are very distinct
#    from the rest of the dataset.
#
# Once you've identified the best parameters for your specific digit-recognition task,
# you can use them in a standard `data-cleaning` workflow to generate final index lists for training.
