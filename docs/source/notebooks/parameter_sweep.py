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
# # Parameter Sweep for Data Cleaning
#
# Analyze the sensitivity of outlier and duplicate detection results across a range of statistical and clustering parameters using the `parameter-sweep` workflow.

# %% [markdown]
# **Who this is for** — T&E engineers and data scientists tuning a data-cleaning
# configuration who need to understand how sensitive their findings are to
# detection thresholds before committing to a setting.
#
# **Where this fits** — A parameter sweep is a calibration step within the data
# quality stage of the T&E workflow: rather than guessing thresholds for
# [data cleaning](data_cleaning), you sweep a range, read the sensitivity tables,
# and lock in defensible parameters for the operational cleaning run. See the
# [Reproducibility](../concepts/Reproducibility.md) concept page for how the
# sweep reuses intermediate computation across runs via config-keyed caching.

# %% [markdown]
# ## What you'll do
#
# - Load a subset of the MilitaryVehicles classification dataset
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
# - How to tell a parameter that matters for your data from one that does not — and why a
#   flat column is a result worth having rather than a broken run

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the dataset
#
# [MilitaryVehicles](https://huggingface.co/datasets/leibnitz-lab/military_vehicles) is a
# classification dataset of 9,444 images across 24 vehicle types, curated from open sources so
# each vehicle appears in a wide range of settings. That is what makes it a realistic subject
# for calibration: the imagery is visually diverse, sizes vary from roughly 100x100 to 224x224,
# and a handful of frames are genuine republications of the same photograph.
#
# `maite-datasets` downloads it, and `as_datamaite=True` writes it back out as a
# class-per-directory tree — the same HuggingFace **ImageFolder** layout `dataeval-flow` reads
# with `task="image_classification"`.
#
# :::{note}
# The download is ~141 MB spread over 9,466 individual files. Without an `HF_TOKEN` in the
# environment, `maite-datasets` fetches them one at a time to stay under the anonymous rate
# limit, which takes the better part of an hour; with a token it downloads in parallel. Set
# one before the first run if you can.
# :::

# %% tags=["remove_output"]
from pathlib import Path

from dataeval.config import set_seed
from maite_datasets.image_classification import MilitaryVehicles

# Seed the random state used by the BoVW vocabulary (MiniBatchKMeans) and the
# KMeans clusterer so the sweep produces identical numbers across machines and
# CI runs. Without this, results vary because random_state defaults to None.
set_seed(42)

data_root = Path("./data")

# One download shared by every export; a re-run reads what is already on disk.
MilitaryVehicles(root=data_root, image_set="base", download=True)
MilitaryVehicles(root=data_root, image_set="train", as_datamaite=True)

# The export nests its images one level down, under the split name.
data_path = data_root / "militaryvehicles_datamaite_train" / "train"
print(f"Reading from {data_path}")

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
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.workflow import run_task

# Define the sweep workflow
sweep_workflow = ParameterSweepWorkflowConfig(
    name="mv_sensitivity_sweep",
    # Outlier parameters (sweeping threshold)
    outlier_method=["adaptive"],
    outlier_threshold=[2.5, 3.5, 4.5],
    outlier_flags=["dimension", "pixel", "visual"],
    # Duplicate parameters (sweeping sensitivity across its documented 0.1-3.0 range)
    # Low sensitivity (0.5) only flags pairs that are extremely close together
    # High sensitivity (3.0) also flags pairs farther apart relative to the cluster spread
    duplicate_cluster_sensitivity=[0.5, 2.0, 3.0],
    duplicate_cluster_algorithm=["hdbscan"],
    duplicate_merge_near=True,
)

# Define the task referencing the sweep workflow
task = ParameterSweepTaskConfig(
    name="mv_param_sweep", workflow="mv_sensitivity_sweep", sources="mv_src", extractor="bovw_ext"
)

# Build the full pipeline config
config = PipelineConfig(
    datasets=[
        HuggingFaceDatasetConfig(name="mv_train", path=str(data_path), task="image_classification"),
    ],
    views=[
        # Shuffle before limiting: the export is stored one directory per class, so a bare
        # Limit would sample the first few classes alphabetically rather than the dataset.
        ViewConfig(
            name="sample1000",
            operations=[
                ViewOperation(type="Shuffle", params={"seed": 42}),
                ViewOperation(type="Limit", params={"size": 1000}),
            ],
        ),
    ],
    sources=[
        SourceConfig(name="mv_src", dataset="mv_train", view="sample1000"),
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
# ### Reading the two tables
#
# The two tables answer the same question about different knobs — *how much does this
# parameter change what I find?* — and on this dataset they answer it very differently.
#
# **`outlier_threshold` matters.** The count falls by roughly 40% across the swept range, and
# the steps are uneven: the drop from 2.5 to 3.5 is much larger than from 3.5 to 4.5. That
# flattening is the useful signal. Images still flagged at 4.5 are far enough from the rest of
# the collection that no reasonable threshold would forgive them, so 3.5 to 4.5 is the
# defensible band to pick from, and the exact value inside it barely matters.
#
# **`duplicate_cluster_sensitivity` does not.** The `Near Duplicates` count is identical at
# every setting. That is not a broken run — it is a property of this data, and it is worth
# understanding rather than working around.
#
# Cluster-based duplicate detection flags a pair when the distance between them is less than
# `sensitivity x cluster_std`. On these embeddings the typical pair inside a cluster sits at a
# distance of about 0.75 with a spread of only 0.08 — a mean roughly **nine times** the
# standard deviation. The widest setting swept here puts the cutoff at `3.0 x 0.08 = 0.25`,
# while even the closest 1% of non-identical pairs sit at 0.62. Every setting in the
# parameter's documented 0.1-3.0 range therefore selects exactly the same thing: the pairs at
# distance zero, which are real duplicates. You would need a sensitivity near 7.5 before the
# knob began to bite, and long before that it would be flagging distinct vehicles.
#
# :::{note}
# **A flat column is a result, not a failure.** It says the finding is robust to that
# parameter, so you can leave it at its default and spend your calibration effort elsewhere.
# The same sweep on a dataset of near-identical imagery — handwritten digits, frames pulled
# from one video — would show the opposite, because there the within-cluster spread is
# comparable to the distances themselves. Which regime *your* data is in is not something you
# can tell by reading the parameter documentation; it is what running the sweep tells you.
# :::
#
# So the calibration decision from this run is: pick an `outlier_threshold` in the 3.5-4.5
# band, leave `duplicate_cluster_sensitivity` alone, and carry both into a standard
# `data-cleaning` workflow to generate the final index lists for training.

# %% [markdown]
# ## Related guides
#
# - **Concept** — [Reproducibility](../concepts/Reproducibility.md): how
#   config-keyed caching lets the sweep reuse cached embeddings and statistics
#   across runs.
# - **Tutorial** — [Clean a dataset](data_cleaning): apply the parameters you select
#   here in a standard `data-cleaning` run.
# - **How-to: Configure outlier detection** — [Configure outlier detection](../how_to/configure_outlier_detection.md)
#   for what each swept parameter actually controls, and which are worth sweeping rather than guessing.
# - **How-to: Reuse results with the disk cache** — [Reuse results with the disk cache](../how_to/reuse_results_with_cache.md)
#   to persist embeddings and statistics so each grid point does not recompute them.
# - **How-to: Run workflows in containers** — [Containerized workflows](../how_to/containerized_workflows.md)
#   to run a parameter sweep from a YAML config inside a container.
