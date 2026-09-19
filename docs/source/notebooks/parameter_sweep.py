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
# **Target audience**: You are a T&E engineer or data scientist tuning a
# data-cleaning configuration who needs to evaluate sensitivity to detection
# thresholds before selecting operational settings.
#
# **Workflow role**: You can run a parameter sweep during the data-quality stage
# of your T&E workflow. Instead of guessing thresholds for [data cleaning](data_cleaning),
# you can sweep a range, inspect sensitivity tables, and select defensible parameters.
# See [Reproducibility](../concepts/Reproducibility.md) for details on config-keyed
# caching across runs.

# %% [markdown]
# ## What you will do
#
# - Load a subset of the MilitaryVehicles classification dataset.
# - Configure a `parameter-sweep` workflow across outlier thresholds and clustering sensitivities.
# - Run the sweep and reuse computed embeddings and statistics across trials.
# - Inspect the sweep pivot tables to compare results across parameter combinations.
# - Select defensible data-cleaning parameters based on sensitivity results.

# %% [markdown]
# ## What you will learn
#
# - How to automate hyperparameter tuning for data cleaning with `parameter-sweep`.
# - How to specify parameter sequences in workflow configurations.
# - How DataEval Flow caches intermediate computations across parameter runs.
# - How to interpret flat parameter curves versus sensitive parameters.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`).
# - Install `maite-datasets` to download MilitaryVehicles.
# - Ensure network access for the initial dataset download.

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the dataset
#
# [MilitaryVehicles](https://huggingface.co/datasets/leibnitz-lab/military_vehicles) is
# a classification dataset containing 9,444 images across 24 vehicle types. The dataset
# includes varying image resolutions, diverse visual backgrounds, and exact duplicate
# images.
#
# You can use `maite-datasets` to download the dataset. Setting `as_datamaite=True`
# writes the data in class-per-directory ImageFolder format, which you can load with
# `task="image_classification"`.
#
# :::{note}
# The download is approximately 141 MB across 9,466 files. You should set `HF_TOKEN`
# in your environment to enable parallel downloads and avoid anonymous rate limits.
# :::

# %% tags=["remove_output"]
from pathlib import Path

from dataeval.config import set_seed
from maite_datasets.image_classification import MilitaryVehicles

# Seed the random state used by MiniBatchKMeans and KMeans clusterers
# so the sweep produces deterministic results across machines.
set_seed(42)

data_root = Path("./data")

# Download once; subsequent runs read existing disk files.
MilitaryVehicles(root=data_root, image_set="base", download=True)
MilitaryVehicles(root=data_root, image_set="train", as_datamaite=True)

# The export nests images under the split directory.
data_path = data_root / "militaryvehicles_datamaite_train" / "train"
print(f"Reading from {data_path}")

# %% [markdown]
# ## Step 1: Build the Sweep Configuration
#
# The standard `data-cleaning` workflow takes scalar values. In contrast, you
# provide sequences (lists) for swept parameters in `parameter-sweep`.
#
# You will sweep:
# - `outlier_threshold`: Test values from 2.5 (aggressive) to 4.5 (conservative).
# - `duplicate_cluster_sensitivity`: Test sensitivities for near-duplicate clustering.
#
# You can use Bag of Visual Words (BoVW) embeddings for cluster-based detection.

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
    # Outlier parameters: sweep thresholds
    outlier_method=["adaptive"],
    outlier_threshold=[2.5, 3.5, 4.5],
    outlier_flags=["dimension", "pixel", "visual"],
    # Duplicate parameters: sweep sensitivity across documented 0.1-3.0 range
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
        # Shuffle before limiting to avoid sampling only the first alphabetical classes.
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
# When you run the task, the orchestrator executes the Cartesian product of all
# swept parameters.
#
# DataEval Flow computes intermediate image statistics and BoVW embeddings once,
# then reuses them across all 9 parameter combinations.

# %%
result = run_task(task, config, cache_dir=Path("./cache"))

# %% [markdown]
# ## Step 3: Analyze the Results
#
# The generated report contains one pivot table per outcome. Each table includes
# only the parameters that affect that outcome:
#
# - **Outliers Sweep**: Shows counts across `outlier_threshold`.
# - **Near Duplicates Sweep**: Shows counts across `duplicate_cluster_sensitivity`.
#
# Exact duplicates do not depend on swept inputs and appear in
# `result.data.raw.results`.

# %%
print(result.report())

# %% [markdown]
# ### Reading the two tables
#
# You should evaluate how each parameter changes the detected issue counts.
#
# **`outlier_threshold` affects outlier counts.** The outlier count decreases by
# roughly 40% across the swept range. The drop from 2.5 to 3.5 is significantly
# larger than the drop from 3.5 to 4.5. This flattening indicates that images flagged
# at 4.5 are distant outliers. You can select a threshold in the 3.5 to 4.5 range
# with stable results.
#
# **`duplicate_cluster_sensitivity` does not affect near-duplicate counts on this data.**
# The near-duplicate count remains identical at every swept setting.
#
# Cluster-based duplicate detection flags a pair when distance is less than
# `sensitivity * cluster_std`. For these embeddings, the average within-cluster pair
# distance is approximately 0.75 with a standard deviation of 0.08. At the widest
# setting of 3.0, the cutoff distance is `3.0 * 0.08 = 0.25`, whereas non-identical
# pairs have distances of 0.62 or higher. Every setting in the tested 0.1 to 3.0 range
# detects only pairs at distance zero (exact duplicates).
#
# :::{note}
# A flat column indicates that your results are robust to that parameter on this
# dataset. You can keep default sensitivity settings and focus calibration on parameters
# that change outputs.
# :::
#
# Based on these results, you can select `outlier_threshold` between 3.5 and 4.5,
# keep default `duplicate_cluster_sensitivity`, and run standard `data-cleaning`.

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Configure parameter sweeps using parameter sequences in workflow configurations.
# - Execute parameter sweeps while reusing intermediate embeddings and statistics across trials.
# - Inspect sweep pivot tables to analyze metric sensitivities across parameters.
# - Select stable data-cleaning thresholds based on sensitivity plateaus.

# %% [markdown]
# ## Next steps
#
# - **Data cleaning**: Apply chosen threshold parameters in an operational [Clean a dataset](data_cleaning) run.
# - **Drift tuning**: Sweep detection parameters for [Monitor incoming data for drift](drift_monitoring).

# %% [markdown]
# ## Related guides
#
# - **Concept**: [Reproducibility](../concepts/Reproducibility.md) explains how
#   config-keyed caching reuses embeddings and statistics across runs.
# - **Tutorial**: [Clean a dataset](data_cleaning) shows how to apply chosen
#   parameters in an operational `data-cleaning` run.
# - **How-to**: [Configure outlier detection](../how_to/configure_outlier_detection.md)
#   explains outlier parameters and selection strategies.
# - **How-to**: [Reuse results with the disk cache](../how_to/reuse_results_with_cache.md)
#   shows how to persist embeddings and statistics across runs.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md)
#   shows how to execute parameter sweeps from YAML configurations in containers.
