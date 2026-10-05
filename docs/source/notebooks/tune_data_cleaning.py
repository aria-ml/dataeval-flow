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
# # Tune data cleaning with a matrix
#
# Run one `data-cleaning` entry over a grid of outlier thresholds and near-duplicate sensitivities, and compare what
# each combination finds in one table.

# %% [markdown]
# **Target audience**: You are a T&E engineer or data scientist tuning a
# data-cleaning configuration who needs to see how sensitive its findings are to the
# detection thresholds before choosing operational settings.
#
# **Workflow role**: You can run a matrix during the data-quality stage of your T&E
# workflow. Instead of guessing thresholds for {doc}`data cleaning <data_cleaning>`,
# you can run a range of them, compare the findings side by side, and choose settings
# you can defend. See [Reproducibility](../concepts/Reproducibility.md) for how the
# runs share one draw of the data and one cache.

# %% [markdown]
# ## What you will do
#
# - Load a subset of the MilitaryVehicles classification dataset.
# - Configure a `data-cleaning` entry, and a task whose `matrix` varies two of its settings.
# - Run the task once per combination, sharing the statistics, embeddings and clusters across runs.
# - Read the comparison table, and one run's result in full.
# - Choose data-cleaning settings from what the table shows.

# %% [markdown]
# ## What you will learn
#
# - How a task's `matrix` runs its entry once per combination of the values it lists.
# - How to read a matrix result's comparison table, and reach each run's result.
# - How to tell a setting that changes the findings from one that changes nothing on your data.

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

from maite_datasets.image_classification import MilitaryVehicles

data_root = Path("./data")

# Download once; subsequent runs read existing disk files.
MilitaryVehicles(root=data_root, image_set="base", download=True)
MilitaryVehicles(root=data_root, image_set="train", as_datamaite=True)

# The export nests images under the split directory.
data_path = data_root / "militaryvehicles_datamaite_train" / "train"
print(f"Reading from {data_path}")

# %% [markdown]
# ## Step 1: Build the configuration
#
# The `data-cleaning` entry is an ordinary one, and must be valid on its own: it sets
# `outliers.outlier_threshold` and `outliers.flags`, which have no defaults. Its cluster settings,
# `duplicates.cluster_algorithm` here, turn on cluster-based near-duplicate detection,
# which reads the BoVW embeddings.
#
# The task's `matrix` names the settings to vary and the values each takes:
#
# - `outliers.outlier_threshold`: `adaptive` with the bound 2.5 (aggressive), 3.5 and 4.5 (conservative).
# - `duplicates.cluster_sensitivity`: 0.5, 2.0 and 3.0, within the 0.1 to 3.0 that DataEval
#   documents as its typical range.
#
# One grid crosses every key with every other, so the task runs 3 × 3 = 9 times. The
# entry's own values for these two settings, if it set any, would be replaced in each run.

# %%
from dataeval_flow import PipelineConfig, run_task
from dataeval_flow.config import (
    HuggingFaceDatasetConfig,
    SourceConfig,
    TaskConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig

cleaning = DataCleaningConfig(
    name="mv_cleaning",
    outliers={"flags": ["dimension", "pixel", "visual"], "outlier_threshold": "adaptive"},
    duplicates={"cluster_algorithm": "hdbscan", "merge_near_duplicates": True},
)

task = TaskConfig(
    name="mv_tuning",
    workflow="mv_cleaning",
    sources="mv_src",
    extractor="bovw_ext",
    matrix={
        "outliers.outlier_threshold": [["adaptive", 2.5], ["adaptive", 3.5], ["adaptive", 4.5]],
        "duplicates.cluster_sensitivity": [0.5, 2.0, 3.0],
    },
)

config = PipelineConfig(
    # Each run starts from this seed, so no run's clustering depends on the runs before it.
    seed=42,
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
    workflows=[cleaning],
    tasks=[task],
)

# %% [markdown]
# ## Step 2: Run the matrix
#
# `run_task` checks every run's configuration first, then runs the entry once per
# combination and returns one `MatrixResult`. The runs share one draw of the source's
# view, and the cache computes the image statistics, the BoVW embeddings and the
# clusters once for all nine.

# %%
result = run_task(task, config, cache_dir=Path("./cache"))
print(type(result).__name__, len(result.runs), "runs")

# %% [markdown]
# ## Step 3: Read the comparison table
#
# `report(detailed=False)` is the short form the console prints: the matrix's health, then
# a table with a row per run. Each row gives the run's number, the values it ran with, its
# health (`[!!]` where it warned), and a column per finding, each cell the finding's
# severity marker and brief. The table is wide, so `width=160` gives it room; at the
# console's 80 columns it wraps its cells further.

# %%
print(result.report(detailed=False, width=160))

# %% [markdown]
# ### Reading the table
#
# Every run warns, with 3 warnings each and 27 in all: Image Outliers, Classwise Outliers
# and Image Duplicates are past their thresholds in every run, and Class Imbalance is `info`.
#
# **`outliers.outlier_threshold` changes the outlier findings.** The adaptive method flags 157
# images (15.7%) at 2.5, 112 (11.2%) at 3.5 and 96 (9.6%) at 4.5. The count falls by 45
# images from 2.5 to 3.5, then by 16 from 3.5 to 4.5: it changes less the higher the
# threshold. The class with the largest share of outliers changes with it: 2S19 MSTA
# (25.0%) at 2.5, and Tornado at 3.5 (20.4%) and 4.5 (16.3%). Even at 4.5, 9.6% of the
# sample is flagged, over the 3% at which `image-outliers` warns by default, and 23 of the
# 24 classes are over the classwise threshold at every setting.
#
# **`duplicates.cluster_sensitivity` changes nothing on this sample.** Runs 1, 2 and 3
# differ only in it, and their rows are identical; so are runs 4 to 6, and runs 7 to 9.
# Every run counts 4 images as exact duplicates (0.4%) and 6 as near duplicates (0.6%).
# Step 4 shows why.
#
# **The label distribution depends on neither setting**: 24 classes, 1,000 items and an
# imbalance of 3.0:1 in every run.

# %% [markdown]
# ## Step 4: Read one run in full
#
# Each run's result is the result the task would return run alone with those values.
# `result.runs` holds the runs in order, each with its number, its values and its
# result, here a `ChainResult`. Run 4 is the fourth:

# %%
run = result.runs[3]
print(run.number, run.label)
for finding in run.result.findings:
    print(f"{finding.severity:<8} {finding.title:<20} {finding.brief}")

# %% [markdown]
# Its findings are row 4 of the table. Its steps hold each step's output, as a lone
# `data-cleaning` run's do. The `duplicates` step's output is DataEval's duplicates output, and
# its `data()` lists the groups behind the Image Duplicates finding, with the methods that found
# each:

# %%
print(run.result.steps["duplicates"].output.data())

# %% [markdown]
# The two exact groups are pairs that `xxhash` matched. Of the three near groups, one is a
# pair the perceptual hashes (`dhash`, `phash`) found, and the other two are the same two
# exact pairs, found again by the cluster pass. The cluster pass is the only part of
# duplicate detection that `duplicates.cluster_sensitivity` changes, and at 0.5, 2.0 and
# 3.0 alike it finds the exact copies and nothing else, so the count does not move.
#
# `result.report(detailed=True)`, the default, adds each run's full report under the
# table, in a `Runs` section, each headed with its number and values, as `dataeval-flow -v`
# prints it.

# %% [markdown]
# ### Choosing settings
#
# From this table:
#
# - **`outliers.outlier_threshold` matters.** The count is still falling at 4.5, but by about a
#   third of what it fell from 2.5 to 3.5, so a value from 3.5 to 4.5 depends less on the
#   exact choice than one below 3.5. Even 4.5 flags more than the default 3% warning, so
#   look at the flagged images, as {doc}`Clean a dataset <data_cleaning>` does, before
#   raising `checks.image-outliers.warning` for a collection this varied.
# - **`duplicates.cluster_sensitivity` does not matter here**: 0.5, 2.0 and 3.0 give the
#   same groups. Keep any of them, or leave it and `duplicates.cluster_algorithm` unset to
#   skip the cluster pass, which found no group the hashes missed.

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Vary a `data-cleaning` entry's settings with a `matrix` on its task.
# - Run every combination over one draw of the data, sharing the statistics, embeddings and clusters.
# - Read the comparison table, and reach each run's result.
# - Choose data-cleaning settings from the settings that change the findings.

# %% [markdown]
# ## Next steps
#
# - **Data cleaning**: Apply the chosen settings in an operational {doc}`Clean a dataset <data_cleaning>` run.
# - **Other settings**: Vary any workflow's or evaluator's settings, its sources or its extractor, as
#   {doc}`Sweep settings with a matrix <../how_to/run_a_matrix>` shows.

# %% [markdown]
# ## Related guides
#
# - **How-to**: {doc}`Sweep settings with a matrix <../how_to/run_a_matrix>` covers every
#   way to write a matrix's values and keys, and what each run shares.
# - **Concept**: [Reproducibility](../concepts/Reproducibility.md) explains how
#   config-keyed caching reuses embeddings and statistics across runs.
# - **Tutorial**: {doc}`Clean a dataset <data_cleaning>` shows how to apply chosen
#   settings in an operational `data-cleaning` run.
# - **How-to**: [Configure outlier detection](../how_to/configure_outlier_detection.md)
#   explains outlier settings and how to choose them.
# - **How-to**: [Reuse results with the disk cache](../how_to/reuse_results_with_cache.md)
#   shows how to persist embeddings and statistics across runs.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md)
#   shows how to run a matrix from a YAML configuration in a container.
