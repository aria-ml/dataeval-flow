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
# # Clean a dataset
#
# Flag outliers and duplicates in CPPE-5 using the config-driven `data-cleaning` workflow.

# %% [markdown]
# ## What you'll do
#
# - Download the CPPE-5 dataset from HuggingFace and save it to disk
# - Build a workflow configuration using BoVW (Bag of Visual Words) for embedding extraction
# - Run `run_tasks()` to detect outliers and duplicates (including cluster-based detection)
# - View the built-in **cleaning report** for a high-level summary
# - Visually inspect flagged **outlier** and **duplicate** images with `dataeval-plots`
# - Use **preparatory mode** to get clean/flagged index lists for downstream filtering

# %% [markdown]
# ## What you'll learn
#
# - How to configure and run the `data-cleaning` workflow via `run_tasks()`
# - How to use BoVW (Bag of Visual Words) for lightweight embedding-based detection — no model file needed
# - What outlier and duplicate detection parameters are available
# - How to configure **health thresholds** to control when findings trigger warnings
# - How to read the built-in **cleaning report** (`result.report()`) for a quick summary with health status
# - How to visually inspect outliers and duplicates with `dataeval-plots`
# - The difference between **advisory** mode (report only) and **preparatory** mode (clean indices)

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datasets`, `maite-datasets`, `pydantic`)
# - `dataeval-plots` (for visualizing flagged images)
# - Internet connection (to download CPPE-5 from HuggingFace Hub on first run)

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load and prepare the dataset
#
# Download [CPPE-5](https://huggingface.co/datasets/rishitdagli/cppe-5) from HuggingFace and save it
# to disk. CPPE-5 is a well-known object detection dataset with 5 classes and 1K images.
# We'll work with a subset of the `train` split for simplicity sake.

# %% tags=["remove_output"]
from typing import cast

from datasets import Dataset
from datasets import load_dataset as hf_load

cppe5_train = cast(Dataset, hf_load("rishitdagli/cppe-5", split="train"))

# %% tags=["remove_output"]
from pathlib import Path

# Save to disk in HuggingFace arrow format
data_path = Path("./data/cppe5/train")
cppe5_train.save_to_disk(str(data_path))

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# The `data-cleaning` workflow requires explicit parameters (no hidden defaults). We'll configure
# outlier detection using **adaptive** thresholding across dimension, pixel, and visual statistics,
# and use the default hash-based duplicate detection. The BoVW (Bag of Visual Words) extractor learns
# a visual vocabulary directly from the dataset images — no model file or preprocessing needed.
#
# Adaptive thresholding automatically picks between Z-score and modified Z-score
# per metric based on the data distribution. A threshold of 3.5 keeps the outlier
# rate conservative without over-flagging on metrics with low variance.

# %%
from dataeval.config import set_max_processes

from dataeval_flow.config import (
    BoVWExtractorConfig,
    DataCleaningTaskConfig,
    DataCleaningWorkflowConfig,
    HuggingFaceDatasetConfig,
    PipelineConfig,
    SelectionConfig,
    SelectionStep,
    SourceConfig,
)
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.cleaning.params import DataCleaningHealthThresholds

set_max_processes(8)  # Set max processes for parallel execution (adjust as needed)

advisory_workflow = DataCleaningWorkflowConfig(
    name="cppe5_advisory_clean",
    mode="advisory",
    outlier_method="adaptive",  # Use adaptive thresholding for outliers
    outlier_threshold=3.5,
    outlier_flags=["dimension", "pixel", "visual"],  # All image stat groups
    outlier_cluster_threshold=3.5,  # Cluster-based detection in embedding space (requires extractor).
    outlier_cluster_algorithm="hdbscan",
    outlier_n_clusters=5,  # CPPE-5 has 5 classes
    duplicate_cluster_sensitivity=0.5,  # Duplicate detection — hash-based plus cluster-based.
    duplicate_cluster_algorithm="hdbscan",
    duplicate_n_clusters=5,
    health_thresholds=DataCleaningHealthThresholds(
        exact_duplicates=0.0,  # No exact duplicates allowed (default)
        near_duplicates=5.0,  # Up to 5% near duplicates before warning (default)
        image_outliers=5.0,  # Relaxed from 3% default — CPPE-5 has diverse images
        target_outliers=10.0,  # Relaxed from 3% default — object detection has annotation variance
        classwise_outliers=12.0,  # Relaxed from 3% default — some classes are visually diverse
        class_label_imbalance=5.0,  # CPPE-5 has moderate imbalance (default)
    ),
)

task = DataCleaningTaskConfig(
    name="cppe5_clean",
    workflow="cppe5_advisory_clean",
    sources="cppe5_src",
    extractor="bovw_ext",
)

# Build the full pipeline config — datasets, sources, extractors, selections, workflows, and tasks
config = PipelineConfig(
    datasets=[
        HuggingFaceDatasetConfig(name="cppe5_train", path=str(data_path)),
    ],
    selections=[
        SelectionConfig(name="first500", steps=[SelectionStep(type="Limit", params={"size": 500})]),
    ],
    sources=[
        SourceConfig(name="cppe5_src", dataset="cppe5_train", selection="first500"),
    ],
    extractors=[
        BoVWExtractorConfig(name="bovw_ext", vocab_size=512, batch_size=32),
    ],
    workflows=[advisory_workflow],
    tasks=[task],
)

# %% [markdown]
# ## Step 2: Run the data cleaning workflow

# %%
result = run_task(task, config, cache_dir=Path("./cache"))

# %% [markdown]
# ### Cleaning report
#
# The workflow result has a built-in `report()` method that renders a formatted
# text summary — outlier counts, duplicate groups, label stats, and **health status**
# in one view.

# %%
print(result.report())

# %% [markdown]
# ### Understanding health status
#
# The **Health** line at the bottom of the summary tells you whether any findings
# exceeded their configured thresholds. Each finding is either:
#
# - **info** — `[ok]` within the allowable threshold (no action needed)
# - **warning** `[!!]` — exceeds the threshold (review recommended)
#
# Health thresholds are configured via `DataCleaningHealthThresholds` on
# the `health_thresholds` parameter. The defaults are:
#
# | Metric | Default | When to adjust |
# |---|---|---|
# | `exact_duplicates` | 0% | Raise above 0 only if your pipeline intentionally repeats images |
# | `near_duplicates` | 5% | Lower to 1–2% for curated benchmarks; raise to 10–15% for web-scraped data |
# | `image_outliers` | 3% | Lower to 1% for safety-critical data; raise to 5–10% for visually diverse collections |
# | `target_outliers` | 3% | Lower to 1% for annotation audits; raise to 5–10% for dense object detection |
# | `classwise_outliers` | 3% | Lower to 1% for label-quality audits; raise to 5–10% for diverse classes |
# | `class_label_imbalance` | 5:1 | Lower to 3:1 for binary; raise to 10–20:1 for large hierarchies (25+ classes) |
#
# In this tutorial we **relax** several thresholds above their defaults because
# CPPE-5 is a diverse object-detection dataset where moderate outlier rates and
# class imbalance are expected.
#
# To tighten thresholds for a stricter audit:
#
# ```python
# from dataeval_flow.workflows.cleaning.params import DataCleaningHealthThresholds
#
# strict = DataCleaningHealthThresholds(
#     exact_duplicates=0.0,   # no exact duplicates (default)
#     near_duplicates=2.0,    # stricter near-duplicate limit
#     image_outliers=1.0,     # flag at 1% for safety-critical data
#     class_label_imbalance=3.0,  # tight balance for binary classification
# )
# params = DataCleaningParameters(..., health_thresholds=strict)
# ```

# %% [markdown]
# ### Inspecting flagged images
#
# The report tells us *how many* outliers and duplicates were found. Now let's
# actually **look** at them. We'll use `dataeval-plots` to render the flagged
# images directly in the notebook so we can judge whether they are genuine
# quality issues or acceptable variation.
#
# The result object carries the resolved, post-selection dataset so we can
# index into it directly — no need to reload from disk.

# %%
assert result.dataset is not None
ds = result.dataset

# %% [markdown]
# #### Outlier images
#
# Extract the image indices flagged as outliers and plot a sample. These are
# images whose statistics (brightness, entropy, dimensions, …) fall outside the
# expected range.

# %%
raw = result.data.raw

# Collect unique outlier image indices, grouped by image
outlier_issues = raw.img_outliers["issues"]
outlier_grouped: dict[int, list[str]] = {
    idx: [i["metric_name"] for i in outlier_issues if i["item_index"] == idx]
    for idx in {i["item_index"] for i in outlier_issues}
}

outlier_indices = sorted(outlier_grouped)
print(f"Image outliers: {len(outlier_indices)} images flagged, {len(outlier_issues)} total flags")

# %%
from dataeval_plots import plot

# Show a sample of outlier images (first 9)
if outlier_indices:
    sample = outlier_indices[:9]
    print("Outlier sample — flagged metrics per image:")
    for idx in sample:
        print(f"  Image {idx:>5d}: {', '.join(outlier_grouped[idx])}")
    _ = plot(ds, indices=sample, images_per_row=3, figsize=(12, 8), show_labels=True)

# %% [markdown]
# #### Duplicate images
#
# Plot each duplicate group side by side — both exact and near duplicates — so
# you can visually confirm whether the images are truly redundant.

# %%
exact_groups = raw.duplicates["items"].get("exact", [])
near_groups = raw.duplicates["items"].get("near", [])

print(f"Exact duplicate groups: {len(exact_groups)}")
print(f"Near  duplicate groups: {len(near_groups)}")

# %%
# Plot exact duplicate groups (if any)
for i, group in enumerate(exact_groups[:3]):
    indices = group if isinstance(group, list) else group["indices"]
    print(f"\nExact group {i}: indices {indices}")
    _ = plot(ds, indices=indices, images_per_row=len(indices), figsize=(4 * len(indices), 4), show_labels=True)

# %%
# Plot near duplicate groups (if any)
for i, group in enumerate(near_groups[:3]):
    indices = group["indices"]
    methods = group.get("methods", [])
    print(f"\nNear group {i}: indices {indices}  (methods: {methods})")
    _ = plot(ds, indices=indices, images_per_row=len(indices), figsize=(4 * len(indices), 4), show_labels=True)

# %% [markdown]
# ## Step 3: Preparatory mode — get clean indices
#
# Re-run with `mode="preparatory"` to compute which indices to keep and which to remove.
# This is useful for building a filtered dataset downstream.

# %%
# Define a preparatory pipeline — same params but mode="preparatory"
# Copy the advisory workflow and change name + mode
prep_workflow = advisory_workflow.model_copy(
    update={"name": "cppe5_prep_clean", "mode": "preparatory"},
)

task_prep = DataCleaningTaskConfig(
    name="cppe5-clean-prep",
    workflow="cppe5_prep_clean",
    sources="cppe5_src",
    extractor="bovw_ext",
)

config_prep = PipelineConfig(
    datasets=config.datasets,
    selections=config.selections,
    sources=config.sources,
    extractors=config.extractors,
    workflows=[advisory_workflow, prep_workflow],
    tasks=[task_prep],
)

result_prep = run_task(task_prep, config_prep, cache_dir=Path("./cache"))

# %% tags=["remove_cell"]
if not result_prep.success:
    print(f"Workflow failed: {result_prep.errors}")
assert result_prep.success

# %%
meta = result_prep.metadata
print(f"Mode: {meta.mode}")
print(f"Flagged for removal : {meta.removed_count}")
print(f"Retained (clean)    : {len(meta.clean_indices)}")

if meta.flagged_indices:
    print(f"\nFirst 20 flagged indices: {meta.flagged_indices[:20]}")

# %% [markdown]
# ## Results Exploration: Export results

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")
print(json_str[:500] + "\n...")

# %% [markdown]
# ## Conclusion
#
# In this tutorial you learned how to:
#
# - **Configure** the `data-cleaning` workflow with explicit outlier and duplicate detection parameters
# - **Use BoVW** (Bag of Visual Words) for lightweight embedding extraction — no model file or preprocessing needed
# - **Set health thresholds** to control when findings are elevated to warnings
# - **Run** the workflow via `run_tasks()` on a CPPE-5 split
# - **Read the cleaning report** — a single `result.report()` call for a formatted summary with health status
# - **Visually inspect** flagged outliers and duplicates with `dataeval-plots`
# - **Use preparatory mode** to get `flagged_indices` and `clean_indices` for downstream filtering
# - **Export** results to JSON for integration with automated pipelines

# %% [markdown]
# ## What's next
#
# - **Run in Docker** — See the [Containerized Workflows how-to](../how_to/containerized_workflows.md) to
#   build a container image, write a YAML config, and run a workflow with `docker run`
# - **Using ONNX models** — See the ONNX embeddings how-to (coming soon!) for
#   configuring a pretrained ResNet50 model with preprocessing transforms for higher-fidelity embeddings
# - **Data analysis** — Use the `data-analysis` workflow for a comprehensive multi-split quality report
#   including cross-split leakage, distribution shift, and bias analysis
# - **Custom thresholds** — Tune `outlier_threshold`, switch to `"iqr"` method, or adjust
#   `health_thresholds` for different sensitivity and warning profiles
