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
# # Clean a dataset
#
# Flag outliers and duplicates in SkySeaLand using the config-driven `data-cleaning` workflow.

# %% [markdown]
# **Who this is for** — T&E engineers and data scientists who need to vet an
# operational dataset for quality problems before it is used downstream.
#
# **Where this fits** — Data cleaning is the first stage of preparing an
# operational dataset: you flag and remove outliers and duplicates here so that
# later stages — [dataset splitting](dataset_splitting), [drift monitoring](drift_monitoring),
# and model training — run on trustworthy data. See the
# [Data quality and cleaning](../concepts/DataQualityAndCleaning.md) concept page
# for the ideas behind the checks.

# %% [markdown]
# ## What you'll do
#
# - Load the SkySeaLand overhead-imagery dataset through `maite-datasets` in datamaite format
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
# - `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`)
# - `dataeval-plots` (for visualizing flagged images)
# - `maite-datasets` (provides SkySeaLand)
# - Internet connection — SkySeaLand downloads on first run (about 262 MB)

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the dataset
#
# [SkySeaLand](https://www.kaggle.com/datasets/mdzahidhasanriad/skysealand) is an overhead
# imagery detection dataset: 1,307 frames collected at four sites around the world and
# annotated with 19,102 objects across four classes — `airplane`, `boat`, `car` and `ship`.
#
# `maite-datasets` downloads it, and `as_datamaite=True` writes it back out in a format
# `dataeval-flow` reads directly. There is no conversion code to maintain here: the export
# is named after both the dataset and the `image_set`, so loading a second split later adds
# a folder rather than overwriting this one.

# %% tags=["remove_output"]
from pathlib import Path

from maite_datasets.object_detection import SkySeaLand

# Downloads to ./data/skysealand on first run (~262 MB), then writes the datamaite-format
# export beside it. A re-run reuses the export instead of rebuilding it.
SkySeaLand(root="./data", image_set="base", download=True, as_datamaite=True)

data_path = Path("./data/skysealand_datamaite_base")

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
#
# :::{note}
# **The view shuffles before it limits.** SkySeaLand stores its frames grouped by
# collection site, so a bare `Limit` would hand the workflow the first site and little
# else — the statistics below would then describe that site rather than the dataset. On
# this data a contiguous 500-frame slice is 11.2:1 across classes where the dataset as a
# whole is 1.9:1. Shuffling with a fixed `seed` keeps the sample representative and the
# run reproducible. See [Narrow a dataset with views](../how_to/build_dataset_views.md).
# :::

# %%
from dataeval.config import set_max_processes

from dataeval_flow.config import (
    BoVWExtractorConfig,
    CocoDatasetConfig,
    DataCleaningTaskConfig,
    DataCleaningWorkflowConfig,
    PipelineConfig,
    SourceConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.cleaning.params import DataCleaningHealthThresholds

# Each worker decodes its own images, so peak memory is roughly workers x batch x image
# size. Four keeps a 300-frame pass comfortable on a 16 GB machine; raise it if you have
# the headroom.
set_max_processes(4)

advisory_workflow = DataCleaningWorkflowConfig(
    name="skysealand_advisory_clean",
    mode="advisory",
    outlier_method="adaptive",  # Use adaptive thresholding for outliers
    outlier_threshold=3.5,
    outlier_flags=["dimension", "pixel", "visual"],  # All image stat groups
    outlier_cluster_threshold=3.5,  # Cluster-based detection in embedding space (requires extractor).
    outlier_cluster_algorithm="hdbscan",
    outlier_n_clusters=4,  # SkySeaLand has 4 classes
    duplicate_cluster_sensitivity=0.5,  # Duplicate detection — hash-based plus cluster-based.
    duplicate_cluster_algorithm="hdbscan",
    duplicate_n_clusters=4,
    health_thresholds=DataCleaningHealthThresholds(
        exact_duplicates=0.0,  # No exact duplicates allowed (default)
        near_duplicates=5.0,  # Up to 5% near duplicates before warning (default)
        image_outliers=5.0,  # Relaxed from 3% default — four collection sites, varied sensors
        target_outliers=10.0,  # Relaxed from 3% default — object detection has annotation variance
        classwise_outliers=12.0,  # Relaxed from 3% default — some classes are visually diverse
        class_label_imbalance=5.0,  # Default; SkySeaLand sits near 1.9:1, well inside it
    ),
)

task = DataCleaningTaskConfig(
    name="skysealand_clean",
    workflow="skysealand_advisory_clean",
    sources="skysealand_src",
    extractor="bovw_ext",
)

# Build the full pipeline config — datasets, sources, extractors, views, workflows, and tasks
config = PipelineConfig(
    datasets=[
        CocoDatasetConfig(name="skysealand_base", path=str(data_path)),
    ],
    views=[
        ViewConfig(
            name="sample300",
            operations=[
                ViewOperation(type="Shuffle", params={"seed": 0}),
                ViewOperation(type="Limit", params={"size": 300}),
            ],
        ),
    ],
    sources=[
        SourceConfig(name="skysealand_src", dataset="skysealand_base", view="sample300"),
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
# SkySeaLand is a diverse collection: its frames come from four sites with different
# sensors, altitudes and lighting, so a moderate outlier rate is expected rather than
# alarming.
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
# The result object carries the resolved, post-view dataset so we can
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
#
# This sample finds none. SkySeaLand's frames are distinct captures, so the two loops below
# produce no output — which is the result you want from a curated collection, and worth
# seeing so you recognise it. On data that does carry redundancy — a sensor repeating a
# frame, augmented copies kept alongside their originals, the same image pulled from two
# sources — each group renders here for you to judge.

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
    update={"name": "skysealand_prep_clean", "mode": "preparatory"},
)

task_prep = DataCleaningTaskConfig(
    name="skysealand-clean-prep",
    workflow="skysealand_prep_clean",
    sources="skysealand_src",
    extractor="bovw_ext",
)

config_prep = PipelineConfig(
    datasets=config.datasets,
    views=config.views,
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
# - **Run** the workflow via `run_tasks()` on a SkySeaLand view
# - **Read the cleaning report** — a single `result.report()` call for a formatted summary with health status
# - **Visually inspect** flagged outliers and duplicates with `dataeval-plots`
# - **Use preparatory mode** to get `flagged_indices` and `clean_indices` for downstream filtering
# - **Export** results to JSON for integration with automated pipelines

# %% [markdown]
# ## What's next
#
# - **Data analysis** — Use the `data-analysis` workflow for a comprehensive multi-split quality report
#   including cross-split leakage, distribution shift, and bias analysis
# - **Custom thresholds** — Tune `outlier_threshold`, switch to `"iqr"` method, or adjust
#   `health_thresholds` for different sensitivity and warning profiles

# %% [markdown]
# ## Related guides
#
# - **Concept** — [Data quality and cleaning](../concepts/DataQualityAndCleaning.md):
#   the outlier and duplicate detection ideas behind this workflow.
# - **How-to: Configure outlier detection** — [Configure outlier detection](../how_to/configure_outlier_detection.md)
#   to pick a statistical method, choose which statistics to test, add cluster-based detection, and set the
#   thresholds that turn a finding into a warning.
# - **How-to: Read evaluation outputs** — [Read evaluation outputs](../how_to/read_evaluation_outputs.md)
#   to interpret the report, export the result envelope, and pull the flagged indices out for inspection.
# - **How-to: Narrow a dataset with views** — [Narrow a dataset with views](../how_to/build_dataset_views.md)
#   to limit, filter, or sample the dataset before this workflow sees it.
# - **How-to: Run workflows in containers** — [Containerized workflows](../how_to/containerized_workflows.md)
#   to build a container image, write a YAML config, and run this workflow with `docker run`.
# - **How-to: Use an ONNX model for embeddings** — [ONNX embeddings](onnx_embeddings)
#   to configure a pretrained ResNet50 model with preprocessing transforms for higher-fidelity embeddings.
# - **How-to: Use a torchvision dataset** — [torchvision datasets](torchvision_datasets)
#   to feed a `torchvision` classification or detection dataset straight into this workflow.
