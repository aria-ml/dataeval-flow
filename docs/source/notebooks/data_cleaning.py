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
# **Target audience**: You are a T&E engineer or data scientist vetting an
# operational dataset for quality issues before downstream use.
#
# **Workflow role**: You should run data cleaning as the initial stage of preparing
# an operational dataset. You can flag and remove outliers and duplicates before
# [Split a dataset](dataset_splitting), [Monitor incoming data for drift](drift_monitoring),
# or model training. See [Data quality and cleaning](../concepts/DataQualityAndCleaning.md)
# for detection concepts.

# %% [markdown]
# ## What you will do
#
# - Load the SkySeaLand object-detection dataset using `maite-datasets`.
# - Configure a `data-cleaning` workflow with BoVW (Bag of Visual Words) embeddings.
# - Run `run_task()` to detect statistical outliers and image duplicates.
# - Inspect the cleaning report and evaluate health status indicators.
# - Visually inspect flagged outlier and duplicate images using `dataeval-plots`.
# - Run preparatory mode to generate index lists of clean and flagged samples.

# %% [markdown]
# ## What you will learn
#
# - How to configure and execute the `data-cleaning` workflow with `run_task()`.
# - How to use BoVW feature extractors without external pretrained model files.
# - How to configure outlier detection parameters and duplicate sensitivity.
# - How to set health thresholds to trigger warning statuses.
# - How to interpret the formatted cleaning report.
# - How to inspect flagged samples with `dataeval-plots`.
# - How advisory mode (reporting only) differs from preparatory mode (index filtering).

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`).
# - Install `dataeval-plots` to visualize flagged images.
# - Install `maite-datasets` to access SkySeaLand.
# - Ensure network access for the initial dataset download (~262 MB).

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the dataset
#
# [SkySeaLand](https://www.kaggle.com/datasets/mdzahidhasanriad/skysealand) is an overhead
# object-detection dataset containing 1,307 frames across four classes: `airplane`, `boat`,
# `car`, and `ship`.
#
# You can use `maite-datasets` to download the dataset and export it with `as_datamaite=True`
# in a format readable by DataEval Flow. Subsequent runs load the export from disk.

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
# You must specify parameters explicitly in `data-cleaning`. In this configuration,
# you will configure outlier detection using **adaptive** thresholding across dimension,
# pixel, and visual statistics, along with hash-based and cluster-based duplicate detection.
# The BoVW (Bag of Visual Words) extractor learns visual words directly from dataset
# images without requiring an external model file.
#
# Adaptive thresholding automatically selects between Z-score and modified Z-score
# per metric based on data distribution. A threshold of 3.5 provides conservative
# outlier flagging.
#
# :::{note}
# You should shuffle before limiting a view. SkySeaLand frames are grouped by collection
# site on disk. Applying `Shuffle` with a fixed seed ensures that the 300-frame sample
# represents all collection sites. See [Narrow a dataset with views](../how_to/build_dataset_views.md).
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
    duplicate_cluster_sensitivity=0.5,  # Duplicate detection: hash-based plus cluster-based.
    duplicate_cluster_algorithm="hdbscan",
    duplicate_n_clusters=4,
    health_thresholds=DataCleaningHealthThresholds(
        exact_duplicates=0.0,  # No exact duplicates allowed (default)
        near_duplicates=5.0,  # Up to 5% near duplicates before warning (default)
        image_outliers=5.0,  # Relaxed from 3% default for four collection sites and varied sensors
        target_outliers=10.0,  # Relaxed from 3% default for annotation variance in object detection
        classwise_outliers=12.0,  # Relaxed from 3% default for diverse class appearances
        class_label_imbalance=5.0,  # Default; SkySeaLand sits near 1.9:1, well inside it
    ),
)

task = DataCleaningTaskConfig(
    name="skysealand_clean",
    workflow="skysealand_advisory_clean",
    sources="skysealand_src",
    extractor="bovw_ext",
)

# Build the pipeline configuration: datasets, sources, extractors, views, workflows, and tasks
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
# You can call `result.report()` to display outlier counts, duplicate groups,
# label statistics, and health statuses in a formatted text summary.

# %%
print(result.report())

# %% [markdown]
# ### Understanding health status
#
# The **Health** summary line indicates whether findings exceeded configured
# thresholds:
#
# - **info** (`[ok]`): Finding is within the allowable threshold.
# - **warning** (`[!!]`): Finding exceeds the threshold and requires review.
#
# You can configure health thresholds using `DataCleaningHealthThresholds` on
# `health_thresholds`. The default values are:
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
# In this tutorial, thresholds are relaxed because SkySeaLand includes four distinct
# capture sites with differing sensors, altitudes, and lighting conditions.
#
# To apply stricter thresholds, specify tighter tolerances:
#
# ```python
# from dataeval_flow.workflows.cleaning.params import DataCleaningHealthThresholds
#
# strict = DataCleaningHealthThresholds(
#     exact_duplicates=0.0,
#     near_duplicates=2.0,
#     image_outliers=1.0,
#     class_label_imbalance=3.0,
# )
# ```

# %% [markdown]
# ### Inspecting flagged images
#
# You can inspect flagged images with `dataeval-plots` to determine whether
# detected anomalies represent data quality errors or acceptable operational variation.
#
# You can access `result.dataset` directly to retrieve images without reloading from disk.

# %%
assert result.dataset is not None
ds = result.dataset

# %% [markdown]
# #### Outlier images
#
# You can extract image indices flagged by statistical checks (such as brightness,
# entropy, or dimensions) and display a sample.

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
    print("Outlier sample: flagged metrics per image:")
    for idx in sample:
        print(f"  Image {idx:>5d}: {', '.join(outlier_grouped[idx])}")
    _ = plot(ds, indices=sample, images_per_row=3, figsize=(12, 8), show_labels=True)

# %% [markdown]
# #### Duplicate images
#
# You can plot duplicate groups side by side (both exact and near duplicates) to
# verify whether images are redundant.
#
# SkySeaLand contains distinct captures without duplicates in this sample. When
# duplicate images are detected in a dataset, each group renders here for visual inspection.

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
# ## Step 3: Preparatory mode: Get clean indices
#
# Run the workflow with `mode="preparatory"` to compute clean indices and flagged
# indices for downstream filtering.

# %%
# Define a preparatory pipeline with mode="preparatory"
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
# In this tutorial, you learned how to:
#
# - Configure the `data-cleaning` workflow with outlier and duplicate detection parameters.
# - Use BoVW feature extractors without external model dependencies.
# - Set health thresholds to control warning generation.
# - Run the workflow via `run_task()` on a dataset view.
# - Read the cleaning report and evaluate health statuses.
# - Visually inspect flagged outliers and duplicates using `dataeval-plots`.
# - Use preparatory mode to extract clean and flagged index lists for downstream filtering.
# - Export cleaning results to JSON format.

# %% [markdown]
# ## Next steps
#
# - **Data analysis**: Use the `data-analysis` workflow for cross-split leakage,
#   distribution shift, and bias analysis.
# - **Threshold tuning**: Adjust `outlier_threshold`, test alternative outlier methods
#   like IQR, or sweep parameters with `parameter-sweep`.

# %% [markdown]
# ## Related guides
#
# - **Concept**: [Data quality and cleaning](../concepts/DataQualityAndCleaning.md) explains
#   the outlier and duplicate detection algorithms used in this workflow.
# - **How-to**: [Configure outlier detection](../how_to/configure_outlier_detection.md) explains
#   statistical methods, visual metrics, and health thresholds.
# - **How-to**: [Read evaluation outputs](../how_to/read_evaluation_outputs.md) explains
#   how to parse reports and export result envelopes.
# - **How-to**: [Narrow a dataset with views](../how_to/build_dataset_views.md) covers
#   limiting, filtering, and sampling datasets.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md) explains
#   how to run cleaning workflows in Docker.
# - **Guide**: [Use an ONNX model for embeddings](onnx_embeddings) shows how to configure
#   pretrained ONNX models for feature extraction.
# - **Guide**: [Use a torchvision dataset with DataEval Flow](torchvision_datasets) shows
#   how to adapt torchvision datasets for DataEval workflows.
