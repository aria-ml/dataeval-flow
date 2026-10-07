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
# Flag outliers and duplicates in SkySeaLand using the config-driven `quality` workflow, and take the dataset
# without them.

# %% [markdown]
# **Target audience**: You are a T&E engineer or data scientist vetting an
# operational dataset for quality issues before downstream use.
#
# **Workflow role**: You should run data cleaning as the initial stage of preparing
# an operational dataset. You can flag and remove outliers and duplicates before
# [Split a dataset](dataset_splitting), {doc}`Monitor incoming data for drift <drift_monitoring>`,
# or model training. See [Data quality and cleaning](../concepts/DataQualityAndCleaning.md)
# for detection concepts.

# %% [markdown]
# ## What you will do
#
# - Load the SkySeaLand object-detection dataset using `maite-datasets`.
# - Configure a `quality` workflow with BoVW (Bag of Visual Words) embeddings.
# - Run `run_task()` to detect statistical outliers and image duplicates.
# - Inspect the cleaning report and evaluate health status indicators.
# - Visually inspect flagged outlier and duplicate images using `dataeval-plots`.
# - Take the cleaned dataset from the workflow's `clean` step, and see how an `export` step writes it.

# %% [markdown]
# ## What you will learn
#
# - How to configure and execute the `quality` workflow with `run_task()`.
# - How to use BoVW feature extractors without external pretrained model files.
# - How to configure outlier detection parameters and duplicate sensitivity.
# - How to set check thresholds to trigger warning statuses.
# - How to interpret the formatted cleaning report.
# - How to read the result's findings, and what each of the workflow's steps found.
# - How to inspect flagged samples with `dataeval-plots`.
# - How to get the dataset without what the workflow flagged, and write it to disk.

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
# You must specify parameters explicitly in `quality`. In this configuration,
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
from dataeval_flow import run_task
from dataeval_flow.config import (
    CocoDatasetConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.workflows.quality import (
    ClassOutliersSettings,
    DuplicatesSettings,
    ImageDuplicatesSettings,
    ImageOutliersSettings,
    OutliersSettings,
    QualityChecks,
    QualityConfig,
    TargetOutliersSettings,
)

workflow = QualityConfig(
    name="skysealand_cleaning",
    outliers=OutliersSettings(
        outlier_threshold=("adaptive", 3.5),  # Adaptive thresholding for outliers, bound 3.5
        flags=["dimension", "pixel", "visual"],  # All image stat groups
        cluster_threshold=3.5,  # Cluster-based detection in embedding space (requires extractor).
        cluster_algorithm="hdbscan",
        n_clusters=4,  # SkySeaLand has 4 classes
    ),
    duplicates=DuplicatesSettings(  # Duplicate detection: hash-based plus cluster-based.
        cluster_sensitivity=0.5,
        cluster_algorithm="hdbscan",
        n_clusters=4,
    ),
    checks=QualityChecks(
        image_duplicates=ImageDuplicatesSettings(
            exact=0.0,  # No exact duplicates allowed (default)
            near=5.0,  # Up to 5% near duplicates before warning (default)
        ),
        image_outliers=ImageOutliersSettings(
            warning=5.0  # Relaxed from 3% default for four collection sites and varied sensors
        ),
        target_outliers=TargetOutliersSettings(
            warning=10.0  # Relaxed from 3% default for annotation variance in object detection
        ),
        class_outliers=ClassOutliersSettings(
            warning=12.0  # Relaxed from 3% default for diverse class appearances
        ),
    ),
)

task = TaskConfig(
    name="skysealand_clean",
    workflow="skysealand_cleaning",
    sources="skysealand_src",
    extractor="bovw_ext",
)

# Build the pipeline configuration: datasets, sources, extractors, views, workflows, and tasks
config = PipelineConfig(
    seed=0,
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
    workflows=[workflow],
    tasks=[task],
)

# %% [markdown]
# ## Step 2: Run the data cleaning workflow

# %%
result = run_task(config, task, cache_dir=Path("./cache"))

# %% [markdown]
# ### Cleaning report
#
# You can call `result.report()` to display outlier counts, duplicate groups,
# label statistics, and health statuses in a formatted text summary.

# %%
print(result.report())

# %% [markdown]
# ### Findings and steps
#
# `quality` is a preset: its settings expand to a chain of steps. Evaluators find the outliers and duplicates
# and count the labels, checks judge what they found against `checks`, and `clean` removes what was
# flagged. The report above gives each finding a section, with the evaluators it judged below it, then a section
# for each step no finding showed, such as `clean`, and a table of every step. `run_task()` returns a `ChainResult`
# holding each step by name, in run order:

# %%
print(list(result.steps))

# %% [markdown]
# `result.findings` holds the checks' findings, the ones the report's summary lists:

# %%
for finding in result.findings:
    print(f"{finding.severity:<8} {finding.title:<20} {finding.brief}")

# %% [markdown]
# ### Understanding health status
#
# The **Health** summary line indicates whether findings exceeded configured
# thresholds:
#
# - **ok** (`[ok]`): Nothing was flagged.
# - **info** (`[..]`): Finding is within the allowable threshold.
# - **warning** (`[!!]`): Finding exceeds the threshold and requires review.
#
# You can configure when findings warn with `checks`, keyed by check type. The default values are:
#
# | Metric | Default | When to adjust |
# |---|---|---|
# | `image-duplicates.exact` | 0% | Raise above 0 only if your pipeline intentionally repeats images |
# | `image-duplicates.near` | 5% | Lower to 1–2% for curated benchmarks; raise to 10–15% for web-scraped data |
# | `image-outliers.warning` | 3% | Lower to 1% for safety-critical data; raise to 5–10% for visually diverse collections |
# | `target-outliers.warning` | 3% | Lower to 1% for annotation reviews; raise to 5–10% for dense object detection |
# | `class-outliers.warning` | 3% | Lower to 1% for label-quality reviews; raise to 5–10% for diverse classes |
#
# Class imbalance is judged by the `bias` preset, not this one.
#
# In this tutorial, thresholds are relaxed because SkySeaLand includes four distinct
# capture sites with differing sensors, altitudes, and lighting conditions.
#
# To apply stricter thresholds, specify tighter tolerances:
#
# ```python
# from dataeval_flow.workflows.quality import (
#     QualityChecks,
#     ImageDuplicatesSettings,
#     ImageOutliersSettings,
# )
#
# strict = QualityChecks(
#     image_duplicates=ImageDuplicatesSettings(exact=0.0, near=2.0),
#     image_outliers=ImageOutliersSettings(warning=1.0),
# )
# ```

# %% [markdown]
# ### Inspecting flagged images
#
# You can inspect flagged images with `dataeval-plots` to determine whether
# detected anomalies represent data quality errors or acceptable operational variation.
#
# `result.sources` holds the dataset each source read, after its view: here the 300 frames the workflow ran on, which
# the flagged indices refer to. You can retrieve images from it without reloading from disk.

# %%
ds = result.sources["skysealand_src"]

# %% [markdown]
# #### Outlier images
#
# The `outliers` step's output is DataEval's own Outliers output. Its `data()` gives one row per flag: the image,
# the box where a box was flagged, the metric (such as brightness, entropy, or a dimension), its value, and the limit
# it crossed. You can extract the images flagged as a whole and display a sample.

# %%
outliers = result.steps["outliers"].output
issues = outliers.data()

# Image-level flags name no box; the rest flag single boxes.
image_issues = issues.filter(issues["target_index"].is_null())
outlier_grouped: dict[int, list[str]] = {}
for row in image_issues.iter_rows(named=True):
    outlier_grouped.setdefault(row["item_index"], []).append(row["metric_name"])

outlier_indices = sorted(outlier_grouped)
print(f"Image outliers: {len(outlier_indices)} images flagged, {image_issues.height} total flags")

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
# verify whether images are redundant. The `duplicates` step's output is DataEval's own
# Duplicates output, and its `items` holds the groups of whole images.
#
# SkySeaLand contains distinct captures without duplicates in this sample. When
# duplicate images are detected in a dataset, each group renders here for visual inspection.

# %%
duplicates = result.steps["duplicates"].output.items
exact_groups = duplicates.exact
near_groups = duplicates.near

print(f"Exact duplicate groups: {len(exact_groups)}")
print(f"Near  duplicate groups: {len(near_groups)}")

# %%
# Plot exact duplicate groups (if any)
for i, indices in enumerate(exact_groups[:3]):
    print(f"\nExact group {i}: indices {indices}")
    _ = plot(ds, indices=indices, images_per_row=len(indices), figsize=(4 * len(indices), 4), show_labels=True)

# %%
# Plot near duplicate groups (if any)
for i, (indices, methods) in enumerate(near_groups[:3]):
    print(f"\nNear group {i}: indices {indices}  (methods: {methods})")
    _ = plot(ds, indices=indices, images_per_row=len(indices), figsize=(4 * len(indices), 4), show_labels=True)

# %% [markdown]
# ## Step 3: Take the cleaned dataset
#
# The chain's last step, `clean`, removes each flagged image and box, and each duplicate but the first of its group.
# Its output is a DataEval `View` of the images that survived, which you can go on to train on or evaluate from
# Python. Its `details` count what it removed at each level, images (`items`) and boxes (`detections`), and under
# `by_plan` what each plan named, `duplicates` and `outliers`.

# %%
clean = result.steps["clean"].output
print(f"Images before cleaning: {len(ds)}")
print(f"Images after cleaning:  {len(clean)}")
print(f"Removed: {result.steps['clean'].details['removed']}")

# %% [markdown]
# To write the cleaned dataset to disk, run the `quality` entry as a step of a custom workflow, and add an
# `export` step that reads the step's `clean` output, `cleaning.clean`. `export` writes object-detection datasets,
# which SkySeaLand is. `data_cleaning.yaml`, beside this notebook, holds this tutorial's pipeline with that workflow
# and a task to run it:
#
# ```yaml
# workflows:
#   - name: clean_export
#     inputs: [data]
#     steps:
#       - {name: cleaning, workflow: skysealand_cleaning, input: data}
#       - {name: dataset, transform: export, input: cleaning.clean, format: coco}
#
# tasks:
#   - name: skysealand_export
#     workflow: clean_export
#     sources: skysealand_src
#     extractor: bovw_ext
# ```
#
# Run from this notebook's directory, `dataeval-flow --config data_cleaning.yaml --output ./output` runs both tasks.
# In `skysealand_export`, the `cleaning` step runs the chain above as `cleaning/outliers`, `cleaning/labels`, and so on
# to `cleaning/clean`, and `dataset` writes the cleaned dataset in COCO format under
# `output/datasets/skysealand_export.dataset/`. Without `--output`, nothing is written, and the export step is skipped.
# See {doc}`Chain steps into a workflow of your own <../how_to/write_a_custom_workflow>`.

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
# - Configure the `quality` workflow with outlier and duplicate detection parameters.
# - Use BoVW feature extractors without external model dependencies.
# - Set check thresholds to control warning generation.
# - Run the workflow via `run_task()` on a dataset view.
# - Read the cleaning report, its findings, and evaluate health statuses.
# - Visually inspect flagged outliers and duplicates using `dataeval-plots`.
# - Take the cleaned dataset from the `clean` step, and write it with an `export` step.
# - Export cleaning results to JSON format.

# %% [markdown]
# ## Next steps
#
# - **Audit**: Use the `audit` preset, as in {doc}`Audit a set of splits before training <audit>`, for cross-split
#   leakage, distribution shift, and shortcut risk before training.
# - **Threshold tuning**: Adjust `outliers.outlier_threshold`, test alternative outlier methods
#   like IQR, or compare several settings in one run with a task matrix, as
#   {doc}`Tune data cleaning with a matrix <tune_data_cleaning>` does.

# %% [markdown]
# ## Related guides
#
# - **Concept**: [Data quality and cleaning](../concepts/DataQualityAndCleaning.md) explains
#   the outlier and duplicate detection algorithms used in this workflow.
# - **How-to**: [Configure outlier detection](../how_to/configure_outlier_detection.md) explains
#   statistical methods, visual metrics, and check thresholds.
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
