# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: dataeval-flow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analyze dataset quality across splits
#
# Produce a comprehensive quality report across dataset splits using the config-driven
# `data-analysis` workflow.

# %% [markdown]
# **Target audience**: You are a T&E engineer or data scientist who needs a
# comprehensive quality evaluation across dataset splits before approving data
# for training or evaluation.
#
# **Workflow role**: You should use dataset analysis to audit splits before model
# training. Dataset analysis verifies that your partitions are free from cross-split
# leakage, excessive class imbalance, and distribution shifts. This workflow builds
# on [Clean a dataset](data_cleaning) and guides [Split a dataset](dataset_splitting).
# See [Data quality and cleaning](../concepts/DataQualityAndCleaning.md) for background.

# %% [markdown]
# ## What you will do
#
# - Download SkySeaLand using `maite-datasets` and export its train, validation, and test splits.
# - Configure a multi-split `data-analysis` workflow.
# - Execute the workflow with `run_task()`.
# - Inspect the analysis report across image quality, redundancy, label health, and bias.
# - Evaluate cross-split comparisons, including label overlap, duplicate leakage, and distribution parity.
# - Configure health thresholds to govern warning severities.
# - Export evaluation results to JSON format.

# %% [markdown]
# ## What you will learn
#
# - How to configure and execute the `data-analysis` workflow.
# - How to interpret findings and severity indicators in `result.report()`.
# - What each of the five assessment areas evaluates.
# - How to tune health thresholds for your quality standards.
# - How to distinguish statistical significance from practical divergence in parity testing.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`).
# - Install `maite-datasets[datamaite]` to download and export SkySeaLand.
# - Ensure network access for the initial dataset download (~262 MB).

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the splits the dataset ships with
#
# [SkySeaLand](https://www.kaggle.com/datasets/mdzahidhasanriad/skysealand) is an overhead
# object-detection dataset containing 1,307 frames from four sites across `airplane`, `boat`,
# `car`, and `ship`. The dataset provides `train` (1,048 frames), `val` (132), and `test`
# (127) partitions. You can use this workflow to evaluate whether partitioned splits agree
# with each other.
#
# You can use `maite-datasets` with `as_datamaite=True` to export the splits into COCO format.
# Each split is exported into a distinct directory.

# %% tags=["remove_output"]
from pathlib import Path

from maite_datasets.object_detection import SkySeaLand

data_root = Path("./data")

# One ~262 MB download into ./data/skysealand, shared by all three exports below.
# A re-run reads what is already on disk instead of downloading again.
SkySeaLand(root=data_root, image_set="base", download=True)

split_paths = {name: data_root / f"skysealand_datamaite_{name}" for name in ("train", "val", "test")}
for image_set in split_paths:
    SkySeaLand(root=data_root, image_set=image_set, as_datamaite=True)

print("\n".join(f"{name}: {path}" for name, path in split_paths.items()))

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# You must specify workflow parameters explicitly. In this configuration, you will
# configure outlier detection using **adaptive** thresholding across dimension, pixel,
# and visual statistics, and enable **balance** and **diversity** analysis for bias checks.
#
# Note these three configuration decisions:
#
# - **Sample `train`; profile `val` and `test` completely**: Sampling 300 training
#   frames reduces memory and computation while matching the sample scale of validation
#   and test sets. Remove the view to profile the full training set.
# - **Shuffle before limiting**: SkySeaLand is organized by collection site on disk.
#   Applying `Shuffle` before `Limit` ensures that the sample represents all collection
#   sites.
# - **Declare a metadata policy for bias factors**: SkySeaLand provides image metadata
#   measured directly from pixel data. Setting `reference_split="train"` ensures that all
#   three splits use identical bin cuts for comparable statistics.

# %%
from dataeval.config import set_max_processes

from dataeval_flow.config import (
    CocoDatasetConfig,
    DataAnalysisTaskConfig,
    DataAnalysisWorkflowConfig,
    PipelineConfig,
    SourceConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.config.schemas import MetadataPolicyConfig
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.analysis.params import DataAnalysisHealthThresholds

# Limit concurrency to 4 processes for memory management during image decoding.
set_max_processes(4)

analysis_workflow = DataAnalysisWorkflowConfig(
    name="skysealand_analysis",
    outlier_method="adaptive",
    outlier_flags=["dimension", "pixel", "visual"],
    outlier_threshold=4.0,
    balance=True,
    diversity_method="simpson",
    metadata="skysealand_factors",
    health_thresholds=DataAnalysisHealthThresholds(
        image_outliers=5.0,  # Relaxed from 3% for diverse overhead imagery
        exact_duplicates=0.0,  # No exact duplicates allowed (default)
        near_duplicates=5.0,  # Up to 5% near duplicates before warning (default)
        class_label_imbalance=5.0,  # Default; SkySeaLand sits near 2.5:1 in every split
        distribution_shift=0.5,  # Default
    ),
)

task = DataAnalysisTaskConfig(
    name="skysealand-quality-check",
    workflow="skysealand_analysis",
    sources=["train", "val", "test"],
)

config = PipelineConfig(
    metadata=[
        MetadataPolicyConfig(
            name="skysealand_factors",
            # Image statistics evaluated as factors for bias analysis
            intrinsic_factors=["visual", "pixel"],
            # Exclude constant metadata fields
            exclude=["label_file_exists"],
            # Shared encoding reference for cross-split factor comparability
            reference_split="train",
        )
    ],
    datasets=[CocoDatasetConfig(name=f"skysealand_{name}", path=str(path)) for name, path in split_paths.items()],
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
        SourceConfig(name="train", dataset="skysealand_train", view="sample300"),
        SourceConfig(name="val", dataset="skysealand_val"),
        SourceConfig(name="test", dataset="skysealand_test"),
    ],
    workflows=[analysis_workflow],
    tasks=[task],
)

print("Configuration ready:")
print(f"  Workflow:   {analysis_workflow.name} (type={analysis_workflow.type})")
print(f"  Task:       {task.name} -> {task.workflow}")
print(f"  Sources:    {task.sources}")

# %% [markdown]
# ## Step 2: Run the data analysis workflow

# %%
result = run_task(task, config, cache_dir=Path("./cache"))

# %% [markdown]
# %% [markdown]
# :::{note}
# The execution outputs two expected informational notices:
#
# Identifier columns (`file_name`, `file_path`, `label_file`, `original_id`) are
# dropped because unique per-row values cannot serve as categorical factors.
#
# Continuous factors are binned automatically when no explicit cuts are provided.
# You can use [Metadata triage](metadata_triage) to establish explicit factor binning
# policies.
# :::

# %% tags=["remove_cell"]
if not result.success:
    print(f"Workflow failed: {result.errors}")
assert result.success

# %% [markdown]
# ## Step 3: View the analysis report
#
# You can call `result.report()` to display findings and severities across all
# assessment areas:
#
# - `[ok]`: Finding is within configured health thresholds.
# - `[!!]`: Finding exceeds configured health thresholds and requires review.
#
# The report covers five assessment areas:
#
# | Area | What it checks |
# |---|---|
# | Image Quality | Outlier images (unusual dimensions, brightness, entropy) |
# | Redundancy | Exact and near-duplicate images within each split |
# | Label Health | Class distribution, imbalance ratio, empty images |
# | Bias | Metadata factor correlations (Balance MI, Diversity) |
# | Cross-split | Label overlap, label parity, duplicate leakage, distribution shift |

# %%
print(result.report())

# %% [markdown]
# ### What this run found
#
# You should review the three flagged findings:
#
# - **Image quality**: Flags 5.7%, 7.6%, and 7.9% across the splits against a 5%
#   threshold. The primary flags are `zeros` and `aspect_ratio`, reflecting cropping
#   swaths standard in overhead sensor captures.
# - **Bias**: Shows class identity correlates with image statistics (`unit_mean` and
#   `unit_skew` with mutual information ~0.87). This indicates that sensor environments
#   correlate with target classes across collection sites.
# - **Label parity**: Detects proportional class shifts between validation and other splits.

# %% [markdown]
# ### Understanding health thresholds
#
# You can configure health thresholds via `DataAnalysisHealthThresholds`:
#
# | Threshold | Default | When to adjust |
# |---|---|---|
# | `image_outliers` | 3% | Lower to 1% for safety-critical data; raise to 5-10% for diverse collections |
# | `exact_duplicates` | 0% | Raise above 0 only if your pipeline intentionally repeats images |
# | `near_duplicates` | 5% | Lower to 1-2% for curated benchmarks; raise to 10-15% for web-scraped data |
# | `class_label_imbalance` | 5:1 | Lower to 3:1 for binary; raise to 10-20:1 for large hierarchies |
# | `distribution_shift` | 0.5 | Lower for stricter cross-split consistency requirements |
#
# To apply stricter thresholds, specify custom limits:
#
# ```python
# from dataeval_flow.workflows.analysis.params import DataAnalysisHealthThresholds
#
# strict = DataAnalysisHealthThresholds(
#     image_outliers=1.0,
#     exact_duplicates=0.0,
#     near_duplicates=2.0,
#     class_label_imbalance=3.0,
# )
# ```

# %% [markdown]
# ## Step 4: Explore cross-split comparisons
#
# When analyzing multiple splits, the report provides pairwise cross-split comparisons.
# You can inspect label overlap, class proportions, and distribution parity across splits.

# %%
import polars as pl

raw = result.data.raw

for pair_name, comparison in raw.cross_split.items():
    overlap = comparison.label_health.label_overlap

    # Check for split-exclusive classes
    split_only = {k: v for k, v in overlap.items() if k.endswith("_only") and v}
    if split_only:
        print(f"--- {pair_name}: MISSING CLASSES ---")
        for key, val in split_only.items():
            print(f"  {key}: {val}")
    else:
        shared = overlap.get("shared_classes", [])
        print(f"--- {pair_name}: all {len(shared)} classes present in both splits ---")

    # Proportion comparison table
    prop = overlap.get("proportion_comparison", {})
    if prop:
        prop_rows = []
        first = next(iter(prop.values()))
        pair_splits = [k for k in first if k != "difference"]
        for cls_name, vals in prop.items():
            row = {"Class": cls_name}
            for s in pair_splits:
                row[f"{s} (%)"] = round(vals[s] * 100, 1)
            row["Diff (pp)"] = round(vals["difference"] * 100, 1)
            prop_rows.append(row)
        df = pl.DataFrame(prop_rows).sort("Diff (pp)", descending=True)
        large_diffs = [c for c, v in prop.items() if abs(v["difference"]) > 0.05]
        if large_diffs:
            print(f"  {len(large_diffs)} class(es) differ by >5 percentage points between splits")
        print(df)
    print()

# %% [markdown]
# ### Label parity (chi-squared test)
#
# Is there a statistically significant difference between label distributions across splits?
# A significant result (p < 0.05) suggests the splits were not drawn from the same label
# distribution, indicating potential sampling bias.

# %%
for pair_name, comparison in raw.cross_split.items():
    lp = comparison.label_health.label_parity
    if lp:
        if lp["significant"]:
            print(
                f"{pair_name}: SIGNIFICANT difference (chi2={lp['chi_squared']:.2f}, "
                f"p={lp['p_value']:.4g}) -- splits may not share the same label distribution"
            )
        else:
            print(f"{pair_name}: no significant difference (chi2={lp['chi_squared']:.2f}, p={lp['p_value']:.4g})")
    else:
        print(f"{pair_name}: label parity not computed")

# %% [markdown]
# ### Significant is not the same as meaningful
#
# Large sample sizes often produce statistically significant p-values for minor
# distribution shifts. You should evaluate both p-values and absolute percentage
# point differences to assess practical impact:
#
# - **train vs test**: All classes remain within 3.3 percentage points (p = 0.008).
#   The distributions align closely for operational evaluation.
# - **train vs val** and **val vs test**: The `boat` class accounts for 33% of annotations
#   in `val` compared to 16% in `train` and `test` (p < 1e-60). This significant shift indicates
#   that models tuned on `val` face a different class balance than `test`.
#
# You can repartition splits using [Split a dataset](dataset_splitting) to resolve validation
# distribution divergence.

# %% [markdown]
# ### Per-split duplicates
#
# You can inspect duplicate groups within each split via `split_data.redundancy`.
#
# SkySeaLand contains no exact or near-duplicate frames in this sample. When duplicates
# exist, you can use these index groups for targeted visual inspection.

# %%
for split_name, split_data in raw.splits.items():
    rd = split_data.redundancy
    print(f"{split_name}: {len(rd.exact_groups)} exact, {len(rd.near_groups)} near duplicate group(s)")
    for i, group in enumerate(rd.exact_groups):
        print(f"  exact group {i + 1}: {[f'{split_name}[{idx}]' for idx in group]}")
    for i, group in enumerate(rd.near_groups):
        print(f"  near group {i + 1}: {[f'{split_name}[{idx}]' for idx in group]}")

# %% [markdown]
# ### Cross-split leakage
#
# You should verify that images from training sets do not leak into evaluation splits.
# Data leakage between train and test artifically inflates evaluation performance.
#
# SkySeaLand has disjoint splits with zero cross-split duplicates. If duplicates are
# detected in your datasets, you can render matched pairs side by side to verify leakage.

# %%
import matplotlib.pyplot as plt
import numpy as np

assert result.sources is not None

for pair_name, comparison in raw.cross_split.items():
    leakage = comparison.redundancy.duplicate_leakage
    exact_count = leakage.get("exact_count", 0)
    near_count = leakage.get("near_count", 0)
    if exact_count == 0 and near_count == 0:
        print(f"{pair_name}: No cross-split duplicates -- split integrity preserved")
        continue

    print(f"{pair_name}: DATA LEAKAGE DETECTED -- {exact_count} exact, {near_count} near duplicates")

    # Render exact duplicate groups: images from both splits side by side
    for i, group in enumerate(leakage.get("exact_groups", [])):
        all_images = []
        all_labels = []
        for split_name, indices in group.items():
            for idx in indices:
                img = np.array(result.sources[split_name][idx][0])
                if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                    img = img.transpose(1, 2, 0)
                all_images.append(img)
                all_labels.append(f"{split_name}[{idx}]")
        if all_images:
            n = len(all_images)
            fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
            if n == 1:
                axes = [axes]
            for ax, img, label in zip(axes, all_images, all_labels, strict=True):
                ax.imshow(img)
                ax.set_title(label, fontsize=10)
                ax.axis("off")
            fig.suptitle(f"Cross-split exact duplicate group {i + 1}", fontsize=12, fontweight="bold")
            fig.tight_layout()

    # Print near duplicate leakage groups
    for i, group in enumerate(leakage.get("near_groups", [])):
        labels = []
        for split_name, indices in group.items():
            labels.extend(f"{split_name}[{idx}]" for idx in indices)
        if labels:
            print(f"  Near duplicate group {i + 1}: {labels}")

# %% [markdown]
# ## Step 5: Export results
#
# You can export results to JSON format for CI/CD integration and archiving.

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")
print(json_str[:500] + "\n...")

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Configure the `data-analysis` workflow across multiple dataset splits.
# - Set health thresholds to govern finding severities.
# - Execute multi-split analysis using `run_task()`.
# - Read the analysis report covering image quality, redundancy, label health, bias, and cross-split metrics.
# - Inspect cross-split label overlap, proportion differences, parity tests, and duplicate leakage.
# - Distinguish statistical significance from practical divergence in distribution parity.
# - Export evaluation results to JSON.

# %% [markdown]
# ## Next steps
#
# - **Dataset splitting**: Use [Split a dataset](dataset_splitting) to generate balanced,
#   stratified partitions when published splits diverge.
# - **Data cleaning**: Use [Clean a dataset](data_cleaning) to detect and remove flagged
#   outliers and duplicates.
# - **ONNX embeddings**: Configure an ONNX model to enable embedding-based distribution shift analysis.

# %% [markdown]
# ## Related guides
#
# - **Concept**: [Data quality and cleaning](../concepts/DataQualityAndCleaning.md) explains
#   assessment areas and cross-split checks.
# - **How-to**: [Read evaluation outputs](../how_to/read_evaluation_outputs.md) explains
#   result structure and finding severity levels.
# - **How-to**: [Narrow a dataset with views](../how_to/build_dataset_views.md) explains
#   dataset sampling and filtering operations.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md) explains
#   how to execute workflows in Docker.
# - **Guide**: [Use an ONNX model for embeddings](onnx_embeddings) shows how to configure
#   pretrained extractors for cross-split shift analysis.
# - **Tutorial**: [Triage a dataset's metadata](metadata_triage) explains how to define
#   explicit metadata binning policies.
