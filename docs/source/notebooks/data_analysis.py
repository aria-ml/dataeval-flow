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
# # Analyze dataset quality across splits
#
# Produce a comprehensive quality report across dataset splits using the config-driven
# `data-analysis` workflow.

# %% [markdown]
# ## What you'll do
#
# - Download the CPPE-5 dataset from HuggingFace and prepare it as a multi-split dataset
# - Build a workflow configuration for multi-split analysis
# - Run the `data-analysis` workflow via `run_task()`
# - View the built-in **analysis report** for a high-level summary of all assessment areas
# - Explore cross-split comparisons — label overlap, duplicate leakage, and distribution parity
# - Configure **health thresholds** to control when findings trigger warnings
# - Export results to JSON for downstream tooling

# %% [markdown]
# ## What you'll learn
#
# - How to configure and run the `data-analysis` workflow via `run_task()`
# - How to read the built-in **analysis report** (`result.report()`) for a quick summary
# - What the five assessment areas cover: image quality, redundancy, label health, bias, and
#   cross-split comparisons
# - How to configure **health thresholds** to control warning severity
# - The difference between **advisory** mode (report only) and **preparatory** mode

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datasets`, `maite-datasets`, `pydantic`)
# - Internet connection (to download CPPE-5 from HuggingFace Hub on first run)

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load and prepare the dataset
#
# Download [CPPE-5](https://huggingface.co/datasets/cppe-5) from HuggingFace and save it
# to disk. CPPE-5 is a small (~1.5K images) object detection dataset for medical personal
# protective equipment with 5 classes (Coverall, Face Shield, Gloves, Goggles, Mask). It
# ships with `train` and `test` splits, making it ideal for demonstrating cross-split analysis.

# %% tags=["remove_output"]
from datasets import load_dataset as hf_load

# Download CPPE-5 from HuggingFace (train + test splits)
cppe5 = hf_load("cppe-5")
print(f"Splits: {list(cppe5.keys())}")
for name, ds in cppe5.items():
    print(f"  {name}: {len(ds)} images")

# %% tags=["remove_output"]
from pathlib import Path

# Save to disk in HuggingFace arrow format
data_path = Path("./data/cppe5")
cppe5.save_to_disk(str(data_path))
print(f"Saved to {data_path}")

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# The `data-analysis` workflow requires explicit parameters (no hidden defaults). We'll
# configure outlier detection using **z-score** thresholding across dimension, pixel, and
# visual statistics, and enable **balance** and **diversity** analysis to surface metadata
# bias signals.
#
# Health thresholds control when findings are elevated to warnings. We relax several
# thresholds because CPPE-5 is a diverse object-detection dataset where moderate outlier
# rates and class imbalance are expected.

# %%
from dataeval.config import set_max_processes

from dataeval_flow.config import (
    DataAnalysisTaskConfig,
    DataAnalysisWorkflowConfig,
    HuggingFaceDatasetConfig,
    PipelineConfig,
    SelectionConfig,
    SelectionStep,
    SourceConfig,
)
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.analysis.params import DataAnalysisHealthThresholds

set_max_processes(8)

analysis_workflow = DataAnalysisWorkflowConfig(
    name="cppe5_analysis",
    outlier_method="adaptive",
    outlier_flags=["dimension", "pixel", "visual"],
    outlier_threshold=4.0,
    balance=True,
    diversity_method="simpson",
    include_image_stats=True,
    health_thresholds=DataAnalysisHealthThresholds(
        image_outliers=5.0,  # Relaxed from 3% — CPPE-5 has diverse images
        exact_duplicates=0.0,  # No exact duplicates allowed (default)
        near_duplicates=5.0,  # Up to 5% near duplicates before warning (default)
        class_label_imbalance=5.0,  # CPPE-5 has moderate imbalance (default)
        distribution_shift=0.5,  # Default
    ),
)

task = DataAnalysisTaskConfig(
    name="cppe5-quality-check",
    workflow="cppe5_analysis",
    sources=["cppe5_trn_src", "cppe5_val_src", "cppe5_tst_src"],
)

config = PipelineConfig(
    datasets=[
        HuggingFaceDatasetConfig(name="cppe5_train", path=str(data_path), split="train"),
        HuggingFaceDatasetConfig(name="cppe5_test", path=str(data_path), split="test"),
    ],
    selections=[
        SelectionConfig(name="trn-500", steps=[SelectionStep(type="Limit", params={"size": 500})]),
        SelectionConfig(
            name="val-50", steps=[SelectionStep(type="Indices", params={"indices": {"start": 500, "stop": 550}})]
        ),
    ],
    sources=[
        SourceConfig(name="cppe5_trn_src", dataset="cppe5_train", selection="trn-500"),
        SourceConfig(name="cppe5_val_src", dataset="cppe5_train", selection="val-50"),
        SourceConfig(name="cppe5_tst_src", dataset="cppe5_test"),
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

# %% tags=["remove_cell"]
if not result.success:
    print(f"Workflow failed: {result.errors}")
assert result.success

# %% [markdown]
# ## Step 3: View the analysis report
#
# The workflow result has a built-in `report()` method that renders a formatted text
# summary. Each assessment area produces one or more **findings** — a concise summary
# with a severity level:
#
# - `[ok]` — within the configured health threshold (no action needed)
# - `[!!]` — exceeds the threshold (review recommended)
#
# The report covers all five assessment areas per split, plus cross-split comparisons
# when multiple splits are present:
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
# ### Understanding health thresholds
#
# Health thresholds are configured via `DataAnalysisHealthThresholds` on the
# `health_thresholds` parameter. The defaults are:
#
# | Threshold | Default | When to adjust |
# |---|---|---|
# | `image_outliers` | 3% | Lower to 1% for safety-critical data; raise to 5-10% for diverse collections |
# | `exact_duplicates` | 0% | Raise above 0 only if your pipeline intentionally repeats images |
# | `near_duplicates` | 5% | Lower to 1-2% for curated benchmarks; raise to 10-15% for web-scraped data |
# | `class_label_imbalance` | 5:1 | Lower to 3:1 for binary; raise to 10-20:1 for large hierarchies |
# | `distribution_shift` | 0.5 | Lower for stricter cross-split consistency requirements |
#
# To tighten thresholds for a stricter audit:
#
# ```python
# from dataeval_flow.workflows.analysis.params import DataAnalysisHealthThresholds
#
# strict = DataAnalysisHealthThresholds(
#     image_outliers=1.0,        # flag at 1% for safety-critical data
#     exact_duplicates=0.0,      # no exact duplicates (default)
#     near_duplicates=2.0,       # stricter near-duplicate limit
#     class_label_imbalance=3.0, # tight balance for binary classification
# )
# ```

# %% [markdown]
# ## Step 4: Explore cross-split comparisons
#
# When analyzing multiple splits, the report includes pairwise cross-split findings.
# Let's look at the raw cross-split data for the most interesting comparisons —
# label overlap and proportion differences between train and test.

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
#
# > NOTE: Due to the size disparity in CPPE-5's very small test split, the chi-squared test
# > may produce inaccurate results.

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
# ### Per-split duplicates
#
# Each split's `RedundancyResult` now exposes the actual duplicate group indices.
# We plot exact duplicate groups so you can visually confirm they are true duplicates.
# Near duplicate groups are printed for reference but not rendered (the hash-based
# detector can be noisy).

# %%
import matplotlib.pyplot as plt
import numpy as np

assert result.sources is not None


def plot_duplicate_group(dataset: object, indices: list[int], source_name: str, group_label: str) -> plt.Figure:
    """Plot a single duplicate group with source[idx] labels."""
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices, strict=True):
        img = np.array(dataset[idx][0])  # type: ignore[index]
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = img.transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f"{source_name}[{idx}]", fontsize=10)
        ax.axis("off")
    fig.suptitle(group_label, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


for split_name, split_data in raw.splits.items():
    rd = split_data.redundancy
    if rd.exact_groups:
        print(f"\n{split_name}: {len(rd.exact_groups)} exact duplicate group(s)")
        for i, group in enumerate(rd.exact_groups):
            plot_duplicate_group(result.sources[split_name], group, split_name, f"Exact duplicate group {i + 1}")
    if rd.near_groups:
        print(f"\n{split_name}: {len(rd.near_groups)} near duplicate group(s)")
        for i, group in enumerate(rd.near_groups):
            print(f"  Group {i + 1}: {[f'{split_name}[{idx}]' for idx in group]}")

# %% [markdown]
# ### Cross-split leakage
#
# Does the same image (or a near-duplicate) appear in multiple splits? Data leakage between
# train and test silently inflates evaluation metrics.
#
# When exact duplicates are found we render them side by side so you can visually confirm
# the leakage. Near duplicate leakage is printed for reference.

# %%
for pair_name, comparison in raw.cross_split.items():
    leakage = comparison.redundancy.duplicate_leakage
    exact_count = leakage.get("exact_count", 0)
    near_count = leakage.get("near_count", 0)
    if exact_count == 0 and near_count == 0:
        print(f"{pair_name}: No cross-split duplicates -- train/test integrity preserved")
        continue

    print(f"{pair_name}: DATA LEAKAGE DETECTED -- {exact_count} exact, {near_count} near duplicates")

    # Render exact duplicate groups — images from both splits side by side
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

    # Print near duplicate leakage groups (skip rendering)
    for i, group in enumerate(leakage.get("near_groups", [])):
        labels = []
        for split_name, indices in group.items():
            labels.extend(f"{split_name}[{idx}]" for idx in indices)
        if labels:
            print(f"  Near duplicate group {i + 1}: {labels}")

# %% [markdown]
# ## Step 5: Export results
#
# Export the full result to JSON for integration with automated pipelines or archival.

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")
print(json_str[:500] + "\n...")

# %% [markdown]
# ## Conclusion
#
# In this tutorial you learned how to:
#
# - **Configure** the `data-analysis` workflow with explicit outlier, bias, and divergence parameters
# - **Set health thresholds** to control when findings are elevated to warnings
# - **Run** the workflow via `run_task()` on a multi-split dataset (CPPE-5 train + test)
# - **Read the analysis report** -- a single `result.report()` call for a formatted summary
#   covering image quality, redundancy, label health, bias, and cross-split comparisons
# - **Explore cross-split data** -- label overlap, proportion differences, parity testing,
#   and leakage detection
# - **Export** results to JSON for integration with automated pipelines

# %% [markdown]
# ## What's next
#
# - **Data cleaning** -- Use the `data-cleaning` workflow for actionable outlier and duplicate
#   detection with visual inspection via `dataeval-plots`
# - **Custom extractors** -- Add an ONNX model configuration to enable embedding-based
#   cross-split divergence analysis (distribution shift)
# - **Container deployment** -- Mount your dataset and config YAML, then run `dataeval-flow`
#   as a container with the same configuration
