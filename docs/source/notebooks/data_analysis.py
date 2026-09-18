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
# **Who this is for** — T&E engineers and data scientists who need a holistic,
# multi-split quality report before signing off on a dataset for evaluation.
#
# **Where this fits** — Dataset analysis is a gate in the T&E workflow: before you
# split data or train a model, you confirm there is no cross-split leakage, no
# excessive imbalance, and no distribution shift between train and test. It builds
# on per-dataset [data cleaning](data_cleaning) and feeds [dataset splitting](dataset_splitting).
# See the [Data quality and cleaning](../concepts/DataQualityAndCleaning.md) concept
# page for the underlying assessment areas.

# %% [markdown]
# ## What you'll do
#
# - Download SkySeaLand with `maite-datasets` and export the three splits it ships with
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
# - Why a significant parity test and a meaningful one are not the same question

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`)
# - `maite-datasets[datamaite]` (to download SkySeaLand and export it for `dataeval-flow`)
# - Internet connection (a ~262 MB download on first run)

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the splits the dataset ships with
#
# [SkySeaLand](https://www.kaggle.com/datasets/mdzahidhasanriad/skysealand) is an overhead
# imagery detection dataset: 1,307 frames collected at four sites around the world and
# annotated with 19,102 objects across `airplane`, `boat`, `car` and `ship`. It arrives
# already partitioned by its publisher into `train` (1,048 frames), `val` (132) and `test`
# (127), which is what makes it the right dataset for this workflow. The question here is
# not "is this data any good" — that is [data cleaning](data_cleaning) — but "do these three
# partitions agree with each other", and that question only exists once somebody has drawn
# the lines.
#
# `maite-datasets` downloads the dataset, and `as_datamaite=True` writes it back out in a
# format `dataeval-flow` reads directly. Each export is named after both the dataset and the
# `image_set`, so the three splits land in three folders instead of overwriting one another.

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
# The `data-analysis` workflow requires explicit parameters (no hidden defaults). We
# configure outlier detection using **adaptive** thresholding across dimension, pixel and
# visual statistics, and enable **balance** and **diversity** analysis to surface metadata
# bias signals.
#
# Three choices below are worth explaining.
#
# **`train` is sampled; `val` and `test` are read whole.** Profiling all 1,048 train frames
# decodes about 15 GB of pixels. 300 frames characterize the split well enough, and they
# leave the three sources close enough in size that the cross-split statistics compare like
# with like. Remove the view to profile the whole split.
#
# **The sample is shuffled before it is limited.** SkySeaLand is stored grouped by collection
# site, so a bare `Limit` would describe one site rather than the split: the first 300 frames
# put `ship` at 11% of annotations against 20% across the whole split. Shuffling first brings
# every class within about five points of its share of the full split.
#
# **A metadata policy declares the bias factors.** SkySeaLand ships no telemetry — no
# altitude, no sensor, no time of day — so the only factors available are the ones measured
# from the imagery, which `intrinsic_factors` asks for. `reference_split` matters as soon as
# more than one split is analyzed: splits binned independently land on different cutpoints
# for the same factor, and their per-factor statistics stop being comparable. Naming one
# split's encoding makes all three read the same cuts.

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

# Four workers rather than eight: peak memory is workers x batch x decoded image size, and
# these are ~1.3 MP frames.
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
        image_outliers=5.0,  # Relaxed from 3% — overhead frames vary widely in size and framing
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
            # Statistics measured from the imagery, injected as factors for bias analysis.
            intrinsic_factors=["visual", "pixel"],
            # Bookkeeping the export carries. `label_file_exists` is true for every frame,
            # so it separates nothing; the export's other bookkeeping columns hold a
            # different value in every row and are dropped before this policy is read.
            exclude=["label_file_exists"],
            # One encoding for all three splits, so their per-factor statistics compare.
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
# :::{note}
# The run prints two warnings, and each is the workflow reporting something true about this
# dataset rather than something wrong with the configuration.
#
# `file_name`, `file_path`, `label_file` and `original_id` are **dropped**: the export records
# where each frame came from, and a column holding a different value in every row identifies
# rows rather than grouping them, so bias analysis has nothing to read in it.
#
# The remaining continuous factors are **binned automatically**, because no cutpoints were
# declared for them. Bins derived this way come from the sample in front of them, so the same
# factor measured on a different sample may not land on the same cuts — which is fine for a
# first look and not fine for numbers you intend to compare across runs.
# [Metadata triage](metadata_triage) is the workflow that turns those derived bins into a
# declared policy you can keep.
# :::

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
# ### What this run found
#
# Three findings are flagged, and they say very different things.
#
# **Image quality** flags 5.7%, 7.6% and 7.9% of the three splits against a 5% threshold.
# That the three rates are close to each other is the useful part: this is a property of the
# collection, not of any one split. The dominant flags are `zeros` and `aspect_ratio` —
# overhead frames arrive cropped to the sensor swath, so black padding and extreme aspect
# ratios are normal here rather than corruption.
#
# **Bias** reports that class identity is strongly predicted by image statistics alone
# (`unit_mean` and `unit_skew` at MI ≈ 0.87). With four sites imaged under different
# conditions, a model can learn "frame looks like this → airplane" without learning what an
# airplane looks like. That is a shortcut risk worth testing for, not a defect in the data.
#
# **Label parity** is the finding to act on, and the next section unpacks it.

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
# label overlap and proportion differences between the splits.

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
# All three pairs come back "significantly different", which on its own says very little:
# with thousands of annotations per split, chi-squared will find a difference in almost any
# pair. The proportion table above is what separates them.
#
# - **train vs test** — every class within 3.3 percentage points, at `p = 0.008`. Significant,
#   and not worth acting on. For practical purposes these two splits describe the same world.
# - **train vs val** and **val vs test** — `boat` is 33% of the annotations in `val` against
#   16% in each of the other two, at `p = 6e-97` and `p = 3e-63`. That difference is large
#   enough to change decisions: a model tuned against this validation split is tuned against
#   a boat-heavy world it will not meet at test time.
#
# So read the p-value and the proportions together. The p-value tells you the difference is
# not chance; only the proportions tell you whether it matters. Here the publisher's `val`
# split is the odd one out, and a team that needs a validation set matching test conditions
# should redraw it — which is what [dataset splitting](dataset_splitting) is for.

# %% [markdown]
# ### Per-split duplicates
#
# Each split's `RedundancyResult` exposes the duplicate group indices, so you can get from a
# rate in the summary to the specific images behind it.
#
# SkySeaLand has none — no two frames in the collection share a hash — which is what the
# `[ok]` on the Redundancy line reports, and the expected outcome for a curated release. On
# a dataset that does have duplicates these lists are the input to visual inspection; the
# [data cleaning](data_cleaning) tutorial renders them with `dataeval-plots`.

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
# Does the same image, or a near-duplicate of it, appear in more than one split? Leakage
# between train and test silently inflates every evaluation metric computed afterwards. It is
# the one finding on this page that invalidates results rather than merely describing them.
#
# SkySeaLand's publisher drew disjoint splits, so nothing is found here and the cell below
# prints three clean lines. It stays executable rather than illustrative because this is the
# check worth pointing at your own data: when duplicates are found, the images are rendered
# side by side so you can confirm the match before acting on it.

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
# - **Run** the workflow via `run_task()` across the three splits SkySeaLand ships with
# - **Read the analysis report** -- a single `result.report()` call for a formatted summary
#   covering image quality, redundancy, label health, bias, and cross-split comparisons
# - **Explore cross-split data** -- label overlap, proportion differences, parity testing,
#   and leakage detection
# - **Separate a significant difference from a meaningful one** when reading the parity test
# - **Export** results to JSON for integration with automated pipelines

# %% [markdown]
# ## What's next
#
# - **Dataset splitting** -- Redraw the splits yourself when the publisher's partition does
#   not match your evaluation conditions, as `val` does not here
# - **Data cleaning** -- Use the `data-cleaning` workflow for actionable outlier and duplicate
#   detection with visual inspection via `dataeval-plots`
# - **Custom extractors** -- Add an ONNX model configuration to enable embedding-based
#   cross-split divergence analysis (distribution shift)

# %% [markdown]
# ## Related guides
#
# - **Concept** — [Data quality and cleaning](../concepts/DataQualityAndCleaning.md):
#   the five assessment areas and cross-split checks behind this report.
# - **How-to: Read evaluation outputs** — [Read evaluation outputs](../how_to/read_evaluation_outputs.md)
#   to interpret the report's severities and reach the per-split raw results behind each finding.
# - **How-to: Narrow a dataset with views** — [Narrow a dataset with views](../how_to/build_dataset_views.md)
#   to limit, filter, or sample each split before it is profiled.
# - **How-to: Run workflows in containers** — [Containerized workflows](../how_to/containerized_workflows.md)
#   to mount your dataset and config YAML, then run `dataeval-flow` as a container with the
#   same configuration.
# - **How-to: Use an ONNX model for embeddings** — [ONNX embeddings](onnx_embeddings)
#   to add a pretrained model and enable embedding-based cross-split divergence analysis.
# - **Tutorial: Triage a dataset's metadata** — [Metadata triage](metadata_triage)
#   to replace the automatically derived factor bins above with a policy you declare once
#   and reuse across runs.
