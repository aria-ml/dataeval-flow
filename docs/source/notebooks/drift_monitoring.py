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
# # Monitor incoming data for drift
#
# Detect distribution drift between a reference dataset and incoming data using
# the config-driven `drift-monitoring` workflow.

# %% [markdown]
# ## What you'll do
#
# - Download MNIST from HuggingFace and prepare a **reference** set of 2 000 clean digit images
# - Synthesize **incoming data** where the second half is blurred — simulating drift that starts partway through
# - Write both datasets to disk as flat image folders (preserving temporal order)
# - Configure the `drift-monitoring` workflow with **K-Neighbors**, **MMD**, and **Univariate (CVM)** detectors
# - Enable **per-detector chunking** — chunked for K-Neighbors/MMD, non-chunked for Univariate
# - Review the drift report and chunk-level breakdown

# %% [markdown]
# ## What you'll learn
#
# - How to configure and run the `drift-monitoring` workflow via `run_tasks()`
# - How **per-detector chunking** lets you mix chunked and non-chunked detectors in one run
# - How to read the built-in drift report with per-detector and per-chunk results
# - The difference between **K-Neighbors** (distance-based), **MMD** (distribution-wide),
#   and **Univariate CVM** (per-feature) detectors

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datasets`, `maite-datasets`, `pydantic`)
# - Internet connection (to download MNIST from HuggingFace Hub on first run)

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Download MNIST and prepare image folders
#
# We'll download the MNIST test split (10 000 images) from HuggingFace and use
# 2 000 as our **reference** dataset — clean, unmodified digit images.
#
# For the **incoming** dataset we'll take a fresh sample of 1 000 images and apply
# a Gaussian blur to the **second half** (images 500–999). This simulates a
# real-world scenario where data quality degrades partway through an ingestion
# window — exactly the kind of temporal drift that chunked analysis can detect.
#
# :::{important}
# Both datasets are saved as **flat** image folders (no class subdirectories) with
# sequentially numbered filenames. This preserves the temporal ordering of the
# incoming data, which is essential for chunked drift analysis to correctly
# identify *when* drift started.
# :::

# %% tags=["remove_output"]
from typing import cast

from datasets import Dataset
from datasets import load_dataset as hf_load

# Download MNIST test split (10 000 images of 28×28 handwritten digits)
mnist_test = cast(Dataset, hf_load("ylecun/mnist", split="test"))

# %% [markdown]
# ### Write the reference set to disk
#
# We'll save 2 000 clean images as our reference baseline. A larger reference
# gives the statistical tests more power. Images are saved as flat numbered
# files — drift detectors compare embedding distributions, so class labels
# aren't needed here.

# %% tags=["remove_output"]
from pathlib import Path

from PIL import Image

data_dir = Path("./data/drift_tutorial")

ref_dir = data_dir / "reference"
ref_dir.mkdir(parents=True, exist_ok=True)

# Use the first 2000 images as reference — flat directory, no class subdirs
for i in range(2000):
    sample = mnist_test[i]
    img: Image.Image = sample["image"]  # type: ignore[index]
    img.save(ref_dir / f"{i:05d}.png")

print(f"Reference: 2000 images saved to {ref_dir}")

# %% [markdown]
# ### Write the incoming set — clean first half, blurred second half
#
# The incoming dataset has 1 000 images (indices 2000–2999 from MNIST). The first
# 500 are saved as-is; the last 500 get a Gaussian blur applied. This creates a
# clear temporal boundary at the midpoint that our chunked detectors should find.
#
# We use a gentle blur (`radius=1.5`) — enough to visibly soften the digits
# and shift the pixel distribution, while keeping them recognizable.

# %% tags=["remove_output"]
from PIL import ImageFilter

incoming_dir = data_dir / "incoming"
incoming_dir.mkdir(parents=True, exist_ok=True)

n_incoming = 1000
n_clean = 500  # first half: clean
blur_radius = 1.5  # gentle blur — digits become perceptibly softer

for j in range(n_incoming):
    idx = 2000 + j  # use different images than reference
    sample = mnist_test[idx]
    img = sample["image"]

    # Apply blur to the second half (indices 500–999)
    if j >= n_clean:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Flat directory with sequential names — preserves temporal order
    img.save(incoming_dir / f"{j:05d}.png")

print(f"Incoming: {n_incoming} images saved to {incoming_dir}")
print(f"  Clean (first {n_clean}): unmodified digits")
print(f"  Blurred (last {n_incoming - n_clean}): Gaussian blur radius={blur_radius}")

# %% [markdown]
# Let's visualize a few images from each group to see the difference:

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for row in range(2):
    for col in range(5):
        sample = mnist_test[(2500 if row else 0) + col]
        img = sample["image"].filter(ImageFilter.GaussianBlur(radius=blur_radius)) if row else sample["image"]
        axes[row, col].imshow(img, cmap="gray")  # type: ignore[index]
        axes[row, col].set_title(f"Digit {sample['label']}")  # type: ignore[index]
        axes[row, col].axis("off")

fig.suptitle("Reference (top) vs Blurred incoming (bottom)", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# The `drift-monitoring` workflow needs:
#
# 1. **Two datasets** — the first is the reference, the rest are test (incoming) data
# 2. **An extractor** — to compute embeddings that detectors compare
# 3. **Detector configuration** — which statistical tests to run
# 4. **Chunking** — to split incoming data into temporal windows
#
# We'll use the **Flatten** extractor which reshapes each image's pixel data into
# a 1-D feature vector — simple, fast, and well suited for small grayscale images
# like MNIST (28×28 = 784 features). For detectors we configure both **K-Neighbors**
# (distance-based) and **MMD** (distribution-wide kernel test).

# %%
from dataeval_flow.config import FlattenExtractorConfig, ImageFolderDatasetConfig, PipelineConfig, SourceConfig

# --- Datasets ---
# First dataset = reference, second = incoming test data
# Flat image folders (no class subdirs) — preserves temporal ordering
ref_dataset = ImageFolderDatasetConfig(
    name="reference",
    path=str(ref_dir),
)

incoming_dataset = ImageFolderDatasetConfig(
    name="incoming",
    path=str(incoming_dir),
)

# --- Extractor ---
# Flatten reshapes each 28×28 grayscale image into a 784-D feature vector — no model file needed
extractor_config = FlattenExtractorConfig(name="flatten", batch_size=64)

# %% [markdown]
# ### Configure drift detectors and chunking
#
# We'll set up three complementary detectors:
#
# | Detector | What it tests | Chunked? | Strengths |
# |---|---|---|---|
# | **kneighbors** | Test points farther from reference neighbors | Yes | Robust in high dimensions, non-parametric |
# | **mmd** | Overall distribution distance via kernel trick | Yes | Sensitive to multivariate shifts |
# | **univariate (CVM)** | Per-feature CDF distance | No | Per-feature breakdown, high power |
#
# **K-Neighbors** drift detection checks whether incoming samples are farther
# from their k-nearest reference neighbors than expected. This works well with
# high-dimensional raw pixel vectors because it operates on pairwise distances
# rather than per-feature statistics — avoiding the multiple-testing burden that
# makes univariate tests noisy on 784 features.
#
# **CVM (Cramér-von Mises)** is a univariate test that measures the integrated
# squared distance between empirical CDFs for each feature independently.  It
# has higher statistical power than the default Kolmogorov-Smirnov test for
# detecting subtle distributional shifts.  We run it **without chunking** so
# the report shows the contrast between a single overall verdict and the
# temporal chunk breakdown from the other two detectors.
#
# **Chunking** is configured **per detector**.  Since the first 500 images are
# clean and the last 500 are blurred, we expect chunks covering the second half
# to show drift while the first half remains clean.

# %%
from dataeval_flow.config import DriftMonitoringTaskConfig, DriftMonitoringWorkflowConfig
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.drift.params import (
    ChunkingConfig,
    DriftDetectorKNeighbors,
    DriftDetectorMMD,
    DriftDetectorUnivariate,
    DriftHealthThresholds,
)

# --- Drift workflow ---
# Chunking is configured per detector — K-Neighbors and MMD get chunked,
# while the univariate CVM detector runs a single overall test.
drift_overall = DriftMonitoringTaskConfig(
    name="mnist-drift-overall",
    workflow="mnist-drift",
    sources=["ref_src", "inc_src"],
    extractor="flatten",
)

config = PipelineConfig(
    datasets=[ref_dataset, incoming_dataset],
    sources=[
        SourceConfig(name="ref_src", dataset="reference"),
        SourceConfig(name="inc_src", dataset="incoming"),
    ],
    extractors=[extractor_config],
    workflows=[
        DriftMonitoringWorkflowConfig(
            name="mnist-drift",
            detectors=[
                DriftDetectorKNeighbors(k=10, chunking=ChunkingConfig(chunk_size=200, threshold_multiplier=4.0)),
                DriftDetectorMMD(n_permutations=100, chunking=ChunkingConfig(chunk_size=200, threshold_multiplier=4.0)),
                DriftDetectorUnivariate(test="cvm"),  # non-chunked overall test
            ],
            health_thresholds=DriftHealthThresholds(
                chunk_drift_pct_warning=15.0,  # warn if >15% of chunks drift
                consecutive_chunks_warning=2,  # warn on 2+ consecutive drifted chunks
            ),
        ),
    ],
    tasks=[drift_overall],
)

# %% [markdown]
# ## Step 2: Run the drift monitoring workflow

# %%
result = run_task(drift_overall, config, cache_dir=Path("./cache"))

# %% [markdown]
# ## Results Exploration: Drift report
#
# The workflow produces a text report summarizing each detector's findings. With
# chunking enabled, you'll see a per-chunk breakdown showing exactly which windows
# drifted.

# %%
print(result.report())

# %% [markdown]
# ### Understanding the chunk results
#
# With 1 000 incoming images and `chunk_size=200`, we get 5 chunks:
#
# | Chunk | Image range | Expected |
# |---|---|---|
# | `[0:200]` | 0–199 | **Clean** — all unmodified digits |
# | `[200:400]` | 200–399 | **Clean** — still unmodified |
# | `[400:600]` | 400–599 | **Mixed** — transition zone (half clean, half blurred) |
# | `[600:800]` | 600–799 | **Blurred** — full drift |
# | `[800:1000]` | 800–999 | **Blurred** — full drift |
#
# The detectors should flag chunks 3–5 (or at least 4–5) as drifted, confirming
# that drift started around the midpoint of the incoming data.

# %% [markdown]
# ### Inspect chunk-level details programmatically
#
# The raw output gives you structured access to per-detector, per-chunk results
# for custom analysis or visualization.

# %%
import polars as pl

pl.Config.set_tbl_hide_dataframe_shape(True)

raw = result.data.raw
print(f"Reference size: {raw.reference_size}")
print(f"Test size:      {raw.test_size}")
print()

for method, det_result in raw.detectors.items():
    print(f"── {method} ({det_result['metric_name']}) ──")
    print(f"  Overall drifted: {det_result['drifted']}")
    print(f"  Distance:        {det_result['distance']:.6g}")

    chunks = det_result.get("chunks", [])
    if chunks:
        df = pl.DataFrame(chunks).select("key", "value", "lower_threshold", "upper_threshold", "drifted")
        print(df)
    print()

# %% [markdown]
# ### Visualize chunk drift over time
#
# A simple bar chart makes the temporal drift pattern immediately visible.

# %%
# Only plot detectors that have chunk results
chunked_methods = [m for m, r in raw.detectors.items() if r.get("chunks")]
fig, axes = plt.subplots(1, len(chunked_methods), figsize=(6 * len(chunked_methods), 4))
if len(chunked_methods) == 1:
    axes = [axes]  # type: ignore[list-item]

for ax, method in zip(axes, chunked_methods, strict=True):
    chunks = raw.detectors[method]["chunks"]  # type: ignore[typeddict-item]

    labels = [c["key"] for c in chunks]
    values = [c["value"] for c in chunks]
    colors = ["#e74c3c" if c["drifted"] else "#2ecc71" for c in chunks]

    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)

    # Draw threshold lines and expand y-axis to include them
    upper_thresh = chunks[0].get("upper_threshold")
    lower_thresh = chunks[0].get("lower_threshold")
    all_y = list(values)
    if upper_thresh is not None:
        ax.axhline(
            y=upper_thresh,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"upper={upper_thresh:.4g}",
        )
        all_y.append(upper_thresh)
    if lower_thresh is not None:
        ax.axhline(
            y=lower_thresh,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=f"lower={lower_thresh:.4g}",
        )
        all_y.append(lower_thresh)
    y_min, y_max = min(all_y), max(all_y)
    margin = (y_max - y_min) * 0.15 or abs(y_max) * 0.1 or 0.01
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.legend(fontsize=8)
    ax.set_title(method, fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance")
    ax.set_xlabel("Chunk")
    ax.tick_params(axis="x", rotation=30)

fig.suptitle("Chunk-level drift — green = ok, red = drift detected", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Results Exploration: Export results
#
# The JSON output contains all raw detector results, chunk data, and metadata —
# ready for integration with monitoring dashboards or automated pipelines.

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")
print(json_str[:600] + "\n...")

# %% [markdown]
# ## Conclusion
#
# In this tutorial you learned how to:
#
# - **Prepare** reference and incoming datasets as flat image folders
# - **Simulate temporal drift** by applying Gaussian blur to the second half of incoming data
# - **Configure** the `drift-monitoring` workflow with multiple detectors (K-Neighbors, MMD, Univariate CVM)
# - **Use per-detector chunking** — chunked analysis for some detectors, non-chunked for others
# - **Read the drift report** — per-detector summaries and per-chunk breakdowns
# - **Visualize** chunk-level results to see the temporal drift pattern
# - **Export** structured JSON results for downstream automation

# %% [markdown]
# ## What's next
#
# - **Classwise drift** — Add `classwise: true` and use labeled image folders
#   (`infer_labels: true`) to detect which digit classes are most affected
# - **Different detectors** — Try `domain_classifier` (trains a binary classifier to
#   distinguish ref from test) or other univariate tests (`ks`, `mwu`, `anderson`, `bws`)
# - **Real-world models** — Use an ONNX model (e.g. ResNet) with preprocessing for
#   richer embeddings that capture higher-level features
# - **Production deployment** — See the Docker deployment guide for running drift
#   monitoring on a schedule against live data pipelines
