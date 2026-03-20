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
# # Detect out-of-distribution samples
#
# Find individual samples in incoming data that fall outside the reference
# distribution using the config-driven `ood-detection` workflow.

# %% [markdown]
# ## What you'll do
#
# - Download MNIST from HuggingFace and wrap it as an in-memory MAITE dataset
# - Synthesize **incoming data** that mixes normal digits with out-of-distribution
#   samples: **color-inverted** digits (via an in-memory transform)
# - Configure the `ood-detection` workflow with **K-Neighbors** and
#   **Domain Classifier** detectors
# - Enable **metadata insights** to explain *why* flagged samples are OOD
# - Review the OOD report and inspect per-sample results

# %% [markdown]
# ## What you'll learn
#
# - How to configure and run the `ood-detection` workflow via `run_task()`
# - The difference between **K-Neighbors** (distance-based) and
#   **Domain Classifier** (LightGBM-based) OOD detectors
# - How **metadata insights** (`factor_deviation`, `factor_predictors`) explain
#   which metadata factors correlate with OOD status
# - How to use `DatasetProtocolConfig` to pass in-memory datasets directly
# - How to read the built-in OOD report and drill into per-sample scores

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datasets`, `maite-datasets`, `pydantic`)
# - Internet connection (to download MNIST from HuggingFace Hub on first run)

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Download MNIST and build in-memory datasets
#
# We'll download the MNIST training split from HuggingFace and create two
# in-memory datasets — no files written to disk:
#
# - **Reference** — 2 000 normal MNIST digits (the "known good" distribution)
# - **Incoming** — 500 digits where **odd-digit classes (1, 3, 5, 7, 9) are
#   color-inverted** (pixel values flipped: white digit on black → black digit
#   on white), while even-digit classes remain normal
#
# Unlike drift detection — which compares distributions as a whole — OOD
# detection flags **individual samples** that don't belong. The inverted images
# look visually different from the reference despite sharing the same digit
# classes, so a good detector should flag them. Because the inversion is
# class-driven, metadata insights should identify `class_label` as a strong
# predictor of OOD status.
#
# :::{note}
# Color inversion flips pixel intensities end-to-end, producing a strong
# embedding shift that OOD detectors should reliably catch.
# :::

# %% tags=["remove_output"]
from typing import Any, cast

from datasets import Dataset
from datasets import load_dataset as hf_load
from maite_datasets.adapters import from_huggingface

# Download MNIST training split (60 000 images of 28x28 handwritten digits)
mnist_train = cast(Dataset, hf_load("ylecun/mnist", split="train"))

# Reference: first 2 000 images, wrapped as a MAITE dataset
ref_maite = from_huggingface(mnist_train.select(range(2000)))

print(f"Reference: {len(ref_maite)} images")
print(f"Sample shape: image={ref_maite[0][0].shape}, dtype={ref_maite[0][0].dtype}, label={ref_maite[0][1]}")

# %% [markdown]
# ### Build the incoming dataset with an in-memory inversion transform
#
# Instead of writing images to disk, we create a thin wrapper around the MAITE
# dataset that **color-inverts samples by class**. All odd-digit classes
# (1, 3, 5, 7, 9) are inverted while even digits remain unchanged. Since MNIST
# is grayscale with pixel values in `[0, 255]` inversion is simply `255 - pixel_value`.

# %%
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray


class InvertedTailDataset:
    """Wraps a MAITE dataset and color-inverts images by class label.

    Parameters
    ----------
    dataset
        A MAITE-compatible dataset returning (image, target, metadata) tuples.
    classes_to_invert
        Class labels to color-invert.
    """

    def __init__(self, dataset: Any, classes_to_invert: list[int] | None = None) -> None:
        self._dataset = dataset
        self._classes_to_invert = classes_to_invert or []
        self.metadata = {"id": f"inverted_tail_{len(self._classes_to_invert)}", "original_metadata": dataset.metadata}

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[NDArray[Any], Any, Mapping[str, Any]]:
        image, target, metadata = self._dataset[index]
        if np.argmax(target) in self._classes_to_invert:
            image = 255 - np.asarray(image)
        return image, target, metadata


# Incoming: next 500 images (indices 2000–2500), odd-digit classes get inverted
incoming_hf = mnist_train.select(range(2000, 2500))
incoming_maite = from_huggingface(incoming_hf)
incoming_dataset = InvertedTailDataset(incoming_maite, classes_to_invert=[1, 3, 5, 7, 9])

n_inverted = sum(1 for i in range(len(incoming_dataset)) if np.argmax(incoming_dataset[i][1]) in [1, 3, 5, 7, 9])
print(f"Incoming: {len(incoming_dataset)} images")
print(f"  In-distribution (even digits): {len(incoming_dataset) - n_inverted}")
print(f"  Out-of-distribution (inverted odd digits): {n_inverted}")

# %% [markdown]
# Let's visualize some in-distribution and OOD samples side by side:

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 8, figsize=(12, 4))

# Row 0: sample reference images
for col in range(8):
    img_arr = np.asarray(ref_maite[col][0]).squeeze()
    axes[0, col].imshow(img_arr, cmap="gray")
    axes[0, col].set_title("ref", fontsize=9)
    axes[0, col].axis("off")

# Row 1: color-inverted OOD samples (odd-digit classes)
ood_indices = [i for i in range(len(incoming_dataset)) if np.argmax(incoming_dataset[i][1]) in [1, 3, 5, 7, 9]]
for col in range(8):
    img_arr = np.asarray(incoming_dataset[ood_indices[col]][0]).squeeze()
    label = int(np.argmax(incoming_dataset[ood_indices[col]][1]))
    axes[1, col].imshow(img_arr, cmap="gray")
    axes[1, col].set_title(f"inverted\n(digit {label})", fontsize=8)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Reference", fontsize=9)
axes[1, 0].set_ylabel("Inverted", fontsize=9)

fig.suptitle("Reference digits (top) vs inverted odd-digit OOD samples (bottom)", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# The `ood-detection` workflow needs:
#
# 1. **Two datasets** — the first is the reference, the rest are test (incoming) data
# 2. **A model/extractor** — to compute embeddings that detectors compare
# 3. **Detector configuration** — which OOD detection methods to run
# 4. **Health thresholds** — what percentage of OOD samples triggers a warning
#
# We'll use the **Flatten** extractor (reshapes each 28x28 image into a 784-D
# vector) and configure two complementary detectors:
#
# | Detector | How it works | Strengths |
# |---|---|---|
# | **K-Neighbors** | Flags samples whose k nearest reference neighbors are unusually far | Fast, non-parametric, works in high dimensions |
# | **Domain Classifier** | Trains a LightGBM to distinguish ref from test; flags easily-separated samples | Powerful with many features, captures complex boundaries |

# %%
from dataeval_flow.config import DatasetProtocolConfig, FlattenExtractorConfig, PipelineConfig, SourceConfig

# --- Datasets (in-memory via DatasetProtocolConfig) ---
ref_config = DatasetProtocolConfig(
    name="reference",
    format="maite",
    dataset=ref_maite,
)

incoming_config = DatasetProtocolConfig(
    name="incoming",
    format="maite",
    dataset=incoming_dataset,
)

# --- Extractor ---
# Flatten reshapes each 28x28 grayscale image into a 784-D feature vector
extractor_config = FlattenExtractorConfig(name="flatten", batch_size=64)

# %% [markdown]
# ### Configure OOD detectors
#
# **K-Neighbors** (cosine distance, k=10) measures how far each incoming sample
# is from its nearest reference neighbors. Samples that are far away in
# embedding space are likely OOD. We use `threshold_perc=95` which means the
# top 5% of reference self-distances set the OOD boundary.
#
# **Domain Classifier** trains a LightGBM model to distinguish reference from
# incoming data using repeated cross-validation. Samples that the classifier
# consistently identifies as "incoming" are likely OOD. We use 3 folds and 3
# repeats to keep it fast for the tutorial.
#
# We also enable **metadata insights** — after OOD samples are identified,
# the workflow analyzes which metadata factors (like class label) deviate most
# for the flagged samples.

# %%
from pathlib import Path

from dataeval_flow.config import OODDetectionTaskConfig, OODDetectionWorkflowConfig
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.ood.params import (
    OODDetectorDomainClassifier,
    OODDetectorKNeighbors,
    OODHealthThresholds,
)

task = OODDetectionTaskConfig(
    name="mnist-ood-check",
    workflow="mnist-ood",
    sources=["ref_src", "inc_src"],
    extractor="flatten",
)

config = PipelineConfig(
    datasets=[ref_config, incoming_config],
    sources=[
        SourceConfig(name="ref_src", dataset="reference"),
        SourceConfig(name="inc_src", dataset="incoming"),
    ],
    extractors=[extractor_config],
    workflows=[
        OODDetectionWorkflowConfig(
            name="mnist-ood",
            detectors=[
                OODDetectorKNeighbors(
                    k=10,
                    distance_metric="cosine",
                    threshold_perc=99.0,
                ),
                OODDetectorDomainClassifier(
                    n_folds=3,
                    n_repeats=3,
                    threshold_perc=99.0,
                ),
            ],
            health_thresholds=OODHealthThresholds(
                ood_pct_warning=5.0,  # warn if >5% of samples are OOD
                ood_pct_info=1.0,  # info if >1% of samples are OOD
            ),
            metadata_insights=True,
            max_ood_insights=50,
        ),
    ],
)

# %% [markdown]
# ## Step 2: Run the OOD detection workflow

# %%
result = run_task(task, config, cache_dir=Path("./cache"))

# %% [markdown]
# ## Results Exploration: OOD report
#
# The workflow produces a text report summarizing each detector's findings —
# how many samples were flagged OOD, per-sample scores, and metadata insights.

# %%
print(result.report())

# %% [markdown]
# ## Understanding the results
#
# Let's look at the raw results to understand what the detectors found.

# %% [markdown]
# ### Per-detector summary
#
# Each detector independently scores every incoming sample. Higher scores mean
# more likely to be OOD. Samples above the threshold (set during fit on
# reference data) are flagged.

# %%
raw = result.data.raw

print(f"Reference size:  {raw.reference_size}")
print(f"Test size:       {raw.test_size}")
print(f"OOD samples:     {len(raw.ood_indices)} (union across all detectors)")
print()

for method, det_result in raw.detectors.items():
    print(f"-- {method} --")
    print(f"  OOD count:     {det_result['ood_count']} / {det_result['total_count']}")
    print(f"  OOD percentage: {det_result['ood_percentage']:.1f}%")
    print(f"  Threshold:     {det_result['threshold_score']:.4f}")
    print()

# %% [markdown]
# ### Visualize OOD scores
#
# A histogram of per-sample scores shows the separation between in-distribution
# and OOD samples. Ideally, OOD samples cluster at higher scores.

# %%
fig, axes = plt.subplots(1, len(raw.detectors), figsize=(6 * len(raw.detectors), 4))
if len(raw.detectors) == 1:
    axes = [axes]

ood_set = set(raw.ood_indices)

for ax, (method, det_result) in zip(axes, raw.detectors.items(), strict=True):
    samples = det_result.get("samples", [])
    if not samples:
        continue

    in_scores = [s["score"] for s in samples if not s["is_ood"]]
    ood_scores = [s["score"] for s in samples if s["is_ood"]]

    ax.hist(in_scores, bins=30, alpha=0.6, label=f"In-dist ({len(in_scores)})", color="#2ecc71")
    ax.hist(ood_scores, bins=30, alpha=0.6, label=f"OOD ({len(ood_scores)})", color="#e74c3c")
    ax.axvline(x=det_result["threshold_score"], color="orange", linestyle="--", label="Threshold")
    ax.set_xlabel("OOD Score")
    ax.set_ylabel("Count")
    ax.set_title(method, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

fig.suptitle("OOD Score Distributions by Detector", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Inspect OOD samples by score
#
# For each detector we show the **top 4** (highest score, most confidently OOD)
# and **bottom 4** (lowest score above threshold, borderline) flagged samples.

# %%
cols = 4

for method, det_result in raw.detectors.items():
    samples = det_result.get("samples", [])
    ood_samples = sorted(
        [(s["index"], s["score"]) for s in samples if s["is_ood"]],
        key=lambda x: -x[1],
    )
    if not ood_samples:
        continue

    top = ood_samples[:cols]
    bottom = ood_samples[-cols:]

    fig, axes = plt.subplots(2, cols, figsize=(1.5 * cols, 4))

    for row_idx, (row_label, row_items) in enumerate([("Top 4 (most OOD)", top), ("Bottom 4 (least OOD)", bottom)]):
        for col, (idx, score) in enumerate(row_items):
            img_data, target, _ = incoming_dataset[idx]
            img_arr = np.asarray(img_data)
            if img_arr.ndim == 3 and img_arr.shape[0] in (1, 3):
                img_arr = img_arr.squeeze(0) if img_arr.shape[0] == 1 else np.moveaxis(img_arr, 0, -1)
            axes[row_idx, col].imshow(img_arr, cmap="gray")
            label = int(np.argmax(target)) if np.asarray(target).ndim > 0 else int(target)
            axes[row_idx, col].set_title(f"idx={idx}\nlabel={label}\n{score:.3f}", fontsize=7)
            axes[row_idx, col].axis("off")
        for col in range(len(row_items), cols):
            axes[row_idx, col].axis("off")
        axes[row_idx, 0].set_ylabel(row_label, fontsize=8)

    fig.suptitle(f"{method}: most vs least confident OOD samples", fontsize=12)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Metadata insights
#
# When metadata insights are enabled, the workflow analyzes which metadata
# factors correlate with OOD status. **Factor predictors** show mutual
# information between each factor and the OOD flag — higher values mean the
# factor is a stronger predictor of OOD. **Factor deviations** show how
# individual OOD samples differ from the reference distribution per factor.

# %%
if raw.factor_predictors:
    print("Factor Predictors (mutual information with OOD status):")
    print("-" * 50)
    for factor, mi in raw.factor_predictors.items():
        bar = "#" * int(mi * 20)
        print(f"  {factor:20s}  {mi:.4f} bits  {bar}")
    print()

if raw.factor_deviations:
    print(f"Factor Deviations (top {min(10, len(raw.factor_deviations))} OOD samples):")
    print("-" * 50)
    for dev in raw.factor_deviations[:10]:
        top_factors = list(dev["deviations"].items())[:3]
        factors_str = ", ".join(f"{k}={v:.2f}" for k, v in top_factors)
        print(f"  Sample {dev['index']:4d}: {factors_str}")

# %% [markdown]
# ## Results Exploration: Export results
#
# The JSON output contains all raw detector results, per-sample scores, and
# metadata insights — ready for integration with monitoring dashboards or
# automated pipelines.

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")
print(json_str[:600] + "\n...")

# %% [markdown]
# ## Conclusion
#
# In this tutorial you learned how to:
#
# - **Prepare** reference and incoming datasets with known OOD samples
# - **Configure** the `ood-detection` workflow with K-Neighbors and Domain
#   Classifier detectors
# - **Run** the workflow and read the OOD report with per-detector summaries
# - **Inspect** per-sample OOD scores and visualize the score distributions
# - **Use metadata insights** to understand which factors correlate with OOD
#   status
# - **Export** structured JSON results for downstream automation
#
# The key difference from **drift monitoring** is granularity: drift detection
# answers "has the distribution changed?" while OOD detection answers "which
# specific samples don't belong?"

# %% [markdown]
# ## What's next
#
# - **Threshold tuning** — Adjust `threshold_perc` to trade off between
#   catching more OOD samples (lower threshold) and reducing false positives
#   (higher threshold)
# - **Real-world models** — Use an ONNX model (e.g. ResNet) with preprocessing
#   for richer embeddings that capture higher-level features
# - **Drift + OOD pipeline** — Combine `drift-monitoring` and `ood-detection`
#   workflows in the same pipeline config to detect both distribution-level
#   shifts and individual outliers
# - **Production deployment** — See the Docker deployment guide for running OOD
#   detection on a schedule against live data pipelines
