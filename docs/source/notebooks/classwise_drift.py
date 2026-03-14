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
# # Detect classwise drift — which classes are changing?
#
# Detect distribution drift that affects **specific classes** rather than the
# dataset as a whole. This tutorial simulates progressive sensor degradation
# that only impacts digits 1, 4, and 7, then uses **classwise drift detection**
# to pinpoint the affected classes.

# %% [markdown]
# ## What you'll do
#
# - Load MNIST from HuggingFace into memory using `from_huggingface`
# - Build a **wrapper dataset** that applies increasing Gaussian blur to
#   classes 1, 4, and 7 — simulating sensor degradation that worsens over time
# - Use `SelectionConfig` with `Indices` to subset the dataset into reference
#   and incoming slices — no need to write images to disk
# - **Phase 1**: Run **overall** drift detection with a chunked **Domain
#   Classifier** to confirm drift exists and see *when* it started
# - **Phase 2**: Follow up with **classwise** drift detection using MMD and
#   Univariate CVM to identify *which* classes are affected

# %% [markdown]
# ## What you'll learn
#
# - How to use in-memory HuggingFace datasets with `DatasetProtocolConfig`
# - How to use `SelectionConfig` and `Indices` to subset datasets in the workflow config
# - How to simulate class-specific degradation with a thin dataset wrapper
# - How to run a two-phase drift analysis: chunked overall first, then classwise
# - How per-detector `classwise: true` breaks drift results down by class

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datasets`, `maite-datasets`, `pydantic`)
# - Internet connection (to download MNIST from HuggingFace Hub on first run)

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load MNIST and convert to MAITE format
#
# We'll load the MNIST test split directly from HuggingFace and convert it to
# the MAITE dataset protocol using `from_huggingface`. Everything stays
# in memory — no need to write images to disk.

# %% tags=["remove_output"]
from collections.abc import Mapping
from typing import cast

from datasets import Dataset
from datasets import load_dataset as hf_load
from maite_datasets.adapters import from_huggingface

# Download MNIST test split (10 000 images of 28×28 handwritten digits)
mnist_test = cast(Dataset, hf_load("ylecun/mnist", split="test"))

# Convert to MAITE protocol — gives us (image, label, metadata) tuples
mnist_maite = from_huggingface(mnist_test)

print(f"Loaded {len(mnist_maite)} MNIST test images via MAITE adapter")
print(f"Sample shape: image={mnist_maite[0][0].shape}, label={mnist_maite[0][1]}")

# %% [markdown]
# ## Data Preparation: Build a degradation wrapper
#
# We'll create a thin wrapper around the MAITE dataset that applies
# **increasing Gaussian blur** to specific digit classes. This simulates a
# real-world scenario where a sensor defect progressively worsens and only
# affects certain types of inputs (e.g., thin strokes in digits 1, 4, 7).
#
# The blur radius increases linearly with the sample index — early samples
# are nearly clean, while later samples are heavily blurred. This models
# temporal degradation that gets worse over time.

# %%
from typing import Any

import numpy as np
from maite_datasets.adapters import HFImageClassificationDataset, HFObjectDetectionDataset
from numpy.typing import NDArray
from PIL import Image, ImageFilter


class DegradedDataset:
    """Wraps a MAITE dataset and applies progressive Gaussian blur to selected classes.

    Parameters
    ----------
    dataset
        A MAITE-compatible dataset returning (image, target, metadata) tuples.
    degraded_classes
        Set of class labels to apply blur to.
    max_blur_radius
        Maximum blur radius applied to the last sample.
    """

    def __init__(
        self,
        dataset: HFImageClassificationDataset | HFObjectDetectionDataset,
        degraded_classes: set[int],
        max_blur_radius: float = 3.0,
    ) -> None:
        self._dataset = dataset
        self._degraded_classes = degraded_classes
        self.max_blur_radius: float = max_blur_radius

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[NDArray[Any], Any, Mapping[str, Any]]:
        image, target, metadata = self._dataset[index]
        t = np.asarray(target)
        label = int(np.argmax(t)) if t.ndim == 1 and t.size > 1 else int(t)

        if label in self._degraded_classes:
            # Blur increases linearly with index — simulates progressive degradation
            progress = index / max(len(self) - 1, 1)
            radius = self.max_blur_radius * progress

            if radius > 0.1:  # skip negligible blur
                img_array = np.asarray(image)
                # CHW → HWC if needed
                if img_array.ndim == 3 and img_array.shape[0] in (1, 3):
                    img_array = np.transpose(img_array, (1, 2, 0))

                if img_array.dtype in (np.float32, np.float64):
                    img_pil = Image.fromarray((img_array.squeeze() * 255).astype(np.uint8), mode="L")
                else:
                    img_pil = Image.fromarray(img_array.squeeze(), mode="L")

                img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
                blurred = np.array(img_pil, dtype=np.float32) / 255.0
                image = blurred[np.newaxis, :, :]  # back to CHW

        return image, target, metadata


# %% [markdown]
# Now let's prepare the datasets. We use the **same** underlying MAITE dataset
# for both reference and incoming — the workflow's `SelectionConfig` with
# `Indices` will subset each to the right index range. The incoming dataset
# gets wrapped with `DegradedDataset` to apply class-specific blur.
#
# - **Reference**: first 2 000 images — clean, unmodified
# - **Incoming**: next 2 000 images (HF indices 3 000–5 999)
#   — classes 1, 4, 7 get progressively blurred

# %% tags=["remove_output"]

# Wrap the incoming slice with degradation for classes 1, 4, 7
incoming_hf = mnist_test.select(range(2000, 4000))
incoming_maite = from_huggingface(incoming_hf)
incoming_dataset = DegradedDataset(
    incoming_maite,
    degraded_classes={1, 4, 7},
    max_blur_radius=3.0,
)

print(f"Reference: {len(mnist_maite)} total images (will select first 2000)")
print(f"Incoming:  {len(incoming_dataset)} images (classes 1,4,7 progressively blurred)")

# %% [markdown]
# Let's visualize the degradation. For each affected class, we show the same
# digit at different positions in the incoming dataset — early (nearly clean)
# to late (heavily blurred):

# %%
import matplotlib.pyplot as plt

degraded_classes = [1, 4, 7]
positions = [0.0, 0.25, 0.5, 0.75, 1.0]

fig, axes = plt.subplots(len(degraded_classes), len(positions), figsize=(12, 7))

for row, cls in enumerate(degraded_classes):
    for col, frac in enumerate(positions):
        start = int(frac * len(incoming_dataset) * 0.9)
        for i in range(start, len(incoming_dataset)):
            img, target, _ = incoming_dataset[i]
            t = np.asarray(target)
            label = int(np.argmax(t)) if t.ndim == 1 and t.size > 1 else int(t)
            if label == cls:
                img_arr = np.asarray(img).squeeze()
                axes[row, col].imshow(img_arr, cmap="gray", vmin=0, vmax=1)
                radius = incoming_dataset.max_blur_radius * (i / max(len(incoming_dataset) - 1, 1))
                axes[row, col].set_title(f"r={radius:.1f}", fontsize=9)
                break
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(f"Digit {cls}", fontsize=11, rotation=0, labelpad=40)

fig.suptitle("Progressive blur on classes 1, 4, 7 (r = blur radius)", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 1: Phase 1 — Overall drift detection with chunking
#
# First we'll run **overall** drift detection (no classwise) to confirm that
# drift exists. We use the **K-Neighbors** detector with **chunking**
# enabled. K-Neighbors checks whether incoming samples are farther from their
# k-nearest reference neighbors than expected — a Mann-Whitney U test on
# these distances detects distributional shift. Chunking splits the incoming
# data into temporal windows so we can see not just *whether* drift occurred,
# but *when* it started.
#
# We use `DatasetProtocolConfig` for in-memory datasets and `SelectionConfig`
# with `Indices` to subset the reference to the first 4 000 images.

# %%

from dataeval_flow.config.models import FlattenExtractorConfig, ModelConfig, PipelineConfig
from dataeval_flow.config.schemas.dataset import DatasetProtocolConfig
from dataeval_flow.config.schemas.params import DriftMonitoringWorkflowConfig
from dataeval_flow.config.schemas.selection import SelectionConfig, SelectionStep
from dataeval_flow.config.schemas.task import DriftMonitoringTaskConfig
from dataeval_flow.workflow.orchestrator import run_task
from dataeval_flow.workflows.drift.params import ChunkingConfig, DriftDetectorKNeighbors, DriftHealthThresholds

# --- Datasets (in-memory via DatasetProtocolConfig) ---
ref_config = DatasetProtocolConfig(
    name="reference",
    format="maite",
    dataset=mnist_maite,  # full 10k — selection will subset it
)

incoming_config = DatasetProtocolConfig(
    name="incoming",
    format="maite",
    dataset=incoming_dataset,
)

# --- Selections ---
# Use Indices to select the first 2000 images as the reference baseline
ref_selection = SelectionConfig(
    name="ref-first-2k",
    steps=[SelectionStep(type="Indices", params={"indices": list(range(2000))})],
)

# --- Model ---
model_config = ModelConfig(
    name="flatten",
    extractor=FlattenExtractorConfig(),
)

# --- Phase 1: Overall drift with chunking (no classwise) ---
overall_config = PipelineConfig(
    datasets=[ref_config, incoming_config],
    models=[model_config],
    selections=[ref_selection],
    workflows=[
        DriftMonitoringWorkflowConfig(
            name="overall-drift",
            detectors=[
                DriftDetectorKNeighbors(k=10, chunking=ChunkingConfig(chunk_count=5, threshold_multiplier=1.5)),
            ],
            health_thresholds=DriftHealthThresholds(
                chunk_drift_pct_warning=15.0,
                consecutive_chunks_warning=2,
            ),
        ),
    ],
)

overall_task = DriftMonitoringTaskConfig(
    name="mnist-overall-drift",
    workflow="overall-drift",
    datasets=["reference", "incoming"],
    models="flatten",
    selections={"reference": "ref-first-2k"},
    batch_size=64,
    cache_dir="./cache",
)

# %% [markdown]
# ## Step 2: Run overall drift detection

# %%
overall_result = run_task(overall_task, overall_config)

# %% [markdown]
# ### Review the overall report
#
# The K-Neighbors should flag drift in the later chunks — the chunked
# view shows *when* drift started, but not *which classes* are responsible.

# %%
print(overall_result.report(format="text"))

# %% [markdown]
# ## Step 3: Phase 2 — Classwise drift detection
#
# The overall K-Neighbors confirmed drift. Now we want to know **which
# classes** are affected. We enable `classwise=True` on each detector and use
# MMD alongside Univariate CVM to get two complementary views:
#
# - **MMD** — the same kernel-based test, now run per class. Works well even
#   on the ~300 samples per class because the RBF kernel captures
#   distributional differences without suffering from distance concentration.
# - **Univariate CVM** — tests each of the 784 pixel features independently,
#   giving a per-feature breakdown of where the shift occurs.

# %%
from dataeval_flow.workflows.drift.params import DriftDetectorMMD, DriftDetectorUnivariate

classwise_config = PipelineConfig(
    datasets=[ref_config, incoming_config],
    models=[model_config],
    selections=[ref_selection],
    workflows=[
        DriftMonitoringWorkflowConfig(
            name="classwise-drift",
            detectors=[
                DriftDetectorMMD(n_permutations=100, classwise=True),
                DriftDetectorUnivariate(test="cvm", classwise=True),
            ],
            health_thresholds=DriftHealthThresholds(
                classwise_any_drift_is_warning=True,
            ),
        ),
    ],
)

classwise_task = DriftMonitoringTaskConfig(
    name="mnist-classwise-drift",
    workflow="classwise-drift",
    datasets=["reference", "incoming"],
    models="flatten",
    selections={"reference": "ref-first-2k"},
    batch_size=64,
    cache_dir="./cache",
)

# %% [markdown]
# ## Step 4: Run classwise drift detection

# %%
classwise_result = run_task(classwise_task, classwise_config)

# %% [markdown]
# ### Review the classwise report
#
# The report now includes a **classwise pivot table**: rows are classes (0–9),
# columns are detectors. Only classes 1, 4, and 7 should show drift — the
# others remain clean.

# %%
print(classwise_result.report(format="text"))

# %% [markdown]
# ## Results Exploration: Classwise results

# %%
import polars as pl

pl.Config.set_tbl_hide_dataframe_shape(True)

raw = classwise_result.data.raw

# Overall results (from the classwise run)
print("── Overall Drift ──")
for method, det_result in raw.detectors.items():
    status = "DRIFT" if det_result["drifted"] else "ok"
    print(f"  {method}: {status} (distance={det_result['distance']:.4f})")
print()

# Classwise results
if raw.classwise:
    print("── Classwise Drift ──")
    for cw in raw.classwise:
        print(f"\n  Detector: {cw['detector']}")
        rows = [
            {
                "class": r["class_name"],
                "drifted": r["drifted"],
                "distance": round(r["distance"], 4),
                "p_val": round(r["p_val"], 6) if r.get("p_val") is not None else None,  # type:ignore
            }
            for r in cw["rows"]
        ]
        df = pl.DataFrame(rows)
        print(df)

# %% [markdown]
# ### Visualize classwise drift
#
# A horizontal bar chart makes it immediately clear which classes are drifting
# and which are stable.

# %%
assert raw.classwise is not None  # classwise=True was set on each detector above
detectors = [cw["detector"] for cw in raw.classwise]

fig, axes = plt.subplots(1, len(detectors), figsize=(6 * len(detectors), 4))
if len(detectors) == 1:
    axes = [axes]

for ax, cw in zip(axes, raw.classwise, strict=True):
    class_names = [r["class_name"] for r in cw["rows"]]
    distances = [r["distance"] for r in cw["rows"]]
    drifted = [r["drifted"] for r in cw["rows"]]
    colors = ["#e74c3c" if d else "#2ecc71" for d in drifted]

    ax.barh(class_names, distances, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Class")
    ax.set_title(cw["detector"], fontsize=12, fontweight="bold")
    ax.invert_yaxis()

fig.suptitle("Classwise drift — green = ok, red = drift detected", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusion
#
# In this tutorial you learned how to:
#
# - **Load datasets in memory** using `from_huggingface` and `DatasetProtocolConfig`
# - **Subset datasets** using `SelectionConfig` with `Indices` — no disk I/O needed
# - **Simulate class-specific degradation** with a lightweight dataset wrapper
#   that applies progressive Gaussian blur to selected classes
# - **Phase 1: Detect drift overall** — a chunked K-Neighbors confirms
#   drift exists and shows *when* it started
# - **Phase 2: Drill into classwise drift** — enable `classwise=True` on
#   MMD and Univariate CVM detectors to pinpoint exactly which classes
#   (1, 4, 7) are affected
#
# This two-phase approach mirrors real-world practice: first confirm drift
# exists, then investigate which classes are responsible so you can take
# targeted corrective action.

# %% [markdown]
# ## What's next
#
# - **Production pipelines** — Run the two-phase approach on a schedule: chunked
#   overall detection as a fast gate, classwise as a deeper diagnostic
# - **Different extractors** — Use an ONNX model (e.g. ResNet) for richer
#   embeddings that may detect subtler class-specific shifts
# - **Threshold tuning** — Adjust `health_thresholds` to control when classwise
#   drift triggers warnings vs. informational findings
