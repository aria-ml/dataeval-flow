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
# # Detect classwise drift — which classes are changing?
#
# Detect distribution drift that affects **specific classes** rather than the
# dataset as a whole. This tutorial simulates progressive sensor degradation that only
# reaches three of twenty-four vehicle types, then uses **classwise drift detection**
# to pinpoint which ones.

# %% [markdown]
# **Who this is for** — T&E engineers diagnosing *which* classes are responsible
# for a drift signal, so corrective action can be targeted rather than dataset-wide.
#
# **Where this fits** — Classwise drift is the diagnostic follow-up to overall
# [drift monitoring](drift_monitoring) in the T&E workflow: once a fast overall
# check flags that drift exists, you drill into per-class results to decide where
# to collect data or retrain. See the
# [Distribution shift](../concepts/DistributionShift.md) concept page for background.

# %% [markdown]
# ## What you'll do
#
# - Load MilitaryVehicles and read it with datamaite's `huggingface_vision` loader
# - Build a **wrapper dataset** that applies increasing Gaussian blur to `BMP-1`,
#   `BTR-80` and `T-72` — simulating sensor degradation that worsens over time
# - Embed both halves with a **pretrained ResNet-18**, so the two are described in the
#   same representation
# - Use `ViewConfig` with `Indices` to subset the dataset into reference
#   and incoming slices
# - **Phase 1**: Run **overall** drift detection with a chunked **K-Neighbors** check to
#   confirm drift exists and see *when* it started
# - **Phase 2**: Follow up with **classwise** drift detection using MMD and
#   Univariate CVM to identify *which* classes are affected

# %% [markdown]
# ## What you'll learn
#
# - How to use in-memory datasets with `DatasetProtocolConfig`
# - How to use `ViewConfig` and `Indices` to subset datasets in the workflow config
# - How to simulate class-specific degradation with a thin dataset wrapper
# - How to run a two-phase drift analysis: chunked overall first, then classwise
# - How per-detector `classwise: true` breaks drift results down by class
# - **Why the extractor decides whether any of this works** — drift is measured in
#   whatever representation you hand the detectors, and a poor one reports drift
#   between two samples of unchanged data

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`)
# - `maite-datasets[datamaite]` (to download MilitaryVehicles and export it)
# - `torch` and `torchvision` (for the pretrained ResNet-18 used as the extractor)
# - Internet connection on the first run; everything after that comes from disk

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the dataset
#
# [MilitaryVehicles](https://huggingface.co/datasets/leibnitz-lab/military_vehicles) is a
# classification dataset of 9,444 images across 24 vehicle types. `as_datamaite=True`
# writes it as a class-per-directory tree, which datamaite's `huggingface_vision` loader
# reads directly — no materialization step to write here.
#
# The loader returns samples grouped by class folder, and this tutorial needs an order
# that stands in for collection time so degradation can worsen along it. The dataset
# carries no timestamps, so we impose a fixed shuffled order and treat position in it as
# "when the frame arrived". Seeded, so every run sees the same sequence.

# %% tags=["remove_output"]
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from dataeval.data import Indices, Limit, Shuffle, View
from datamaite import load_ic
from maite_datasets.image_classification import MilitaryVehicles
from numpy.typing import NDArray

data_root = Path("./data")
MilitaryVehicles(root=data_root, image_set="base", download=True)
MilitaryVehicles(root=data_root, image_set="train", as_datamaite=True)

vehicles_raw = load_ic(data_root / "militaryvehicles_datamaite_train" / "train", dataset_format="huggingface_vision")

# The loader groups samples by class folder, so raw index order is alphabetical by class
# rather than anything resembling arrival. `Shuffle` imposes one seeded sequence that the
# rest of this tutorial treats as collection order — it is what makes "index i arrived
# after index i-1" mean something, which the degradation below and the chunked view in
# Phase 1 both rely on.
#
# `View` is the same machinery a `ViewConfig` drives from the pipeline config, used
# directly because the degradation wrapper needs the ordered dataset in hand before the
# workflow ever sees it. `resolve_indices()` hands back the source indices in selection
# order, which is how the incoming slice below is cut without re-deriving the shuffle.
vehicles = View(vehicles_raw, [Shuffle(seed=42), Limit(4000)])
_order = vehicles.resolve_indices()

index2label = vehicles_raw.metadata.get("index2label", {})
DEGRADED = {4, 10, 19}
print(f"Loaded {len(vehicles)} images across {len(index2label)} classes")
print(f"Degrading: {[index2label[c] for c in sorted(DEGRADED)]}")

# %% [markdown]
# ## Data Preparation: Build a degradation wrapper
#
# We'll create a thin wrapper around the MAITE dataset that applies
# **increasing Gaussian blur** to three of the twenty-four classes. This stands in for a
# collection defect that reaches only part of the data — the realistic cause being
# provenance rather than subject matter: one platform's optics degrade, and that platform
# happens to supply most of the frames for certain vehicle types.
#
# The blur radius increases linearly with the sample index — early samples
# are nearly clean, while later samples are heavily blurred. This models
# temporal degradation that gets worse over time.

# %%
from PIL import Image, ImageFilter


class DegradedDataset:
    """Wraps a MAITE dataset and applies progressive Gaussian blur to selected classes.

    Parameters
    ----------
    dataset
        A MAITE-compatible dataset returning (image, target, metadata) tuples,
        with images as CHW uint8 arrays (datamaite's native image format).
    degraded_classes
        Set of class labels to apply blur to.
    max_blur_radius
        Maximum blur radius applied to the last sample.
    """

    def __init__(
        self,
        dataset: Any,
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
                chw = np.asarray(image)
                hwc = np.transpose(chw, (1, 2, 0))  # CHW -> HWC
                img_pil = Image.fromarray(hwc, mode="RGB").filter(ImageFilter.GaussianBlur(radius=radius))
                image = np.transpose(np.array(img_pil, dtype=chw.dtype), (2, 0, 1))  # back to CHW

        return image, target, metadata


# %% [markdown]
# Now let's prepare the datasets. We use the **same** underlying MAITE dataset
# for both reference and incoming — the workflow's `ViewConfig` with
# `Indices` will subset each to the right index range. The incoming dataset
# gets wrapped with `DegradedDataset` to apply class-specific blur.
#
# - **Reference**: first 2 000 frames — clean, unmodified
# - **Incoming**: next 2 000 frames (positions 2 000–3 999)
#   — `BMP-1`, `BTR-80` and `T-72` get progressively blurred

# %% tags=["remove_output"]

# Wrap the incoming slice with degradation for classes 1, 4, 7
incoming_maite = View(vehicles_raw, [Indices(_order[2000:4000])])
incoming_dataset = DegradedDataset(
    incoming_maite,
    degraded_classes=DEGRADED,
    max_blur_radius=6.0,
)

print(f"Reference: {len(vehicles)} total frames (will select first 2000)")
print(f"Incoming:  {len(incoming_dataset)} frames ({len(DEGRADED)} classes progressively blurred)")

# %% [markdown]
# Let's visualize the degradation. For each affected class, we show the same
# frame at different positions in the incoming dataset — early (nearly clean)
# to late (heavily blurred):

# %%
import matplotlib.pyplot as plt

degraded_classes = sorted(DEGRADED)
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
                img_arr = np.transpose(np.asarray(img), (1, 2, 0))  # CHW -> HWC
                axes[row, col].imshow(img_arr)
                radius = incoming_dataset.max_blur_radius * (i / max(len(incoming_dataset) - 1, 1))
                axes[row, col].set_title(f"r={radius:.1f}", fontsize=9)
                break
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(index2label[cls], fontsize=11, rotation=0, labelpad=60)

fig.suptitle("Progressive blur on three of twenty-four classes (r = blur radius)", fontsize=13)
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
# We use `DatasetProtocolConfig` for in-memory datasets and `ViewConfig`
# with `Indices` to subset the reference to the first 2 000 frames.
#
# :::{important}
# **The extractor decides whether any of this works.** Drift is measured in whatever
# representation you hand the detectors, so the extractor is not an implementation detail
# here — it is the experiment.
#
# A `flatten` extractor turns each image into a raw pixel vector. On MNIST that is
# defensible: every sample is a centred 28x28 glyph, so two sets of them differ mainly in
# what was written. On natural imagery it is not. Distance becomes a function of scene
# content, and two different halves of a 24-class collection look far apart before
# anything has degraded — a chunked check over flattened pixels flags drift on data with
# **no degradation applied at all**.
#
# A pretrained network is the fix: its representation is fixed in advance, so reference
# and incoming are described in the same terms no matter which is embedded first. We use
# **ResNet-18** — small (~45 MB), fast on CPU, and familiar. Its `avgpool` layer gives a
# 512-dimensional feature per image.
# :::

# %% tags=["remove_output"]
import torch
import torchvision

# torchvision fetches the weights once (~45 MB) into its own cache; saving the model
# beside the notebook is what `TorchExtractorConfig` loads.
model_dir = Path("./models")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "resnet18.pt"
if not model_path.exists():
    torch.save(torchvision.models.resnet18(weights="IMAGENET1K_V1").eval(), model_path)
print(f"Extractor model: {model_path}")

# %%
from dataeval_flow.config import (
    DatasetProtocolConfig,
    DriftMonitoringTaskConfig,
    DriftMonitoringWorkflowConfig,
    PipelineConfig,
    PreprocessorConfig,
    SourceConfig,
    TorchExtractorConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.preprocessing import PreprocessingStep
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.drift.params import ChunkingConfig, DriftDetectorKNeighbors, DriftHealthThresholds

# --- Datasets (in-memory via DatasetProtocolConfig) ---
ref_config = DatasetProtocolConfig(
    name="reference",
    format="maite",
    dataset=vehicles,  # full 4k — view will subset it
)

incoming_config = DatasetProtocolConfig(
    name="incoming",
    format="maite",
    dataset=incoming_dataset,
)

# --- Views ---
# Use Indices to select the first 2000 images as the reference baseline
ref_view = ViewConfig(
    name="ref-first-2k",
    operations=[ViewOperation(type="Indices", params={"indices": list(range(2000))})],
)

# --- Sources ---
ref_source_config = SourceConfig(name="reference_2k", dataset="reference", view="ref-first-2k")
inc_source_config = SourceConfig(name="incoming_2k", dataset="incoming")

# --- Extractors ---
# The preprocessor is ResNet's own: resize to the resolution it was trained at, scale to
# float, and normalize with the ImageNet statistics. It also settles the variable image
# sizes in this collection, which a raw-pixel extractor could not have handled at all.
preprocessor_config = PreprocessorConfig(
    name="imagenet",
    steps=[
        PreprocessingStep(step="Resize", params={"size": [224, 224], "antialias": True}),
        PreprocessingStep(step="ToDtype", params={"dtype": "float32", "scale": True}),
        PreprocessingStep(
            step="Normalize",
            params={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        ),
    ],
)

extractor_config = TorchExtractorConfig(
    name="resnet18",
    model_path=str(model_path),
    layer_name="avgpool",  # 512-d feature, before the classification head
    preprocessor="imagenet",
    batch_size=32,
)

# --- Workflows ---
drift_workflow_config = DriftMonitoringWorkflowConfig(
    name="overall-drift",
    detectors=[
        DriftDetectorKNeighbors(k=10, chunking=ChunkingConfig(chunk_count=5, threshold_multiplier=1.5)),
    ],
    health_thresholds=DriftHealthThresholds(
        chunk_drift_pct_warning=15.0,
        consecutive_chunks_warning=2,
    ),
)

# --- Phase 1: Overall drift with chunking (no classwise) ---
overall_task = DriftMonitoringTaskConfig(
    name="vehicles-overall-drift",
    workflow="overall-drift",
    sources=["reference_2k", "incoming_2k"],
    extractor="resnet18",
)

overall_config = PipelineConfig(
    datasets=[ref_config, incoming_config],
    views=[ref_view],
    sources=[ref_source_config, inc_source_config],
    preprocessors=[preprocessor_config],
    extractors=[extractor_config],
    workflows=[drift_workflow_config],
    tasks=[overall_task],
)

# %% [markdown]
# ## Step 2: Run overall drift detection

# %%
overall_result = run_task(overall_task, overall_config, cache_dir=Path("./cache"))

# %% [markdown]
# ### Review the overall report
#
# The chunked view shows *when* drift started, not *which classes* are responsible. The
# first chunks sit inside the reference band and the later ones cross it, with the
# distance climbing monotonically — which is what progressive degradation looks like from
# the outside. On a clean incoming set the same check stays inside the band for all five
# chunks, so the pattern below is the degradation rather than the two halves differing.

# %%
print(overall_result.report())

# %% [markdown]
# ## Step 3: Phase 2 — Classwise drift detection
#
# The overall K-Neighbors confirmed drift. Now we want to know **which
# classes** are affected. We enable `classwise=True` on each detector and use
# MMD alongside Univariate CVM to get two complementary views:
#
# - **MMD** — the same kernel-based test, now run per class. It works on the ~80 samples
#   each class contributes to a 2 000-frame slice because the RBF kernel captures
#   distributional differences without suffering from distance concentration.
# - **Univariate CVM** — tests each of the 512 embedding dimensions independently,
#   giving a per-feature breakdown of where the shift occurs.

# %%
from dataeval_flow.workflows.drift.params import DriftDetectorMMD, DriftDetectorUnivariate

classwise_task = DriftMonitoringTaskConfig(
    name="vehicles-classwise-drift",
    workflow="classwise-drift",
    sources=["reference_2k", "incoming_2k"],
    extractor="resnet18",
)

classwise_config = PipelineConfig(
    datasets=[ref_config, incoming_config],
    views=[ref_view],
    sources=[ref_source_config, inc_source_config],
    preprocessors=[preprocessor_config],
    extractors=[extractor_config],
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
    tasks=[classwise_task],
)

# %% [markdown]
# ## Step 4: Run classwise drift detection

# %%
classwise_result = run_task(classwise_task, classwise_config, cache_dir=Path("./cache"))

# %% [markdown]
# ### Review the classwise report
#
# The report now includes a **classwise pivot table**: rows are the 24 vehicle types,
# columns are detectors. The three degraded classes should stand out — not only by being
# flagged, but by the size of their distance.
#
# Read the table by effect size rather than by the flag alone. A class can cross the
# significance threshold on a difference far too small to act on, and here the degraded
# classes sit an order of magnitude above any such borderline hit. The flag tells you a
# difference is unlikely to be chance; only the distance tells you whether it matters.

# %%
print(classwise_result.report())

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
# - **Load a dataset** exported by `maite-datasets` with datamaite's `huggingface_vision`
#   loader, then pass the result to `DatasetProtocolConfig`
# - **Subset datasets** using `ViewConfig` with `Indices` — no additional
#   disk I/O needed beyond the initial load
# - **Simulate class-specific degradation** with a lightweight dataset wrapper
#   that applies progressive Gaussian blur to selected classes
# - **Phase 1: Detect drift overall** — a chunked K-Neighbors confirms
#   drift exists and shows *when* it started
# - **Phase 2: Drill into classwise drift** — enable `classwise=True` on
#   MMD and Univariate CVM detectors to pinpoint which of the 24 classes are affected
# - **Choose an extractor deliberately** — drift is measured in the representation you
#   provide, and a raw-pixel one reports drift between two samples of unchanged data
#
# This two-phase approach mirrors real-world practice: first confirm drift
# exists, then investigate which classes are responsible so you can take
# targeted corrective action.

# %% [markdown]
# ## What's next
#
# - **Production pipelines** — Run the two-phase approach on a schedule: chunked
#   overall detection as a fast gate, classwise as a deeper diagnostic
# - **Different extractors** — Swap ResNet-18 for a larger backbone, or an ONNX model via
#   [ONNX embeddings](onnx_embeddings), when subtler class-specific shifts matter
# - **Threshold tuning** — Adjust `health_thresholds` to control when classwise
#   drift triggers warnings vs. informational findings

# %% [markdown]
# ## Related guides
#
# - **Concept** — [Distribution shift](../concepts/DistributionShift.md):
#   the overall-then-classwise diagnostic pattern and what each detector measures.
# - **Tutorial** — [Monitor incoming data for drift](drift_monitoring): the overall
#   drift check this tutorial drills into.
# - **How-to: Run workflows in containers** — [Containerized workflows](../how_to/containerized_workflows.md)
#   to run the two-phase approach on a schedule from a container.
