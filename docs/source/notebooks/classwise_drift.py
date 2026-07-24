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
# dataset as a whole. This tutorial simulates progressive sensor degradation
# that only impacts digits 1, 4, and 7, then uses **classwise drift detection**
# to pinpoint the affected classes.

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
# - Download MNIST from HuggingFace and materialize it into an ImageFolder
#   layout that datamaite's `huggingface_vision` loader reads
# - Build a **wrapper dataset** that applies increasing Gaussian blur to
#   classes 1, 4, and 7 — simulating sensor degradation that worsens over time
# - Use `SelectionConfig` with `Indices` to subset the dataset into reference
#   and incoming slices
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
# - `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`)
# - `datasets` (to download MNIST from HuggingFace Hub)
# - Internet connection (to download MNIST from HuggingFace Hub on first run)

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load MNIST and convert to MAITE format
#
# datamaite has no in-memory constructor — every dataset comes from a
# filesystem loader. We'll download the MNIST test split from HuggingFace,
# then materialize a 4 000-image subset to a temporary **ImageFolder** layout
# (`<label>/<file>.png`) that datamaite's `huggingface_vision` loader can read.
# That loader groups samples by class folder for a deterministic layout, which
# discards the original download order, so we restore it right after loading —
# needed below to simulate progressive degradation over "time".

# %% tags=["remove_output"]
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from datamaite import load_ic
from datasets import Dataset
from datasets import load_dataset as hf_load
from numpy.typing import NDArray

# Download MNIST test split, keep the first 4 000 images (reference + incoming)
mnist_test = cast(Dataset, hf_load("ylecun/mnist", split="test")).select(range(4000))

# Materialize as an ImageFolder tree: <root>/<label>/<seq>.png. The zero-padded
# sequence number lets us recover the original download order after reload,
# since datamaite's huggingface_vision loader groups samples by class folder.
mnist_root = Path(tempfile.mkdtemp(prefix="dataeval_flow_mnist_"))
for seq, example in enumerate(mnist_test):
    label_dir = mnist_root / str(example["label"])
    label_dir.mkdir(parents=True, exist_ok=True)
    example["image"].save(label_dir / f"{seq:05d}.png")

mnist_maite_raw = load_ic(mnist_root, dataset_format="huggingface_vision")
_order = sorted(range(len(mnist_maite_raw)), key=lambda i: int(Path(mnist_maite_raw.samples[i].image_id).stem))


class OrderedView:
    """Presents *dataset* through a fixed list of source indices.

    datamaite's ``huggingface_vision`` loader groups samples by class folder
    for a deterministic layout, discarding upload order. We recover it via
    ``_order`` (built from the sequence number embedded in each file name) so
    the rest of this tutorial can rely on "index i is the i-th downloaded
    image" — used below to simulate progressive degradation over "time" and
    to slice out the incoming split.
    """

    def __init__(self, dataset: Any, indices: list[int]) -> None:
        self._dataset = dataset
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> tuple[NDArray[Any], Any, Mapping[str, Any]]:
        return self._dataset[self._indices[index]]

    @property
    def metadata(self) -> Any:
        return self._dataset.metadata


# Reference: all 4 000 images, restored to original download order
mnist_maite = OrderedView(mnist_maite_raw, _order)

print(f"Loaded {len(mnist_maite)} MNIST images via datamaite (huggingface_vision)")
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
import numpy as np
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
# for both reference and incoming — the workflow's `SelectionConfig` with
# `Indices` will subset each to the right index range. The incoming dataset
# gets wrapped with `DegradedDataset` to apply class-specific blur.
#
# - **Reference**: first 2 000 images — clean, unmodified
# - **Incoming**: next 2 000 images (indices 2 000–3 999)
#   — classes 1, 4, 7 get progressively blurred

# %% tags=["remove_output"]

# Wrap the incoming slice with degradation for classes 1, 4, 7
incoming_maite = OrderedView(mnist_maite_raw, _order[2000:4000])
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
                img_arr = np.transpose(np.asarray(img), (1, 2, 0))  # CHW -> HWC
                axes[row, col].imshow(img_arr)
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
from dataeval_flow.config import (
    DatasetProtocolConfig,
    DriftMonitoringTaskConfig,
    DriftMonitoringWorkflowConfig,
    FlattenExtractorConfig,
    PipelineConfig,
    SelectionConfig,
    SelectionStep,
    SourceConfig,
)
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.drift.params import ChunkingConfig, DriftDetectorKNeighbors, DriftHealthThresholds

# --- Datasets (in-memory via DatasetProtocolConfig) ---
ref_config = DatasetProtocolConfig(
    name="reference",
    format="maite",
    dataset=mnist_maite,  # full 4k — selection will subset it
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

# --- Sources ---
ref_source_config = SourceConfig(name="reference_2k", dataset="reference", selection="ref-first-2k")
inc_source_config = SourceConfig(name="incoming_2k", dataset="incoming")

# --- Extractors ---
extractor_config = FlattenExtractorConfig(
    name="flatten",
    batch_size=64,
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
    name="mnist-overall-drift",
    workflow="overall-drift",
    sources=["reference_2k", "incoming_2k"],
    extractor="flatten",
)

overall_config = PipelineConfig(
    datasets=[ref_config, incoming_config],
    selections=[ref_selection],
    sources=[ref_source_config, inc_source_config],
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
# The K-Neighbors should flag drift in the later chunks — the chunked
# view shows *when* drift started, but not *which classes* are responsible.

# %%
print(overall_result.report())

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

classwise_task = DriftMonitoringTaskConfig(
    name="mnist-classwise-drift",
    workflow="classwise-drift",
    sources=["reference_2k", "incoming_2k"],
    extractor="flatten",
)

classwise_config = PipelineConfig(
    datasets=[ref_config, incoming_config],
    selections=[ref_selection],
    sources=[ref_source_config, inc_source_config],
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
# The report now includes a **classwise pivot table**: rows are classes (0–9),
# columns are detectors. Only classes 1, 4, and 7 should show drift — the
# others remain clean.

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
# - **Materialize and load datasets** by writing a HuggingFace dataset to an
#   ImageFolder layout and loading it with datamaite, then passing the result
#   to `DatasetProtocolConfig`
# - **Subset datasets** using `SelectionConfig` with `Indices` — no additional
#   disk I/O needed beyond the initial load
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

# %% [markdown]
# ## Related guides
#
# - **Concept** — [Distribution shift](../concepts/DistributionShift.md):
#   the overall-then-classwise diagnostic pattern and what each detector measures.
# - **Tutorial** — [Monitor incoming data for drift](drift_monitoring): the overall
#   drift check this tutorial drills into.
# - **How-to: Run workflows in containers** — [Containerized workflows](../how_to/containerized_workflows.md)
#   to run the two-phase approach on a schedule from a container.
