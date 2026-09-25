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
# # Detect out-of-distribution samples
#
# Find individual samples in incoming data that fall outside the reference
# distribution using the config-driven `ood-detection` workflow.

# %% [markdown]
# **Target audience**: You are a T&E engineer who needs to flag individual incoming
# samples that fall outside the operational reference distribution.
#
# **Workflow role**: OOD detection complements {doc}`Monitor incoming data for drift <drift_monitoring>`.
# While drift monitoring evaluates aggregate distribution shift, OOD detection
# identifies anomalous individual samples for quarantine or routing before model
# inference. See [Distribution shift](../concepts/DistributionShift.md) for conceptual
# details.

# %% [markdown]
# ## What you will do
#
# - Load MilitaryVehicles as a MAITE dataset using datamaite.
# - Synthesize incoming test data containing misaligned imagery: ship images inserted under vehicle labels.
# - Configure the `ood-detection` workflow with K-Neighbors and Domain Classifier detectors using ResNet-18 embeddings.
# - Enable metadata insights to identify factors correlated with OOD status.
# - Inspect the OOD report, score distributions, and flagged samples.
# - Evaluate detector performance on Gaussian noise sensor corruption.

# %% [markdown]
# ## What you will learn
#
# - How to configure and execute the `ood-detection` workflow with `run_task()`.
# - How K-Neighbors (distance-based) and Domain Classifier (LightGBM-based) detectors operate.
# - How feature representations influence OOD boundaries.
# - How metadata insights (`factor_deviation`, `factor_predictors`) explain OOD flags.
# - How to pass in-memory datasets via `DatasetProtocolConfig`.
# - How to interpret per-sample OOD scores and evaluation reports.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`).
# - Install `maite-datasets[datamaite]` to access MilitaryVehicles and Ships.
# - Install `torch` and `torchvision` for ResNet-18 feature extraction.
# - Ensure network access for initial dataset downloads.

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the dataset
#
# [MilitaryVehicles](https://huggingface.co/datasets/leibnitz-lab/military_vehicles)
# contains 9,444 images across 24 vehicle types. Setting `as_datamaite=True` writes
# the dataset in class-per-directory ImageFolder format.
#
# You will apply a seeded shuffle to draw reference and incoming samples evenly across
# all vehicle classes.

# %% tags=["remove_output"]
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from dataeval.data import Indices, Limit, Shuffle, View
from datamaite import load_ic
from maite_datasets.image_classification import MilitaryVehicles, Ships
from numpy.typing import NDArray

data_root = Path("./data")
MilitaryVehicles(root=data_root, image_set="base", download=True)
MilitaryVehicles(root=data_root, image_set="train", as_datamaite=True)

vehicles_raw = load_ic(data_root / "militaryvehicles_datamaite_train" / "train", dataset_format="huggingface_vision")

# The loader groups samples by class folder, so raw index order is alphabetical by class.
# `Shuffle` imposes one seeded sequence, which is what keeps the reference and incoming
# slices below from being drawn from different corners of the label space. `View` is the
# same machinery a `ViewConfig` drives from the pipeline config; `resolve_indices()` hands
# back the shuffled source indices so each slice can be cut with `Indices`.
_order = View(vehicles_raw, [Shuffle(seed=42), Limit(2500)]).resolve_indices()

# Reference: the first 2 000 frames, unmodified
ref_maite = View(vehicles_raw, [Indices(_order[:2000])])

print(f"Reference: {len(ref_maite)} frames")
print(f"Sample shape: image={ref_maite[0][0].shape}, dtype={ref_maite[0][0].dtype}")

# %% [markdown]
# ### Build the incoming dataset: right label, wrong image
#
# You will create an incoming dataset where a subset of samples contains satellite
# imagery of ships paired with vehicle labels.
#
# This setup simulates data ingestion errors where labels remain syntactically valid
# while underlying image content is out of distribution.

# %%
ships = Ships(root=data_root, download=True)
print(f"Foreign source: {len(ships)} satellite frames, shape {ships[0][0].shape}")


class MisalignedImages:
    """Keeps each datum's annotation and replaces only its pixels.

    Parameters
    ----------
    dataset
        A MAITE-compatible dataset returning (image, target, metadata) tuples.
    foreign
        The dataset supplying replacement imagery.
    start, count
        The contiguous run of positions whose images are replaced.
    """

    def __init__(self, dataset: Any, foreign: Any, start: int, count: int) -> None:
        self._dataset = dataset
        self._foreign = foreign
        self._span = range(start, start + count)
        self.metadata = {"id": "misaligned_images", "original_metadata": dataset.metadata}

    def __len__(self) -> int:
        return len(self._dataset)

    def swapped(self, index: int) -> bool:
        """Whether this position carries foreign imagery."""
        return index in self._span

    def __getitem__(self, index: int) -> tuple[NDArray[Any], Any, Mapping[str, Any]]:
        image, target, metadata = self._dataset[index]
        if index in self._span:
            image = np.asarray(self._foreign[(index - self._span.start) % len(self._foreign)][0])
        return image, target, metadata


# Incoming: the next 500 frames, with positions 200-299 carrying ships imagery
incoming_maite = View(vehicles_raw, [Indices(_order[2000:2500])])
incoming_dataset = MisalignedImages(incoming_maite, ships, start=200, count=100)

n_swapped = sum(1 for i in range(len(incoming_dataset)) if incoming_dataset.swapped(i))
print(f"Incoming: {len(incoming_dataset)} frames")
print(f"  In-distribution (vehicles):        {len(incoming_dataset) - n_swapped}")
print(f"  Out-of-distribution (ships under vehicle labels): {n_swapped}")

# %% [markdown]
# You can compare reference images with foreign images that inherited vehicle labels:

# %%
import matplotlib.pyplot as plt

index2label = vehicles_raw.metadata.get("index2label", {})
fig, axes = plt.subplots(2, 8, figsize=(12, 4))

for col in range(8):
    img_arr = np.transpose(np.asarray(ref_maite[col][0]), (1, 2, 0))  # CHW -> HWC
    axes[0, col].imshow(img_arr)
    axes[0, col].set_title("reference", fontsize=9)
    axes[0, col].axis("off")

swapped_indices = [i for i in range(len(incoming_dataset)) if incoming_dataset.swapped(i)]
for col in range(8):
    idx = swapped_indices[col * 4]
    image, target, _ = incoming_dataset[idx]
    img_arr = np.transpose(np.asarray(image), (1, 2, 0))  # CHW -> HWC
    axes[1, col].imshow(img_arr)
    axes[1, col].set_title(f"labeled\n{index2label[int(np.argmax(target))]}", fontsize=8)
    axes[1, col].axis("off")

fig.suptitle("Reference vehicles (top) vs foreign frames carrying vehicle labels (bottom)", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# To configure the `ood-detection` workflow, specify:
#
# 1. **Datasets**: A reference dataset and incoming test datasets.
# 2. **Extractor**: Pretrained models or algorithms to produce embeddings.
# 3. **Detectors**: Statistical or classifier-based OOD detectors.
# 4. **Health thresholds**: Percentage thresholds that trigger warnings.
#
# You will extract 512-dimensional features from a pretrained ResNet-18 model and evaluate
# two complementary detectors:
#
# | Detector | How it works | Strengths |
# |---|---|---|
# | **K-Neighbors** | Flags samples whose k nearest reference neighbors are unusually distant | Fast, non-parametric, effective in high dimensions |
# | **Domain Classifier** | Trains a LightGBM model to separate reference from incoming data | Captures complex non-linear decision boundaries |

# %%
import torch
import torchvision

from dataeval_flow import PipelineConfig
from dataeval_flow.config import (
    DatasetProtocolConfig,
    PreprocessingStep,
    PreprocessorConfig,
    SourceConfig,
)
from dataeval_flow.config.extractors import TorchExtractorConfig

# Cache weights locally for TorchExtractorConfig
model_dir = Path("./models")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "resnet18.pt"
if not model_path.exists():
    torch.save(torchvision.models.resnet18(weights="IMAGENET1K_V1").eval(), model_path)

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
    layer_name="avgpool",
    preprocessor="imagenet",
    batch_size=32,
)

# %% [markdown]
# ### Configure OOD detectors
#
# **K-Neighbors** uses cosine distance with k=10 to evaluate distance to reference
# neighbors. Setting `threshold_perc=99.0` establishes the detection boundary at the
# 99th percentile of reference baseline distances.
#
# **Domain Classifier** uses LightGBM with 3-fold cross-validation repeated 3 times.
# Samples consistently predicted as incoming data receive high OOD probabilities.
#
# Setting `metadata_insights=True` directs the workflow to analyze metadata factors
# that correlate with flagged OOD samples.

# %%
from dataeval_flow import run_task
from dataeval_flow.config import TaskConfig
from dataeval_flow.workflows.ood_detection import (
    OODDetectionConfig,
    OODDetectionHealthThresholds,
    OODDetectorDomainClassifier,
    OODDetectorKNeighbors,
)

task = TaskConfig(
    name="vehicles-ood-check",
    workflow="vehicles-ood",
    sources=["ref_src", "inc_src"],
    extractor="resnet18",
)

config = PipelineConfig(
    datasets=[ref_config, incoming_config],
    sources=[
        SourceConfig(name="ref_src", dataset="reference"),
        SourceConfig(name="inc_src", dataset="incoming"),
    ],
    preprocessors=[preprocessor_config],
    extractors=[extractor_config],
    workflows=[
        OODDetectionConfig(
            name="vehicles-ood",
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
            health_thresholds=OODDetectionHealthThresholds(
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
# Call `result.report()` to view OOD sample counts, detector summaries, and
# metadata factor correlations.

# %%
print(result.report())

# %% [markdown]
# ## Understanding the results
#
# You can evaluate detection performance by comparing flagged samples against
# ground truth and analyzing score distributions.

# %% [markdown]
# ### Per-detector summary
#
# Each detector scores incoming samples independently. Samples with scores
# exceeding reference thresholds are flagged as OOD.

# %%
raw = result.output.raw

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
# ### Benchmark detector recall and precision
#
# Because sample replacement positions are known, you can evaluate detector
# recall and precision directly against ground-truth labels.
#
# **Recall** is the share of planted frames a detector flagged; **precision** is the
# share of its flags that were actually planted.

# %%
truth = [incoming_dataset.swapped(i) for i in range(len(incoming_dataset))]
planted = sum(truth)

for method, det_result in raw.detectors.items():
    flagged = [sample["index"] for sample in det_result.get("samples", []) if sample.get("is_ood")]
    hits = sum(1 for i in flagged if truth[i])
    recall = hits / planted * 100 if planted else 0.0
    precision = hits / len(flagged) * 100 if flagged else 0.0
    print(
        f"{method:>18}: flagged {len(flagged):>3} of {planted} planted -- recall {recall:.0f}%, precision {precision:.0f}%"
    )

# %% [markdown]
# ### Visualize OOD scores
#
# You can plot score histograms to evaluate separation between in-distribution
# and out-of-distribution samples relative to the threshold.

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
# You can display samples with the highest OOD scores alongside borderline samples
# closest to the threshold.

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
# When metadata insights are enabled, the workflow computes correlations between
# metadata factors and OOD status. **Factor predictors** report mutual information
# with OOD flags. **Factor deviations** show per-factor metric deviations for individual
# OOD samples.

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
# ## A second failure: Corrupted imagery and detector differences
#
# You can evaluate how detectors perform when incoming frames undergo sensor
# degradation rather than semantic substitution. In this experiment, you will add
# heavy Gaussian noise to five selected vehicle classes.

# %%
NOISY_CLASSES = [1, 5, 11, 16, 23]


class NoisyClasses:
    """Adds heavy sensor noise to frames of the named classes."""

    def __init__(self, dataset: Any, classes: list[int], sigma: float = 60.0) -> None:
        self._dataset = dataset
        self._classes = set(classes)
        self._sigma = sigma
        self._rng = np.random.default_rng(0)
        self.metadata = {"id": "noisy_classes", "original_metadata": dataset.metadata}

    def __len__(self) -> int:
        return len(self._dataset)

    def corrupted(self, index: int) -> bool:
        return int(np.argmax(self._dataset[index][1])) in self._classes

    def __getitem__(self, index: int) -> tuple[NDArray[Any], Any, Mapping[str, Any]]:
        image, target, metadata = self._dataset[index]
        if int(np.argmax(target)) in self._classes:
            noise = self._rng.normal(0, self._sigma, np.asarray(image).shape)
            image = np.clip(np.asarray(image).astype(np.int16) + noise, 0, 255).astype(np.asarray(image).dtype)
        return image, target, metadata


noisy_dataset = NoisyClasses(View(vehicles_raw, [Indices(_order[2000:2500])]), NOISY_CLASSES)
print(f"Corrupting: {[index2label[c] for c in NOISY_CLASSES]}")

noisy_task = TaskConfig(
    name="vehicles-ood-noise",
    workflow="vehicles-ood",
    sources=["ref_src", "noisy_src"],
    extractor="resnet18",
)
noisy_config = config.model_copy(
    update={
        "datasets": [ref_config, DatasetProtocolConfig(name="noisy", format="maite", dataset=noisy_dataset)],
        "sources": [
            SourceConfig(name="ref_src", dataset="reference"),
            SourceConfig(name="noisy_src", dataset="noisy"),
        ],
        "tasks": [noisy_task],
    }
)

noisy_result = run_task(noisy_task, noisy_config, cache_dir=Path("./cache"))

# %%
noisy_truth = [noisy_dataset.corrupted(i) for i in range(len(noisy_dataset))]
noisy_planted = sum(noisy_truth)

for method, det_result in noisy_result.output.raw.detectors.items():
    flagged = [sample["index"] for sample in det_result.get("samples", []) if sample.get("is_ood")]
    hits = sum(1 for i in flagged if noisy_truth[i])
    recall = hits / noisy_planted * 100 if noisy_planted else 0.0
    precision = hits / len(flagged) * 100 if flagged else 0.0
    print(
        f"{method:>18}: flagged {len(flagged):>3} of {noisy_planted} corrupted -- recall {recall:.0f}%, precision {precision:.0f}%"
    )

# %% [markdown]
# ### Evaluating detector complementarity
#
# On misaligned images, both detectors showed high recall. Under sensor noise, the
# Domain Classifier maintains high recall, whereas K-Neighbors detects fewer corrupted
# samples.
#
# K-Neighbors flags samples that fall outside reference embedding clusters. When
# noise moves samples along dimensions where reference embeddings already have broad
# variance, K-Neighbors may not exceed distance thresholds. In contrast, the Domain
# Classifier trains a supervised model to separate reference and incoming distributions,
# capturing consistent feature patterns induced by noise.
#
# You should combine multiple detectors in operational workflows to detect diverse
# anomaly types.

# %% [markdown]
# ## Results Exploration: Export results
#
# You can export raw detector outputs, per-sample scores, and metadata insights
# to JSON format.

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")
print(json_str[:600] + "\n...")

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Configure the `ood-detection` workflow with K-Neighbors and Domain Classifier detectors.
# - Extract embedding features using pretrained ResNet-18 models.
# - Execute OOD detection workflows and inspect formatted reports.
# - Score detector precision and recall against known anomaly labels.
# - Visualize per-sample OOD score distributions and identify boundary samples.
# - Use metadata insights to identify factors correlated with OOD status.
# - Export structured OOD findings to JSON format.

# %% [markdown]
# ## Next steps
#
# - **Threshold tuning**: Adjust `threshold_perc` to tune the balance between detection
#   recall and false-positive rates.
# - **Integrated pipelines**: Combine `drift-monitoring` and `ood-detection` tasks in
#   a single pipeline configuration to track both batch drift and individual anomalies.

# %% [markdown]
# ## Related guides
#
# - **Concept**: [Distribution shift](../concepts/DistributionShift.md) explains
#   relationships between OOD detection and drift monitoring.
# - **How-to**: [Read evaluation outputs](../how_to/read_evaluation_outputs.md) details
#   how to parse per-sample OOD scores and export envelopes.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md) explains
#   how to schedule OOD monitoring tasks in Docker.
# - **Guide**: [Use an ONNX model for embeddings](onnx_embeddings) shows how to configure
#   ONNX models for feature extraction.
