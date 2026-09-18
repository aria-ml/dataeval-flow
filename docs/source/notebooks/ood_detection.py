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
# **Who this is for** — T&E engineers who need to flag *individual* incoming samples
# that fall outside the distribution a model was evaluated on.
#
# **Where this fits** — OOD detection is the sample-level complement to
# [drift monitoring](drift_monitoring) in the operational T&E workflow: drift answers
# "has the distribution changed?" while OOD answers "which specific samples don't
# belong?" — letting you quarantine or route anomalous inputs before they reach the
# model. See the [Distribution shift](../concepts/DistributionShift.md) concept page
# for how the two relate.

# %% [markdown]
# ## What you'll do
#
# - Load MilitaryVehicles as a MAITE dataset via datamaite
# - Synthesize **incoming data** in which a run of frames carries the **right label and
#   the wrong image** — imagery from a different collection entirely, as though a folder
#   had been dropped in the wrong place
# - Configure the `ood-detection` workflow with **K-Neighbors** and
#   **Domain Classifier** detectors, embedding with a pretrained **ResNet-18**
# - Enable **metadata insights** to explain *why* flagged samples are OOD
# - Review the OOD report and inspect per-sample results
# - Run a second pass on **corrupted** imagery, where the two detectors disagree

# %% [markdown]
# ## What you'll learn
#
# - How to configure and run the `ood-detection` workflow via `run_task()`
# - The difference between **K-Neighbors** (distance-based) and
#   **Domain Classifier** (LightGBM-based) OOD detectors — including a case where one
#   finds almost nothing the other finds easily
# - Why "out of distribution" is a property of the **representation**, not of the image
# - How **metadata insights** (`factor_deviation`, `factor_predictors`) explain
#   which metadata factors correlate with OOD status
# - How to use `DatasetProtocolConfig` to pass in-memory datasets directly
# - How to read the built-in OOD report and drill into per-sample scores

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`)
# - `maite-datasets[datamaite]` (for MilitaryVehicles and Ships)
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
# reads directly.
#
# The loader returns samples grouped by class folder. We impose one fixed shuffled order
# so the reference and incoming slices below each see a mix of classes rather than a few
# alphabetically adjacent ones.

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
# The out-of-distribution samples here are not corrupted vehicles. They are frames from a
# **different collection entirely** — satellite imagery of ships — carrying the vehicle
# labels that belonged to the frames they displaced.
#
# That models a specific and unglamorous failure: a folder lands in the wrong place, an
# index goes out of step, a sync half-completes. Every annotation is still correct,
# internally consistent, and in its right proportions. Nothing that inspects labels can
# see the problem, because nothing is wrong with the labels — the images beneath them
# simply are not what they claim to be.
#
# It is also why the labels are left untouched below. Had we invented a class for the
# foreign frames, class label would predict OOD status perfectly and the metadata
# insights at the end would be reporting our own construction back to us.

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
        The contiguous run of positions whose images are replaced — contiguous because
        that is what a misplaced folder looks like, and because it makes the planted
        samples easy to check against what the detectors flag.
    """

    def __init__(self, dataset: Any, foreign: Any, start: int, count: int) -> None:
        self._dataset = dataset
        self._foreign = foreign
        self._span = range(start, start + count)
        self.metadata = {"id": "misaligned_images", "original_metadata": dataset.metadata}

    def __len__(self) -> int:
        return len(self._dataset)

    def swapped(self, index: int) -> bool:
        """Whether this position carries foreign imagery — the ground truth to score against."""
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
# Let's look at what slipped in. The top row is reference imagery; the bottom row is what
# the incoming stream carries at the swapped positions — with the vehicle label each frame
# inherited printed above it.

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
    axes[1, col].set_title(f"labelled\n{index2label[int(np.argmax(target))]}", fontsize=8)
    axes[1, col].axis("off")

fig.suptitle("Reference vehicles (top) vs foreign frames carrying vehicle labels (bottom)", fontsize=12)
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
# The extractor is the part worth deciding deliberately, because **"out of distribution"
# is a property of the representation, not of the image**. A `flatten` extractor compares
# raw pixels, which works on MNIST-shaped data and misleads on natural imagery: measured
# on this dataset, colour-inverting a frame — the most violent thing you can do to it in
# pixel space — moves a pretrained network's features so little that a distance-based
# detector finds 3% of the planted samples. Networks trained with colour augmentation are
# built to ignore exactly that.
#
# We use a pretrained **ResNet-18** and take its 512-dimensional `avgpool` features, with
# ResNet's own preprocessing in front. Two complementary detectors run on top:
#
# | Detector | How it works | Strengths |
# |---|---|---|
# | **K-Neighbors** | Flags samples whose k nearest reference neighbors are unusually far | Fast, non-parametric, works in high dimensions |
# | **Domain Classifier** | Trains a LightGBM to distinguish ref from test; flags easily-separated samples | Powerful with many features, captures complex boundaries |

# %%
import torch
import torchvision

from dataeval_flow.config import (
    DatasetProtocolConfig,
    PipelineConfig,
    PreprocessorConfig,
    SourceConfig,
    TorchExtractorConfig,
)
from dataeval_flow.preprocessing import PreprocessingStep

# torchvision fetches the weights once (~45 MB) into its own cache; saving the model
# beside the notebook is what `TorchExtractorConfig` loads.
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
# ResNet's own preprocessing. It also settles the mismatch in frame sizes: vehicle frames
# are around 180x280 and the ships imagery is 80x80, and both arrive at the network as
# 224x224.
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
from dataeval_flow.config import OODDetectionTaskConfig, OODDetectionWorkflowConfig
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.ood.params import (
    OODDetectorDomainClassifier,
    OODDetectorKNeighbors,
    OODHealthThresholds,
)

task = OODDetectionTaskConfig(
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
        OODDetectionWorkflowConfig(
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
# ### Did they find what we planted?
#
# We know exactly which frames carry foreign imagery, so we can score the detectors
# rather than take their word for it. On your own data you will not have this luxury —
# which is the reason to establish on data you *have* labelled whether a detector finds
# the failures you care about.
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
# ## A second failure: corrupted imagery, and where the detectors disagree
#
# The misalignment above is caught cleanly by both detectors, which makes it a poor guide
# to choosing between them. So run a different failure through the same pipeline: the same
# frames, from the same collection, with heavy sensor noise on five of the twenty-four
# classes. Nothing foreign has arrived — the imagery is simply degraded.

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

noisy_task = OODDetectionTaskConfig(
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

for method, det_result in noisy_result.data.raw.detectors.items():
    flagged = [sample["index"] for sample in det_result.get("samples", []) if sample.get("is_ood")]
    hits = sum(1 for i in flagged if noisy_truth[i])
    recall = hits / noisy_planted * 100 if noisy_planted else 0.0
    precision = hits / len(flagged) * 100 if flagged else 0.0
    print(
        f"{method:>18}: flagged {len(flagged):>3} of {noisy_planted} corrupted -- recall {recall:.0f}%, precision {precision:.0f}%"
    )

# %% [markdown]
# ### The detectors are not interchangeable
#
# On the misaligned frames both detectors were near-perfect. On noise they part company:
# the Domain Classifier recovers nearly all of it, while K-Neighbors finds under half — at
# high precision, so what it does flag is right, there is simply much it never sees.
#
# The reason is in how each one decides. K-Neighbors asks whether a sample sits further
# from its reference neighbours than reference samples sit from each other, and noise
# moves a frame in a direction the reference set already spreads along, so many corrupted
# frames stay inside the existing spread. The Domain Classifier instead *learns* whatever
# separates the two sets, and a consistent noise signature is exactly the kind of cue a
# gradient-boosted model picks up on.
#
# Neither is the better detector. They fail differently, which is the argument for running
# both and reading the aggregate finding: agreement is evidence, and disagreement tells you
# which kind of difference you are looking at.

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
#   Classifier detectors, embedding with a pretrained ResNet-18
# - **Run** the workflow and read the OOD report with per-detector summaries
# - **Score the detectors against ground truth**, rather than reading a count of flags
#   and assuming they landed on the right samples
# - **Inspect** per-sample OOD scores and visualize the score distributions
# - **Use metadata insights** to understand which factors correlate with OOD
#   status
# - **Export** structured JSON results for downstream automation
#
# Two things are worth carrying away beyond the mechanics.
#
# **"Out of distribution" is a property of the representation.** The same frame can be
# wildly anomalous in pixel space and unremarkable to a pretrained network, or the
# reverse. Choosing the extractor is choosing what the question means.
#
# **The detectors fail differently.** Both caught foreign imagery almost perfectly; only
# one caught sensor noise. Running both and reading where they agree is worth more than
# picking the one that scored best on somebody else's data.
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
# - **Drift + OOD pipeline** — Combine `drift-monitoring` and `ood-detection`
#   workflows in the same pipeline config to detect both distribution-level
#   shifts and individual outliers

# %% [markdown]
# ## Related guides
#
# - **Concept** — [Distribution shift](../concepts/DistributionShift.md):
#   how OOD detection relates to drift monitoring and what each detector measures.
# - **How-to: Read evaluation outputs** — [Read evaluation outputs](../how_to/read_evaluation_outputs.md)
#   to interpret per-sample OOD scores and pull the flagged images out of the result for review.
# - **How-to: Run workflows in containers** — [Containerized workflows](../how_to/containerized_workflows.md)
#   to run OOD detection on a schedule against live data pipelines from a container.
# - **How-to: Use an ONNX model for embeddings** — [ONNX embeddings](onnx_embeddings)
#   to use a pretrained model (e.g. ResNet) with preprocessing for richer embeddings.
