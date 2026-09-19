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
# # Prioritize unlabeled data for labeling
#
# Use the `data-prioritization` workflow to rank incoming data by what to
# label next, given an already-labeled reference dataset and a trained model.

# %% [markdown]
# **Target audience**: You are a model developer or data scientist managing an
# active-learning or labeling pipeline who needs to prioritize unlabeled data
# under budget constraints.
#
# **Workflow role**: Prioritization operates during data acquisition. Given
# reference data and a trained model, it ranks unlabeled inputs to target novel
# or challenging cases before retraining. See [Prioritization](../concepts/Prioritization.md)
# for background on ranking policies.

# %% [markdown]
# ## What you will do
#
# - Load MilitaryVehicles and split it into labeled reference and unlabeled pools.
# - Train a classifier on the labeled pool covering 20 vehicle types, holding out 4 Air Defense systems.
# - Inject corrupted and duplicate frames into the unlabeled pool to test automated pruning.
# - Execute the `data-prioritization` workflow to rank unlabeled samples.
# - Measure ranking efficiency against random selection baselines and model uncertainty sampling.

# %% [markdown]
# ## What you will learn
#
# - How to configure and execute `data-prioritization` with `run_task()`.
# - How reference datasets and extractor representations direct prioritization.
# - How `hard_first` ordering prioritizes out-of-distribution or challenging samples.
# - How to benchmark prioritization gains against random sampling baselines.
# - Why model uncertainty sampling fails to detect novel classes and how distance-based prioritization resolves this.
# - How pre-prioritization pruning removes outliers and duplicates from the ranking pool.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `torch`) and `torchvision`.
# - Install `maite-datasets[datamaite]` to download and export MilitaryVehicles.
# - Ensure network access for the initial dataset download.

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: a labeled pool and an unlabeled one
#
# [MilitaryVehicles](https://huggingface.co/datasets/leibnitz-lab/military_vehicles)
# contains 9,444 images across 24 vehicle types. The dataset includes coarse categories
# such as tanks, BMPs, BTRs, self-propelled artillery, and air defense systems.
#
# In this tutorial, you will hold out the four Air Defense systems (`30N6E`, `Iskander`,
# `Pantsir-S1`, `Rs-24`) to simulate encountering an unannotated operational category:
#
# - **Labeled pool**: 4,000 frames drawn from the 20 known vehicle types.
# - **Unlabeled pool**: 1,500 frames drawn from remaining data, containing leftover known types and held-out Air Defense systems.

# %% tags=["remove_output"]
from pathlib import Path

import numpy as np
from dataeval.data import ClassFilter, Indices, Limit, Shuffle, View
from datamaite import load_ic
from maite_datasets.image_classification import MilitaryVehicles

from dataeval_flow.preprocessing import PreprocessingStep

data_root = Path("./data")
MilitaryVehicles(root=data_root, image_set="base", download=True)
MilitaryVehicles(root=data_root, image_set="train", as_datamaite=True)

vehicles = load_ic(data_root / "militaryvehicles_datamaite_train" / "train", dataset_format="huggingface_vision")
index2label = vehicles.metadata.get("index2label", {})

HELD_OUT_NAMES = {"30N6E", "Iskander", "Pantsir-S1", "Rs-24"}
held_out = {i for i, name in index2label.items() if name in HELD_OUT_NAMES}
known_classes = [i for i in sorted(index2label) if i not in held_out]

# Both pools are constructed using View operations.
# You build them in Python here because the classification model trains on the
# labeled pool directly before pipeline execution.
#
# Labeled pool: The twenty known classes only.
labeled_dataset = View(vehicles, [ClassFilter(known_classes), Shuffle(seed=42), Limit(4000)])
labeled_indices = labeled_dataset.resolve_indices()

# Unlabeled pool: The remaining frames, containing Air Defense systems and leftover known types.
pool_dataset = View(vehicles, [Indices(labeled_indices, exclude=True), Shuffle(seed=7), Limit(1500)])

pool_labels = np.array([int(np.argmax(pool_dataset[i][1])) for i in range(len(pool_dataset))])
n_novel = int(np.isin(pool_labels, sorted(held_out)).sum())
print(f"Held out (Air Defense): {sorted(index2label[i] for i in held_out)}")
print(f"Labeled pool:   {len(labeled_dataset)} frames, {len(known_classes)} types")
print(
    f"Unlabeled pool: {len(pool_dataset)} frames, {n_novel} of them Air Defense "
    f"({n_novel / len(pool_dataset) * 100:.0f}%)"
)

# %% [markdown]
# ### Inject corrupted and duplicate images into the test data
#
# You can simulate incoming data corruption using an in-memory wrapper that
# corrupts a slice of the test set on the fly:
#
# | Indices   | Corruption            |
# |-----------|-----------------------|
# | 900–919   | Duplicates of 200–219 |
# | 920–939   | Gaussian blur         |
# | 940–959   | Random noise          |
# | 960–979   | Overexposure          |
# | 980–999   | Underexposure         |

# %% tags=["remove_output"]
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter


class CorruptedTestDataset:
    """Wraps a MAITE dataset and injects synthetic corruptions into indices 900-999.

    Labels are left untouched throughout, so ground truth stays available for
    verification even though the pixel content at those indices is corrupted.
    """

    _DUPLICATE_OFFSET = 700  # maps index i in [900, 920) to source index i - 700

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[NDArray[Any], Any, Mapping[str, Any]]:
        if 900 <= index < 920:
            image, _, _ = self._dataset[index - self._DUPLICATE_OFFSET]
            _, target, metadata = self._dataset[index]
            return np.array(image, copy=True), target, metadata

        image, target, metadata = self._dataset[index]

        if 920 <= index < 940:
            image = self._blur(image)
        elif 940 <= index < 960:
            image = self._add_noise(image, seed=index)
        elif 960 <= index < 980:
            image = self._adjust_exposure(image, 100)
        elif 980 <= index < 1000:
            image = self._adjust_exposure(image, -200, halve=True)

        return image, target, metadata

    @staticmethod
    def _blur(image: NDArray[Any]) -> NDArray[Any]:
        hwc = np.transpose(image, (1, 2, 0))  # CHW -> HWC
        blurred = Image.fromarray(hwc, mode="RGB").filter(ImageFilter.GaussianBlur(radius=3))
        return np.transpose(np.array(blurred, dtype=image.dtype), (2, 0, 1))  # back to CHW

    @staticmethod
    def _add_noise(image: NDArray[Any], *, seed: int) -> NDArray[Any]:
        arr = image.astype(np.int16)
        noise = np.random.default_rng(seed).integers(-80, 80, arr.shape, dtype=np.int16)
        return np.clip(arr + noise, 0, 255).astype(image.dtype)

    @staticmethod
    def _adjust_exposure(image: NDArray[Any], delta: int, *, halve: bool = False) -> NDArray[Any]:
        arr = np.clip(image.astype(np.int16) + delta, 0, 255)
        if halve:
            arr = arr // 2
        return arr.astype(image.dtype)


test_dataset = CorruptedTestDataset(pool_dataset)

print(
    "Test data corruption wrapper ready: 20 exact duplicates (900-919), "
    "20 blurred (920-939), 20 noisy (940-959), "
    "20 bright (960-979), 20 dark (980-999)"
)

# %% [markdown]
# ## Step 1: Train a classifier on the labeled pool
#
# You should use a feature extractor trained on your specific label space so that
# embeddings reflect domain characteristics.
#
# You will fine-tune a linear head on top of a frozen pretrained ResNet-18 trunk.
# You will extract embeddings from the 128-dimensional `embed` layer connecting trunk
# and classification head.

# %% tags=["remove_output"]
import random

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision

random.seed(42)
torch.manual_seed(42)


class VehicleNet(nn.Module):
    """A frozen pretrained trunk, a learned embedding, and a head over known classes.

    ``embed`` bundles Linear + ReLU so a forward hook captures the activated output:
    the representation the classifier was trained on.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        trunk = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        trunk.fc = nn.Identity()
        self.trunk = trunk
        self.embed = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(self.trunk(x)))


model = VehicleNet(len(known_classes))
for parameter in model.trunk.parameters():
    parameter.requires_grad = False
model.trunk.eval()

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _trunk_features(dataset: Any) -> tuple[torch.Tensor, list[int]]:
    """Run the frozen trunk once over a dataset to generate training features for the head."""
    out: list[torch.Tensor] = []
    labels: list[int] = []
    with torch.no_grad():
        for start in range(0, len(dataset), 64):
            batch = []
            for i in range(start, min(start + 64, len(dataset))):
                image, target, _ = dataset[i]
                labels.append(int(np.argmax(target)))
                img = torch.tensor(np.asarray(image), dtype=torch.float32) / 255.0
                img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
                batch.append((img.squeeze(0) - _MEAN) / _STD)
            out.append(model.trunk(torch.stack(batch)))
    return torch.cat(out), labels


# %% tags=["remove_output"]
_remap = {c: k for k, c in enumerate(known_classes)}
features, labeled_labels = _trunk_features(labeled_dataset)
targets = torch.tensor([_remap[c] for c in labeled_labels], dtype=torch.long)

head = nn.Sequential(model.embed, model.classifier)
optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
for _epoch in range(40):
    optimizer.zero_grad()
    loss = F.cross_entropy(head(features), targets)
    loss.backward()
    optimizer.step()
model.eval()

accuracy = (head(features).argmax(1) == targets).float().mean().item()
print(f"Head trained: loss={loss.item():.3f}, accuracy on the labeled pool={accuracy * 100:.1f}%")

# %% [markdown]
# ### Save the trained model to disk
#
# Save the model to disk so `TorchExtractorConfig` can load it and hook the
# `embed` layer.

# %% tags=["remove_output"]
model_path = Path("./models/vehiclenet.pt")
model_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, str(model_path))
print(f"Model saved to {model_path}")

# %% [markdown]
# ## Step 2: Build the prioritization workflow configuration
#
# You will configure:
#
# - **Reference**: Labeled pool (4,000 frames, 20 vehicle types).
# - **Incoming data**: Unlabeled pool (1,500 frames, including held-out Air Defense and corrupted samples).
# - **Extractor**: Trained VehicleNet hooking the `embed` layer for 128-dimensional embeddings.
# - **Method**: KNN with `hard_first` ordering to rank samples farthest from reference neighbors.
# - **Pruning**: Outlier and duplicate detection to filter invalid samples before ranking.
# - **Mode**: `preparatory` mode to output explicit clean and flagged index lists.

# %%
from dataeval_flow.config import (
    DatasetProtocolConfig,
    PipelineConfig,
    PreprocessorConfig,
    SourceConfig,
)
from dataeval_flow.config.schemas import (
    DataPrioritizationTaskConfig,
    DataPrioritizationWorkflowConfig,
    TorchExtractorConfig,
)

ref_dataset = labeled_dataset

from dataeval_flow.workflows.prioritization.params import CleaningConfig

workflow = DataPrioritizationWorkflowConfig(
    name="vehicles_prioritize",
    method="knn",
    k=5,
    order="hard_first",
    policy="difficulty",
    mode="preparatory",
    cleaning=CleaningConfig(
        outlier_method="adaptive",
        outlier_flags=["dimension", "pixel", "visual"],
        outlier_threshold=3.0,  # lower than default 3.5 to catch subtler corruptions
        duplicate_exact_only=True,
    ),
)

task = DataPrioritizationTaskConfig(
    name="prioritize_pool",
    workflow="vehicles_prioritize",
    sources=["ref_src", "test_src"],
    extractor="cnn_extractor",
)

# %% [markdown]
# ### Assemble the pipeline config
#
# The first source specifies the reference dataset; subsequent sources specify
# unlabeled data to prioritize. `TorchExtractorConfig` points to the saved model
# and hooks the `embed` layer.
#
# You should include a preprocessor that resizes frames to 224x224 and normalizes
# pixel values using ImageNet statistics to match the model training conditions.

# %%
config = PipelineConfig(
    datasets=[
        DatasetProtocolConfig(name="ref_ds", dataset=ref_dataset),
        DatasetProtocolConfig(name="test_ds", dataset=test_dataset),
    ],
    preprocessors=[
        PreprocessorConfig(
            name="imagenet",
            steps=[
                PreprocessingStep(step="Resize", params={"size": [224, 224], "antialias": True}),
                PreprocessingStep(step="ToDtype", params={"dtype": "float32", "scale": True}),
                PreprocessingStep(
                    step="Normalize",
                    params={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                ),
            ],
        ),
    ],
    sources=[
        SourceConfig(name="ref_src", dataset="ref_ds"),
        SourceConfig(name="test_src", dataset="test_ds"),
    ],
    extractors=[
        TorchExtractorConfig(
            name="cnn_extractor",
            model_path=str(model_path),
            layer_name="embed",
            device="cpu",
            preprocessor="imagenet",
            batch_size=32,
        ),
    ],
    workflows=[workflow],
    tasks=[task],
)

# %% [markdown]
# ## Step 3: Run the prioritization workflow
#
# Execute `run_task()` to prune outliers and rank unlabeled samples.

# %%
from dataeval_flow.workflow import run_task

result = run_task(task, config, cache_dir=Path("./cache"))

# %% tags=["remove_cell"]
if not result.success:
    print(f"Workflow failed: {result.errors}")
assert result.success

# %% [markdown]
# ### Prioritization report
#
# The report summarizes how many items were ranked and by what method.

# %%
print(result.report())

# %% [markdown]
# ## Step 4: Inspect what pruning removed
#
# The pruning phase detects outliers and duplicates in both reference and incoming
# data. In `preparatory` mode, `result.metadata` records clean indices and dropped
# sample counts. You can verify whether pruning removed the injected corrupted samples.

# %%
raw = result.data.raw
meta = result.metadata

print(f"Pruning enabled: {meta.cleaning_enabled}")
print(f"Items removed by pruning: {meta.items_removed_by_cleaning}")

if raw.cleaning_summary is not None:
    cs = raw.cleaning_summary
    print(f"  Outliers flagged:    {cs['outliers_flagged']}")
    print(f"  Duplicates flagged:  {cs['duplicates_flagged']}")
    print(f"  Total removed:       {cs['total_removed']}")

# %%
# Which pool indices were pruned?
all_test_indices = set(range(len(test_dataset)))
clean_test_indices = set(meta.per_source_clean_indices.get("test_src", []))
pruned_indices = sorted(all_test_indices - clean_test_indices)

print(f"Pool frames: {len(all_test_indices)} total, {len(clean_test_indices)} clean, {len(pruned_indices)} pruned")

# Check overlap with known corrupted ranges
corrupted_ranges = {
    "duplicates (900-919)": set(range(900, 920)),
    "blurred (920-939)": set(range(920, 940)),
    "noisy (940-959)": set(range(940, 960)),
    "bright (960-979)": set(range(960, 980)),
    "dark (980-999)": set(range(980, 1000)),
}
pruned_set = set(pruned_indices)
print("\nOverlap with injected corruptions:")
for name, indices in corrupted_ranges.items():
    overlap = pruned_set & indices
    print(f"  {name}: {len(overlap)}/{len(indices)} pruned")

planted = set(range(900, 1000)) | set(range(200, 220))
other_pruned = pruned_set - planted
if other_pruned:
    print(f"  Other frames pruned: {len(other_pruned)} of {len(all_test_indices) - len(planted)}")

# %% [markdown]
# Pruned frames are excluded from the final prioritization ranking so labeling
# budgets target clean data.

# %% [markdown]
# ## Step 5: Does the ranking beat picking at random?
#
# You should benchmark the prioritized ranking against random selection. In this
# evaluation, you measure the proportion of held-out Air Defense samples retrieved
# across various labeling budgets.


# %%
prioritized = result.data.raw.prioritizations[0]
top_indices = prioritized["prioritized_indices"]

# `pool_labels` was read straight off the view; the corruption wrapper alters pixels only,
# so a frame's ground-truth label is the same whether or not it was corrupted.
is_novel = np.isin(pool_labels[top_indices], sorted(held_out))
baseline_pct = 100.0 * is_novel.mean()

print(f"Baseline (share of the ranked set): {baseline_pct:.0f}% Air Defense\n")
print(f"{'Budget':>8}  {'Air Defense':>12}  {'vs random':>10}")
for n in (50, 100, 200, 500):
    pct = 100.0 * is_novel[:n].mean()
    print(f"{n:>8}  {pct:>11.0f}%  {pct / baseline_pct:>9.2f}x")

# %% [markdown]
# The prioritized ranking exceeds the random baseline across budgets. The relative
# advantage is greatest at small budgets (such as the top 50 to 100 samples) and
# gradually converges toward the baseline as larger fractions of the pool are labeled.

# %%
cumulative = np.cumsum(is_novel)
n_items = np.arange(1, len(cumulative) + 1)
novel_pct = 100.0 * cumulative / n_items

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(n_items, novel_pct, linewidth=2, label="Prioritized (hard_first)")
    ax.axhline(baseline_pct, color="gray", linestyle="--", linewidth=1, label=f"Random baseline ({baseline_pct:.0f}%)")
    ax.set_xlabel("Labeling budget (top-N frames)")
    ax.set_ylabel("% Air Defense")
    ax.set_title("Novel-category concentration across the priority ranking")
    ax.set_xlim(1, len(cumulative))
    ax.set_ylim(0, max(100, novel_pct[:50].max() * 1.2))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
except ImportError:
    for n in (50, 100, 200, 500, len(cumulative)):
        print(f"Top {n:4d}: {novel_pct[n - 1]:.0f}% novel")

# %% [markdown]
# ### What about ranking by model uncertainty?
#
# You can evaluate entropy-based uncertainty sampling against distance-based prioritization.
# Compute the entropy of predicted class probabilities across the unlabeled pool:


# %%
def _normalized(idx: int) -> torch.Tensor:
    """One pool frame, in the shape the model expects."""
    img = torch.tensor(np.asarray(test_dataset[idx][0]), dtype=torch.float32) / 255.0
    img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    return (img.squeeze(0) - _MEAN) / _STD


# Batched, because the whole pool at 224x224 does not fit in memory at once.
_probs = []
with torch.no_grad():
    for start in range(0, len(top_indices), 32):
        batch = torch.stack([_normalized(i) for i in top_indices[start : start + 32]])
        _probs.append(torch.softmax(model(batch), dim=1))
probabilities = torch.cat(_probs).numpy()

entropy = -(probabilities * np.log(probabilities + 1e-9)).sum(axis=1)
by_entropy = np.argsort(-entropy)
novel_by_entropy = is_novel[by_entropy]

print(f"{'Budget':>8}  {'hard_first':>11}  {'by entropy':>11}  {'baseline':>9}")
for n in (50, 100, 200, 500):
    print(
        f"{n:>8}  {100.0 * is_novel[:n].mean():>10.0f}%  "
        f"{100.0 * novel_by_entropy[:n].mean():>10.0f}%  {baseline_pct:>8.0f}%"
    )

# %% [markdown]
# Uncertainty sampling fails to prioritize novel classes because classifiers assign
# confident predictions to unfamiliar inputs matching known feature patterns.
# Uncertainty sampling identifies samples near known decision boundaries, whereas
# distance-based prioritization identifies samples outside known clusters.

# %% [markdown]
# ## Step 6: Visualize the top-ranked images (optional)
#
# You can plot the top-ranked prioritized samples to inspect flagged imagery:

# %%
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        idx = top_indices[i]
        img, target, _ = test_dataset[idx]
        img_arr = np.transpose(np.asarray(img), (1, 2, 0))  # CHW -> HWC
        label = int(np.argmax(target))
        ax.imshow(img_arr)
        ax.set_title(f"#{i + 1} idx={idx}\nlabel={label}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Top 10 prioritized images (hard_first)", fontsize=13)
    plt.tight_layout()
    plt.show()
except ImportError:
    print("Install matplotlib to visualize: pip install matplotlib")

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Train a custom embedding extractor on reference data.
# - Configure the `data-prioritization` workflow with KNN distance metrics and `hard_first` ordering.
# - Prune outliers and duplicates before ranking using integrated cleaning parameters.
# - Execute prioritization workflows via `run_task()`.
# - Benchmark prioritization results against random selection baselines.
# - Contrast distance-based prioritization with model uncertainty sampling.
# - Export clean and prioritized index arrays for labeling queues.

# %% [markdown]
# ## Next steps
#
# - **Alternative ranking methods**: Evaluate `kmeans_distance` or `hdbscan_complexity` policies.
# - **Class-balanced sampling**: Use `policy="class_balanced"` to balance ranking across known classes.
# - **Threshold tuning**: Adjust `outlier_threshold` and `outlier_flags` to control pruning sensitivity.

# %% [markdown]
# ## Related guides
#
# - **Concept**: [Prioritization](../concepts/Prioritization.md) explains ranking
#   methods and ordering policies.
# - **How-to**: [PyTorch model for embeddings](../how_to/torch_embeddings.md) explains
#   hooking intermediate neural network layers.
# - **How-to**: [Read evaluation outputs](../how_to/read_evaluation_outputs.md) covers
#   ranking reports and result envelopes.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md) explains
#   how to run prioritization pipelines in Docker.
