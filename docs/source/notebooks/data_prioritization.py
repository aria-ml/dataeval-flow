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
# **Who this is for** — Model developers and data scientists running an active-learning
# or labeling loop who need to spend a limited labeling budget on the most informative
# samples.
#
# **Where this fits** — Prioritization sits in the data-acquisition stage of the T&E
# workflow: given a trained model and labeled reference data, it ranks unlabeled
# incoming data so the next labeling batch targets novel or hard cases — closing the
# loop back to [dataset splitting](dataset_splitting) and retraining. See the
# [Prioritization](../concepts/Prioritization.md) concept page for the ranking methods.

# %% [markdown]
# ## What you'll do
#
# - Load MilitaryVehicles and split it into a **labelled pool** and an **unlabelled pool**
# - Train a classifier on the labelled pool, which covers 20 of the dataset's 24 vehicle
#   types — the four **Air Defense** systems are held out, standing in for a capability
#   that has not been annotated yet
# - Inject corrupted and duplicate frames into the unlabelled pool, because real pools
#   contain junk you would rather not spend labelling budget on
# - Run the `data-prioritization` workflow to rank the pool
# - **Measure** how much the ranking beats picking at random — and compare it against
#   ranking by model uncertainty, which is the first thing most teams try

# %% [markdown]
# ## What you'll learn
#
# - How to configure and run the `data-prioritization` workflow via `run_task()`
# - How the reference dataset (labelled data) and extractor (trained model) shape prioritization
# - How `hard_first` ordering surfaces novel or challenging samples
# - How to map prioritized indices back to source labels and **score the ranking against a
#   random baseline**, rather than assuming it worked
# - Why ranking by model **uncertainty** does not find novel categories, and what does
# - How optional pruning (outlier/duplicate removal) integrates with prioritization

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (brings in `dataeval`, `datamaite`, and `torch`) plus `torchvision`
# - `maite-datasets[datamaite]` (to download MilitaryVehicles and export it)
# - Internet connection on the first run; everything after that comes from disk

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: a labelled pool and an unlabelled one
#
# [MilitaryVehicles](https://huggingface.co/datasets/leibnitz-lab/military_vehicles) holds
# 9,444 images across 24 vehicle types, and its `hierarchy` groups those types into coarse
# categories — tanks, BMPs, BTRs, self-propelled artillery, air defense.
#
# We hold out an entire category rather than a few arbitrary classes. The four **Air
# Defense** systems (`30N6E`, `Iskander`, `Pantsir-S1`, `Rs-24`) are radar and missile
# platforms, structurally unlike the tracked armour that makes up the rest — and holding
# out a whole category is the realistic version of this scenario: a capability appears in
# theatre that nobody has annotated yet. Holding out four tank variants instead would hold
# out things that look like the training data, which is a different and much weaker test.
#
# - **Labelled pool** — 4,000 frames drawn from the 20 known types
# - **Unlabelled pool** — 1,500 frames sampled at random from everything left, which is
#   where all the Air Defense frames end up, since none of them were ever labelled

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

# Both pools are ordinary view pipelines built from the operations `ViewConfig` exposes.
# We build them in Python rather than in the pipeline config because the model below
# trains on the labelled pool directly, before any workflow runs.
#
# Labelled pool: the twenty known types only. `ClassFilter` is what enforces the holdout —
# it reads each datum's own label, so nothing here re-derives the label space by hand.
labelled_dataset = View(vehicles, [ClassFilter(known_classes), Shuffle(seed=42), Limit(4000)])
labelled_indices = labelled_dataset.resolve_indices()

# Unlabelled pool: everything the labelled pool did not consume, which is where the Air
# Defense frames sit alongside the leftover known types. `Indices(..., exclude=True)` is
# the complement; shuffling before the limit is what keeps the pool honest, since taking a
# prefix of the remainder would concentrate the held-out frames and flatter any ranking.
pool_dataset = View(vehicles, [Indices(labelled_indices, exclude=True), Shuffle(seed=7), Limit(1500)])

pool_labels = np.array([int(np.argmax(pool_dataset[i][1])) for i in range(len(pool_dataset))])
n_novel = int(np.isin(pool_labels, sorted(held_out)).sum())
print(f"Held out (Air Defense): {sorted(index2label[i] for i in held_out)}")
print(f"Labelled pool:   {len(labelled_dataset)} frames, {len(known_classes)} types")
print(
    f"Unlabelled pool: {len(pool_dataset)} frames, {n_novel} of them Air Defense "
    f"({n_novel / len(pool_dataset) * 100:.0f}%)"
)

# %% [markdown]
# ### Inject corrupted and duplicate images into the test data
#
# Real incoming data is messy. We simulate this with an in-memory wrapper that
# corrupts a slice of the (uncorrupted, on-disk) test set on the fly:
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
# ## Step 1: Train a classifier on the labelled pool
#
# The extractor has to be a model shaped by *this* label space, which is what makes the
# ranking meaningful: frames unlike anything in the labelled pool should land far from
# everything the model has organised.
#
# We take a pretrained ResNet-18, freeze it, and train a small head over its features —
# what most teams would actually do with 4,000 labelled frames. Training the trunk from
# scratch on this much fine-grained data would produce a weak embedding space and a
# correspondingly meaningless ranking.
#
# The hook target is the 128-dimensional `embed` layer between trunk and classifier.

# %% tags=["remove_output"]
import random

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision

random.seed(42)
torch.manual_seed(42)


class VehicleNet(nn.Module):
    """A frozen pretrained trunk, a learned embedding, and a head over the known types.

    ``embed`` bundles Linear + ReLU so a forward hook captures the activated output —
    the representation the classifier was actually trained on.
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
    """Run the frozen trunk once over a dataset; the head then trains on the result.

    Labels come back from the same pass — the view already decoded each datum, so there
    is no reason to walk it a second time just to read the targets.
    """
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
features, labelled_labels = _trunk_features(labelled_dataset)
targets = torch.tensor([_remap[c] for c in labelled_labels], dtype=torch.long)

head = nn.Sequential(model.embed, model.classifier)
optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
for _epoch in range(40):
    optimizer.zero_grad()
    loss = F.cross_entropy(head(features), targets)
    loss.backward()
    optimizer.step()
model.eval()

accuracy = (head(features).argmax(1) == targets).float().mean().item()
print(f"Head trained: loss={loss.item():.3f}, accuracy on the labelled pool={accuracy * 100:.1f}%")

# %% [markdown]
# Twenty-way fine-grained vehicle recognition is genuinely hard, and the accuracy above
# reflects that — six of the twenty types are tank variants. That is fine for this purpose.
# We are not shipping this classifier; we are using the embedding space it organises to ask
# which unlabelled frames sit furthest from everything it knows.

# %% [markdown]
# ### Save the trained model to disk
#
# We save the full model so the workflow can load it via `TorchExtractorConfig`.
# The config points to the `.pt` file path and specifies which layer to hook
# for embeddings.

# %% tags=["remove_output"]
model_path = Path("./models/vehiclenet.pt")
model_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, str(model_path))
print(f"Model saved to {model_path}")

# %% [markdown]
# ## Step 2: Build the prioritization workflow configuration
#
# We configure the workflow with:
#
# - **Reference**: the labelled pool (4,000 frames, 20 vehicle types) — what the model knows
# - **Incoming data**: the unlabelled pool (1,500 frames, including Air Defense and the
#   injected junk) — what needs labelling
# - **Extractor**: our trained model, hooking the `embed` layer for 128-dim embeddings
# - **Method**: KNN with `hard_first` — surface frames farthest from their reference neighbours
# - **Pruning**: outlier + duplicate detection to drop corrupted and duplicate frames before ranking
# - **Mode**: `preparatory` so the result includes clean/flagged index lists
#
# The question the next section answers is not whether the ranking *looks* sensible but
# whether it beats picking frames at random — and by how much.

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

ref_dataset = labelled_dataset

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
# The first source is the reference (labeled data); subsequent sources
# are the data to prioritize.  The `TorchExtractorConfig` points to
# the saved `.pt` file and hooks the `embed` layer for 128-dim embeddings.
#
# We add a preprocessor to bring the dataset's images into the shape the model was trained
# on — ResNet's own pipeline, and the same one used to compute the features the head was
# fitted to. It also settles the variable frame sizes in this collection, which range from
# roughly 100x100 to 224x224.

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
# A single `run_task()` call handles dataset loading, embedding extraction,
# and KNN-based prioritization.

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
# The pruning step ran outlier and duplicate detection across both the
# reference and incoming datasets.  In `preparatory` mode, the metadata
# includes the clean indices for each source — let's see what got removed
# from the test data and whether it overlaps with our injected corruptions.

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

# Check overlap with our known corrupted ranges
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

# 900-999 are the frames we corrupted; 200-219 are the originals the duplicates at
# 900-919 were copied from, so either copy of a pair is a fair catch.
planted = set(range(900, 1000)) | set(range(200, 220))
other_pruned = pruned_set - planted
if other_pruned:
    print(f"  Other (non-planted) frames pruned: {len(other_pruned)} of {len(all_test_indices) - len(planted)}")

# %% [markdown]
# Read that table honestly. Cleaning catches what it is good at and misses the rest: the
# exact duplicates and the heavily underexposed frames go 20/20, while blur at radius 3
# gets 2/20 and noise 7/20 — those corruptions leave a frame well inside the spread of
# ordinary imagery. It also prunes frames we never touched, which is the cost of an
# outlier filter rather than a defect in it.
#
# What matters for the next step is only that pruned frames are excluded from the
# ranking, so prioritization operates on the surviving pool.

# %% [markdown]
# ## Step 5: Does the ranking beat picking at random?
#
# This is the question worth asking, and it needs a baseline. Air Defense frames make up a
# known share of the pool; if the ranking carries no signal, the top of it will hold that
# same share. Anything above it is what prioritization bought you.


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
# The ranking clears the baseline at every budget, and the margin narrows as the budget
# grows — strongest over the first hundred frames and decaying steadily from there. That
# shape is the useful part: prioritization earns its keep when you can only label a small
# fraction of the pool, and buys you less and less the more of the pool you work through.
#
# Read the shape, not the digits. The exact multiplier depends on which frames happened to
# land in the labelled pool and which in the unlabelled one — re-drawing both with
# different seeds moved our own top-50 figure between roughly 1.4x and 1.7x. What survives
# the re-draw is the ordering (prioritized above random) and the decay.
#
# It is worth being clear about what this does *not* say. The ranking does not find every
# novel frame, and a good share of what it surfaces is not novel at all — it is ordinary
# imagery that happens to sit at the edge of the labelled distribution. Both are expected.

# %%
cumulative = np.cumsum(is_novel)
n_items = np.arange(1, len(cumulative) + 1)
novel_pct = 100.0 * cumulative / n_items

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(n_items, novel_pct, linewidth=2, label="Prioritized (hard_first)")
    ax.axhline(baseline_pct, color="gray", linestyle="--", linewidth=1, label=f"Random baseline ({baseline_pct:.0f}%)")
    ax.set_xlabel("Labelling budget (top-N frames)")
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
# The obvious alternative is to label whatever the classifier is least sure about. It is
# the first thing most teams reach for, and on this problem it does not work.
#
# We already have the trained model, so we can rank the same pool by the entropy of its
# softmax output and score it the same way.


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
# Uncertainty ranking never clears the baseline. It sits **below random** at every budget
# here — a fifth of the first fifty picks are Air Defense against a 26% baseline — and
# stays under water out to 500. `hard_first` is above it throughout.
#
# The reason is worth internalising: **a classifier trained without a category is not
# uncertain about it.** It has twenty labels to choose from and no option for "something
# else", so it assigns a confident answer to a system it has never seen — an Iskander
# launcher looks enough like artillery for the head to say so, firmly. The frames it
# *is* unsure about are the ones near boundaries it has genuinely learned, which is
# mostly the six tank variants, not the systems missing from its label space.
#
# Uncertainty finds the boundaries *inside* a label space you already have. It does not
# find things outside it. For that you need the distance-based view, which asks a different
# question: not "which class is this?" but "have I seen anything like this before?"

# %% [markdown]
# ## Step 6: Visualize the top-ranked images (optional)
#
# Let's see what the top-10 prioritized images actually look like.

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
# In this tutorial you learned how to:
#
# - **Train a model** on a subset of the label space and use it as an embedding extractor
# - **Inject corruptions** (blur, noise, brightness, duplicates) into the unlabelled pool
# - **Configure** the `data-prioritization` workflow with KNN ranking,
#   `hard_first` ordering, and pre-prioritization pruning
# - **Run** the workflow via `run_task()` to prune and rank unlabelled data
# - **Inspect pruned items** — verify that corrupted/duplicate frames were removed
# - **Score the ranking against a random baseline** instead of assuming it worked
# - **Compare it with uncertainty sampling**, and see why that does not find novel categories
#
# The result to carry away is a measured one rather than a promise. Prioritization
# concentrates novel-category frames at the top of the ranking by a real but finite margin,
# largest when the labelling budget is small and shrinking as it grows. Plan around that
# shape: it is worth most for the first few dozen frames you can afford to label.
#
# And the negative result matters as much as the positive one. Ranking by model uncertainty
# — the intuitive choice — barely beats random at a small budget and falls below it at a
# larger one, because a classifier with no "something else" option answers confidently
# about systems it has never seen. Uncertainty maps the boundaries within a label space;
# distance is what looks outside it.

# %% [markdown]
# ## What's next
#
# - **Try different methods** — Compare `knn`, `kmeans_distance`, `hdbscan_complexity`, etc.
# - **Class-balanced policy** — Use `policy="class_balanced"` to diversify across
#   known classes while still surfacing novel samples
# - **Tune pruning thresholds** — Adjust `outlier_flags` and `health_thresholds`
#   to control how aggressively corrupted images are pruned

# %% [markdown]
# ## Related guides
#
# - **Concept** — [Prioritization](../concepts/Prioritization.md):
#   the ranking methods (`knn`, `kmeans_distance`, `hdbscan_complexity`) and ordering policies.
# - **How-to: Use a PyTorch model for embeddings** — [PyTorch embeddings](../how_to/torch_embeddings.md)
#   to read an intermediate layer of the CNN trained here — the `layer_name: embed` configuration used above,
#   explained in full.
# - **How-to: Read evaluation outputs** — [Read evaluation outputs](../how_to/read_evaluation_outputs.md)
#   to interpret the ranking report and export the ordering for a labeling queue.
# - **How-to: Run workflows in containers** — [Containerized workflows](../how_to/containerized_workflows.md)
#   to run prioritization from a YAML config inside a container for production pipelines.
