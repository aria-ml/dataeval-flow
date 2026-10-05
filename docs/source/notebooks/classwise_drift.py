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
# # Detect classwise drift: Which classes are changing?
#
# Detect distribution drift that affects specific classes rather than the
# dataset as a whole. This tutorial simulates progressive sensor degradation
# across three vehicle types and uses classwise drift detection to identify
# affected classes.

# %% [markdown]
# **Target audience**: You are a T&E engineer diagnosing which classes are
# responsible for a drift signal so you can target data collection and model
# retraining.
#
# **Workflow role**: Classwise drift provides a diagnostic follow-up to
# {doc}`Monitor incoming data for drift <drift_monitoring>`. Once an initial check
# detects drift, you can evaluate per-class drift to identify affected categories.
# See [Distribution shift](../concepts/DistributionShift.md) for conceptual background.

# %% [markdown]
# ## What you will do
#
# - Load MilitaryVehicles using datamaite's `huggingface_vision` loader.
# - Build a wrapper dataset that applies progressive Gaussian blur to three target vehicle classes.
# - Generate embeddings using a pretrained ResNet-18 model.
# - Partition reference and incoming subsets using `ViewConfig` and `Indices`.
# - **Phase 1**: Run overall drift detection with chunked K-Neighbors to confirm drift and identify onset timing.
# - **Phase 2**: Run classwise drift detection using MMD and Univariate CVM to pinpoint affected classes.
# - **Phase 3**: Compare the degraded classes with the rest as two groups, in a custom workflow.

# %% [markdown]
# ## What you will learn
#
# - How to evaluate in-memory datasets with `DatasetProtocolConfig`.
# - How to subset datasets using `ViewConfig` and `Indices`.
# - How to simulate progressive degradation using dataset wrappers.
# - How to execute a two-phase drift workflow: chunked detection followed by classwise diagnostics.
# - How naming a detector in `classwise:` runs it once per class.
# - How to read a classwise step's `PerClassOutput`.
# - How `by:` with `groups:` runs a detector once per group of classes.
# - How extractor feature spaces determine drift detection sensitivity.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`).
# - Install `maite-datasets[datamaite]` to download and export MilitaryVehicles.
# - Install `torch` and `torchvision` for ResNet-18 feature extraction.
# - Ensure network access for the initial dataset download.

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the dataset
#
# [MilitaryVehicles](https://huggingface.co/datasets/leibnitz-lab/military_vehicles)
# contains 9,444 images across 24 vehicle types. Setting `as_datamaite=True` writes the
# dataset as a class-per-directory ImageFolder tree readable by datamaite's `huggingface_vision`
# loader.
#
# Because raw directories are ordered alphabetically by class, you can apply a seeded
# shuffle to simulate chronological collection order. Progressive degradation will be applied
# along this sequence.

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

# The loader groups samples by class folder, so raw index order is alphabetical by class.
# Shuffle imposes a seeded sequence that this tutorial treats as collection order.
# The degradation wrapper and chunked view in Phase 1 rely on this ordering.
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
# You will create a dataset wrapper that applies progressive Gaussian blur to
# three selected classes: `BMP-1`, `BTR-80`, and `T-72`. This models localized sensor
# degradation over time.
#
# The wrapper passes the wrapped dataset's `metadata` through. A classwise run names each class from
# its `index2label`, so a dataset without `metadata` could not be run by class.
#
# The blur radius scales linearly with sample index: early samples have negligible blur,
# while later samples receive heavy blurring.

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

    @property
    def metadata(self) -> Any:
        """The wrapped dataset's metadata, which holds the `index2label` that names each class."""
        return self._dataset.metadata

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[NDArray[Any], Any, Mapping[str, Any]]:
        image, target, metadata = self._dataset[index]
        t = np.asarray(target)
        label = int(np.argmax(t)) if t.ndim == 1 and t.size > 1 else int(t)

        if label in self._degraded_classes:
            # Blur increases linearly with index to simulate progressive degradation
            progress = index / max(len(self) - 1, 1)
            radius = self.max_blur_radius * progress

            if radius > 0.1:  # skip negligible blur
                chw = np.asarray(image)
                hwc = np.transpose(chw, (1, 2, 0))  # CHW -> HWC
                img_pil = Image.fromarray(hwc, mode="RGB").filter(ImageFilter.GaussianBlur(radius=radius))
                image = np.transpose(np.array(img_pil, dtype=chw.dtype), (2, 0, 1))  # back to CHW

        return image, target, metadata


# %% [markdown]
# You can now construct the reference and incoming datasets. Both share the same
# underlying images, partitioned with `ViewConfig` and `Indices`:
#
# - **Reference dataset**: First 2,000 samples (clean, unmodified).
# - **Incoming dataset**: Next 2,000 samples (positions 2,000 to 3,999) with progressive blur applied to target classes.

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
# You can inspect the degradation across positions in the incoming dataset,
# ranging from early (nearly clean) to late (heavily blurred):

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
# ## Step 1: Phase 1: Overall drift detection with chunking
#
# First, you will execute overall drift detection without classwise breakdown.
# You can use the K-Neighbors detector with chunking enabled. K-Neighbors performs
# a Mann-Whitney U test on distances to nearest reference neighbors. Chunking divides
# incoming samples into sequential time windows to pinpoint when drift began.
#
# :::{important}
# You should use a pretrained feature extractor for natural images. Raw pixel vectors
# reflect scene lighting, backgrounds, and framing rather than semantic distribution.
# Pretrained models such as ResNet-18 provide stable semantic representations where
# distributional distances correspond to genuine image shifts.
# :::

# %% tags=["remove_output"]
import torch
import torchvision

# Cache ResNet-18 weights locally for TorchExtractorConfig
model_dir = Path("./models")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "resnet18.pt"
if not model_path.exists():
    torch.save(torchvision.models.resnet18(weights="IMAGENET1K_V1").eval(), model_path)
print(f"Extractor model: {model_path}")

# %%
from dataeval_flow import PipelineConfig, run_task, set_device
from dataeval_flow.config import (
    DatasetProtocolConfig,
    PreprocessingStep,
    PreprocessorConfig,
    SourceConfig,
    TaskConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.config.extractors import TorchExtractorConfig
from dataeval_flow.evaluators.shift import ChunkedDriftConfig, DriftKNeighborsConfig
from dataeval_flow.steps.checks import DriftThresholds
from dataeval_flow.workflows.drift_monitoring import DriftMonitoringChecks, DriftMonitoringConfig

# --- Datasets (in-memory via DatasetProtocolConfig) ---
ref_config = DatasetProtocolConfig(
    name="reference",
    format="maite",
    dataset=vehicles,
)

incoming_config = DatasetProtocolConfig(
    name="incoming",
    format="maite",
    dataset=incoming_dataset,
)

# --- Views ---
# Select the first 2,000 images as reference baseline
ref_view = ViewConfig(
    name="ref-first-2k",
    operations=[ViewOperation(type="Indices", params={"indices": list(range(2000))})],
)

# --- Sources ---
ref_source_config = SourceConfig(name="reference_2k", dataset="reference", view="ref-first-2k")
inc_source_config = SourceConfig(name="incoming_2k", dataset="incoming")

# --- Extractors ---
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
    layer_name="avgpool",  # 512-dimensional features
    preprocessor="imagenet",
    batch_size=32,
)

# --- Workflows ---
drift_workflow_config = DriftMonitoringConfig(
    name="overall-drift",
    detectors=[
        DriftKNeighborsConfig(k=10, chunking=ChunkedDriftConfig(chunk_count=5, threshold=("zscore", 1.5))),
    ],
    checks=DriftMonitoringChecks(drift=DriftThresholds(chunk_percent=15.0, consecutive_chunks=2)),
)

# --- Phase 1: Overall drift with chunking (no classwise) ---
overall_task = TaskConfig(
    name="vehicles-overall-drift",
    workflow="overall-drift",
    sources=["reference_2k", "incoming_2k"],
    extractor="resnet18",
)

set_device("cpu")  # the same numbers on a machine with a GPU

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
# Inspect `result.report()` to evaluate chunk drift progression. Early chunks fall
# within the reference threshold, while later chunks exceed thresholds with steadily
# increasing distances as degradation accumulates.

# %%
print(overall_result.report())

# %% [markdown]
# ## Step 3: Phase 2: Classwise drift detection
#
# Once overall drift is confirmed, you can identify affected classes. Set
# `classwise:` to the names of the detectors to run once per class. A detector is named for its type
# unless its entry sets `name`. In this phase, you will configure MMD and Univariate CVM:
#
# - **MMD**: Evaluates distribution distance per class using kernel maximum mean discrepancy.
# - **Univariate CVM**: Runs Cramer-von Mises tests on individual embedding dimensions.
#
# Each named detector keeps its whole-set step, `drift-mmd`, and gains a step that runs it by class,
# `drift-mmd-classes`. A `drift` check judges each, as `drift-mmd-check` and `drift-mmd-classes-check`.
# A class needs 2 or more items in the reference and in the incoming data to be tested; the step
# lists any it leaves out as skipped.

# %%
from dataeval_flow.evaluators.shift import DriftMMDConfig, DriftUnivariateConfig

classwise_task = TaskConfig(
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
        DriftMonitoringConfig(
            name="classwise-drift",
            detectors=[
                DriftMMDConfig(n_permutations=100),
                DriftUnivariateConfig(method="cvm"),
            ],
            classwise={"drift-mmd": "class", "drift-univariate": "class"},
            checks=DriftMonitoringChecks(drift=DriftThresholds(warn_on_drift=True)),
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
# The report contains a classwise pivot table. You should inspect both detection flags
# and distance values. The degraded classes (`BMP-1`, `BTR-80`, `T-72`) exhibit distances
# substantially higher than unaffected classes.
#
# In this run, five of the 24 classes warn for each detector. `BMP-1`, `BTR-80` and `T-72` are
# the degraded ones, with MMD distances near 0.13 to 0.17 against about zero for most others.
# The other warnings sit just under the significance threshold: MMD flags `BMP-T15` and `T-64`
# at a p-value of 0.04, and CVM flags `BM-30` and `Tornado` on distances close to the unaffected
# classes'. A detector tested at `p_val` flags about that share of unaffected classes by luck, and
# 24 classes is 24 tests, so read the size of a class's distance, not only its flag. The
# [drift how-to](../how_to/monitor_drift.md) says more about reading many verdicts.

# %%
print(classwise_result.report())

# %% [markdown]
# ## Results Exploration: Classwise results
#
# A classwise step's output is a `PerClassOutput`. Its `outputs` hold DataEval's `DriftOutput` for each class, in
# ascending class index, and its `skipped` holds each class left out, with why. The incoming source is named
# `incoming_2k`, so its elements are under that key.

# %%
import polars as pl

pl.Config.set_tbl_hide_dataframe_shape(True)

steps = classwise_result.steps

# Overall results (from the classwise run)
print("── Overall Drift ──")
for name in ("drift-mmd", "drift-univariate"):
    output = (steps[name].elements or {})["incoming_2k"].output
    status = "DRIFT" if output.drifted else "ok"
    print(f"  {name}: {status} (distance={output.distance:.4f})")
print()

# Classwise results
print("── Classwise Drift ──")
per_class = {}
for name in ("drift-mmd", "drift-univariate"):
    per_class[name] = (steps[f"{name}-classes"].elements or {})["incoming_2k"].output
    print(f"\n  Detector: {name}")
    rows = [
        {
            "class": label,
            "drifted": output.drifted,
            "distance": round(output.distance, 4),
            "p_val": round(output.details["p_val"], 6) if "p_val" in output.details else None,
        }
        for label, output in per_class[name].outputs.items()
    ]
    print(pl.DataFrame(rows))
    print(f"  Skipped: {per_class[name].skipped}")

# %% [markdown]
# ### Visualize classwise drift
#
# You can plot per-class distances in a horizontal bar chart to contrast drifting
# and stable classes visually.

# %%
fig, axes = plt.subplots(1, len(per_class), figsize=(6 * len(per_class), 4))

for ax, (name, classes) in zip(axes, per_class.items(), strict=True):
    class_names = list(classes.outputs)
    distances = [output.distance for output in classes.outputs.values()]
    colors = ["#e74c3c" if output.drifted else "#2ecc71" for output in classes.outputs.values()]

    ax.barh(class_names, distances, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Class")
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.invert_yaxis()

fig.suptitle("Classwise drift: green = ok, red = drift detected", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 5: Phase 3: Compare groups of classes
#
# The preset runs each class on its own. A custom workflow can run a detector once per *group* of classes instead,
# with `by:` on its evaluate step and on the check that judges it. Here the three degraded classes are one group
# and the other classes are the other, so each group's distances are tested as one population. A class is listed by
# name or by index, and a name resolves through the first input's `index2label`.

# %%
from dataeval_flow.steps import CustomWorkflowConfig, StepEntry

others = [name for index, name in sorted(index2label.items()) if index not in DEGRADED]
groups = {"class": {"groups": {"degraded": [index2label[c] for c in sorted(DEGRADED)], "others": others}}}

grouped_task = TaskConfig(
    name="vehicles-grouped-drift",
    workflow="grouped-drift",
    sources=["reference_2k", "incoming_2k"],
    extractor="resnet18",
)

grouped_config = PipelineConfig(
    datasets=[ref_config, incoming_config],
    views=[ref_view],
    sources=[ref_source_config, inc_source_config],
    preprocessors=[preprocessor_config],
    extractors=[extractor_config],
    evaluators=[DriftMMDConfig(name="mmd", n_permutations=100)],
    workflows=[
        CustomWorkflowConfig(
            name="grouped-drift",
            inputs=["reference", {"name": "tests", "list": True}],
            steps=[
                StepEntry.model_validate(
                    {"name": "mmd-groups", "evaluator": "mmd", "input": ["reference", "tests"], "by": groups}
                ),
                StepEntry.model_validate(
                    {"name": "mmd-groups-check", "check": "drift", "input": "mmd-groups", "by": "class"}
                ),
            ],
        ),
    ],
    tasks=[grouped_task],
)

grouped_result = run_task(grouped_task, grouped_config, cache_dir=Path("./cache"))
print(grouped_result.report())

# %% [markdown]
# The check's `by:` takes no settings, since its keys come from the step it reads. It rolls the groups into one
# finding, which names the groups that drifted. Where a group holds fewer than `min_items` items (2 by default) in
# either source, it is skipped and the finding says why.

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Load datasets with datamaite's `huggingface_vision` loader into `DatasetProtocolConfig`.
# - Subset datasets using `ViewConfig` with `Indices`.
# - Simulate progressive class-specific degradation using custom dataset wrappers.
# - Execute Phase 1 overall drift detection with chunked K-Neighbors.
# - Execute Phase 2 classwise drift detection with MMD and Univariate CVM detectors.
# - Run a detector once per group of classes with `by:` in a custom workflow.
# - Configure pretrained extractors to evaluate semantic distribution shifts.
#
# You can apply this two-phase methodology to confirm high-level drift and isolate
# specific affected classes for targeted retraining.

# %% [markdown]
# ## Next steps
#
# - **Automated scheduling**: Run chunked overall detection periodically as a fast filter,
#   triggering classwise diagnostics only when warnings appear.
# - **Alternative backbones**: Evaluate larger pretrained models or ONNX extractors via
#   [Use an ONNX model for embeddings](onnx_embeddings).
# - **Health thresholds**: Tune `checks` to control warning triggers.
# - **More recipes**: [Monitor drift with steps](../how_to/monitor_drift.md) merges test sources, compares
#   one group of classes against another, and drifts on detection crops.

# %% [markdown]
# ## Related guides
#
# - **Concept**: [Distribution shift](../concepts/DistributionShift.md) explains
#   drift monitoring methodologies and detector algorithms.
# - **Tutorial**: {doc}`Monitor incoming data for drift <drift_monitoring>` covers full
#   dataset drift monitoring.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md) explains
#   how to run drift pipelines in containers.
