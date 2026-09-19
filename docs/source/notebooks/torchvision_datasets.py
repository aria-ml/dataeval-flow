# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# # Use a torchvision dataset with DataEval Flow
#
# You can pass a torchvision dataset directly into the `data-cleaning`
# workflow with the `"torchvision"` adapter. This adapter converts both
# **image classification** and **object detection** datasets to the MAITE
# protocol.

# %% [markdown]
# ## Used in these tutorials
#
# You can reference this guide from:
#
# - [Clean a dataset](data_cleaning): Feed a `torchvision` classification or
#   detection dataset directly into the `data-cleaning` workflow.

# %% [markdown]
# ## Image classification

# %% tags=["remove_output"]
from torchvision.datasets import FashionMNIST

from dataeval_flow.config import (
    BoVWExtractorConfig,
    DataCleaningWorkflowConfig,
    DatasetProtocolConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.workflow import run_tasks

# 1. Create the torchvision dataset without transforms. The adapter handles conversion.
tv_dataset = FashionMNIST(root="./data", train=True, download=True)

# %%
# 2. Build the pipeline configuration with a subset for faster execution.
datasets = [DatasetProtocolConfig(name="fmnist-train", format="torchvision", dataset=tv_dataset)]
# A bare Limit takes the first N samples in storage order.
# Shuffle first so the subset represents the entire dataset.
views = [
    ViewConfig(
        name="sample500",
        operations=[
            ViewOperation(type="Shuffle", params={"seed": 0}),
            ViewOperation(type="Limit", params={"size": 500}),
        ],
    )
]
sources = [SourceConfig(name="fmnist-src", dataset="fmnist-train", view="sample500")]
extractors = [BoVWExtractorConfig(name="bovw", vocab_size=512, batch_size=64)]

workflows = [
    DataCleaningWorkflowConfig(
        name="adaptive_clean",
        outlier_method="adaptive",
        outlier_threshold=3.5,
        outlier_flags=["dimension", "pixel", "visual"],
    )
]
tasks = [
    TaskConfig(
        name="fmnist-clean",
        workflow="adaptive_clean",
        sources="fmnist-src",
        extractor="bovw",
    )
]

config = PipelineConfig(
    datasets=datasets,
    views=views,
    sources=sources,
    extractors=extractors,
    workflows=workflows,
    tasks=tasks,
)

# %%
# 3. Run
results = run_tasks(config)
print(results[0].report())

# %% [markdown]
# ### What happens under the hood
#
# When you specify `"torchvision"` format, the resolver wraps your dataset in
# a `TorchvisionDataset` adapter. The adapter performs these operations:
#
# - Converts PIL images to CHW float32 numpy arrays
# - Converts integer labels to one-hot vectors using `.classes` when available
# - Exposes `.metadata` with `index2label` derived from `.classes`

# %% [markdown]
# ## Object detection
#
# Torchvision object-detection datasets (such as `CocoDetection` and
# `VOCDetection`) return targets in varying raw formats. You should use
# `wrap_dataset_for_transforms_v2` to normalize targets into a dictionary
# with `"boxes"` (as `BoundingBoxes`) and `"labels"`:
#
# ```python
# from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
#
# from dataeval_flow.config import DatasetProtocolConfig
#
# # 1. Create the raw torchvision detection dataset
# raw_ds = CocoDetection(
#     root="./data/coco/val2017",
#     annFile="./data/coco/annotations/instances_val2017.json",
# )
#
# # 2. Wrap with transforms v2 to produce structured BoundingBoxes and labels
# tv_dataset = wrap_dataset_for_transforms_v2(raw_ds)
#
# # 3. Pass to DataEval via the torchvision adapter
# dataset_config = DatasetProtocolConfig(
#     name="coco-val",
#     format="torchvision",
#     dataset=tv_dataset,
# )
# ```

# %% [markdown]
# ### Bounding box format handling
#
# The adapter converts bounding boxes to XYXY format from any supported
# source `BoundingBoxFormat`:
#
# | Source format | Converted to |
# |---|---|
# | `XYXY` | passed through |
# | `XYWH` | converted to `XYXY` |
# | `CXCYWH` | converted to `XYXY` |
#
# The adapter uses `torchvision.ops.box_convert` to support rotated formats
# (`XYWHR`, `CXCYWHR`, `XYXYXYXY`).

# %% [markdown]
# ## Cache identity
#
# Because torchvision datasets are in-memory objects, DataEval Flow builds the
# cache key from the `name`, `format`, and `version` fields:

# %%
DatasetProtocolConfig(
    name="fmnist-train",
    format="torchvision",
    dataset=tv_dataset,
    version="2",  # bump this when the underlying data changes
)

# %% [markdown]
# When you update `version`, DataEval Flow invalidates cached embeddings and
# statistics from earlier runs.

# %% [markdown]
# ## Tips
#
# - **Do not apply transforms** to your torchvision dataset before passing it
#   to the adapter. The adapter expects raw PIL images or tensors in HWC
#   or CHW layout. Configure DataEval preprocessors for required transforms.
# - **Use `wrap_dataset_for_transforms_v2`** for detection datasets. The
#   adapter converts structured targets (dictionaries with `"boxes"`) to the
#   MAITE `ObjectDetectionTarget` protocol. Raw annotation formats are not
#   supported directly.
# - **Shuffle before you limit.** A bare `Limit` takes samples in disk storage
#   order. Apply `Shuffle` with an explicit `seed` to sample evenly across
#   classes. See [Build dataset views](../how_to/build_dataset_views.md).
# - **Class discovery** relies on the `.classes` attribute. If your dataset
#   lacks `.classes`, the adapter still works, but `index2label` will be empty
#   and integer targets pass through as scalar arrays.
