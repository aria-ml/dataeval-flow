# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# # Use a torchvision dataset with DataEval Workflows
#
# This guide shows how to pass a torchvision dataset into the `data-cleaning`
# workflow using the `"torchvision"` adapter.  The adapter converts both
# **image classification** and **object detection** datasets to the MAITE
# protocol that DataEval expects.

# %% [markdown]
# ## Image classification

# %% tags=["remove_output"]
from torchvision.datasets import FashionMNIST

from dataeval_flow.config import (
    BoVWExtractorConfig,
    DataCleaningWorkflowConfig,
    DatasetProtocolConfig,
    PipelineConfig,
    SelectionConfig,
    SelectionStep,
    SourceConfig,
    TaskConfig,
)
from dataeval_flow.workflow import run_tasks

# 1. Create the torchvision dataset (no transforms — the adapter handles conversion)
tv_dataset = FashionMNIST(root="./data", train=True, download=True)

# %%
# 2. Build the full pipeline config (using a small subset for speed)
datasets = [DatasetProtocolConfig(name="fmnist-train", format="torchvision", dataset=tv_dataset)]
selections = [SelectionConfig(name="first500", steps=[SelectionStep(type="Limit", params={"size": 500})])]
sources = [SourceConfig(name="fmnist-src", dataset="fmnist-train", selection="first500")]
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
    selections=selections,
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
# The `"torchvision"` format tells the resolver to wrap the dataset in a
# `TorchvisionDataset` adapter before passing it to the workflow.  The
# adapter:
#
# - Converts PIL images to CHW float32 numpy arrays
# - Converts integer labels to one-hot vectors (using `.classes` from the
#   dataset when available)
# - Exposes `.metadata` with `index2label` derived from the dataset's
#   `.classes` attribute

# %% [markdown]
# ## Object detection
#
# Torchvision object-detection datasets (e.g. `CocoDetection`,
# `VOCDetection`) return targets in varying raw formats.  The recommended
# approach is to use `wrap_dataset_for_transforms_v2` which normalises
# targets into a dict with `"boxes"` (as `BoundingBoxes`) and `"labels"`:
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
# # 2. Wrap with transforms v2 — this gives structured BoundingBoxes + labels
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
# The adapter automatically converts bounding boxes to XYXY format
# regardless of the source `BoundingBoxFormat`.  This means datasets using
# XYWH (COCO convention), CXCYWH, or any other torchvision format are
# handled transparently:
#
# | Source format | Converted to |
# |---|---|
# | `XYXY` | passed through |
# | `XYWH` | converted to `XYXY` |
# | `CXCYWH` | converted to `XYXY` |
#
# The conversion uses `torchvision.ops.box_convert` so rotated formats
# (`XYWHR`, `CXCYWHR`, `XYXYXYXY`) are also supported.

# %% [markdown]
# ## Cache identity
#
# Since torchvision datasets are in-memory objects, the cache key is built
# from the config's `name`, `format`, and `version` fields:

# %%
DatasetProtocolConfig(
    name="fmnist-train",
    format="torchvision",
    dataset=tv_dataset,
    version="2",  # bump this when the underlying data changes
)

# %% [markdown]
# Changing `version` invalidates cached embeddings and statistics from
# previous runs.

# %% [markdown]
# ## Tips
#
# - **Don't apply transforms** to the torchvision dataset before passing it
#   to the adapter.  The adapter expects raw PIL images (or tensors in HWC
#   or CHW layout).  Use DataEval's preprocessor config for any
#   transforms needed by the workflow.
# - **Use `wrap_dataset_for_transforms_v2`** for detection datasets.  The
#   adapter detects structured targets (dicts with `"boxes"`) and converts
#   them to the MAITE `ObjectDetectionTarget` protocol.  Raw annotation
#   formats (list-of-dicts from `CocoDetection`, XML dicts from
#   `VOCDetection`) are not supported directly.
# - **Class discovery** relies on the `.classes` attribute that most
#   torchvision datasets expose.  If your dataset doesn't have it, the
#   adapter still works but `index2label` will be empty and integer targets
#   will be passed through as scalar arrays.
