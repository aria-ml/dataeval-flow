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
# # Use a torchvision dataset
#
# You can pass a torchvision dataset directly into the [`quality`](../reference/presets.md#quality) workflow with the
# `"torchvision"` adapter. The adapter converts image classification and object detection datasets to the
# {term}`MAITE` protocol. This guide runs `quality` on FashionMNIST and shows how to wrap a detection dataset. The first
# code cell downloads FashionMNIST into `./data`.

# %% [markdown]
# ## Load a classification dataset
#
# Pass the dataset to the adapter without transforms. The adapter expects raw PIL images or tensors in HWC or CHW
# layout. Configure DataEval preprocessors for required transforms.

# %% tags=["remove_output"]
from torchvision.datasets import FashionMNIST

from dataeval_flow import run_tasks
from dataeval_flow.config import (
    DatasetProtocolConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.workflows.quality import QualityConfig

# 1. Create the torchvision dataset without transforms. The adapter handles conversion.
tv_dataset = FashionMNIST(root="./data", train=True, download=True)

# %% [markdown]
# ## Build the pipeline and run `quality`
#
# Shuffle before you limit. A bare `Limit` takes samples in disk storage order. Apply `Shuffle` with an explicit `seed`
# to sample evenly across classes. See [Narrow a dataset with views](../how_to/build_dataset_views.md).

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
    QualityConfig(
        name="adaptive_clean",
        outliers={"outlier_threshold": ("adaptive", 3.5), "flags": ["dimension", "pixel", "visual"]},
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
    max_processes=2,
    datasets=datasets,
    views=views,
    sources=sources,
    extractors=extractors,
    workflows=workflows,
    tasks=tasks,
)

# %%
# 3. Run, then read the findings and the cleaned dataset
results = run_tasks(config)
result = results["fmnist-clean"]

for finding in result.findings:
    print(f"{finding.severity:<8} {finding.title:<20} {finding.brief}")

clean = result.steps["clean"].output
print(f"\nImages kept: {len(clean)} of {len(result.sources['fmnist-src'])}")

# %% [markdown]
# `quality` runs as a chain of steps, so `result.findings` holds the findings its checks made, and
# `result.report()` prints the full report. Its `clean` step's output is the sample without each image it flagged,
# and each duplicate but the first: a DataEval `View` that still reads the torchvision images through the adapter.

# %% [markdown]
# ## Check how the adapter converts the data
#
# When you specify the `"torchvision"` format, the resolver wraps your dataset in a `TorchvisionDataset` adapter. The
# adapter performs these operations:
#
# - Converts PIL images to CHW float32 numpy arrays
# - Converts integer labels to one-hot vectors using `.classes` when available
# - Exposes `.metadata` with `index2label` derived from `.classes`
#
# Class discovery relies on the `.classes` attribute. If your dataset lacks `.classes`, the adapter still works, but
# `index2label` will be empty and integer targets pass through as scalar arrays.

# %% [markdown]
# ## Wrap a detection dataset
#
# Torchvision object-detection datasets, such as `CocoDetection` and `VOCDetection`, return targets in varying raw
# formats. Use `wrap_dataset_for_transforms_v2` to normalize targets into a dictionary with `"boxes"` (as
# `BoundingBoxes`) and `"labels"`. The adapter converts these structured targets (dictionaries with `"boxes"`) to the
# MAITE `ObjectDetectionTarget` protocol. It does not support raw annotation formats directly:
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
# ### Check the bounding box format
#
# The adapter converts bounding boxes to XYXY format from any supported source `BoundingBoxFormat`:
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
# ## Version the dataset for the cache
#
# Torchvision datasets are in-memory objects, so DataEval Flow builds the cache key from the `name`, `format` and
# `version` fields:

# %%
DatasetProtocolConfig(
    name="fmnist-train",
    format="torchvision",
    dataset=tv_dataset,
    version="2",  # bump this when the underlying data changes
)

# %% [markdown]
# When you update `version`, DataEval Flow invalidates cached embeddings and statistics from earlier runs.

# %% [markdown]
# ## Next steps
#
# - {doc}`data_cleaning` — Run `quality` on a dataset end to end and take the cleaned dataset.
# - {doc}`../how_to/build_dataset_views` — Limit, filter, shuffle or index into a dataset before a workflow sees it.
# - {doc}`../how_to/reuse_results_with_cache` — Persist embeddings and statistics across runs, and know what
#   invalidates them.
