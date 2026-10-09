"""The dataset formats the tests build configs for: each format's config class and a valid set of its settings.

Not a test module.
"""

from __future__ import annotations

from typing import Any

from dataeval_flow.config import (
    CocoDatasetConfig,
    DemoDatasetConfig,
    HuggingFaceDatasetConfig,
    ImageFolderDatasetConfig,
    YoloDatasetConfig,
)

FORMATS: dict[str, tuple[type, dict[str, Any]]] = {
    "image_folder": (ImageFolderDatasetConfig, {"recursive": True, "infer_labels": True}),
    "huggingface": (HuggingFaceDatasetConfig, {"split": "train", "task": "object_detection"}),
    "coco": (CocoDatasetConfig, {"annotations_file": "ann.json", "images_dir": "imgs"}),
    "yolo": (YoloDatasetConfig, {"split": "val", "yaml_file": "d.yaml", "ann_dir": "labels"}),
    "demo": (DemoDatasetConfig, {"dataset": "M3FD", "image_set": "train"}),
}


def entry(fmt: str, name: str = "ds", **extra: Any) -> dict[str, Any]:
    """A `datasets:` entry for format `fmt` with every format-specific setting given."""
    return {"name": name, "format": fmt, "path": "data", **FORMATS[fmt][1], **extra}
