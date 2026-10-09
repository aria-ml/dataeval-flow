"""Deterministic synthetic-data fixtures for workflow verification.

These produce mathematically valid but meaningless data so workflow
verification tests can run end-to-end without external datasets, GPUs,
or model downloads.

``SyntheticDataset`` and ``SyntheticMetadata`` are intentionally kept as
separate, coupled objects: ``make_synthetic_dataset`` also constructs a
matching ``SyntheticMetadata`` and exposes it as
``SyntheticDataset.synthetic_metadata`` so workflow tests can grab both
from a single fixture call without rebuilding ``dataeval.Metadata`` from
scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


@dataclass
class SyntheticMetadata:
    """Minimal Metadata-protocol implementation for verification tests."""

    class_labels: NDArray[np.intp]
    factor_data: NDArray[np.int64]
    factor_names: list[str]
    is_discrete: list[bool]
    index2label: dict[int, str] = field(default_factory=dict)


@dataclass
class SyntheticDataset:
    """MAITE-compatible image dataset returning (image, target, metadata) triples."""

    images: NDArray[np.uint8]
    labels: NDArray[np.intp]
    index2label: dict[int, str] = field(default_factory=dict)
    synthetic_metadata: SyntheticMetadata | None = None
    _id: str = "verification-synthetic"

    @property
    def metadata(self) -> dict[str, Any]:
        # Shape matches ``dataeval.protocols.DatasetMetadata`` (TypedDict with
        # required ``id`` and optional ``index2label``). ``cast`` keeps the
        # public return type as ``dict[str, Any]`` while documenting intent.
        payload: dict[str, Any] = {"id": self._id, "index2label": dict(self.index2label)}
        return cast(dict[str, Any], payload)

    def __getitem__(self, idx: int) -> tuple[NDArray[np.uint8], NDArray[np.float32], dict[str, Any]]:
        # DataEval reads a classification target as a one-hot vector (a bare int is rejected).
        target = np.zeros(len(self.index2label) or int(self.labels.max()) + 1, dtype=np.float32)
        target[int(self.labels[idx])] = 1.0
        return self.images[idx], target, {"id": idx}

    def __len__(self) -> int:
        return len(self.images)


def make_synthetic_images(
    n: int = 64,
    shape: tuple[int, int, int] = (3, 8, 8),
    seed: int = 0,
) -> NDArray[np.uint8]:
    """Return a deterministic uint8 image batch of shape (n, *shape)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, *shape), dtype=np.uint8)


def make_synthetic_labels(n: int, n_classes: int = 3, seed: int = 0) -> NDArray[np.intp]:
    """Return a deterministic label vector in [0, n_classes)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_classes, size=n, dtype=np.intp)


def make_synthetic_metadata(n: int, n_factors: int = 3, n_classes: int = 3, seed: int = 0) -> SyntheticMetadata:
    rng = np.random.default_rng(seed)
    class_labels = rng.integers(0, n_classes, size=n, dtype=np.intp)
    factor_data = rng.integers(0, 4, size=(n, n_factors), dtype=np.int64)
    factor_names = [f"factor_{i}" for i in range(n_factors)]
    is_discrete = [True] * n_factors
    index2label = {i: f"class_{i}" for i in range(n_classes)}
    return SyntheticMetadata(
        class_labels=class_labels,
        factor_data=factor_data,
        factor_names=factor_names,
        is_discrete=is_discrete,
        index2label=index2label,
    )


def make_synthetic_dataset(
    n: int = 64,
    n_classes: int = 3,
    shape: tuple[int, int, int] = (3, 8, 8),
    seed: int = 0,
) -> SyntheticDataset:
    """Return a MAITE-compatible dataset with deterministic content.

    The returned dataset also carries a coupled ``SyntheticMetadata`` instance
    (built from the same labels) at ``dataset.synthetic_metadata`` so workflow
    tests can access both halves from a single call.
    """
    images = make_synthetic_images(n=n, shape=shape, seed=seed)
    labels = make_synthetic_labels(n=n, n_classes=n_classes, seed=seed + 1)
    index2label = {i: f"class_{i}" for i in range(n_classes)}
    factor_data = np.random.default_rng(seed + 2).integers(0, 4, size=(n, 3), dtype=np.int64)
    synthetic_metadata = SyntheticMetadata(
        class_labels=labels,
        factor_data=factor_data,
        factor_names=[f"factor_{i}" for i in range(3)],
        is_discrete=[True, True, True],
        index2label=index2label,
    )
    return SyntheticDataset(
        images=images,
        labels=labels,
        index2label=index2label,
        synthetic_metadata=synthetic_metadata,
    )


def make_synthetic_embeddings(n: int = 64, dim: int = 32, seed: int = 0) -> NDArray[np.float32]:
    """Return a deterministic float32 embedding matrix of shape (n, dim)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def write_image_folder(
    root: Path,
    n_per_class: int = 4,
    n_classes: int = 2,
    seed: int = 0,
    size: int = 8,
    low: int = 0,
    high: int = 256,
) -> Path:
    """Write a tiny ImageFolder-style dataset to disk and return its root.

    Pixels are uniform noise in ``[low, high)``; ``size`` is the square edge in pixels.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    for c in range(n_classes):
        class_dir = root / f"class_{c}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = rng.integers(low, high, size=(size, size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"img_{i}.png", compress_level=1)
    return root


def write_coco_dataset(root: Path, n: int = 3, size: int = 16, seed: int = 0) -> Path:
    """Write a COCO object-detection dataset (``images/`` + ``annotations.json``), one box per image."""
    import json

    from PIL import Image

    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for i in range(n):
        Image.fromarray(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)).save(root / "images" / f"{i}.png")
        images.append({"id": i, "file_name": f"{i}.png", "width": size, "height": size})
        annotations.append({"id": i, "image_id": i, "category_id": 1, "bbox": [1, 1, 8, 8], "area": 64, "iscrowd": 0})
    categories = [{"id": 1, "name": "thing"}]
    (root / "annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )
    return root


def write_yolo_dataset(root: Path, n: int = 3, size: int = 16, seed: int = 0) -> Path:
    """Write a YOLO object-detection dataset (``data.yaml`` + ``images/train`` + ``labels/train``)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.fromarray(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)).save(
            root / "images" / "train" / f"{i}.png"
        )
        (root / "labels" / "train" / f"{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nnames:\n  0: thing\n")
    return root


def plant_duplicate_and_outlier(root: Path) -> tuple[list[int], int]:
    """Add one exact duplicate and one pixel outlier to a 2-class ImageFolder of ``img_0..img_9`` per class.

    Returns ``(duplicate_pair, outlier_index)`` as dataset indices (files sort by name, so the copy
    ``class_0/img_copy.png`` is index 10 and the all-white ``class_1/img_white.png`` is index 21).
    """
    import shutil

    from PIL import Image

    shutil.copy(root / "class_0" / "img_0.png", root / "class_0" / "img_copy.png")
    Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8)).save(root / "class_1" / "img_white.png")
    return [0, 10], 21
