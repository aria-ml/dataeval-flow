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

    def __getitem__(self, idx: int) -> tuple[NDArray[np.uint8], int, dict[str, Any]]:
        return self.images[idx], int(self.labels[idx]), {"id": idx}

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


def write_image_folder(root: Path, n_per_class: int = 4, n_classes: int = 2, seed: int = 0) -> Path:
    """Write a tiny ImageFolder-style dataset to disk and return its root."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    for c in range(n_classes):
        class_dir = root / f"class_{c}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"img_{i}.png")
    return root
