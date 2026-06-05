"""Smoke tests for the verification harness itself."""
# ruff: noqa: ICN001
# S101: asserts are standard pytest practice; ICN001: alias is intentional for clarity.

from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.fixtures import (
    SyntheticDataset,
    SyntheticMetadata,
    make_synthetic_dataset,
    make_synthetic_embeddings,
    make_synthetic_images,
    make_synthetic_labels,
    make_synthetic_metadata,
    write_image_folder,
)
from verification.helpers import safe_import


def test_make_synthetic_images_deterministic() -> None:
    """Two calls with the same seed must yield identical arrays."""
    a = make_synthetic_images(n=4, seed=0)
    b = make_synthetic_images(n=4, seed=0)
    assert np.array_equal(a, b)


def test_make_synthetic_labels_in_range() -> None:
    """Generated labels must fall in [0, n_classes)."""
    labels = make_synthetic_labels(n=10, n_classes=3, seed=0)
    assert labels.min() >= 0
    assert labels.max() < 3


def test_synthetic_dataset_satisfies_basic_protocol() -> None:
    """The dataset must expose MAITE-style triples plus index2label metadata."""
    ds = make_synthetic_dataset(n=4, n_classes=2)
    assert isinstance(ds, SyntheticDataset)
    assert len(ds) == 4
    img, target, datum_meta = ds[0]
    assert img.shape == (3, 8, 8)
    assert isinstance(target, int)
    assert "id" in datum_meta
    assert "index2label" in ds.metadata
    assert isinstance(ds.synthetic_metadata, SyntheticMetadata)


def test_make_synthetic_metadata_shape() -> None:
    """SyntheticMetadata factory must produce arrays sized as requested."""
    md = make_synthetic_metadata(n=8, n_factors=2, n_classes=3, seed=0)
    assert md.class_labels.shape == (8,)
    assert md.factor_data.shape == (8, 2)
    assert len(md.factor_names) == 2
    assert len(md.index2label) == 3


def test_make_synthetic_embeddings_shape() -> None:
    """Embeddings must have requested (n, dim) shape and float32 dtype."""
    emb = make_synthetic_embeddings(n=6, dim=12, seed=0)
    assert emb.shape == (6, 12)
    assert emb.dtype == np.float32


def test_write_image_folder_creates_files(tmp_path: Path) -> None:
    """ImageFolder writer must create n_per_class*n_classes PNG files on disk."""
    root = write_image_folder(tmp_path / "imgs", n_per_class=2, n_classes=2)
    assert root.exists()
    pngs = list(root.rglob("*.png"))
    assert len(pngs) == 4


def test_safe_import_known_module() -> None:
    """safe_import must return the actual module object for installed packages."""
    import numpy as expected

    assert safe_import("numpy") is expected


def test_safe_import_missing_module_returns_none() -> None:
    """safe_import must return None instead of raising for missing packages."""
    assert safe_import("definitely_not_a_real_module_xyz") is None
