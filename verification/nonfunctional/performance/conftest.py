"""Shared dataset builder for the performance tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from verification.fixtures import write_image_folder


@pytest.fixture(scope="session")
def image_root(tmp_path_factory: pytest.TempPathFactory) -> Callable[[int, int], Path]:
    """Return ``root(n_images, size)``: a data root holding ``imgs/`` with *n_images* noise images, 10 classes.

    Each (count, size) pair is written once per session and shared by every test that asks for it.
    """
    roots: dict[tuple[int, int], Path] = {}

    def root(n_images: int, size: int) -> Path:
        key = (n_images, size)
        if key not in roots:
            path = tmp_path_factory.mktemp(f"images_{n_images}_{size}")
            write_image_folder(path / "imgs", n_per_class=n_images // 10, n_classes=10, size=size)
            roots[key] = path
        return roots[key]

    return root
