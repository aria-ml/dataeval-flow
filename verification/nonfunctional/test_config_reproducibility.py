"""TC-34-1 (NFR-4) — configuration reproducibility.

The same configuration run twice gives the same result, ignoring the fields that record when and how long a run
took, and a changed configuration gives a different one. The pipeline reads its data through a view that
shuffles the images and keeps some of them; the shuffle is seeded by the pipeline's ``seed``, so the result
depends on the seed and on the number of images kept.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import load_config, run_tasks
from dataeval_flow._cache import DatasetCache
from verification.nonfunctional._support import stable, write_project

pytestmark = pytest.mark.required


@pytest.fixture(autouse=True)
def _fresh_caches() -> Iterator[None]:
    """Each run starts with empty in-memory caches, as a new process would."""
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _run(root: Path, *, keep: int, seed: int | None = 42) -> dict[str, Any]:
    """Run ``clean_task`` on a project that keeps *keep* shuffled images, and return its comparable result."""
    config = write_project(root, seed=seed, view=[{"type": "Shuffle"}, {"type": "Limit", "params": {"size": keep}}])
    DatasetCache.clear_instances()
    result = run_tasks(load_config(config), data_dir=root)["clean_task"]
    assert result.success, result.errors
    return stable(result.to_dict())


class TestConfigReproducibility:
    def test_same_config_produces_identical_output(self, tmp_path: Path) -> None:
        out_a = _run(tmp_path / "a", keep=10)
        out_b = _run(tmp_path / "b", keep=10)
        assert out_a == out_b

    def test_different_config_produces_different_output(self, tmp_path: Path) -> None:
        out_a = _run(tmp_path / "a", keep=10)
        out_b = _run(tmp_path / "b", keep=12)
        assert out_a != out_b

    def test_a_different_seed_changes_which_images_are_kept(self, tmp_path: Path) -> None:
        out_a = _run(tmp_path / "a", keep=8, seed=1)
        out_b = _run(tmp_path / "b", keep=8, seed=2)
        assert out_a != out_b
