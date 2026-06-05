"""TC-15-1 — caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.test_case("15-1")
class TestCache:
    def test_cache_module_importable(self) -> None:
        from dataeval_flow import cache

        assert cache is not None

    def test_dataset_cache_roundtrip(self, tmp_path: Path) -> None:
        """Disk-backed DatasetCache writes an entry and reads it back."""
        from dataeval_flow.cache import DatasetCache

        c = DatasetCache(tmp_path, "ds_verify")
        selection_repr = "sel:all"
        config_json = '{"name": "flat", "model": "flatten"}'
        transforms_repr = "none"
        array = np.arange(12, dtype=np.float32).reshape(3, 4)

        # Initial miss
        assert c.load_embeddings(selection_repr, config_json, transforms_repr) is None

        c.save_embeddings(selection_repr, config_json, transforms_repr, array)
        loaded = c.load_embeddings(selection_repr, config_json, transforms_repr)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, array)

    def test_cache_key_stable_across_instances(self, tmp_path: Path) -> None:
        """Two DatasetCache instances pointing at the same root resolve to the same on-disk entry."""
        from dataeval_flow.cache import DatasetCache

        selection_repr = "sel:all"
        config_json = '{"name": "flat", "model": "flatten"}'
        transforms_repr = "none"
        array = np.arange(6, dtype=np.float32).reshape(2, 3)

        c1 = DatasetCache(tmp_path, "ds_verify")
        c1.save_embeddings(selection_repr, config_json, transforms_repr, array)

        # Fresh instance — no in-memory carryover — should still see the disk entry.
        c2 = DatasetCache(tmp_path, "ds_verify")
        loaded = c2.load_embeddings(selection_repr, config_json, transforms_repr)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, array)

    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        """Loading a never-saved key returns None (documented miss behavior)."""
        from dataeval_flow.cache import DatasetCache

        c = DatasetCache(tmp_path, "ds_verify")
        assert c.load_embeddings("sel:never", "nonexistent-config", "none") is None
