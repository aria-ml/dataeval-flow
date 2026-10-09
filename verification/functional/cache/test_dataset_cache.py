"""TC-14-2 — what `DatasetCache` stores, how its entries are keyed, and how it behaves on a miss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataeval_flow._cache import CACHE_VERSION, DatasetCache

pytestmark = pytest.mark.required

SELECTION = "sel:all"
EXTRACTOR = '{"name": "flat", "model": "flatten"}'
TRANSFORMS = "none"


def _array(offset: float = 0.0) -> np.ndarray:
    return np.arange(12, dtype=np.float32).reshape(3, 4) + offset


def _clusters(offset: int = 0) -> dict[str, np.ndarray]:
    return {
        "clusters": np.array([0, 0, 1, 1]) + offset,
        "mst": np.arange(9, dtype=np.float64).reshape(3, 3),
        "linkage_tree": np.arange(12, dtype=np.float64).reshape(3, 4),
        "membership_strengths": np.linspace(0, 1, 4),
        "k_neighbors": np.arange(8).reshape(4, 2),
        "k_distances": np.linspace(0, 1, 8).reshape(4, 2),
    }


class TestEmbeddingEntries:
    def test_a_saved_array_reads_back_unchanged_and_a_never_saved_key_misses(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path, "ds_a")
        assert cache.load_embeddings(SELECTION, EXTRACTOR, TRANSFORMS) is None  # a miss is None, not an error
        assert not list(tmp_path.rglob("*.npy"))  # and reading it stores nothing

        cache.save_embeddings(SELECTION, EXTRACTOR, TRANSFORMS, _array())
        loaded = cache.load_embeddings(SELECTION, EXTRACTOR, TRANSFORMS)

        assert loaded is not None
        np.testing.assert_array_equal(loaded, _array())
        assert loaded.dtype == np.float32

    def test_a_new_instance_over_the_same_directory_finds_the_entry_on_disk(self, tmp_path: Path) -> None:
        DatasetCache(tmp_path, "ds_a").save_embeddings(SELECTION, EXTRACTOR, TRANSFORMS, _array())

        loaded = DatasetCache(tmp_path, "ds_a").load_embeddings(SELECTION, EXTRACTOR, TRANSFORMS)

        assert loaded is not None
        np.testing.assert_array_equal(loaded, _array())

    @pytest.mark.parametrize(
        ("selection", "extractor", "transforms"),
        [
            ("sel:first-4", EXTRACTOR, TRANSFORMS),
            (SELECTION, '{"name": "flat", "model": "flatten", "batch_size": 4}', TRANSFORMS),
            (SELECTION, EXTRACTOR, "resize(32)"),
        ],
        ids=["view", "extractor", "transforms"],
    )
    def test_the_key_is_the_view_the_extractor_and_the_transforms(
        self, tmp_path: Path, selection: str, extractor: str, transforms: str
    ) -> None:
        cache = DatasetCache(tmp_path, "ds_a")
        cache.save_embeddings(SELECTION, EXTRACTOR, TRANSFORMS, _array())

        assert cache.load_embeddings(selection, extractor, transforms) is None
        cache.save_embeddings(selection, extractor, transforms, _array(100))

        first = cache.load_embeddings(SELECTION, EXTRACTOR, TRANSFORMS)
        second = cache.load_embeddings(selection, extractor, transforms)
        assert first is not None
        assert second is not None
        np.testing.assert_array_equal(first, _array())
        np.testing.assert_array_equal(second, _array(100))

    def test_an_entry_is_isolated_to_its_dataset(self, tmp_path: Path) -> None:
        first, second = DatasetCache(tmp_path, "ds_a"), DatasetCache(tmp_path, "ds_b")
        first.save_embeddings(SELECTION, EXTRACTOR, TRANSFORMS, _array())

        assert second.load_embeddings(SELECTION, EXTRACTOR, TRANSFORMS) is None
        second.save_embeddings(SELECTION, EXTRACTOR, TRANSFORMS, _array(100))

        reread = DatasetCache(tmp_path, "ds_a").load_embeddings(SELECTION, EXTRACTOR, TRANSFORMS)
        assert reread is not None
        np.testing.assert_array_equal(reread, _array())
        assert first.dataset_dir != second.dataset_dir

    def test_the_dataset_directory_name_is_stable_for_one_content_key_and_differs_for_another(
        self, tmp_path: Path
    ) -> None:
        name_a = DatasetCache.get_or_create(tmp_path, "train", "key-1").dataset_name
        name_b = DatasetCache.get_or_create(tmp_path, "train", "key-1").dataset_name
        name_c = DatasetCache.get_or_create(tmp_path, "train", "key-2").dataset_name
        assert name_a == name_b  # a stable key for the same dataset content
        assert name_a != name_c  # and a different one when the content differs

    def test_entries_are_written_under_a_versioned_dataset_directory(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path, "ds_a")
        cache.save_embeddings(SELECTION, EXTRACTOR, TRANSFORMS, _array())

        assert cache.dataset_dir == tmp_path / f"v{CACHE_VERSION}" / "ds_a"
        (entry,) = cache.dataset_dir.rglob("*.npy")
        assert entry.parent.parent == cache.dataset_dir
        assert entry.parent.name.startswith("sel_")
        assert entry.name.startswith("embeddings_")
        assert not list(tmp_path.rglob("*.tmp"))

    def test_entries_of_another_cache_version_are_ignored_not_deleted(self, tmp_path: Path) -> None:
        DatasetCache(tmp_path, "ds_a").save_embeddings(SELECTION, EXTRACTOR, TRANSFORMS, _array())
        (tmp_path / f"v{CACHE_VERSION}").rename(tmp_path / "v0")  # as if an older release wrote it

        assert DatasetCache(tmp_path, "ds_a").load_embeddings(SELECTION, EXTRACTOR, TRANSFORMS) is None
        assert list((tmp_path / "v0").rglob("*.npy"))

    @pytest.mark.parametrize("name", ["../escape", "a/b", "a\\b", ".", ".."])
    def test_a_dataset_name_cannot_leave_the_cache_directory(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ValueError, match="Invalid dataset_name"):
            DatasetCache(tmp_path, name)


class TestClusterEntries:
    def test_a_saved_cluster_result_reads_back_unchanged_and_is_keyed_by_algorithm(self, tmp_path: Path) -> None:
        cache = DatasetCache(tmp_path, "ds_a")
        assert cache.load_cluster_result(SELECTION, EXTRACTOR, TRANSFORMS, "kmeans", 2) is None

        cache.save_cluster_result(SELECTION, EXTRACTOR, TRANSFORMS, "kmeans", 2, _clusters())
        loaded = DatasetCache(tmp_path, "ds_a").load_cluster_result(SELECTION, EXTRACTOR, TRANSFORMS, "kmeans", 2)

        assert loaded is not None
        for key, expected in _clusters().items():
            np.testing.assert_array_equal(loaded[key], expected, err_msg=key)
        assert cache.load_cluster_result(SELECTION, EXTRACTOR, TRANSFORMS, "hdbscan", None) is None
        assert cache.load_cluster_result(SELECTION, EXTRACTOR, TRANSFORMS, "kmeans", 3) is None


class TestMemoryOnlyCache:
    def test_a_cache_without_a_directory_holds_entries_in_memory_and_writes_nothing(self, tmp_path: Path) -> None:
        cache = DatasetCache(None, "ds_a")
        assert not cache.disk_backed
        assert cache.cache_dir is None
        assert cache.dataset_dir is None

        cache.save_embeddings(SELECTION, EXTRACTOR, TRANSFORMS, _array())

        loaded = cache.load_embeddings(SELECTION, EXTRACTOR, TRANSFORMS)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, _array())
        assert DatasetCache(None, "ds_a").load_embeddings(SELECTION, EXTRACTOR, TRANSFORMS) is None  # per instance
        assert not list(tmp_path.iterdir())

    def test_get_or_create_shares_one_in_memory_cache_per_dataset_until_cleared(self) -> None:
        shared = DatasetCache.get_or_create(None, "train", "key-1")
        assert DatasetCache.get_or_create(None, "train", "key-1") is shared
        assert DatasetCache.get_or_create(None, "train", "key-2") is not shared
        shared.save_embeddings(SELECTION, EXTRACTOR, TRANSFORMS, _array())

        DatasetCache.clear_instances()

        fresh = DatasetCache.get_or_create(None, "train", "key-1")
        assert fresh is not shared
        assert fresh.load_embeddings(SELECTION, EXTRACTOR, TRANSFORMS) is None

    def test_a_disk_backed_get_or_create_returns_a_new_instance_over_the_same_entries(self, tmp_path: Path) -> None:
        first = DatasetCache.get_or_create(tmp_path, "train", "key-1")
        first.save_embeddings(SELECTION, EXTRACTOR, TRANSFORMS, _array())

        second = DatasetCache.get_or_create(tmp_path, "train", "key-1")

        assert second is not first
        assert second.disk_backed
        loaded = second.load_embeddings(SELECTION, EXTRACTOR, TRANSFORMS)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, _array())
