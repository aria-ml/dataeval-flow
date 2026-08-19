"""Tests for DatasetCache — disk-backed caching of workflow computations."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from dataeval import Metadata
from dataeval.protocols import DatasetMetadata

from dataeval_flow.cache import (
    CACHE_VERSION,
    FLAG_TO_METRIC,
    METRIC_TO_FLAG,
    DatasetCache,
    _atomic_write,
    _atomic_write_pair,
    _config_hash,
    _make_dataset_id,
    _metadata_config_key,
    active_cache,
    dataset_fingerprint,
    get_or_compute_cluster_result,
    get_or_compute_embeddings,
    get_or_compute_metadata,
    get_or_compute_stats,
    missing_flags,
    scope_key,
    selection_repr,
)
from dataeval_flow.policy import ResolvedPolicy

pytestmark = pytest.mark.required


@pytest.fixture(autouse=True)
def _reset_singleton_registry():
    """Reset the singleton registry between tests."""
    DatasetCache._instances.clear()
    yield
    DatasetCache._instances.clear()


# ---------------------------------------------------------------------------
# _config_hash helper
# ---------------------------------------------------------------------------


class TestConfigHash:
    def test_deterministic(self):
        assert _config_hash("hello") == _config_hash("hello")

    def test_different_inputs_different_hashes(self):
        assert _config_hash("a") != _config_hash("b")

    def test_returns_8_chars(self):
        assert len(_config_hash("anything")) == 8

    def test_hex_string(self):
        h = _config_hash("test")
        int(h, 16)  # should not raise


# ---------------------------------------------------------------------------
# selection_repr helper
# ---------------------------------------------------------------------------


class TestSelectionRepr:
    def test_plain_dataset_returns_sel_all(self):
        """A plain dataset (no resolve_indices) returns 'sel:all'."""
        dataset = MagicMock(spec=[])  # no resolve_indices attribute
        assert selection_repr(dataset) == "sel:all"

    def test_select_wrapper_uses_resolved_indices(self):
        dataset = MagicMock()
        dataset.resolve_indices.return_value = [0, 1, 2, 3, 4]
        result = selection_repr(dataset)
        assert result.startswith("sel:n=5:")
        assert len(result.split(":")[-1]) == 16  # 16-char hex hash

    def test_deterministic_for_same_indices(self):
        ds1 = MagicMock()
        ds1.resolve_indices.return_value = [10, 20, 30]
        ds2 = MagicMock()
        ds2.resolve_indices.return_value = [10, 20, 30]
        assert selection_repr(ds1) == selection_repr(ds2)

    def test_different_indices_different_repr(self):
        ds1 = MagicMock()
        ds1.resolve_indices.return_value = [0, 1, 2]
        ds2 = MagicMock()
        ds2.resolve_indices.return_value = [3, 4, 5]
        assert selection_repr(ds1) != selection_repr(ds2)

    def test_shuffled_indices_differ_from_sorted(self):
        """Different orderings produce different keys (shuffle-aware)."""
        ds_sorted = MagicMock()
        ds_sorted.resolve_indices.return_value = [0, 1, 2, 3, 4]
        ds_shuffled = MagicMock()
        ds_shuffled.resolve_indices.return_value = [3, 1, 4, 0, 2]
        assert selection_repr(ds_sorted) != selection_repr(ds_shuffled)


# ---------------------------------------------------------------------------
# dataset_fingerprint helper
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Minimal dataset for fingerprint tests: returns (image, target, metadata) tuples."""

    def __init__(self, images: list[np.ndarray], targets: list | None = None, metadata: list | None = None):
        self._images = images
        self._targets = targets or [np.array(i) for i in range(len(images))]
        self._metadata = metadata or [{"idx": i} for i in range(len(images))]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple:
        return (self._images[idx], self._targets[idx], self._metadata[idx])


class TestDatasetFingerprint:
    def test_deterministic(self):
        imgs = [np.random.RandomState(i).rand(3, 8, 8).astype(np.float32) for i in range(5)]
        tgts = [np.array(i) for i in range(5)]
        meta = [{"idx": i} for i in range(5)]
        assert dataset_fingerprint(_FakeDataset(imgs, tgts, meta)) == dataset_fingerprint(
            _FakeDataset(list(imgs), list(tgts), list(meta))
        )

    def test_different_images_different_fingerprint(self):
        imgs_a = [np.zeros((3, 8, 8), dtype=np.float32) for _ in range(5)]
        imgs_b = [np.ones((3, 8, 8), dtype=np.float32) for _ in range(5)]
        assert dataset_fingerprint(_FakeDataset(imgs_a)) != dataset_fingerprint(_FakeDataset(imgs_b))

    def test_different_targets_different_fingerprint(self):
        imgs = [np.zeros((3, 8, 8), dtype=np.float32) for _ in range(5)]
        tgts_a = [np.array(0) for _ in range(5)]
        tgts_b = [np.array(1) for _ in range(5)]
        assert dataset_fingerprint(_FakeDataset(imgs, tgts_a)) != dataset_fingerprint(_FakeDataset(imgs, tgts_b))

    def test_different_metadata_different_fingerprint(self):
        imgs = [np.zeros((3, 8, 8), dtype=np.float32) for _ in range(5)]
        meta_a = [{"label": "cat"} for _ in range(5)]
        meta_b = [{"label": "dog"} for _ in range(5)]
        assert dataset_fingerprint(_FakeDataset(imgs, metadata=meta_a)) != dataset_fingerprint(
            _FakeDataset(imgs, metadata=meta_b)
        )

    def test_different_length_different_fingerprint(self):
        img = np.zeros((3, 8, 8), dtype=np.float32)
        assert dataset_fingerprint(_FakeDataset([img] * 3)) != dataset_fingerprint(_FakeDataset([img] * 4))

    def test_samples_middle_for_large_dataset(self):
        """Changing only a middle element should change the fingerprint."""
        n = 20
        imgs_a = [np.zeros((3, 4, 4), dtype=np.float32) for _ in range(n)]
        imgs_b = list(imgs_a)
        imgs_b[n // 2] = np.ones((3, 4, 4), dtype=np.float32)
        assert dataset_fingerprint(_FakeDataset(imgs_a)) != dataset_fingerprint(_FakeDataset(imgs_b))

    def test_all_items_hashed_when_small(self):
        """For <= 15 items, all items are hashed — changing any item changes the fingerprint."""
        imgs_a = [np.zeros((3, 4, 4), dtype=np.float32) for _ in range(12)]
        imgs_b = list(imgs_a)
        imgs_b[7] = np.ones((3, 4, 4), dtype=np.float32)  # middle-ish item
        assert dataset_fingerprint(_FakeDataset(imgs_a)) != dataset_fingerprint(_FakeDataset(imgs_b))

    def test_empty_dataset(self):
        fp = dataset_fingerprint(_FakeDataset([]))
        assert isinstance(fp, str)
        assert len(fp) > 0


# ---------------------------------------------------------------------------
# scope_key helper
# ---------------------------------------------------------------------------


class TestScopeKey:
    def test_default_scope(self):
        assert scope_key() == "img+tgt"

    def test_all_flags(self):
        assert scope_key(True, True) == "img+tgt"

    def test_image_only(self):
        assert scope_key(True, False) == "img"

    def test_target_only(self):
        assert scope_key(False, True) == "tgt"

    def test_none_scope(self):
        assert scope_key(False, False) == "none"


# ---------------------------------------------------------------------------
# missing_flags helper
# ---------------------------------------------------------------------------


class TestMissingFlags:
    def test_all_cached_returns_none(self):
        from dataeval.flags import ImageStats

        cached = {"mean", "std", "var"}
        desired = ImageStats.PIXEL_MEAN | ImageStats.PIXEL_STD | ImageStats.PIXEL_VAR
        assert missing_flags(cached, desired) == ImageStats.NONE

    def test_none_cached_returns_all(self):
        from dataeval.flags import ImageStats

        desired = ImageStats.PIXEL_MEAN | ImageStats.PIXEL_STD
        result = missing_flags(set(), desired)
        assert result != ImageStats.NONE
        # Should include at least the requested flags
        assert ImageStats.PIXEL_MEAN in result
        assert ImageStats.PIXEL_STD in result

    def test_partial_returns_only_missing(self):
        from dataeval.flags import ImageStats

        cached = {"mean"}  # have mean
        desired = ImageStats.PIXEL_MEAN | ImageStats.PIXEL_STD  # want mean + std
        result = missing_flags(cached, desired)
        assert ImageStats.PIXEL_STD in result
        # mean is already cached so it should NOT be in the result
        assert ImageStats.PIXEL_MEAN not in result


# ---------------------------------------------------------------------------
# METRIC_TO_FLAG / FLAG_TO_METRIC mappings
# ---------------------------------------------------------------------------


class TestFlagMetricMappings:
    def test_metric_to_flag_has_all_expected_metrics(self):
        # Spot-check key metrics exist
        assert "mean" in METRIC_TO_FLAG
        assert "brightness" in METRIC_TO_FLAG
        assert "xxhash" in METRIC_TO_FLAG
        assert "width" in METRIC_TO_FLAG
        assert "histogram" in METRIC_TO_FLAG

    def test_flag_to_metric_is_inverse(self):
        for metric, flag in METRIC_TO_FLAG.items():
            assert FLAG_TO_METRIC[flag] == metric

    def test_mappings_are_same_size(self):
        assert len(METRIC_TO_FLAG) == len(FLAG_TO_METRIC)


# ---------------------------------------------------------------------------
# DatasetCache — construction and directory layout
# ---------------------------------------------------------------------------


class TestDatasetCacheInit:
    def test_stores_config(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="my_ds")
        assert cache.cache_dir == tmp_path
        assert cache.dataset_name == "my_ds"

    def test_dataset_dir_creates_hierarchy(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds1")
        dataset_dir = cache.dataset_dir
        assert dataset_dir is not None
        assert dataset_dir == tmp_path / f"v{CACHE_VERSION}" / "ds1"
        assert dataset_dir.is_dir()

    @pytest.mark.parametrize("name", ["../escape", "a/b", "a\\b", ".", ".."])
    def test_rejects_path_traversal(self, tmp_path: Path, name: str):
        with pytest.raises(ValueError, match="Invalid dataset_name"):
            DatasetCache(cache_dir=tmp_path, dataset_name=name)

    def test_allows_dots_in_name(self, tmp_path: Path):
        """Names like 'foo..bar' are valid directory names."""
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="foo..bar")
        assert cache.dataset_name == "foo..bar"


# ---------------------------------------------------------------------------
# Corrupted cache resilience (load_* try/except)
# ---------------------------------------------------------------------------


class TestCorruptedCacheResilience:
    """Corrupted cache files should return None instead of raising."""

    def test_load_embeddings_corrupted_npy(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        # Save a valid embedding first to get the path, then corrupt it
        arr = np.ones((2, 3), dtype=np.float32)
        cache.save_embeddings("sel:all", "cfg", "none", arr)
        path = cache._embeddings_path("sel:all", "cfg", "none")
        assert path is not None
        path.write_bytes(b"corrupted data")
        # Clear in-memory cache so the disk corruption is actually tested
        cache._memory.clear()

        result = cache.load_embeddings("sel:all", "cfg", "none")
        assert result is None

    def test_load_stats_corrupted_parquet(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        s = _config_hash("sel:all")
        h = _config_hash("img+tgt")
        sel_dir = tmp_path / f"v{CACHE_VERSION}" / "ds" / f"sel_{s}"
        sel_dir.mkdir(parents=True)
        # Write corrupted parquet and valid json
        (sel_dir / f"stats_{h}.parquet").write_bytes(b"bad parquet")
        (sel_dir / f"stats_{h}.json").write_text(
            '{"source_index":[],"object_count":[],"invalid_box_count":[],"image_count":0}'
        )

        result = cache.load_stats("sel:all", "img+tgt")
        assert result is None

    def test_load_stats_corrupted_json(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3)
        cache.save_stats("sel:all", "img+tgt", stats)
        # Corrupt the json file
        s = _config_hash("sel:all")
        h = _config_hash("img+tgt")
        sel_dir = tmp_path / f"v{CACHE_VERSION}" / "ds" / f"sel_{s}"
        (sel_dir / f"stats_{h}.json").write_text("not valid json{{{")
        # Clear in-memory cache so the disk corruption is actually tested
        cache._memory.clear()

        result = cache.load_stats("sel:all", "img+tgt")
        assert result is None

    def test_load_metadata_not_an_archive(self, tmp_path: Path):
        """A file that is not a zip at all is a miss, not an exception."""
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        path = _default_metadata_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not a zip archive")

        result = cache.load_metadata("sel:all", _FakeICDataset())  # type: ignore
        assert result is None

    def test_load_metadata_truncated_archive(self, tmp_path: Path):
        """A half-written archive is a miss, not an exception."""
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_metadata("sel:all", _make_metadata())
        path = _metadata_path(tmp_path)
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
        # Clear in-memory cache so the disk corruption is actually tested
        cache._memory.clear()

        result = cache.load_metadata("sel:all", _FakeICDataset())  # type: ignore
        assert result is None


# ---------------------------------------------------------------------------
# Embeddings cache (.npy)
# ---------------------------------------------------------------------------


class TestEmbeddingsCache:
    def test_save_and_load_round_trip(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        arr = np.random.default_rng(42).random((100, 512)).astype(np.float32)

        cache.save_embeddings("sel:all", '{"type":"onnx"}', "none", arr)
        loaded = cache.load_embeddings("sel:all", '{"type":"onnx"}', "none")

        assert loaded is not None
        np.testing.assert_array_equal(loaded, arr)

    def test_load_returns_none_on_miss(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        assert cache.load_embeddings("sel:all", '{"type":"onnx"}') is None

    def test_different_config_different_file(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        arr1 = np.ones((10, 4), dtype=np.float32)
        arr2 = np.zeros((10, 4), dtype=np.float32)

        cache.save_embeddings("sel:all", '{"type":"onnx","path":"a.onnx"}', "none", arr1)
        cache.save_embeddings("sel:all", '{"type":"onnx","path":"b.onnx"}', "none", arr2)

        loaded1 = cache.load_embeddings("sel:all", '{"type":"onnx","path":"a.onnx"}', "none")
        loaded2 = cache.load_embeddings("sel:all", '{"type":"onnx","path":"b.onnx"}', "none")

        assert loaded1 is not None
        assert loaded2 is not None
        np.testing.assert_array_equal(loaded1, arr1)
        np.testing.assert_array_equal(loaded2, arr2)

    def test_different_selections_independent(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cfg = '{"type":"flat"}'
        arr_a = np.ones((5, 3), dtype=np.float32)
        arr_b = np.zeros((5, 3), dtype=np.float32)

        cache.save_embeddings("sel:indices=0-100", cfg, "none", arr_a)
        cache.save_embeddings("sel:indices=100-200", cfg, "none", arr_b)

        loaded_a = cache.load_embeddings("sel:indices=0-100", cfg, "none")
        loaded_b = cache.load_embeddings("sel:indices=100-200", cfg, "none")

        assert loaded_a is not None
        assert loaded_b is not None
        np.testing.assert_array_equal(loaded_a, arr_a)
        np.testing.assert_array_equal(loaded_b, arr_b)

    def test_preserves_dtype(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        arr = np.array([[1.0, 2.0]], dtype=np.float64)
        cache.save_embeddings("sel:all", "cfg", "none", arr)
        loaded = cache.load_embeddings("sel:all", "cfg", "none")
        assert loaded is not None
        assert loaded.dtype == np.float64

    def test_file_goes_to_correct_path(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="my_ds")
        arr = np.ones((2, 2))
        cache.save_embeddings("sel:all", "cfg", "tfm", arr)

        s = _config_hash("sel:all")
        h = _config_hash("cfg|tfm")
        expected = tmp_path / f"v{CACHE_VERSION}" / "my_ds" / f"sel_{s}" / f"embeddings_{h}.npy"
        assert expected.exists()


# ---------------------------------------------------------------------------
# load_or_compute_embeddings
# ---------------------------------------------------------------------------


class TestLoadOrComputeEmbeddings:
    _BUILD_EMBEDDINGS_PATH = "dataeval_flow.embeddings.build_embeddings"

    def _mock_extractor_config(self) -> MagicMock:
        cfg = MagicMock()
        cfg.model_dump_json.return_value = '{"type":"onnx","model_path":"model.onnx"}'
        cfg.model_path = None  # no file to hash — keeps cache key = model_dump_json()
        return cfg

    def test_full_miss_computes_and_saves(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cfg = self._mock_extractor_config()
        mock_array = np.random.default_rng(42).random((10, 64)).astype(np.float32)

        mock_embeddings = MagicMock()
        mock_embeddings.__array__ = MagicMock(return_value=mock_array)

        with patch(self._BUILD_EMBEDDINGS_PATH, return_value=mock_embeddings) as mock_build:
            result = cache.load_or_compute_embeddings(
                "sel:all", cfg.model_dump_json(), "none", MagicMock(), cfg, None, None
            )
            mock_build.assert_called_once()

        np.testing.assert_array_equal(result, mock_array)
        # Verify saved to cache
        cached = cache.load_embeddings("sel:all", cfg.model_dump_json(), "none")
        assert cached is not None
        np.testing.assert_array_equal(cached, mock_array)

    def test_full_hit_skips_compute(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cfg = self._mock_extractor_config()
        arr = np.ones((5, 32), dtype=np.float32)

        # Pre-populate cache
        cache.save_embeddings("sel:all", cfg.model_dump_json(), "none", arr)

        with patch(self._BUILD_EMBEDDINGS_PATH) as mock_build:
            result = cache.load_or_compute_embeddings(
                "sel:all", cfg.model_dump_json(), "none", MagicMock(), cfg, None, None
            )
            mock_build.assert_not_called()

        np.testing.assert_array_equal(result, arr)

    def test_flattens_high_dimensional_output(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cfg = self._mock_extractor_config()
        # Simulate 3-D output (N, H, W)
        mock_array = np.random.default_rng(42).random((5, 4, 8)).astype(np.float32)

        mock_embeddings = MagicMock()
        mock_embeddings.__array__ = MagicMock(return_value=mock_array)

        with patch(self._BUILD_EMBEDDINGS_PATH, return_value=mock_embeddings):
            result = cache.load_or_compute_embeddings(
                "sel:all", cfg.model_dump_json(), "none", MagicMock(), cfg, None, None
            )

        assert result.ndim == 2
        assert result.shape == (5, 32)


class TestGetOrComputeEmbeddings:
    _BUILD_EMBEDDINGS_PATH = "dataeval_flow.embeddings.build_embeddings"

    def _mock_extractor_config(self) -> MagicMock:
        cfg = MagicMock()
        cfg.model_dump_json.return_value = '{"type":"onnx","model_path":"model.onnx"}'
        cfg.model_path = None  # no file to hash — keeps cache key = model_dump_json()
        return cfg

    def test_without_cache_computes_directly(self):
        cfg = self._mock_extractor_config()
        mock_array = np.ones((5, 16), dtype=np.float32)
        mock_embeddings = MagicMock()
        mock_embeddings.__array__ = MagicMock(return_value=mock_array)

        with patch(self._BUILD_EMBEDDINGS_PATH, return_value=mock_embeddings) as mock_build:
            result = get_or_compute_embeddings(dataset=MagicMock(), extractor_config=cfg)
            mock_build.assert_called_once()

        np.testing.assert_array_equal(result, mock_array)

    def test_with_cache_delegates_to_workflow_cache(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cfg = self._mock_extractor_config()
        mock_array = np.ones((5, 16), dtype=np.float32)
        mock_embeddings = MagicMock()
        mock_embeddings.__array__ = MagicMock(return_value=mock_array)

        with (
            active_cache(cache, "sel:all"),
            patch(self._BUILD_EMBEDDINGS_PATH, return_value=mock_embeddings) as mock_build,
        ):
            result = get_or_compute_embeddings(dataset=MagicMock(), extractor_config=cfg)
            mock_build.assert_called_once()

        np.testing.assert_array_equal(result, mock_array)
        # Should be saved to cache
        cached = cache.load_embeddings("sel:all", cfg.model_dump_json(), "none")
        assert cached is not None

    def test_with_cache_full_hit_skips_compute(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cfg = self._mock_extractor_config()
        arr = np.ones((5, 16), dtype=np.float32)

        # Pre-populate cache
        cache.save_embeddings("sel:all", cfg.model_dump_json(), "none", arr)

        with active_cache(cache, "sel:all"), patch(self._BUILD_EMBEDDINGS_PATH) as mock_build:
            result = get_or_compute_embeddings(dataset=MagicMock(), extractor_config=cfg)
            mock_build.assert_not_called()

        np.testing.assert_array_equal(result, arr)

    def test_different_transforms_different_cache_keys(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cfg = self._mock_extractor_config()

        arr_a = np.ones((3, 8), dtype=np.float32)
        arr_b = np.zeros((3, 8), dtype=np.float32)

        def make_mock_embeddings(arr: np.ndarray) -> MagicMock:
            m = MagicMock()
            m.__array__ = MagicMock(return_value=arr)
            return m

        transform_a = MagicMock()
        transform_a.__repr__ = MagicMock(return_value="transform_a")
        transform_b = MagicMock()
        transform_b.__repr__ = MagicMock(return_value="transform_b")

        with active_cache(cache, "sel:all"):
            with patch(self._BUILD_EMBEDDINGS_PATH, return_value=make_mock_embeddings(arr_a)):
                get_or_compute_embeddings(MagicMock(), cfg, transforms=transform_a)
            with patch(self._BUILD_EMBEDDINGS_PATH, return_value=make_mock_embeddings(arr_b)):
                get_or_compute_embeddings(MagicMock(), cfg, transforms=transform_b)

        # Both should be independently cached
        loaded_a = cache.load_embeddings("sel:all", cfg.model_dump_json(), "transform_a")
        loaded_b = cache.load_embeddings("sel:all", cfg.model_dump_json(), "transform_b")
        assert loaded_a is not None
        assert loaded_b is not None
        np.testing.assert_array_equal(loaded_a, arr_a)
        np.testing.assert_array_equal(loaded_b, arr_b)

    def test_no_active_cache_computes_directly(self):
        """When no active_cache context is set, computes directly without caching."""
        cfg = self._mock_extractor_config()
        mock_array = np.ones((3, 8), dtype=np.float32)
        mock_embeddings = MagicMock()
        mock_embeddings.__array__ = MagicMock(return_value=mock_array)

        with patch(self._BUILD_EMBEDDINGS_PATH, return_value=mock_embeddings) as mock_build:
            result = get_or_compute_embeddings(dataset=MagicMock(), extractor_config=cfg)
            mock_build.assert_called_once()

        np.testing.assert_array_equal(result, mock_array)


# ---------------------------------------------------------------------------
# Unified stats cache (.parquet + .json)
# ---------------------------------------------------------------------------


def _make_calc_result(n: int = 5, include_2d: bool = False, include_hashes: bool = False) -> dict[str, Any]:
    """Build a minimal StatsResult-shaped dict for testing."""
    from dataeval.types import SourceIndex

    rng = np.random.default_rng(42)
    stats: dict[str, Any] = {
        "mean": rng.random(n).astype(np.float64),
        "brightness": rng.random(n).astype(np.float64),
    }
    if include_hashes:
        stats["xxhash"] = np.array([f"hash_{i}" for i in range(n)], dtype=object)
    if include_2d:
        stats["histogram"] = rng.random((n, 256)).astype(np.float32)
        stats["percentiles"] = rng.random((n, 5)).astype(np.float32)
        stats["center"] = rng.random((n, 2)).astype(np.float32)

    return {
        "source_index": [SourceIndex(item=i, target=None, channel=None) for i in range(n)],
        "object_count": list(range(n)),
        "invalid_box_count": [0] * n,
        "image_count": n,
        "stats": stats,
    }


class TestStatsCache:
    def test_save_and_load_round_trip(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(5)

        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        assert loaded["image_count"] == 5
        assert len(loaded["source_index"]) == 5
        assert len(loaded["object_count"]) == 5
        assert len(loaded["invalid_box_count"]) == 5
        assert set(loaded["stats"].keys()) == {"mean", "brightness"}

    def test_load_returns_none_on_miss(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        assert cache.load_stats("sel:all", "img+tgt") is None

    def test_float_arrays_round_trip(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3)
        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded["stats"]["mean"], stats["stats"]["mean"])
        np.testing.assert_array_almost_equal(loaded["stats"]["brightness"], stats["stats"]["brightness"])

    def test_string_arrays_round_trip(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3, include_hashes=True)
        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        original_hashes = list(stats["stats"]["xxhash"])
        loaded_hashes = list(loaded["stats"]["xxhash"])
        assert original_hashes == loaded_hashes

    def test_2d_histogram_round_trip(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3, include_2d=True)
        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded["stats"]["histogram"], stats["stats"]["histogram"], decimal=5)
        assert loaded["stats"]["histogram"].shape == (3, 256)

    def test_2d_percentiles_round_trip(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3, include_2d=True)
        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded["stats"]["percentiles"], stats["stats"]["percentiles"], decimal=5)
        assert loaded["stats"]["percentiles"].shape == (3, 5)

    def test_2d_center_round_trip(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3, include_2d=True)
        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded["stats"]["center"], stats["stats"]["center"], decimal=5)
        assert loaded["stats"]["center"].shape == (3, 2)

    def test_source_index_round_trip(self, tmp_path: Path):
        from dataeval.types import SourceIndex

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3)
        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        for orig, loaded_si in zip(stats["source_index"], loaded["source_index"], strict=True):
            assert isinstance(loaded_si, SourceIndex)
            assert loaded_si.item == orig.item
            assert loaded_si.target == orig.target
            assert loaded_si.channel == orig.channel

    def test_source_index_with_targets(self, tmp_path: Path):
        from dataeval.types import SourceIndex

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(2)
        # Override source_index with target/channel data
        stats["source_index"] = [
            SourceIndex(item=0, target=None, channel=None),
            SourceIndex(item=0, target=1, channel=2),
        ]
        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        assert loaded["source_index"][0].target is None
        assert loaded["source_index"][1].target == 1
        assert loaded["source_index"][1].channel == 2

    def test_different_scopes_different_files(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats_a = _make_calc_result(3)
        stats_b = _make_calc_result(3)
        stats_b["stats"]["mean"] = np.zeros(3)  # different data

        cache.save_stats("sel:all", "img+tgt", stats_a)
        cache.save_stats("sel:all", "img", stats_b)

        loaded_a = cache.load_stats("sel:all", "img+tgt")
        loaded_b = cache.load_stats("sel:all", "img")

        assert loaded_a is not None
        assert loaded_b is not None
        # Different scope → different data
        assert not np.array_equal(loaded_a["stats"]["mean"], loaded_b["stats"]["mean"])

    def test_files_are_parquet_and_json(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_stats("sel:all", "img+tgt", _make_calc_result(2))

        s = _config_hash("sel:all")
        h = _config_hash("img+tgt")
        sel_dir = tmp_path / f"v{CACHE_VERSION}" / "ds" / f"sel_{s}"
        pq = sel_dir / f"stats_{h}.parquet"
        js = sel_dir / f"stats_{h}.json"
        assert pq.exists()
        assert js.exists()
        # Verify parquet is readable
        df = pl.read_parquet(pq)
        assert "mean" in df.columns


# ---------------------------------------------------------------------------
# load_or_compute_stats
# ---------------------------------------------------------------------------


class TestComputeStatsArguments:
    """The arguments flow pins on every ``compute_stats`` call."""

    _CALC_STATS_PATH = "dataeval.core.compute_stats"

    def _call(self) -> Any:
        from dataeval.flags import ImageStats

        from dataeval_flow.cache import _do_compute_stats

        with patch(self._CALC_STATS_PATH, return_value=_make_calc_result(3)) as mock_calc:
            _do_compute_stats(MagicMock(), ImageStats.PIXEL_MEAN)
        _, kwargs = mock_calc.call_args
        return kwargs

    def test_normalize_pixel_values_pinned_false(self):
        """Pixel statistics are reported in the units the data is stored in.

        Passed explicitly rather than left to the default so the scale is a
        property of this project, not of the installed dataeval.
        """
        assert self._call()["normalize_pixel_values"] is False

    def test_per_channel_not_passed(self):
        """Deprecated in dataeval 1.1 — channel rows can never reach Metadata."""
        assert "per_channel" not in self._call()


class TestLoadOrComputeStats:
    _CALC_STATS_PATH = "dataeval.core.compute_stats"

    def test_full_miss_computes_and_saves(self, tmp_path: Path):
        from dataeval.flags import ImageStats

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        mock_result = _make_calc_result(3)

        with patch(self._CALC_STATS_PATH, return_value=mock_result) as mock_calc:
            result = cache.load_or_compute_stats("sel:all", "img+tgt", ImageStats.PIXEL_MEAN, MagicMock(), True, True)
            mock_calc.assert_called_once()

        assert "mean" in result["stats"]
        # Cache should be populated
        cached = cache.load_stats("sel:all", "img+tgt")
        assert cached is not None

    def test_full_hit_returns_cached(self, tmp_path: Path):
        from dataeval.flags import ImageStats

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3)
        cache.save_stats("sel:all", "img+tgt", stats)

        with patch(self._CALC_STATS_PATH) as mock_calc:
            result = cache.load_or_compute_stats("sel:all", "img+tgt", ImageStats.PIXEL_MEAN, MagicMock(), True, True)
            # Should NOT call compute_stats — full cache hit
            mock_calc.assert_not_called()

        assert "mean" in result["stats"]

    def test_partial_hit_computes_missing_only(self, tmp_path: Path):
        from dataeval.flags import ImageStats

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        # Pre-populate cache with "mean" only
        stats = _make_calc_result(3)
        stats["stats"] = {"mean": stats["stats"]["mean"]}
        cache.save_stats("sel:all", "img+tgt", stats)

        # Request mean + std
        fresh_result = _make_calc_result(3)
        fresh_result["stats"] = {"std": np.array([0.1, 0.2, 0.3])}

        with patch(self._CALC_STATS_PATH, return_value=fresh_result) as mock_calc:
            result = cache.load_or_compute_stats(
                "sel:all", "img+tgt", ImageStats.PIXEL_MEAN | ImageStats.PIXEL_STD, MagicMock(), True, True
            )
            mock_calc.assert_called_once()
            # Should request only PIXEL_STD (mean is cached)
            call_args = mock_calc.call_args
            requested_flags = call_args[1]["stats"]
            assert ImageStats.PIXEL_STD in requested_flags
            assert ImageStats.PIXEL_MEAN not in requested_flags

        # Merged result should have both
        assert "mean" in result["stats"]
        assert "std" in result["stats"]

    def test_merged_result_persisted(self, tmp_path: Path):
        from dataeval.flags import ImageStats

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        # Pre-populate with "mean"
        stats = _make_calc_result(3)
        stats["stats"] = {"mean": stats["stats"]["mean"]}
        cache.save_stats("sel:all", "img+tgt", stats)

        # Compute "std"
        fresh = _make_calc_result(3)
        fresh["stats"] = {"std": np.array([0.1, 0.2, 0.3])}

        with patch(self._CALC_STATS_PATH, return_value=fresh):
            cache.load_or_compute_stats(
                "sel:all", "img+tgt", ImageStats.PIXEL_MEAN | ImageStats.PIXEL_STD, MagicMock(), True, True
            )

        # Cache should now have both metrics
        cached = cache.load_stats("sel:all", "img+tgt")
        assert cached is not None
        assert "mean" in cached["stats"]
        assert "std" in cached["stats"]


class TestGetOrComputeStats:
    """Tests for the centralized get_or_compute_stats() entry point."""

    _CALC_STATS_PATH = "dataeval.core.compute_stats"

    def test_without_cache_computes_directly(self):
        from dataeval.flags import ImageStats

        mock_result = _make_calc_result(3)

        with patch(self._CALC_STATS_PATH, return_value=mock_result) as mock_calc:
            result = get_or_compute_stats(desired_flags=ImageStats.PIXEL_MEAN, dataset=MagicMock())
            mock_calc.assert_called_once()

        assert "mean" in result["stats"]

    def test_with_cache_delegates_to_workflow_cache(self, tmp_path: Path):
        from dataeval.flags import ImageStats

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        mock_result = _make_calc_result(3)

        with active_cache(cache, "sel:all"), patch(self._CALC_STATS_PATH, return_value=mock_result) as mock_calc:
            result = get_or_compute_stats(desired_flags=ImageStats.PIXEL_MEAN, dataset=MagicMock())
            mock_calc.assert_called_once()

        assert "mean" in result["stats"]
        # Should have been saved to cache
        cached = cache.load_stats("sel:all", "img+tgt")
        assert cached is not None

    def test_with_cache_full_hit_skips_compute(self, tmp_path: Path):
        from dataeval.flags import ImageStats

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3)
        cache.save_stats("sel:all", "img+tgt", stats)

        with active_cache(cache, "sel:all"), patch(self._CALC_STATS_PATH) as mock_calc:
            result = get_or_compute_stats(desired_flags=ImageStats.PIXEL_MEAN, dataset=MagicMock())
            mock_calc.assert_not_called()

        assert "mean" in result["stats"]

    def test_no_active_cache_computes_directly(self):
        """When no active_cache context is set, computes directly without caching."""
        from dataeval.flags import ImageStats

        mock_result = _make_calc_result(3)

        with patch(self._CALC_STATS_PATH, return_value=mock_result) as mock_calc:
            result = get_or_compute_stats(desired_flags=ImageStats.PIXEL_MEAN, dataset=MagicMock())
            mock_calc.assert_called_once()

        assert "mean" in result["stats"]


# ---------------------------------------------------------------------------
# Metadata cache (.dem)
# ---------------------------------------------------------------------------


class _FakeICDataset:
    """Minimal MAITE image-classification dataset: 5 images, 3 classes, 1 factor."""

    _BRIGHTNESS = (0.1, 0.3, 0.5, 0.7, 0.9)
    _LABELS = (0, 1, 0, 1, 2)

    @property
    def metadata(self) -> DatasetMetadata:
        return {"id": "fake-ic", "index2label": {0: "cat", 1: "dog", 2: "bird"}}

    def __len__(self) -> int:
        return len(self._LABELS)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        one_hot = np.zeros(3, dtype=np.float32)
        one_hot[self._LABELS[index]] = 1.0
        return (
            np.zeros((1, 1, 1), dtype=np.float32),
            one_hot,
            {"id": index, "brightness": self._BRIGHTNESS[index]},
        )


class _FakeODTarget:
    def __init__(self, labels: Any, boxes: Any, scores: Any) -> None:
        self.labels = labels
        self.boxes = boxes
        self.scores = scores


class _FakeODDataset:
    """Minimal MAITE object-detection dataset — two levels, uneven detections per image.

    The instance level is what makes the row layout non-trivial: image-level
    factors are propagated onto detection rows, and ``detection_area`` is
    binned at the instance level.  A cached Metadata has to reproduce both.
    """

    _DETECTIONS = (2, 0, 3, 1)

    @property
    def metadata(self) -> dict[str, Any]:
        return {"id": "fake-od", "index2label": {0: "cat", 1: "dog"}}

    def __len__(self) -> int:
        return len(self._DETECTIONS)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        count = self._DETECTIONS[index]
        target = _FakeODTarget(
            np.arange(count, dtype=np.intp) % 2,
            np.tile(np.array([0.0, 0.0, 4.0, 4.0], dtype=np.float32), (count, 1)),
            np.ones(count, dtype=np.float32),
        )
        return (
            np.zeros((1, 8, 8), dtype=np.float32),
            target,
            {
                "id": index,
                "brightness": 0.2 * index,
                "detection_area": [1.5 * (index + d) for d in range(count)],
            },
        )


def _make_metadata(dataset: Any = None, **kwargs: Any) -> Metadata:
    """Build a real, structured Metadata — what DatasetCache actually caches.

    Reaching for the real class rather than a mock is deliberate: the cache
    round-trips Metadata through ``save``/``load``, so a mock would keep passing
    while an upstream format change silently broke every cache hit.
    """
    metadata = Metadata(_FakeICDataset() if dataset is None else dataset, **kwargs)  # type: ignore
    metadata.dataframe  # noqa: B018  # force structuring, as a real workflow would
    return metadata


def _policy(**kwargs: Any) -> ResolvedPolicy:
    """A resolved policy in the shape the cache keys and builds from."""
    if "exclude" in kwargs:
        kwargs["exclude"] = tuple(kwargs["exclude"])
    return ResolvedPolicy(**kwargs)


def _selection_dir(cache_dir: Path, selection: str = "sel:all", dataset_name: str = "ds") -> Path:
    """Where ``DatasetCache`` writes one selection's artifacts."""
    return cache_dir / f"v{CACHE_VERSION}" / dataset_name / f"sel_{_config_hash(selection)}"


def _default_metadata_path(cache_dir: Path, selection: str = "sel:all", dataset_name: str = "ds") -> Path:
    """Where the *default* binning configuration's archive would go, written or not.

    Named rather than globbed, for the tests that plant a corrupt file before any save.
    """
    return _selection_dir(cache_dir, selection, dataset_name) / f"{_metadata_config_key()}.dem"


def _metadata_path(cache_dir: Path, selection: str = "sel:all", dataset_name: str = "ds") -> Path:
    """The metadata archive for a selection.

    Globbed rather than named: an archive is keyed by the binning configuration that
    wrote it, and a test that pinned the hash would be asserting the hash rather than
    the behaviour.  Every caller here writes exactly one.
    """
    archives = sorted(_selection_dir(cache_dir, selection, dataset_name).glob("metadata_*.dem"))
    assert len(archives) == 1, f"expected one archive, found {[p.name for p in archives]}"
    return archives[0]


def _rewrite_manifest(path: Path, **changes: Any) -> None:
    """Rewrite the manifest inside a ``.dem`` archive, leaving every other member."""
    import json
    import zipfile

    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    manifest.update(changes)
    members["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


class _ShortICDataset(_FakeICDataset):
    """Same shape as :class:`_FakeICDataset`, three items instead of five."""

    _BRIGHTNESS = (0.1, 0.3, 0.5)
    _LABELS = (0, 1, 0)


class TestMetadataCache:
    def test_save_creates_archive(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")

        cache.save_metadata("sel:all", _make_metadata())

        path = _metadata_path(tmp_path)
        assert path.exists()
        # The old two-file layout is gone: one self-describing archive replaces it.
        assert not (path.parent / "metadata.parquet").exists()
        assert not (path.parent / "metadata.json").exists()

    def test_save_writes_manifest_and_frames(self, tmp_path: Path):
        """The archive is dataeval's own container: a manifest plus a frame per level."""
        import zipfile

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_metadata("sel:all", _make_metadata())

        with zipfile.ZipFile(_metadata_path(tmp_path)) as archive:
            names = archive.namelist()
        assert "manifest.json" in names
        assert any(n.startswith("frames/") and n.endswith(".parquet") for n in names)

    def test_save_strips_binned_columns(self, tmp_path: Path):
        """Binned (↕) and digitized (#) columns are dropped before saving.

        This is what lets one archive serve every binning configuration a caller
        might read it back with.
        """
        import io
        import zipfile

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_metadata()
        meta.factor_data  # noqa: B018  # force binning, adding ↕/# companion columns
        assert any(c.endswith(("↕", "#")) for c in meta.dataframe.columns)

        cache.save_metadata("sel:all", meta)

        with zipfile.ZipFile(_metadata_path(tmp_path)) as archive:
            frames = [n for n in archive.namelist() if n.startswith("frames/")]
            columns = [c for n in frames for c in pl.read_parquet(io.BytesIO(archive.read(n))).columns]
        assert "brightness" in columns
        assert not [c for c in columns if c.endswith(("↕", "#"))]

    def test_save_persists_factors_only_metadata(self, tmp_path: Path):
        """A factors-only Metadata now round-trips instead of being skipped."""
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        meta = Metadata.from_factors({"brightness": np.array([0.1, 0.5, 0.9])}, np.array([0, 1, 0]), level="unit")

        cache.save_metadata("sel:all", meta)
        cache._memory.clear()  # Force disk round-trip
        loaded = cache.load_metadata("sel:all", None)  # type: ignore

        assert _metadata_path(tmp_path).exists()
        assert loaded is not None
        np.testing.assert_array_equal(loaded.factor_data, meta.factor_data)

    def test_load_returns_none_on_miss(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        assert cache.load_metadata("sel:all", _FakeICDataset()) is None  # type: ignore

    def test_load_returns_none_on_format_drift(self, tmp_path: Path):
        """A file written by an incompatible dataeval is a miss, not a wrong answer.

        This is the designed outcome upstream: ``MetadataFormatError`` means
        recompute, so it must never surface as an exception to a workflow.
        """
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_metadata("sel:all", _make_metadata())
        _rewrite_manifest(_metadata_path(tmp_path), format_version=999)
        cache._memory.clear()  # Force disk round-trip

        assert cache.load_metadata("sel:all", _FakeICDataset()) is None  # type: ignore

    def test_load_returns_none_on_schema_drift(self, tmp_path: Path):
        """Rows laid out against a different level graph are refused, not gathered."""
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_metadata("sel:all", _make_metadata())
        _rewrite_manifest(_metadata_path(tmp_path), levels=["unit", "instance", "invented"])
        cache._memory.clear()  # Force disk round-trip

        assert cache.load_metadata("sel:all", _FakeICDataset()) is None  # type: ignore

    def test_load_returns_none_on_dataset_length_mismatch(self, tmp_path: Path):
        """Rows saved for one dataset must not be served against a different one.

        Upstream raises ``ValueError`` rather than ``MetadataFormatError`` here, so
        the cache has to catch it separately or a mismatched dataset kills the run.
        """
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_metadata("sel:all", _make_metadata(_FakeICDataset()))
        cache._memory.clear()  # Force disk round-trip

        assert cache.load_metadata("sel:all", _ShortICDataset()) is None  # type: ignore

    def test_load_reconstructs_metadata(self, tmp_path: Path):
        """Full save → load round trip with real Metadata reconstruction."""
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        dataset = _FakeICDataset()
        meta = _make_metadata(dataset)

        cache.save_metadata("sel:all", meta)
        cache._memory.clear()  # Force disk round-trip
        loaded = cache.load_metadata("sel:all", dataset)  # type: ignore

        assert loaded is not None
        assert loaded.item_count == 5
        np.testing.assert_array_equal(loaded.class_labels, np.array([0, 1, 0, 1, 2], dtype=np.intp))
        assert loaded.index2label[0] == "cat"
        assert loaded.index2label[1] == "dog"
        assert loaded.index2label[2] == "bird"
        np.testing.assert_array_equal(loaded.item_indices, np.array([0, 1, 2, 3, 4], dtype=np.intp))
        assert list(loaded.factor_names) == list(meta.factor_names)
        # Binning the cached raw data must land exactly where a fresh build does.
        np.testing.assert_array_equal(loaded.factor_data, meta.factor_data)

    def test_load_reconstructs_object_detection_levels(self, tmp_path: Path):
        """Detection rows, their links and their own factors survive the round trip."""
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        dataset = _FakeODDataset()
        meta = _make_metadata(dataset)

        cache.save_metadata("sel:all", meta)
        cache._memory.clear()  # Force disk round-trip
        loaded = cache.load_metadata("sel:all", dataset)  # type: ignore

        assert loaded is not None
        assert loaded.multi_target is True
        assert dict(loaded.level_counts) == dict(meta.level_counts)
        np.testing.assert_array_equal(loaded.class_labels, meta.class_labels)
        np.testing.assert_array_equal(loaded.item_indices, meta.item_indices)
        # detection_area is binned at the instance level, brightness at the unit
        # level and then propagated onto the detection rows — both need the links.
        np.testing.assert_array_equal(loaded.factor_data, meta.factor_data)
        np.testing.assert_array_equal(loaded.at("unit").factor_data, meta.at("unit").factor_data)

    def test_load_leaves_binning_lazy(self, tmp_path: Path):
        """Loaded metadata is structured but not yet binned, so _bin() runs lazily."""
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")

        cache.save_metadata("sel:all", _make_metadata())
        cache._memory.clear()  # Force disk round-trip
        loaded = cache.load_metadata("sel:all", _FakeICDataset())  # type: ignore

        assert loaded is not None
        assert loaded._is_binned is False
        assert loaded._is_structured is True

    def test_load_omits_raw(self, tmp_path: Path):
        """The per-item dicts are not written, and a loaded instance says so.

        Upstream raises rather than returning ``[]``, which would read as a dataset
        that carried no metadata at all.
        """
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")

        cache.save_metadata("sel:all", _make_metadata())
        cache._memory.clear()  # Force disk round-trip
        loaded = cache.load_metadata("sel:all", _FakeICDataset())  # type: ignore

        assert loaded is not None
        with pytest.raises(ValueError, match="loaded from a file"):
            loaded.raw  # noqa: B018

    def test_load_applies_caller_config(self, tmp_path: Path):
        """Loaded metadata has the caller's binning config applied.

        Saved under the same configuration it is read back with, since an archive
        serves the configuration that wrote it — see
        ``test_archive_is_not_reused_across_binning_configs``.
        """
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        config: dict[str, Any] = {
            "auto_bin_method": "clusters",
            "exclude": ["x"],
            "continuous_factor_bins": {"y": [0.0, 1.0]},
        }

        cache.save_metadata("sel:all", _make_metadata(**config), _policy(**config))
        cache._memory.clear()  # Force disk round-trip
        loaded = cache.load_metadata("sel:all", _FakeICDataset(), _policy(**config))  # type: ignore

        assert loaded is not None
        assert loaded.auto_bin_method == "clusters"
        assert loaded.exclude == {"x"}
        assert dict(loaded.continuous_factor_bins) == {"y": [0.0, 1.0]}

    def test_load_preserves_task(self, tmp_path: Path):
        """The structurer is rebuilt, so the level model survives the round trip."""
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_metadata()

        cache.save_metadata("sel:all", meta)
        cache._memory.clear()  # Force disk round-trip
        loaded = cache.load_metadata("sel:all", _FakeICDataset())  # type: ignore

        assert loaded is not None
        assert loaded.multi_target is False
        assert list(loaded.levels) == list(meta.levels)
        assert loaded.item_level == meta.item_level
        assert loaded.label_level == meta.label_level

    def test_load_dataframe_accessible(self, tmp_path: Path):
        """Loaded metadata exposes the cached rows as a flat frame."""
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_metadata()

        cache.save_metadata("sel:all", meta)
        cache._memory.clear()  # Force disk round-trip
        loaded = cache.load_metadata("sel:all", _FakeICDataset())  # type: ignore

        assert loaded is not None
        assert "brightness" in loaded.dataframe.columns
        assert len(loaded.dataframe) == len(meta.dataframe)

    def test_archive_is_not_reused_across_binning_configs(self, tmp_path: Path):
        """An archive carries the encoding it was written under, so it serves only that.

        DataEval v1.1 persists the encoding record and restores it *underneath* whatever
        the reader declares, for every factor the reader did not name.  A shared archive
        therefore handed a run configured for one auto-bin method the edges another run
        derived, with no notice — the archive won, and only the digest showed it.
        """
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")

        cache.save_metadata(
            "sel:all", _make_metadata(auto_bin_method="uniform_width"), _policy(auto_bin_method="uniform_width")
        )
        cache._memory.clear()  # Force disk round-trip

        same = cache.load_metadata("sel:all", _FakeICDataset(), _policy(auto_bin_method="uniform_width"))  # type: ignore
        other = cache.load_metadata("sel:all", _FakeICDataset(), _policy(auto_bin_method="clusters"))  # type: ignore

        assert same is not None, "the configuration that wrote the archive must hit"
        assert same.auto_bin_method == "uniform_width"
        assert other is None, "a different configuration must miss rather than inherit the archive's cuts"

    def test_save_writes_a_readable_policy_sidecar(self, tmp_path: Path):
        """The hash in the filename gets a human-readable expansion beside it."""
        import json

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_metadata("sel:all", _make_metadata())

        sidecar = _metadata_path(tmp_path).with_suffix(".json")
        assert sidecar.exists(), "an archive should name its own policy"
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        assert "factors" in document
        assert all("provenance" in entry for entry in document["factors"].values())


class TestMetadataMemoryCache:
    """Both the in-process cache and the archive are keyed by binning config.

    A memoized ``Metadata`` carries the configuration it was built with, so serving one
    to a caller that asked for different bins hands back the wrong binning silently.
    The archive has the same problem for a different reason — it carries the encoding
    record it was written under, restored underneath whatever the reader declares — so
    both are keyed rather than only the memory entry.
    """

    def _build(self, **kwargs: Any) -> Any:
        return {k: v for k, v in kwargs.items() if v is not None}

    def test_same_config_served_from_memory(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        dataset = _FakeICDataset()

        first = cache.load_or_compute_metadata("sel:all", dataset)  # type: ignore
        second = cache.load_or_compute_metadata("sel:all", dataset)  # type: ignore

        assert second is first

    def test_different_bins_do_not_reuse_memoized_instance(self, tmp_path: Path):
        """The regression: coverage asks for custom bins after analysis used defaults."""
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        dataset = _FakeICDataset()

        default = cache.load_or_compute_metadata("sel:all", dataset)  # type: ignore
        custom = cache.load_or_compute_metadata(  # type: ignore
            "sel:all", dataset, _policy(continuous_factor_bins={"brightness": 2})
        )

        assert custom is not default
        assert dict(custom.continuous_factor_bins) == {"brightness": 2}
        assert dict(default.continuous_factor_bins) == {}

    def test_different_auto_bin_method_does_not_reuse(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        dataset = _FakeICDataset()

        cache.load_or_compute_metadata("sel:all", dataset, _policy(auto_bin_method="uniform_width"))  # type: ignore
        clustered = cache.load_or_compute_metadata("sel:all", dataset, _policy(auto_bin_method="clusters"))  # type: ignore

        assert clustered.auto_bin_method == "clusters"

    def test_different_exclude_does_not_reuse(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        dataset = _FakeICDataset()

        cache.load_or_compute_metadata("sel:all", dataset)  # type: ignore
        excluded = cache.load_or_compute_metadata("sel:all", dataset, _policy(exclude=["brightness"]))  # type: ignore

        assert excluded.exclude == {"brightness"}
        assert "brightness" not in excluded.factor_names

    def test_second_config_rebuilds_rather_than_inheriting(self, tmp_path: Path):
        """A different binning config costs a re-walk, which is the price of not lying.

        It used to cost only a disk read, because one archive served every
        configuration.  That stopped being true when the encoding record began
        travelling with the archive: the cheap read returned the *writer's* cuts.
        Rebuilding is the conservative half of that trade, and each configuration keeps
        its own warm entry afterwards.
        """
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        dataset = _FakeICDataset()

        with patch(
            "dataeval_flow.metadata.build_metadata",
            side_effect=lambda ds, policy=None: _make_metadata(ds, **(policy or _policy()).metadata_kwargs()),
        ) as mock_build:
            cache.load_or_compute_metadata("sel:all", dataset)  # type: ignore
            cache.load_or_compute_metadata(  # type: ignore
                "sel:all", dataset, _policy(continuous_factor_bins={"brightness": 2})
            )

        assert mock_build.call_count == 2


# ---------------------------------------------------------------------------
# DatasetCache.load_or_compute_metadata
# ---------------------------------------------------------------------------


class TestLoadOrComputeMetadata:
    _BUILD_METADATA_PATH = "dataeval_flow.metadata.build_metadata"

    def test_full_miss_computes_and_saves(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_metadata()

        with patch(self._BUILD_METADATA_PATH, return_value=meta) as mock_build:
            result = cache.load_or_compute_metadata("sel:all", MagicMock())
            mock_build.assert_called_once()

        assert result is meta
        # Verify saved to disk, not merely memoized
        cache._memory.clear()
        cached = cache.load_metadata("sel:all", _FakeICDataset())  # type: ignore
        assert cached is not None
        assert cached.item_count == 5

    def test_full_hit_skips_compute(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_metadata()

        # Pre-populate cache
        cache.save_metadata("sel:all", meta)

        with patch(self._BUILD_METADATA_PATH) as mock_build:
            result = cache.load_or_compute_metadata("sel:all", MagicMock())
            mock_build.assert_not_called()

        assert result is not None
        assert result.item_count == 5

    def test_passes_the_policy_to_build(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_metadata()
        dataset = MagicMock()
        policy = _policy(auto_bin_method="uniform_width", exclude=["x"], continuous_factor_bins={"y": [0.0, 1.0]})

        with patch(self._BUILD_METADATA_PATH, return_value=meta) as mock_build:
            cache.load_or_compute_metadata("sel:all", dataset, policy)
            mock_build.assert_called_once_with(dataset, policy)

    def test_different_configs_get_separate_cache_entries(self, tmp_path: Path):
        """Different binning configs are different entries, on disk as well as in memory.

        No memory clearing between the calls: the binning-aware key is what keeps the
        second caller from being handed the first caller's binning, and it now applies
        to the archive too because the archive carries its own encoding record.
        """
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        dataset = _FakeICDataset()

        build = lambda ds, policy=None: _make_metadata(ds, **(policy or _policy()).metadata_kwargs())  # noqa: E731

        with patch(self._BUILD_METADATA_PATH, side_effect=build) as mock_build:
            cache.load_or_compute_metadata("sel:all", dataset, _policy(auto_bin_method="uniform_width"))  # type: ignore
            mock_build.assert_called_once()

        with patch(self._BUILD_METADATA_PATH, side_effect=build) as mock_build:
            result = cache.load_or_compute_metadata("sel:all", dataset, _policy(auto_bin_method="clusters"))  # type: ignore
            mock_build.assert_called_once()

        assert result is not None
        assert result.auto_bin_method == "clusters"

        archives = sorted(p.name for p in _selection_dir(tmp_path).glob("metadata_*.dem"))
        assert len(archives) == 2, f"each configuration wants its own archive, found {archives}"


# ---------------------------------------------------------------------------
# get_or_compute_metadata (top-level function)
# ---------------------------------------------------------------------------


class TestGetOrComputeMetadata:
    _BUILD_METADATA_PATH = "dataeval_flow.metadata.build_metadata"

    def test_without_cache_computes_directly(self):
        meta = _make_metadata()

        with patch(self._BUILD_METADATA_PATH, return_value=meta) as mock_build:
            result = get_or_compute_metadata(dataset=MagicMock())
            mock_build.assert_called_once()

        assert result is meta

    def test_with_cache_delegates_to_workflow_cache(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_metadata()

        with active_cache(cache, "sel:all"), patch(self._BUILD_METADATA_PATH, return_value=meta) as mock_build:
            result = get_or_compute_metadata(dataset=MagicMock())
            mock_build.assert_called_once()

        assert result is meta
        # Should be saved to cache
        cache._memory.clear()
        cached = cache.load_metadata("sel:all", _FakeICDataset())  # type: ignore
        assert cached is not None

    def test_with_cache_full_hit_skips_compute(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_metadata()

        # Pre-populate cache
        cache.save_metadata("sel:all", meta)

        with active_cache(cache, "sel:all"), patch(self._BUILD_METADATA_PATH) as mock_build:
            result = get_or_compute_metadata(dataset=MagicMock())
            mock_build.assert_not_called()

        assert result is not None
        assert result.item_count == 5

    def test_no_active_cache_computes_directly(self):
        """When no active_cache context is set, computes directly without caching."""
        meta = _make_metadata()

        with patch(self._BUILD_METADATA_PATH, return_value=meta) as mock_build:
            result = get_or_compute_metadata(dataset=MagicMock())
            mock_build.assert_called_once()

        assert result is meta

    def test_the_policy_is_forwarded(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_metadata()
        dataset = MagicMock()
        policy = _policy(auto_bin_method="uniform_width", exclude=["a"], continuous_factor_bins={"b": [0.0, 0.5, 1.0]})

        with active_cache(cache, "sel:all"), patch(self._BUILD_METADATA_PATH, return_value=meta) as mock_build:
            get_or_compute_metadata(dataset, policy)
            mock_build.assert_called_once_with(dataset, policy)


# ---------------------------------------------------------------------------
# Cross-dataset isolation
# ---------------------------------------------------------------------------


class TestCacheIsolation:
    def test_different_datasets_independent(self, tmp_path: Path):
        cache_a = DatasetCache(cache_dir=tmp_path, dataset_name="ds_a")
        cache_b = DatasetCache(cache_dir=tmp_path, dataset_name="ds_b")

        arr_a = np.ones((3, 4))
        arr_b = np.zeros((3, 4))

        cache_a.save_embeddings("sel:all", "cfg", "none", arr_a)
        cache_b.save_embeddings("sel:all", "cfg", "none", arr_b)

        loaded_a = cache_a.load_embeddings("sel:all", "cfg", "none")
        loaded_b = cache_b.load_embeddings("sel:all", "cfg", "none")

        assert loaded_a is not None
        assert loaded_b is not None
        np.testing.assert_array_equal(loaded_a, arr_a)
        np.testing.assert_array_equal(loaded_b, arr_b)

    def test_all_component_types_in_same_cache(self, tmp_path: Path):
        """All component types can coexist in the same dataset directory."""
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")

        # Embeddings
        cache.save_embeddings("sel:all", "cfg", "none", np.ones((2, 3)))
        # Stats (unified)
        cache.save_stats("sel:all", "img+tgt", _make_calc_result(2))
        # Metadata
        cache.save_metadata("sel:all", _make_metadata())

        # All should be loadable
        assert cache.load_embeddings("sel:all", "cfg", "none") is not None
        assert cache.load_stats("sel:all", "img+tgt") is not None
        assert cache.load_metadata("sel:all", _FakeICDataset()) is not None  # type: ignore


# ---------------------------------------------------------------------------
# Cache version isolation
# ---------------------------------------------------------------------------


class TestCacheVersioning:
    def test_data_stored_under_version_directory(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_embeddings("sel:all", "cfg", "none", np.ones((2, 3)))
        assert (tmp_path / f"v{CACHE_VERSION}" / "ds").is_dir()

    def test_different_versions_are_isolated(self, tmp_path: Path, monkeypatch: Any):
        """Bumping CACHE_VERSION produces a separate directory."""
        import dataeval_flow.cache as cache_mod

        # Save with current version
        monkeypatch.setattr(cache_mod, "CACHE_VERSION", "1")
        cache_v1 = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        arr_v1 = np.ones((2, 3))
        cache_v1.save_embeddings("sel:all", "cfg", "none", arr_v1)

        # Simulate a version bump
        monkeypatch.setattr(cache_mod, "CACHE_VERSION", "2")
        cache_v2 = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        arr_v2 = np.zeros((2, 3))
        cache_v2.save_embeddings("sel:all", "cfg", "none", arr_v2)

        # Both directories exist side-by-side
        assert (tmp_path / "v1" / "ds").is_dir()
        assert (tmp_path / "v2" / "ds").is_dir()

        # Each version loads its own data
        loaded_v2 = cache_v2.load_embeddings("sel:all", "cfg", "none")
        assert loaded_v2 is not None
        np.testing.assert_array_equal(loaded_v2, arr_v2)

        # Restore original version and verify v1 data
        monkeypatch.setattr(cache_mod, "CACHE_VERSION", "1")
        cache_v1_reload = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        loaded_v1 = cache_v1_reload.load_embeddings("sel:all", "cfg", "none")
        assert loaded_v1 is not None
        np.testing.assert_array_equal(loaded_v1, arr_v1)

    def test_version_miss_after_bump(self, tmp_path: Path, monkeypatch: Any):
        """After a version bump, old data is not returned."""
        import dataeval_flow.cache as cache_mod

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_embeddings("sel:all", "cfg", "none", np.ones((2, 3)))

        # Bump version — old cache should miss
        monkeypatch.setattr(cache_mod, "CACHE_VERSION", "99")
        cache_new = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        assert cache_new.load_embeddings("sel:all", "cfg", "none") is None


# ---------------------------------------------------------------------------
# Integration with WorkflowContext and TaskConfig
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_dataset_context_accepts_cache(self, tmp_path: Path):
        from unittest.mock import MagicMock

        from dataeval_flow.workflow import DatasetContext

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        dc = DatasetContext(name="ds", dataset=MagicMock(), cache=cache)
        assert dc.cache is cache

    def test_dataset_context_cache_defaults_none(self):
        from unittest.mock import MagicMock

        from dataeval_flow.workflow import DatasetContext

        dc = DatasetContext(name="ds", dataset=MagicMock())
        assert dc.cache is None


# ---------------------------------------------------------------------------
# _atomic_write — exception cleanup path (lines 167-169)
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_cleans_up_temp_file_on_error(self, tmp_path: Path):
        """If the data_fn raises, the temp file is removed."""
        target = tmp_path / "output.npy"

        def failing_fn(p: Path) -> None:
            p.write_text("partial")
            raise RuntimeError("write failed")

        with pytest.raises(RuntimeError, match="write failed"):
            _atomic_write(target, failing_fn)

        assert not target.exists()
        # Temp file should also be cleaned up
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# get_or_compute_cluster_result (top-level function, lines 374-377)
# ---------------------------------------------------------------------------


class TestGetOrComputeClusterResult:
    _CLUSTER_PATH = "dataeval.core._clusterer.cluster"

    def test_without_cache_computes_directly(self):
        embeddings = np.random.default_rng(42).random((20, 8)).astype(np.float32)
        mock_result = _MockClusterResult()

        with patch(self._CLUSTER_PATH, return_value=mock_result) as mock_cluster:
            result = get_or_compute_cluster_result(embeddings=embeddings, algorithm="hdbscan", n_clusters=None)
            mock_cluster.assert_called_once()

        assert result is mock_result

    def test_with_extractor_config_and_transforms(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        embeddings = np.random.default_rng(42).random((20, 8)).astype(np.float32)
        mock_result = _MockClusterResult()

        ext_cfg = MagicMock()
        ext_cfg.model_dump_json.return_value = '{"type":"onnx"}'
        transforms = MagicMock()
        transforms.__repr__ = MagicMock(return_value="my_transforms")

        with active_cache(cache, "sel:all"), patch(self._CLUSTER_PATH, return_value=mock_result) as mock_cluster:
            result = get_or_compute_cluster_result(
                embeddings=embeddings, algorithm="kmeans", n_clusters=5, extractor_config=ext_cfg, transforms=transforms
            )
            mock_cluster.assert_called_once()

        assert result is mock_result


# ---------------------------------------------------------------------------
# DatasetCache.get_or_create with cache_dir (line 438)
# ---------------------------------------------------------------------------


class TestGetOrCreateDiskBacked:
    def test_disk_backed_returns_fresh_instance(self, tmp_path: Path):
        cache1 = DatasetCache.get_or_create(tmp_path, name="myds", cache_key='{"name":"ds","path":"/data"}')
        cache2 = DatasetCache.get_or_create(tmp_path, name="myds", cache_key='{"name":"ds","path":"/data"}')

        # Disk-backed always creates fresh instances (not singletons)
        assert cache1 is not cache2
        assert cache1.disk_backed is True
        assert cache1.cache_dir == tmp_path


# ---------------------------------------------------------------------------
# persist_memory=False (instance attribute)
# ---------------------------------------------------------------------------


class TestPersistMemoryDisabled:
    def test_mem_get_returns_none_when_disabled(self):
        cache = DatasetCache(cache_dir=None, dataset_name="ds", persist_memory=False)
        # Manually store something to prove _mem_get bypasses it
        cache._memory.setdefault("sel:all", {})["key"] = "value"
        assert cache._mem_get("sel:all", "key") is None

    def test_mem_set_is_noop_when_disabled(self):
        cache = DatasetCache(cache_dir=None, dataset_name="ds", persist_memory=False)
        cache._mem_set("sel:all", "key", "value")
        assert cache._memory == {}


# ---------------------------------------------------------------------------
# disk_backed property (line 484)
# ---------------------------------------------------------------------------


class TestDiskBackedProperty:
    def test_disk_backed_true(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        assert cache.disk_backed is True

    def test_disk_backed_false(self):
        cache = DatasetCache(cache_dir=None, dataset_name="ds")
        assert cache.disk_backed is False


# ---------------------------------------------------------------------------
# Cluster result cache (lines 644-754)
# ---------------------------------------------------------------------------


class _MockClusterResult:
    """dict()-compatible mock that mimics ClusterResult for save_cluster_result."""

    def __init__(self) -> None:
        self._data = _make_cluster_dict()

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key: str):
        return self._data[key]

    def keys(self):
        return self._data.keys()


def _make_cluster_dict() -> dict[str, Any]:
    """Build a minimal cluster result dict for testing."""
    rng = np.random.default_rng(42)
    return {
        "clusters": rng.integers(0, 3, size=10),
        "mst": rng.random((9, 3)),
        "linkage_tree": rng.random((9, 4)),
        "membership_strengths": rng.random(10),
        "k_neighbors": rng.integers(0, 10, size=(10, 5)),
        "k_distances": rng.random((10, 5)),
    }


class TestClusterResultCache:
    def test_save_and_load_round_trip(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        result = _make_cluster_dict()

        cache.save_cluster_result("sel:all", "cfg", "none", "hdbscan", None, result)
        loaded = cache.load_cluster_result("sel:all", "cfg", "none", "hdbscan", None)

        assert loaded is not None
        for key in result:
            np.testing.assert_array_equal(loaded[key], result[key])

    def test_load_returns_none_on_miss(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        assert cache.load_cluster_result("sel:all", "cfg", "none", "hdbscan", None) is None

    def test_load_returns_none_on_corrupted_file(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        result = _make_cluster_dict()
        cache.save_cluster_result("sel:all", "cfg", "none", "hdbscan", None, result)

        # Corrupt the file
        path = cache._cluster_path("sel:all", "cfg", "none", "hdbscan", None)
        assert path is not None
        path.write_bytes(b"corrupted data")
        cache._memory.clear()

        loaded = cache.load_cluster_result("sel:all", "cfg", "none", "hdbscan", None)
        assert loaded is None

    def test_memory_hit_skips_disk(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        result = _make_cluster_dict()
        cache.save_cluster_result("sel:all", "cfg", "none", "hdbscan", None, result)

        # Second load should hit memory (no disk read needed)
        loaded = cache.load_cluster_result("sel:all", "cfg", "none", "hdbscan", None)
        assert loaded is not None

    def test_save_no_disk_when_not_disk_backed(self):
        cache = DatasetCache(cache_dir=None, dataset_name="ds")
        result = _make_cluster_dict()
        # Should not raise — save is a no-op for disk
        cache.save_cluster_result("sel:all", "cfg", "none", "kmeans", 3, result)
        # But memory cache should still work
        loaded = cache.load_cluster_result("sel:all", "cfg", "none", "kmeans", 3)
        assert loaded is not None

    def test_different_algorithms_different_keys(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        result_a = _make_cluster_dict()
        result_b = _make_cluster_dict()
        result_b["clusters"] = np.zeros(10, dtype=int)

        cache.save_cluster_result("sel:all", "cfg", "none", "hdbscan", None, result_a)
        cache.save_cluster_result("sel:all", "cfg", "none", "kmeans", 5, result_b)

        loaded_a = cache.load_cluster_result("sel:all", "cfg", "none", "hdbscan", None)
        loaded_b = cache.load_cluster_result("sel:all", "cfg", "none", "kmeans", 5)

        assert loaded_a is not None
        assert loaded_b is not None
        assert not np.array_equal(loaded_a["clusters"], loaded_b["clusters"])


class TestLoadOrComputeClusterResult:
    _CLUSTER_PATH = "dataeval.core._clusterer.cluster"

    def test_miss_computes_and_caches(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        embeddings = np.random.default_rng(42).random((20, 8)).astype(np.float32)

        mock_cr = _MockClusterResult()

        with patch(self._CLUSTER_PATH, return_value=mock_cr) as mock_cluster:
            cache.load_or_compute_cluster_result("sel:all", "cfg", "none", embeddings, "hdbscan", None)
            mock_cluster.assert_called_once()

    def test_hit_skips_compute(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        result = _make_cluster_dict()
        cache.save_cluster_result("sel:all", "cfg", "none", "hdbscan", None, result)

        embeddings = np.random.default_rng(42).random((20, 8)).astype(np.float32)

        with patch(self._CLUSTER_PATH) as mock_cluster:
            loaded = cache.load_or_compute_cluster_result("sel:all", "cfg", "none", embeddings, "hdbscan", None)
            mock_cluster.assert_not_called()

        assert loaded is not None


# ---------------------------------------------------------------------------
# _make_dataset_id
# ---------------------------------------------------------------------------


class TestMakeDsId:
    @staticmethod
    def _key(name: str, **kwargs: Any) -> tuple[str, str]:
        """Return (name, cache_key) for _make_dataset_id."""

        from dataeval_flow.config import HuggingFaceDatasetConfig

        defaults: dict[str, Any] = {"path": f"data/{name}", "split": "train", "task": "image_classification"}
        defaults.update(kwargs)
        cfg = HuggingFaceDatasetConfig(name=name, **defaults)
        return name, cfg.model_dump_json(exclude_defaults=False)

    def test_single_dataset_has_name_prefix_and_hash(self):
        result = _make_dataset_id(*self._key("my_dataset"))
        assert result.startswith("my_dataset_")
        hash_suffix = result.rsplit("_", 1)[-1]
        assert len(hash_suffix) == 16
        int(hash_suffix, 16)

    def test_different_split_different_id(self):
        """Changing split (but keeping name) must produce a different cache id."""
        train = _make_dataset_id(*self._key("ds", split="train"))
        test = _make_dataset_id(*self._key("ds", split="test"))
        assert train != test

    def test_different_path_different_id(self):
        """Changing path (but keeping name) must produce a different cache id."""
        a = _make_dataset_id(*self._key("ds", path="data/a"))
        b = _make_dataset_id(*self._key("ds", path="data/b"))
        assert a != b

    def test_same_config_deterministic(self):
        """Same config always produces the same output."""
        key = self._key("my_dataset")
        assert _make_dataset_id(*key) == _make_dataset_id(*key)

    def test_long_name_truncated(self):
        """Result must fit within _MAX_DS_ID_BYTES."""
        result = _make_dataset_id(*self._key("a" * 200))
        assert len(result.encode("utf-8")) <= 100
        hash_suffix = result.rsplit("_", 1)[-1]
        assert len(hash_suffix) == 16
        int(hash_suffix, 16)

    def test_different_configs_different_ids(self):
        """Different dataset configs produce different hashed IDs."""
        a = _make_dataset_id(*self._key("dataset_a"))
        b = _make_dataset_id(*self._key("dataset_b"))
        assert a != b


# ---------------------------------------------------------------------------
# _atomic_write_pair — error cleanup (lines 227-235)
# ---------------------------------------------------------------------------


class TestAtomicWritePairCleanup:
    def test_cleans_up_on_first_write_failure(self, tmp_path: Path):
        target_a = tmp_path / "out_a.parquet"
        target_b = tmp_path / "out_b.json"

        def failing_fn(p: Path) -> None:
            p.write_text("partial")
            raise ValueError("write_a failed")

        def ok_fn(p: Path) -> None:
            p.write_text("{}")

        with pytest.raises(ValueError, match="write_a failed"):
            _atomic_write_pair(target_a, failing_fn, target_b, ok_fn)

        assert not target_a.exists()
        assert not target_b.exists()

    def test_cleans_orphaned_target_on_second_rename_failure(self, tmp_path: Path):
        target_a = tmp_path / "out_a.parquet"
        target_b = tmp_path / "out_b.json"

        def ok_fn_a(p: Path) -> None:
            p.write_bytes(b"data_a")

        def ok_fn_b(p: Path) -> None:
            p.write_bytes(b"data_b")

        call_count = {"n": 0}
        original_rename = Path.rename

        def patched_rename(self_path, target):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise OSError("second rename failed")
            return original_rename(self_path, target)

        with patch.object(Path, "rename", patched_rename), pytest.raises(OSError, match="second rename failed"):
            _atomic_write_pair(target_a, ok_fn_a, target_b, ok_fn_b)

        assert not target_a.exists()
        assert not target_b.exists()


# ---------------------------------------------------------------------------
# DatasetCache.clear_instances() (lines 634-635)
# ---------------------------------------------------------------------------


class TestClearInstances:
    def test_empties_registry(self):
        DatasetCache.get_or_create(None, name="ds_a", cache_key="key_a")
        DatasetCache.get_or_create(None, name="ds_b", cache_key="key_b")
        assert len(DatasetCache._instances) >= 2

        DatasetCache.clear_instances()
        assert DatasetCache._instances == {}

    def test_new_creation_after_clear(self):
        first = DatasetCache.get_or_create(None, name="ds", cache_key="key")
        DatasetCache.clear_instances()
        second = DatasetCache.get_or_create(None, name="ds", cache_key="key")
        assert first is not second


# ---------------------------------------------------------------------------
# _embeddings_path / save_embeddings when not disk-backed (lines 730, 777)
# ---------------------------------------------------------------------------


class TestEmbeddingsNoDisk:
    def test_embeddings_path_returns_none(self):
        cache = DatasetCache(cache_dir=None, dataset_name="ds")
        assert cache._embeddings_path("sel:all", '{"type":"onnx"}', "none") is None

    def test_save_embeddings_noop_stores_in_memory(self):
        cache = DatasetCache(cache_dir=None, dataset_name="ds")
        arr = np.ones((3, 4), dtype=np.float32)
        cache.save_embeddings("sel:all", "cfg", "none", arr)

        loaded = cache.load_embeddings("sel:all", "cfg", "none")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, arr)


# ---------------------------------------------------------------------------
# load_cluster_result from disk (lines 863-872)
# ---------------------------------------------------------------------------


class TestLoadClusterResultFromDisk:
    def test_round_trip_after_memory_clear(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cluster_dict = _make_cluster_dict()

        cache.save_cluster_result("sel:all", "cfg", "none", "hdbscan", None, cluster_dict)
        cache._memory.clear()

        loaded = cache.load_cluster_result("sel:all", "cfg", "none", "hdbscan", None)
        assert loaded is not None
        for key in cluster_dict:
            np.testing.assert_array_equal(loaded[key], cluster_dict[key])

    def test_populates_memory_after_disk_load(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_cluster_result("sel:all", "cfg", "none", "hdbscan", None, _make_cluster_dict())
        cache._memory.clear()

        cache.load_cluster_result("sel:all", "cfg", "none", "hdbscan", None)

        obj_key = f"clusters_{_config_hash('cfg|none|hdbscan|None')}"
        assert cache._memory.get("sel:all", {}).get(obj_key) is not None


# ---------------------------------------------------------------------------
# _stats_paths / save_stats / load_stats (lines 1135, 1168-1190, 1217)
# ---------------------------------------------------------------------------


class TestStatsNoDisk:
    def test_stats_paths_returns_none_tuple(self):
        cache = DatasetCache(cache_dir=None, dataset_name="ds")
        pq, js = cache._stats_paths("sel:all", "img+tgt")
        assert pq is None
        assert js is None

    def test_save_stats_noop_stores_in_memory(self):
        cache = DatasetCache(cache_dir=None, dataset_name="ds")
        stats = _make_calc_result(2)
        cache.save_stats("sel:all", "img+tgt", stats)

        loaded = cache.load_stats("sel:all", "img+tgt")
        assert loaded is not None
        assert loaded["image_count"] == 2


class TestLoadStatsFromDisk:
    def test_numeric_column_round_trip(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(5)
        cache.save_stats("sel:all", "img+tgt", stats)
        cache._memory.clear()

        loaded = cache.load_stats("sel:all", "img+tgt")
        assert loaded is not None
        assert "mean" in loaded["stats"]
        np.testing.assert_array_almost_equal(loaded["stats"]["mean"], stats["stats"]["mean"])

    def test_utf8_column_round_trip(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(4, include_hashes=True)
        cache.save_stats("sel:all", "img+tgt", stats)
        cache._memory.clear()

        loaded = cache.load_stats("sel:all", "img+tgt")
        assert loaded is not None
        assert "xxhash" in loaded["stats"]
        assert loaded["stats"]["xxhash"].dtype == object

    def test_list_column_round_trip(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3, include_2d=True)
        cache.save_stats("sel:all", "img+tgt", stats)
        cache._memory.clear()

        loaded = cache.load_stats("sel:all", "img+tgt")
        assert loaded is not None
        assert "histogram" in loaded["stats"]
        assert loaded["stats"]["histogram"].ndim == 2

    def test_populates_memory_after_disk_load(self, tmp_path: Path):
        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(2)
        cache.save_stats("sel:all", "img+tgt", stats)
        cache._memory.clear()

        cache.load_stats("sel:all", "img+tgt")
        obj_key = f"stats_{_config_hash('img+tgt')}"
        assert cache._memory.get("sel:all", {}).get(obj_key) is not None

    def test_aux_fields_round_trip(self, tmp_path: Path):
        from dataeval.types import SourceIndex

        cache = DatasetCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3)
        cache.save_stats("sel:all", "img+tgt", stats)
        cache._memory.clear()

        loaded = cache.load_stats("sel:all", "img+tgt")
        assert loaded is not None
        assert loaded["image_count"] == 3
        assert len(loaded["source_index"]) == 3
        for si in loaded["source_index"]:
            assert isinstance(si, SourceIndex)
