"""Tests for WorkflowCache — disk-backed caching of workflow computations."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl

from dataeval_app.cache import (
    CACHE_VERSION,
    FLAG_TO_METRIC,
    METRIC_TO_FLAG,
    WorkflowCache,
    _config_hash,
    get_or_compute_embeddings,
    get_or_compute_metadata,
    get_or_compute_stats,
    missing_flags,
    scope_key,
    selection_repr,
)

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
# scope_key helper
# ---------------------------------------------------------------------------


class TestScopeKey:
    def test_default_scope(self):
        assert scope_key() == "img+tgt"

    def test_all_flags(self):
        assert scope_key(True, True, True) == "img+tgt+ch"

    def test_image_only(self):
        assert scope_key(True, False, False) == "img"

    def test_target_only(self):
        assert scope_key(False, True, False) == "tgt"

    def test_none_scope(self):
        assert scope_key(False, False, False) == "none"


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

    def test_resolves_dependencies(self):
        from dataeval.flags import ImageStats

        # entropy depends on histogram
        cached: set[str] = set()
        result = missing_flags(cached, ImageStats.PIXEL_ENTROPY)
        assert ImageStats.PIXEL_HISTOGRAM in result


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
# WorkflowCache — construction and directory layout
# ---------------------------------------------------------------------------


class TestWorkflowCacheInit:
    def test_stores_config(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="my_ds")
        assert cache.cache_dir == tmp_path
        assert cache.dataset_name == "my_ds"

    def test_dataset_dir_creates_hierarchy(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds1")
        dataset_dir = cache.dataset_dir
        assert dataset_dir == tmp_path / f"v{CACHE_VERSION}" / "ds1"
        assert dataset_dir.is_dir()


# ---------------------------------------------------------------------------
# Embeddings cache (.npy)
# ---------------------------------------------------------------------------


class TestEmbeddingsCache:
    def test_save_and_load_round_trip(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        arr = np.random.default_rng(42).random((100, 512)).astype(np.float32)

        cache.save_embeddings("sel:all", '{"type":"onnx"}', "none", arr)
        loaded = cache.load_embeddings("sel:all", '{"type":"onnx"}', "none")

        assert loaded is not None
        np.testing.assert_array_equal(loaded, arr)

    def test_load_returns_none_on_miss(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        assert cache.load_embeddings("sel:all", '{"type":"onnx"}') is None

    def test_different_config_different_file(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
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
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
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
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        arr = np.array([[1.0, 2.0]], dtype=np.float64)
        cache.save_embeddings("sel:all", "cfg", "none", arr)
        loaded = cache.load_embeddings("sel:all", "cfg", "none")
        assert loaded is not None
        assert loaded.dtype == np.float64

    def test_file_goes_to_correct_path(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="my_ds")
        arr = np.ones((2, 2))
        cache.save_embeddings("sel:all", "cfg", "tfm", arr)

        s = _config_hash("sel:all")
        h = _config_hash("cfg|tfm")
        expected = tmp_path / f"v{CACHE_VERSION}" / "my_ds" / f"embeddings_{s}_{h}.npy"
        assert expected.exists()


# ---------------------------------------------------------------------------
# load_or_compute_embeddings
# ---------------------------------------------------------------------------


class TestLoadOrComputeEmbeddings:
    _BUILD_EMBEDDINGS_PATH = "dataeval_app.embeddings.build_embeddings"

    def _mock_extractor_config(self) -> MagicMock:
        cfg = MagicMock()
        cfg.model_dump_json.return_value = '{"type":"onnx","model_path":"model.onnx"}'
        return cfg

    def test_full_miss_computes_and_saves(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        cfg = self._mock_extractor_config()
        mock_array = np.random.default_rng(42).random((10, 64)).astype(np.float32)

        mock_embeddings = MagicMock()
        mock_embeddings.__array__ = MagicMock(return_value=mock_array)

        with patch(self._BUILD_EMBEDDINGS_PATH, return_value=mock_embeddings) as mock_build:
            result = cache.load_or_compute_embeddings(
                "sel:all",
                cfg.model_dump_json(),
                "none",
                MagicMock(),
                cfg,
                None,
                None,
            )
            mock_build.assert_called_once()

        np.testing.assert_array_equal(result, mock_array)
        # Verify saved to cache
        cached = cache.load_embeddings("sel:all", cfg.model_dump_json(), "none")
        assert cached is not None
        np.testing.assert_array_equal(cached, mock_array)

    def test_full_hit_skips_compute(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        cfg = self._mock_extractor_config()
        arr = np.ones((5, 32), dtype=np.float32)

        # Pre-populate cache
        cache.save_embeddings("sel:all", cfg.model_dump_json(), "none", arr)

        with patch(self._BUILD_EMBEDDINGS_PATH) as mock_build:
            result = cache.load_or_compute_embeddings(
                "sel:all",
                cfg.model_dump_json(),
                "none",
                MagicMock(),
                cfg,
                None,
                None,
            )
            mock_build.assert_not_called()

        np.testing.assert_array_equal(result, arr)

    def test_flattens_high_dimensional_output(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        cfg = self._mock_extractor_config()
        # Simulate 3-D output (N, H, W)
        mock_array = np.random.default_rng(42).random((5, 4, 8)).astype(np.float32)

        mock_embeddings = MagicMock()
        mock_embeddings.__array__ = MagicMock(return_value=mock_array)

        with patch(self._BUILD_EMBEDDINGS_PATH, return_value=mock_embeddings):
            result = cache.load_or_compute_embeddings(
                "sel:all",
                cfg.model_dump_json(),
                "none",
                MagicMock(),
                cfg,
                None,
                None,
            )

        assert result.ndim == 2
        assert result.shape == (5, 32)


class TestGetOrComputeEmbeddings:
    _BUILD_EMBEDDINGS_PATH = "dataeval_app.embeddings.build_embeddings"

    def _mock_extractor_config(self) -> MagicMock:
        cfg = MagicMock()
        cfg.model_dump_json.return_value = '{"type":"onnx","model_path":"model.onnx"}'
        return cfg

    def test_without_cache_computes_directly(self):
        cfg = self._mock_extractor_config()
        mock_array = np.ones((5, 16), dtype=np.float32)
        mock_embeddings = MagicMock()
        mock_embeddings.__array__ = MagicMock(return_value=mock_array)

        with patch(self._BUILD_EMBEDDINGS_PATH, return_value=mock_embeddings) as mock_build:
            result = get_or_compute_embeddings(
                dataset=MagicMock(),
                extractor_config=cfg,
            )
            mock_build.assert_called_once()

        np.testing.assert_array_equal(result, mock_array)

    def test_with_cache_delegates_to_workflow_cache(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        cfg = self._mock_extractor_config()
        mock_array = np.ones((5, 16), dtype=np.float32)
        mock_embeddings = MagicMock()
        mock_embeddings.__array__ = MagicMock(return_value=mock_array)

        with patch(self._BUILD_EMBEDDINGS_PATH, return_value=mock_embeddings) as mock_build:
            result = get_or_compute_embeddings(
                dataset=MagicMock(),
                extractor_config=cfg,
                cache=cache,
                selection_key="sel:all",
            )
            mock_build.assert_called_once()

        np.testing.assert_array_equal(result, mock_array)
        # Should be saved to cache
        cached = cache.load_embeddings("sel:all", cfg.model_dump_json(), "none")
        assert cached is not None

    def test_with_cache_full_hit_skips_compute(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        cfg = self._mock_extractor_config()
        arr = np.ones((5, 16), dtype=np.float32)

        # Pre-populate cache
        cache.save_embeddings("sel:all", cfg.model_dump_json(), "none", arr)

        with patch(self._BUILD_EMBEDDINGS_PATH) as mock_build:
            result = get_or_compute_embeddings(
                dataset=MagicMock(),
                extractor_config=cfg,
                cache=cache,
                selection_key="sel:all",
            )
            mock_build.assert_not_called()

        np.testing.assert_array_equal(result, arr)

    def test_different_transforms_different_cache_keys(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
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

        with patch(self._BUILD_EMBEDDINGS_PATH, return_value=make_mock_embeddings(arr_a)):
            get_or_compute_embeddings(
                MagicMock(),
                cfg,
                transforms=transform_a,
                cache=cache,
                selection_key="sel:all",
            )
        with patch(self._BUILD_EMBEDDINGS_PATH, return_value=make_mock_embeddings(arr_b)):
            get_or_compute_embeddings(
                MagicMock(),
                cfg,
                transforms=transform_b,
                cache=cache,
                selection_key="sel:all",
            )

        # Both should be independently cached
        loaded_a = cache.load_embeddings("sel:all", cfg.model_dump_json(), "transform_a")
        loaded_b = cache.load_embeddings("sel:all", cfg.model_dump_json(), "transform_b")
        assert loaded_a is not None
        assert loaded_b is not None
        np.testing.assert_array_equal(loaded_a, arr_a)
        np.testing.assert_array_equal(loaded_b, arr_b)

    def test_cache_none_selection_key_provided_computes_directly(self):
        cfg = self._mock_extractor_config()
        mock_array = np.ones((3, 8), dtype=np.float32)
        mock_embeddings = MagicMock()
        mock_embeddings.__array__ = MagicMock(return_value=mock_array)

        with patch(self._BUILD_EMBEDDINGS_PATH, return_value=mock_embeddings) as mock_build:
            result = get_or_compute_embeddings(
                dataset=MagicMock(),
                extractor_config=cfg,
                cache=None,
                selection_key="sel:all",
            )
            mock_build.assert_called_once()

        np.testing.assert_array_equal(result, mock_array)


# ---------------------------------------------------------------------------
# Unified stats cache (.parquet + .json)
# ---------------------------------------------------------------------------


def _make_calc_result(n: int = 5, include_2d: bool = False, include_hashes: bool = False) -> dict[str, Any]:
    """Build a minimal CalculationResult-shaped dict for testing."""
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
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
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
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        assert cache.load_stats("sel:all", "img+tgt") is None

    def test_float_arrays_round_trip(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3)
        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded["stats"]["mean"], stats["stats"]["mean"])
        np.testing.assert_array_almost_equal(loaded["stats"]["brightness"], stats["stats"]["brightness"])

    def test_string_arrays_round_trip(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3, include_hashes=True)
        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        original_hashes = list(stats["stats"]["xxhash"])
        loaded_hashes = list(loaded["stats"]["xxhash"])
        assert original_hashes == loaded_hashes

    def test_2d_histogram_round_trip(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3, include_2d=True)
        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded["stats"]["histogram"], stats["stats"]["histogram"], decimal=5)
        assert loaded["stats"]["histogram"].shape == (3, 256)

    def test_2d_percentiles_round_trip(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3, include_2d=True)
        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded["stats"]["percentiles"], stats["stats"]["percentiles"], decimal=5)
        assert loaded["stats"]["percentiles"].shape == (3, 5)

    def test_2d_center_round_trip(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3, include_2d=True)
        cache.save_stats("sel:all", "img+tgt", stats)
        loaded = cache.load_stats("sel:all", "img+tgt")

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded["stats"]["center"], stats["stats"]["center"], decimal=5)
        assert loaded["stats"]["center"].shape == (3, 2)

    def test_source_index_round_trip(self, tmp_path: Path):
        from dataeval.types import SourceIndex

        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
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

        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
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
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
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
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_stats("sel:all", "img+tgt", _make_calc_result(2))

        s = _config_hash("sel:all")
        h = _config_hash("img+tgt")
        pq = tmp_path / f"v{CACHE_VERSION}" / "ds" / f"stats_{s}_{h}.parquet"
        js = tmp_path / f"v{CACHE_VERSION}" / "ds" / f"stats_{s}_{h}.json"
        assert pq.exists()
        assert js.exists()
        # Verify parquet is readable
        df = pl.read_parquet(pq)
        assert "mean" in df.columns


# ---------------------------------------------------------------------------
# load_or_compute_stats
# ---------------------------------------------------------------------------


class TestLoadOrComputeStats:
    _CALC_STATS_PATH = "dataeval.core._calculate_stats.calculate_stats"

    def test_full_miss_computes_and_saves(self, tmp_path: Path):
        from dataeval.flags import ImageStats

        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_result = _make_calc_result(3)

        with patch(self._CALC_STATS_PATH, return_value=mock_result) as mock_calc:
            result = cache.load_or_compute_stats(
                "sel:all", "img+tgt", ImageStats.PIXEL_MEAN, MagicMock(), True, True, False
            )
            mock_calc.assert_called_once()

        assert "mean" in result["stats"]
        # Cache should be populated
        cached = cache.load_stats("sel:all", "img+tgt")
        assert cached is not None

    def test_full_hit_returns_cached(self, tmp_path: Path):
        from dataeval.flags import ImageStats

        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3)
        cache.save_stats("sel:all", "img+tgt", stats)

        with patch(self._CALC_STATS_PATH) as mock_calc:
            result = cache.load_or_compute_stats(
                "sel:all", "img+tgt", ImageStats.PIXEL_MEAN, MagicMock(), True, True, False
            )
            # Should NOT call calculate_stats — full cache hit
            mock_calc.assert_not_called()

        assert "mean" in result["stats"]

    def test_partial_hit_computes_missing_only(self, tmp_path: Path):
        from dataeval.flags import ImageStats

        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        # Pre-populate cache with "mean" only
        stats = _make_calc_result(3)
        stats["stats"] = {"mean": stats["stats"]["mean"]}
        cache.save_stats("sel:all", "img+tgt", stats)

        # Request mean + std
        fresh_result = _make_calc_result(3)
        fresh_result["stats"] = {"std": np.array([0.1, 0.2, 0.3])}

        with patch(self._CALC_STATS_PATH, return_value=fresh_result) as mock_calc:
            result = cache.load_or_compute_stats(
                "sel:all", "img+tgt", ImageStats.PIXEL_MEAN | ImageStats.PIXEL_STD, MagicMock(), True, True, False
            )
            mock_calc.assert_called_once()
            # Should request only PIXEL_STD (mean is cached)
            call_args = mock_calc.call_args
            requested_flags = call_args[0][2]
            assert ImageStats.PIXEL_STD in requested_flags
            assert ImageStats.PIXEL_MEAN not in requested_flags

        # Merged result should have both
        assert "mean" in result["stats"]
        assert "std" in result["stats"]

    def test_merged_result_persisted(self, tmp_path: Path):
        from dataeval.flags import ImageStats

        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        # Pre-populate with "mean"
        stats = _make_calc_result(3)
        stats["stats"] = {"mean": stats["stats"]["mean"]}
        cache.save_stats("sel:all", "img+tgt", stats)

        # Compute "std"
        fresh = _make_calc_result(3)
        fresh["stats"] = {"std": np.array([0.1, 0.2, 0.3])}

        with patch(self._CALC_STATS_PATH, return_value=fresh):
            cache.load_or_compute_stats(
                "sel:all", "img+tgt", ImageStats.PIXEL_MEAN | ImageStats.PIXEL_STD, MagicMock(), True, True, False
            )

        # Cache should now have both metrics
        cached = cache.load_stats("sel:all", "img+tgt")
        assert cached is not None
        assert "mean" in cached["stats"]
        assert "std" in cached["stats"]


class TestGetOrComputeStats:
    """Tests for the centralized get_or_compute_stats() entry point."""

    _CALC_STATS_PATH = "dataeval.core._calculate_stats.calculate_stats"

    def test_without_cache_computes_directly(self):
        from dataeval.flags import ImageStats

        mock_result = _make_calc_result(3)

        with patch(self._CALC_STATS_PATH, return_value=mock_result) as mock_calc:
            result = get_or_compute_stats(
                desired_flags=ImageStats.PIXEL_MEAN,
                dataset=MagicMock(),
            )
            mock_calc.assert_called_once()

        assert "mean" in result["stats"]

    def test_with_cache_delegates_to_workflow_cache(self, tmp_path: Path):
        from dataeval.flags import ImageStats

        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_result = _make_calc_result(3)

        with patch(self._CALC_STATS_PATH, return_value=mock_result) as mock_calc:
            result = get_or_compute_stats(
                desired_flags=ImageStats.PIXEL_MEAN,
                dataset=MagicMock(),
                cache=cache,
                selection_key="sel:all",
            )
            mock_calc.assert_called_once()

        assert "mean" in result["stats"]
        # Should have been saved to cache
        cached = cache.load_stats("sel:all", "img+tgt")
        assert cached is not None

    def test_with_cache_full_hit_skips_compute(self, tmp_path: Path):
        from dataeval.flags import ImageStats

        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        stats = _make_calc_result(3)
        cache.save_stats("sel:all", "img+tgt", stats)

        with patch(self._CALC_STATS_PATH) as mock_calc:
            result = get_or_compute_stats(
                desired_flags=ImageStats.PIXEL_MEAN,
                dataset=MagicMock(),
                cache=cache,
                selection_key="sel:all",
            )
            mock_calc.assert_not_called()

        assert "mean" in result["stats"]

    def test_cache_none_selection_key_provided_computes_directly(self):
        """When cache is None but selection_key is provided, computes directly."""
        from dataeval.flags import ImageStats

        mock_result = _make_calc_result(3)

        with patch(self._CALC_STATS_PATH, return_value=mock_result) as mock_calc:
            result = get_or_compute_stats(
                desired_flags=ImageStats.PIXEL_MEAN,
                dataset=MagicMock(),
                cache=None,
                selection_key="sel:all",
            )
            mock_calc.assert_called_once()

        assert "mean" in result["stats"]


# ---------------------------------------------------------------------------
# Metadata cache (.parquet + .json)
# ---------------------------------------------------------------------------


def _make_mock_metadata() -> MagicMock:
    """Build a mock Metadata with the attributes WorkflowCache serializes."""
    meta = MagicMock()
    meta.dataframe = pl.DataFrame(
        {
            "class_label": [0, 1, 0, 1, 2],
            "target_index": [None, None, None, None, None],
            "brightness": [0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )
    meta.class_labels = np.array([0, 1, 0, 1, 2], dtype=np.intp)
    meta.index2label = {0: "cat", 1: "dog", 2: "bird"}
    meta.item_indices = np.array([0, 1, 2, 3, 4], dtype=np.intp)
    meta.item_count = 5
    meta._image_factors = {"brightness"}  # noqa: SLF001
    meta._target_factors = set()  # noqa: SLF001
    meta._has_targets = False  # noqa: SLF001
    return meta


class TestMetadataCache:
    def test_save_creates_files(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_mock_metadata()

        cache.save_metadata("sel:all", meta)

        s = _config_hash("sel:all")
        pq = tmp_path / f"v{CACHE_VERSION}" / "ds" / f"metadata_{s}.parquet"
        js = tmp_path / f"v{CACHE_VERSION}" / "ds" / f"metadata_{s}.json"
        assert pq.exists()
        assert js.exists()

    def test_save_parquet_contains_dataframe(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_mock_metadata()

        cache.save_metadata("sel:all", meta)

        s = _config_hash("sel:all")
        df = pl.read_parquet(tmp_path / f"v{CACHE_VERSION}" / "ds" / f"metadata_{s}.parquet")
        assert "brightness" in df.columns
        assert len(df) == 5

    def test_save_strips_binned_columns(self, tmp_path: Path):
        """Binned (↕) and digitized (#) columns are dropped before saving."""
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_mock_metadata()
        # Add binned/digitized columns to the mock dataframe
        meta.dataframe = meta.dataframe.with_columns(
            pl.Series("brightness↕", [0, 1, 0, 1, 2]),
            pl.Series("brightness#", [0, 1, 0, 1, 2]),
        )

        cache.save_metadata("sel:all", meta)

        s = _config_hash("sel:all")
        df = pl.read_parquet(tmp_path / f"v{CACHE_VERSION}" / "ds" / f"metadata_{s}.parquet")
        assert "brightness" in df.columns
        assert "brightness↕" not in df.columns
        assert "brightness#" not in df.columns

    def test_save_json_contains_auxiliary(self, tmp_path: Path):
        import json

        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        meta = _make_mock_metadata()

        cache.save_metadata("sel:all", meta)

        s = _config_hash("sel:all")
        with open(tmp_path / f"v{CACHE_VERSION}" / "ds" / f"metadata_{s}.json") as f:
            aux = json.load(f)

        assert aux["item_count"] == 5
        assert aux["class_labels"] == [0, 1, 0, 1, 2]
        assert aux["index2label"]["0"] == "cat"
        assert aux["image_factors"] == ["brightness"]
        assert aux["target_factors"] == []
        assert aux["has_targets"] is False

    def test_load_returns_none_on_miss(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        assert cache.load_metadata("sel:all", MagicMock()) is None

    def test_load_returns_none_when_only_parquet_exists(self, tmp_path: Path):
        """Both files must exist for a cache hit."""
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")

        # Create just the parquet file in the versioned path
        s = _config_hash("sel:all")
        ds_dir = tmp_path / f"v{CACHE_VERSION}" / "ds"
        ds_dir.mkdir(parents=True)
        pl.DataFrame({"x": [1]}).write_parquet(ds_dir / f"metadata_{s}.parquet")

        assert cache.load_metadata("sel:all", MagicMock()) is None

    def test_load_reconstructs_metadata(self, tmp_path: Path):
        """Full save → load round trip with real Metadata reconstruction."""
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()

        cache.save_metadata("sel:all", mock_meta)
        loaded = cache.load_metadata("sel:all", MagicMock())

        assert loaded is not None
        # Verify reconstructed attributes
        assert loaded.item_count == 5
        np.testing.assert_array_equal(loaded.class_labels, np.array([0, 1, 0, 1, 2], dtype=np.intp))
        assert loaded.index2label[0] == "cat"
        assert loaded.index2label[1] == "dog"
        assert loaded.index2label[2] == "bird"
        np.testing.assert_array_equal(loaded.item_indices, np.array([0, 1, 2, 3, 4], dtype=np.intp))
        # Loaded metadata is unbinned — factor_names comes from _factors dict
        assert "brightness" in loaded._factors  # noqa: SLF001

    def test_load_sets_is_binned_false(self, tmp_path: Path):
        """Loaded metadata has _is_binned=False so _bin() runs lazily."""
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()

        cache.save_metadata("sel:all", mock_meta)
        loaded = cache.load_metadata("sel:all", MagicMock())

        assert loaded is not None
        assert loaded._is_binned is False  # noqa: SLF001
        assert loaded._is_structured is True  # noqa: SLF001

    def test_load_applies_caller_config(self, tmp_path: Path):
        """Loaded metadata has the caller's binning config applied."""
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()

        cache.save_metadata("sel:all", mock_meta)
        loaded = cache.load_metadata(
            "sel:all",
            MagicMock(),
            auto_bin_method="clusters",
            exclude=["x"],
            continuous_factor_bins={"y": [0.0, 1.0]},
        )

        assert loaded is not None
        assert loaded._auto_bin_method == "clusters"  # noqa: SLF001
        assert loaded._exclude == {"x"}  # noqa: SLF001
        assert loaded._continuous_factor_bins == {"y": [0.0, 1.0]}  # noqa: SLF001

    def test_load_sets_has_targets(self, tmp_path: Path):
        """Loaded metadata preserves the has_targets flag."""
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()

        cache.save_metadata("sel:all", mock_meta)
        loaded = cache.load_metadata("sel:all", MagicMock())

        assert loaded is not None
        assert loaded._has_targets is False  # noqa: SLF001

    def test_load_dataframe_accessible(self, tmp_path: Path):
        """Loaded metadata exposes the cached DataFrame."""
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()

        cache.save_metadata("sel:all", mock_meta)
        loaded = cache.load_metadata("sel:all", MagicMock())

        assert loaded is not None
        df = loaded._dataframe  # noqa: SLF001
        assert "brightness" in df.columns
        assert len(df) == 5

    def test_same_cache_entry_reused_across_configs(self, tmp_path: Path):
        """Different binning configs reuse the same raw cache entry."""
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()

        cache.save_metadata("sel:all", mock_meta)

        loaded_a = cache.load_metadata("sel:all", MagicMock(), auto_bin_method="uniform_width")
        loaded_b = cache.load_metadata("sel:all", MagicMock(), auto_bin_method="clusters")

        assert loaded_a is not None
        assert loaded_b is not None
        assert loaded_a._auto_bin_method == "uniform_width"  # noqa: SLF001
        assert loaded_b._auto_bin_method == "clusters"  # noqa: SLF001
        # Both loaded from the same cache files
        assert loaded_a.item_count == loaded_b.item_count


# ---------------------------------------------------------------------------
# WorkflowCache.load_or_compute_metadata
# ---------------------------------------------------------------------------


class TestLoadOrComputeMetadata:
    _BUILD_METADATA_PATH = "dataeval_app.metadata.build_metadata"

    def test_full_miss_computes_and_saves(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()

        with patch(self._BUILD_METADATA_PATH, return_value=mock_meta) as mock_build:
            result = cache.load_or_compute_metadata("sel:all", MagicMock())
            mock_build.assert_called_once()

        assert result is mock_meta
        # Verify saved to cache
        cached = cache.load_metadata("sel:all", MagicMock())
        assert cached is not None
        assert cached.item_count == 5

    def test_full_hit_skips_compute(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()

        # Pre-populate cache
        cache.save_metadata("sel:all", mock_meta)

        with patch(self._BUILD_METADATA_PATH) as mock_build:
            result = cache.load_or_compute_metadata("sel:all", MagicMock())
            mock_build.assert_not_called()

        assert result is not None
        assert result.item_count == 5

    def test_passes_config_kwargs_to_build(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()
        dataset = MagicMock()

        with patch(self._BUILD_METADATA_PATH, return_value=mock_meta) as mock_build:
            cache.load_or_compute_metadata(
                "sel:all",
                dataset,
                auto_bin_method="uniform_width",
                exclude=["x"],
                continuous_factor_bins={"y": [0.0, 1.0]},
            )
            mock_build.assert_called_once_with(
                dataset,
                auto_bin_method="uniform_width",
                exclude=["x"],
                continuous_factor_bins={"y": [0.0, 1.0]},
            )

    def test_different_configs_share_same_cache_entry(self, tmp_path: Path):
        """Different binning configs reuse the same raw cache entry."""
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()

        # First call — cache miss, computes and saves
        with patch(self._BUILD_METADATA_PATH, return_value=mock_meta) as mock_build:
            cache.load_or_compute_metadata("sel:all", MagicMock(), auto_bin_method="uniform_width")
            mock_build.assert_called_once()

        # Second call with different config — cache hit, skips compute
        with patch(self._BUILD_METADATA_PATH) as mock_build:
            result = cache.load_or_compute_metadata("sel:all", MagicMock(), auto_bin_method="clusters")
            mock_build.assert_not_called()

        assert result is not None
        assert result._auto_bin_method == "clusters"  # noqa: SLF001


# ---------------------------------------------------------------------------
# get_or_compute_metadata (top-level function)
# ---------------------------------------------------------------------------


class TestGetOrComputeMetadata:
    _BUILD_METADATA_PATH = "dataeval_app.metadata.build_metadata"

    def test_without_cache_computes_directly(self):
        mock_meta = _make_mock_metadata()

        with patch(self._BUILD_METADATA_PATH, return_value=mock_meta) as mock_build:
            result = get_or_compute_metadata(dataset=MagicMock())
            mock_build.assert_called_once()

        assert result is mock_meta

    def test_with_cache_delegates_to_workflow_cache(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()

        with patch(self._BUILD_METADATA_PATH, return_value=mock_meta) as mock_build:
            result = get_or_compute_metadata(
                dataset=MagicMock(),
                cache=cache,
                selection_key="sel:all",
            )
            mock_build.assert_called_once()

        assert result is mock_meta
        # Should be saved to cache
        cached = cache.load_metadata("sel:all", MagicMock())
        assert cached is not None

    def test_with_cache_full_hit_skips_compute(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()

        # Pre-populate cache
        cache.save_metadata("sel:all", mock_meta)

        with patch(self._BUILD_METADATA_PATH) as mock_build:
            result = get_or_compute_metadata(
                dataset=MagicMock(),
                cache=cache,
                selection_key="sel:all",
            )
            mock_build.assert_not_called()

        assert result is not None
        assert result.item_count == 5

    def test_cache_none_selection_key_provided_computes_directly(self):
        mock_meta = _make_mock_metadata()

        with patch(self._BUILD_METADATA_PATH, return_value=mock_meta) as mock_build:
            result = get_or_compute_metadata(
                dataset=MagicMock(),
                cache=None,
                selection_key="sel:all",
            )
            mock_build.assert_called_once()

        assert result is mock_meta

    def test_config_kwargs_forwarded(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        mock_meta = _make_mock_metadata()
        dataset = MagicMock()

        with patch(self._BUILD_METADATA_PATH, return_value=mock_meta) as mock_build:
            get_or_compute_metadata(
                dataset=dataset,
                auto_bin_method="uniform_width",
                exclude=["a"],
                continuous_factor_bins={"b": [0.0, 0.5, 1.0]},
                cache=cache,
                selection_key="sel:all",
            )
            mock_build.assert_called_once_with(
                dataset,
                auto_bin_method="uniform_width",
                exclude=["a"],
                continuous_factor_bins={"b": [0.0, 0.5, 1.0]},
            )


# ---------------------------------------------------------------------------
# Cross-dataset isolation
# ---------------------------------------------------------------------------


class TestCacheIsolation:
    def test_different_datasets_independent(self, tmp_path: Path):
        cache_a = WorkflowCache(cache_dir=tmp_path, dataset_name="ds_a")
        cache_b = WorkflowCache(cache_dir=tmp_path, dataset_name="ds_b")

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
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")

        # Embeddings
        cache.save_embeddings("sel:all", "cfg", "none", np.ones((2, 3)))
        # Stats (unified)
        cache.save_stats("sel:all", "img+tgt", _make_calc_result(2))
        # Metadata
        cache.save_metadata("sel:all", _make_mock_metadata())

        # All should be loadable
        assert cache.load_embeddings("sel:all", "cfg", "none") is not None
        assert cache.load_stats("sel:all", "img+tgt") is not None
        assert cache.load_metadata("sel:all", MagicMock()) is not None


# ---------------------------------------------------------------------------
# Cache version isolation
# ---------------------------------------------------------------------------


class TestCacheVersioning:
    def test_data_stored_under_version_directory(self, tmp_path: Path):
        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_embeddings("sel:all", "cfg", "none", np.ones((2, 3)))
        assert (tmp_path / f"v{CACHE_VERSION}" / "ds").is_dir()

    def test_different_versions_are_isolated(self, tmp_path: Path, monkeypatch: Any):
        """Bumping CACHE_VERSION produces a separate directory."""
        import dataeval_app.cache as cache_mod

        # Save with current version
        monkeypatch.setattr(cache_mod, "CACHE_VERSION", "1")
        cache_v1 = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        arr_v1 = np.ones((2, 3))
        cache_v1.save_embeddings("sel:all", "cfg", "none", arr_v1)

        # Simulate a version bump
        monkeypatch.setattr(cache_mod, "CACHE_VERSION", "2")
        cache_v2 = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
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
        cache_v1_reload = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        loaded_v1 = cache_v1_reload.load_embeddings("sel:all", "cfg", "none")
        assert loaded_v1 is not None
        np.testing.assert_array_equal(loaded_v1, arr_v1)

    def test_version_miss_after_bump(self, tmp_path: Path, monkeypatch: Any):
        """After a version bump, old data is not returned."""
        import dataeval_app.cache as cache_mod

        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        cache.save_embeddings("sel:all", "cfg", "none", np.ones((2, 3)))

        # Bump version — old cache should miss
        monkeypatch.setattr(cache_mod, "CACHE_VERSION", "99")
        cache_new = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        assert cache_new.load_embeddings("sel:all", "cfg", "none") is None


# ---------------------------------------------------------------------------
# Integration with WorkflowContext and TaskConfig
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_task_config_accepts_cache_dir(self):
        from dataeval_app.config.schemas.task import TaskConfig

        task = TaskConfig(
            name="test",
            workflow="data-cleaning",
            datasets="ds",
            cache_dir="/cache",
        )
        assert task.cache_dir == "/cache"

    def test_task_config_cache_dir_defaults_none(self):
        from dataeval_app.config.schemas.task import TaskConfig

        task = TaskConfig(
            name="test",
            workflow="data-cleaning",
            datasets="ds",
        )
        assert task.cache_dir is None

    def test_workflow_context_accepts_cache(self, tmp_path: Path):
        from dataeval_app.workflow import WorkflowContext

        cache = WorkflowCache(cache_dir=tmp_path, dataset_name="ds")
        ctx = WorkflowContext(cache=cache)
        assert ctx.cache is cache

    def test_workflow_context_cache_defaults_none(self):
        from dataeval_app.workflow import WorkflowContext

        ctx = WorkflowContext()
        assert ctx.cache is None
