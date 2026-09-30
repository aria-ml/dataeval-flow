"""parameter-sweep's copy of the embedding and cluster-merge helpers."""

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from dataeval_flow.workflows.data_cleaning import DataCleaningConfig
from dataeval_flow.workflows.parameter_sweep._cleaning import (
    CleaningRunContext,
    _compute_embeddings,
    _merge_duplicate_results,
    _merge_outlier_outputs,
)

pytestmark = pytest.mark.required


def _make_params(**overrides: object) -> DataCleaningConfig:
    """Build DataCleaningConfig with defaults for testing."""
    defaults: dict[str, object] = {
        "outlier_method": "adaptive",
        "outlier_flags": ["dimension", "pixel"],
        "outlier_threshold": None,
    }
    defaults.update(overrides)
    return DataCleaningConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _compute_embeddings
# ---------------------------------------------------------------------------


class TestComputeEmbeddings:
    @patch("dataeval_flow.workflows.parameter_sweep._cleaning.get_or_compute_embeddings")
    def test_with_extractor_config(self, mock_get_emb: MagicMock):
        """Uses cached embeddings when extractor_config is available."""
        mock_get_emb.return_value = "cached_embeddings"
        dataset = MagicMock()
        extractor_cfg = MagicMock()
        extractor_cfg.model = "onnx"  # implementation expects implemented model name
        run_ctx = CleaningRunContext(extractor_config=extractor_cfg, transforms=None, batch_size=32)

        result = _compute_embeddings(dataset, MagicMock(), run_ctx)
        assert result == "cached_embeddings"
        mock_get_emb.assert_called_once()

    def test_without_extractor_config(self):
        """Falls back to direct extractor call when no extractor_config."""
        import sys
        import types

        # Stub the lazy-imported module
        original = sys.modules.get("dataeval.utils")
        arrays_mod = types.ModuleType("dataeval.utils")
        arrays_mod.flatten_samples = lambda x: x  # type: ignore[attr-defined]
        arrays_mod.to_numpy = lambda x: x  # type: ignore[attr-defined]
        sys.modules["dataeval.utils"] = arrays_mod
        try:
            dataset = [(np.zeros((3, 32, 32))), (np.zeros((3, 32, 32)))]
            extractor = MagicMock(return_value=np.zeros((2, 64)))

            result = _compute_embeddings(dataset, extractor, run_ctx=None)  # type: ignore
            assert result is not None
            extractor.assert_called_once()
        finally:
            if original is not None:
                sys.modules["dataeval.utils"] = original
            else:
                sys.modules.pop("dataeval.utils", None)


# ---------------------------------------------------------------------------
# _merge_outlier_outputs
# ---------------------------------------------------------------------------


class TestMergeOutlierOutputs:
    @patch("dataeval_flow.workflows.parameter_sweep._cleaning.get_or_compute_cluster_result")
    def test_merges_stats_and_cluster(self, mock_cluster: MagicMock):
        """Stats-based and cluster-based outlier issues are concatenated."""
        # Mock stats output
        stats_df = pl.DataFrame(
            {
                "item_index": [0],
                "metric_name": ["brightness"],
                "metric_value": [0.1],
            }
        )
        stats_output = MagicMock()
        stats_output.data.return_value = stats_df

        # Mock cluster output via outliers_eval.from_clusters
        cluster_df = pl.DataFrame(
            {
                "item_index": [1],
                "metric_name": ["cluster_dist"],
                "metric_value": [0.9],
            }
        )
        outliers_eval = MagicMock()
        cluster_output = MagicMock()
        cluster_output.data.return_value = cluster_df
        outliers_eval.from_clusters.return_value = cluster_output

        mock_cluster.return_value = MagicMock()
        params = _make_params(outlier_cluster_threshold=2.5, outlier_cluster_algorithm="hdbscan")
        embeddings = np.zeros((10, 64), dtype=np.float32)

        result = _merge_outlier_outputs(outliers_eval, stats_output, embeddings, params, run_ctx=None)
        # Result is an OutliersOutput wrapping the merged DataFrame
        merged = result.data()
        assert merged.shape[0] == 2
        assert set(merged["item_index"].to_list()) == {0, 1}


# ---------------------------------------------------------------------------
# _merge_duplicate_results
# ---------------------------------------------------------------------------


class TestMergeDuplicateResults:
    @patch("dataeval_flow.workflows.parameter_sweep._cleaning.get_or_compute_cluster_result")
    @patch("dataeval_flow.workflows.parameter_sweep._cleaning.Duplicates")
    def test_merge_hash_and_cluster(self, mock_dup_cls: MagicMock, mock_cluster: MagicMock):
        """Hash and cluster duplicate results are merged with re-numbered group IDs."""
        hash_df = pl.DataFrame(
            {
                "group_id": [0, 0],
                "level": ["item", "item"],
                "dup_type": ["exact", "exact"],
                "item_indices": [[0, 1], [0, 1]],
            }
        )
        hash_result = MagicMock()
        hash_result.data.return_value = hash_df

        cluster_df = pl.DataFrame(
            {
                "group_id": [0, 0],
                "level": ["item", "item"],
                "dup_type": ["near", "near"],
                "item_indices": [[2, 3], [2, 3]],
            }
        )
        cluster_result = MagicMock()
        cluster_result.data.return_value = cluster_df
        mock_dup_instance = MagicMock()
        mock_dup_instance.from_clusters.return_value = cluster_result
        mock_dup_cls.return_value = mock_dup_instance
        mock_cluster.return_value = MagicMock()

        params = _make_params(duplicate_cluster_sensitivity=0.5)
        embeddings = np.zeros((10, 64), dtype=np.float32)
        result = _merge_duplicate_results(hash_result, embeddings, params, run_ctx=None)
        merged = result.data()
        # Cluster group IDs should be re-numbered to avoid collision
        assert merged.shape[0] == 4
        group_ids = set(merged["group_id"].to_list())
        assert len(group_ids) == 2  # original 0 + re-numbered 1

    @patch("dataeval_flow.workflows.parameter_sweep._cleaning.get_or_compute_cluster_result")
    @patch("dataeval_flow.workflows.parameter_sweep._cleaning.Duplicates")
    def test_cluster_duplicates_pass_merge_near_duplicates(self, mock_dup_cls: MagicMock, mock_cluster: MagicMock):
        """`duplicate_merge_near` must reach the cluster-mode `Duplicates`, not just the hash-mode one."""
        mock_dup_instance = MagicMock()
        mock_dup_instance.from_clusters.return_value = MagicMock(data=lambda: pl.DataFrame({"group_id": []}))
        mock_dup_cls.return_value = mock_dup_instance
        mock_cluster.return_value = MagicMock()

        hash_result = MagicMock()
        hash_result.data.return_value = pl.DataFrame({"group_id": []})

        params = _make_params(duplicate_cluster_sensitivity=0.5, duplicate_merge_near=False)
        embeddings = np.zeros((10, 64), dtype=np.float32)
        _merge_duplicate_results(hash_result, embeddings, params, run_ctx=None)

        mock_dup_cls.assert_called_once_with(cluster_sensitivity=0.5, merge_near_duplicates=False)

    @patch("dataeval_flow.workflows.parameter_sweep._cleaning.get_or_compute_cluster_result")
    @patch("dataeval_flow.workflows.parameter_sweep._cleaning.Duplicates")
    def test_empty_cluster_returns_hash(self, mock_dup_cls: MagicMock, mock_cluster: MagicMock):
        """Empty cluster result returns hash result as-is."""
        hash_result = MagicMock()
        hash_result.data.return_value = pl.DataFrame(
            {
                "group_id": [0],
                "level": ["item"],
                "dup_type": ["exact"],
                "item_indices": [[0, 1]],
            }
        )

        empty_cluster = MagicMock()
        empty_cluster.data.return_value = pl.DataFrame(
            {
                "group_id": [],
                "level": [],
                "dup_type": [],
                "item_indices": [],
            }
        )
        mock_dup_instance = MagicMock()
        mock_dup_instance.from_clusters.return_value = empty_cluster
        mock_dup_cls.return_value = mock_dup_instance
        mock_cluster.return_value = MagicMock()

        params = _make_params(duplicate_cluster_sensitivity=0.5)
        embeddings = np.zeros((10, 64), dtype=np.float32)
        result = _merge_duplicate_results(hash_result, embeddings, params, run_ctx=None)
        assert result is hash_result


# ---------------------------------------------------------------------------
# _merge_outlier_outputs — missing target_index added
# ---------------------------------------------------------------------------


class TestMergeOutlierOutputsMissingTargetIndex:
    @patch("dataeval_flow.workflows.parameter_sweep._cleaning.get_or_compute_cluster_result")
    def test_adds_target_index_when_missing(self, mock_cluster: MagicMock):
        stats_df = pl.DataFrame({"item_index": [0], "metric_name": ["brightness"], "metric_value": [0.1]})
        stats_output = MagicMock()
        stats_output.data.return_value = stats_df

        cluster_df = pl.DataFrame({"item_index": [1], "metric_name": ["cluster_dist"], "metric_value": [0.9]})
        outliers_eval = MagicMock()
        cluster_output = MagicMock()
        cluster_output.data.return_value = cluster_df
        outliers_eval.from_clusters.return_value = cluster_output
        mock_cluster.return_value = MagicMock()

        params = _make_params(outlier_cluster_threshold=2.5, outlier_cluster_algorithm="hdbscan")
        embeddings = np.zeros((10, 64), dtype=np.float32)

        result = _merge_outlier_outputs(outliers_eval, stats_output, embeddings, params, run_ctx=None)
        merged = result.data()
        assert merged.shape[0] == 2
        assert "target_index" not in merged.columns  # all null → dropped


# ---------------------------------------------------------------------------
# _merge_duplicate_results — column alignment
# ---------------------------------------------------------------------------


class TestMergeDuplicateResultsColumnAlignment:
    @patch("dataeval_flow.workflows.parameter_sweep._cleaning.get_or_compute_cluster_result")
    @patch("dataeval_flow.workflows.parameter_sweep._cleaning.Duplicates")
    def test_hash_missing_col_added_from_cluster(self, mock_dup_cls: MagicMock, mock_cluster: MagicMock):
        hash_df = pl.DataFrame({"group_id": [0], "level": ["item"], "dup_type": ["exact"], "item_indices": [[0, 1]]})
        hash_result = MagicMock()
        hash_result.data.return_value = hash_df

        cluster_df = pl.DataFrame(
            {
                "group_id": [0],
                "level": ["item"],
                "dup_type": ["near"],
                "item_indices": [[2, 3]],
                "orientation": ["same"],
            }
        )
        cluster_result = MagicMock()
        cluster_result.data.return_value = cluster_df
        mock_dup_instance = MagicMock()
        mock_dup_instance.from_clusters.return_value = cluster_result
        mock_dup_cls.return_value = mock_dup_instance
        mock_cluster.return_value = MagicMock()

        params = _make_params(duplicate_cluster_sensitivity=0.5)
        embeddings = np.zeros((10, 64), dtype=np.float32)
        result = _merge_duplicate_results(hash_result, embeddings, params, run_ctx=None)
        merged = result.data()
        assert merged.shape[0] == 2
        assert "orientation" in merged.columns
