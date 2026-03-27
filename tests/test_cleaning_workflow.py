"""Tests for cleaning workflow — serializers, findings, flagged indices, execute."""

import logging
from typing import Literal
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import dataeval_flow.embeddings
import dataeval_flow.metadata
import dataeval_flow.selection  # noqa: F401
from dataeval_flow.config import OnnxExtractorConfig, SelectionStep
from dataeval_flow.workflow import DatasetContext, WorkflowContext
from dataeval_flow.workflows.cleaning.outputs import (
    DataCleaningMetadata,
    DataCleaningOutputs,
    DataCleaningRawOutputs,
    DataCleaningReport,
    is_cleaning_result,
)
from dataeval_flow.workflows.cleaning.params import DataCleaningParameters
from dataeval_flow.workflows.cleaning.workflow import (
    CleaningRunContext,
    DataCleaningWorkflow,
    _build_class_labels_df,
    _build_duplicates,
    _build_outliers,
    _compute_classwise_pivot,
    _compute_embeddings,
    _compute_label_stats,
    _merge_duplicate_results,
    _merge_outlier_outputs,
    _resolve_flags,
    _run_cleaning,
    _run_duplicate_detection,
    _serialize_duplicates,
    _serialize_outlier_issues,
    _split_outlier_issues,
    _validate_cluster_params,
)


def _make_params(**overrides: object) -> DataCleaningParameters:
    """Build DataCleaningParameters with defaults for testing."""
    defaults: dict[str, object] = {
        "outlier_method": "adaptive",
        "outlier_flags": ["dimension", "pixel"],
        "outlier_threshold": None,
    }
    defaults.update(overrides)
    return DataCleaningParameters(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _build_outliers / _build_duplicates
# ---------------------------------------------------------------------------


class TestBuildOutliers:
    @patch("dataeval_flow.workflows.cleaning.workflow.Outliers")
    def test_basic_flags(self, mock_outliers_cls: MagicMock):
        params = _make_params(outlier_flags=["dimension", "visual"])
        _build_outliers(params)
        mock_outliers_cls.assert_called_once()
        call_kwargs = mock_outliers_cls.call_args[1]
        assert call_kwargs["outlier_threshold"] == ("adaptive", None)

    @patch("dataeval_flow.workflows.cleaning.workflow.Outliers")
    def test_with_extractor(self, mock_outliers_cls: MagicMock):
        mock_fe = MagicMock()
        params = _make_params()
        _build_outliers(params, extractor=mock_fe)
        call_kwargs = mock_outliers_cls.call_args[1]
        assert call_kwargs["extractor"] is mock_fe

    @patch("dataeval_flow.workflows.cleaning.workflow.Outliers")
    def test_with_threshold(self, mock_outliers_cls: MagicMock):
        params = _make_params(outlier_flags=["dimension"], outlier_threshold=2.5)
        _build_outliers(params)
        call_kwargs = mock_outliers_cls.call_args[1]
        assert call_kwargs["outlier_threshold"] == ("adaptive", 2.5)

    @patch("dataeval_flow.workflows.cleaning.workflow.Outliers")
    def test_cluster_params_passed(self, mock_outliers_cls: MagicMock):
        """Cluster params are passed through to Outliers when extractor provided."""
        mock_fe = MagicMock()
        params = _make_params(outlier_cluster_threshold=3.0, outlier_cluster_algorithm="kmeans", outlier_n_clusters=5)
        _build_outliers(params, extractor=mock_fe)
        call_kwargs = mock_outliers_cls.call_args[1]
        assert call_kwargs["cluster_threshold"] == 3.0
        assert call_kwargs["cluster_algorithm"] == "kmeans"
        assert call_kwargs["n_clusters"] == 5

    @patch("dataeval_flow.workflows.cleaning.workflow.Outliers")
    def test_cluster_threshold_none_passed_explicitly(self, mock_outliers_cls: MagicMock):
        """cluster_threshold=None is passed explicitly to override DataEval default."""
        params = _make_params()
        _build_outliers(params)
        call_kwargs = mock_outliers_cls.call_args[1]
        assert call_kwargs["cluster_threshold"] is None

    def test_cluster_without_extractor_raises(self):
        """Cluster params without extractor raises ValueError."""
        import pytest

        params = _make_params(outlier_cluster_threshold=2.5)
        with pytest.raises(ValueError, match="requires an extractor"):
            _build_outliers(params)


class TestBuildDuplicates:
    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
    def test_basic(self, mock_dup_cls: MagicMock):
        params = _make_params()
        _build_duplicates(params)
        mock_dup_cls.assert_called_once()

    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
    def test_with_extractor(self, mock_dup_cls: MagicMock):
        mock_fe = MagicMock()
        params = _make_params()
        _build_duplicates(params, extractor=mock_fe)
        call_kwargs = mock_dup_cls.call_args[1]
        assert call_kwargs["extractor"] is mock_fe

    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
    def test_with_flags(self, mock_dup_cls: MagicMock):
        """Explicit duplicate_flags are passed through."""
        params = _make_params(duplicate_flags=["hash_basic", "hash_d4"])
        _build_duplicates(params)
        call_kwargs = mock_dup_cls.call_args[1]
        assert "flags" in call_kwargs

    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
    def test_merge_near_false(self, mock_dup_cls: MagicMock):
        params = _make_params(duplicate_merge_near=False)
        _build_duplicates(params)
        call_kwargs = mock_dup_cls.call_args[1]
        assert call_kwargs["merge_near_duplicates"] is False

    def test_cluster_without_extractor_raises(self):
        import pytest

        params = _make_params(duplicate_cluster_sensitivity=2.5)
        with pytest.raises(ValueError, match="requires an extractor"):
            _build_duplicates(params)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


class TestSerializeOutlierIssues:
    def test_serializes_polars_df(self):
        df = pl.DataFrame({"item_index": [0, 1, 2], "metric_name": ["brightness"] * 3, "metric_value": [0.1, 0.2, 0.3]})
        out = _serialize_outlier_issues(df)
        assert out["count"] == 3
        assert len(out["issues"]) == 3
        assert out["issues"][0]["item_index"] == 0

    def test_empty_df(self):
        df = pl.DataFrame({"item_index": [], "metric_name": [], "metric_value": []})
        out = _serialize_outlier_issues(df)
        assert out["count"] == 0
        assert out["issues"] == []


class TestSerializeDuplicates:
    def test_exact_and_near(self):
        df = pl.DataFrame(
            {
                "group_id": [0, 1, 2],
                "level": ["item", "item", "item"],
                "dup_type": ["exact", "exact", "near"],
                "item_indices": [[0, 1], [2, 3], [4, 5]],
                "target_indices": [None, None, None],
                "methods": [None, None, ["hash"]],
                "orientation": [None, None, "same"],
            }
        )
        result = MagicMock()
        result.data.return_value = df

        out = _serialize_duplicates(result)
        assert len(out["items"]["exact"]) == 2  # type: ignore[typeddict-item]
        assert out["items"]["near"][0]["orientation"] == "same"  # type: ignore[typeddict-item]
        assert "exact" not in out["targets"]

    def test_no_duplicates(self):
        df = pl.DataFrame(
            {
                "group_id": [],
                "level": [],
                "dup_type": [],
                "item_indices": [],
                "target_indices": [],
                "methods": [],
                "orientation": [],
            }
        )
        result = MagicMock()
        result.data.return_value = df

        out = _serialize_duplicates(result)
        assert out["items"] == {}
        assert out["targets"] == {}

    def test_source_index_targets(self):
        """Target-level duplicates with target_indices produce SourceIndexDict entries."""
        df = pl.DataFrame(
            {
                "group_id": [0, 1, 2],
                "level": ["item", "target", "target"],
                "dup_type": ["exact", "exact", "near"],
                "item_indices": [[0, 1], [0, 1], [4, 5]],
                "target_indices": [None, [2, 3], [0, 1]],
                "methods": [None, None, ["hash"]],
                "orientation": [None, None, "same"],
            }
        )
        result = MagicMock()
        result.data.return_value = df

        out = _serialize_duplicates(result)

        # item-level exact should remain plain ints
        assert out["items"]["exact"] == [[0, 1]]  # type: ignore[typeddict-item]
        # target-level exact: SourceIndexDict entries
        assert out["targets"]["exact"] == [  # type: ignore[typeddict-item]
            [
                {"item": 0, "target": 2, "channel": None},
                {"item": 1, "target": 3, "channel": None},
            ]
        ]
        # target-level near: SourceIndexDict entries
        near_indices = out["targets"]["near"][0]["indices"]  # type: ignore[typeddict-item]
        assert near_indices == [
            {"item": 4, "target": 0, "channel": None},
            {"item": 5, "target": 1, "channel": None},
        ]


# ---------------------------------------------------------------------------
# _compute_label_stats
# ---------------------------------------------------------------------------


class TestComputeLabelStats:
    def test_basic_stats(self):
        metadata = MagicMock()
        metadata.class_labels = [0, 0, 1, 1, 1, 2]
        metadata.index2label = {0: "cat", 1: "dog", 2: "bird"}
        metadata.item_count = 6

        stats = _compute_label_stats(metadata)
        assert stats["item_count"] == 6  # type: ignore[typeddict-item]
        assert stats["class_count"] == 3  # type: ignore[typeddict-item]
        assert stats["index2label"] == {0: "cat", 1: "dog", 2: "bird"}  # type: ignore[typeddict-item]
        assert stats["label_counts_per_class"]["cat"] == 2  # type: ignore[typeddict-item]
        assert stats["label_counts_per_class"]["dog"] == 3  # type: ignore[typeddict-item]
        assert stats["label_counts_per_class"]["bird"] == 1  # type: ignore[typeddict-item]

    def test_empty_class_labels(self):
        metadata = MagicMock()
        metadata.class_labels = []
        metadata.index2label = {}
        metadata.item_count = 0

        stats = _compute_label_stats(metadata)
        assert stats["item_count"] == 0  # type: ignore[typeddict-item]
        assert stats["class_count"] == 0  # type: ignore[typeddict-item]
        assert stats["label_counts_per_class"] == {}  # type: ignore[typeddict-item]

    def test_label_not_in_index2label(self):
        """Labels not in index2label fall back to str(label_idx)."""
        metadata = MagicMock()
        metadata.class_labels = [0, 99]
        metadata.index2label = {0: "cat"}
        metadata.item_count = 2

        stats = _compute_label_stats(metadata)
        # class_count reflects index2label size (known classes), not distinct labels in data
        assert stats["class_count"] == 1  # type: ignore[typeddict-item]
        assert stats["label_counts_per_class"]["cat"] == 1  # type: ignore[typeddict-item]
        assert stats["label_counts_per_class"]["99"] == 1  # type: ignore[typeddict-item]


# ---------------------------------------------------------------------------
# DataCleaningWorkflow.execute
# ---------------------------------------------------------------------------


class TestDataCleaningWorkflowExecute:
    def _make_exec_params(self, mode: Literal["advisory", "preparatory"] = "advisory") -> DataCleaningParameters:
        return _make_params(mode=mode)

    def test_properties(self):
        wf = DataCleaningWorkflow()
        assert wf.name == "data-cleaning"
        assert wf.description
        assert wf.params_schema is DataCleaningParameters
        assert wf.output_schema is DataCleaningOutputs

    def test_rejects_non_context(self):
        wf = DataCleaningWorkflow()
        result = wf.execute("not a context", self._make_exec_params())  # type: ignore[arg-type]
        assert not result.success
        assert "WorkflowContext" in result.errors[0]

    def test_rejects_no_params(self):
        wf = DataCleaningWorkflow()
        ctx = MagicMock(spec=WorkflowContext)
        result = wf.execute(ctx, None)
        assert not result.success
        assert "required" in result.errors[0].lower()

    def test_rejects_wrong_params_type(self):
        wf = DataCleaningWorkflow()
        ctx = MagicMock(spec=WorkflowContext)
        result = wf.execute(ctx, MagicMock())
        assert not result.success
        assert "DataCleaningParameters" in result.errors[0]

    @patch("dataeval_flow.metadata.Metadata", side_effect=RuntimeError("model crashed"))
    def test_execution_error_returns_failed_result(
        self,
        mock_meta_cls: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ):
        """Runtime errors during execution return WorkflowResult(success=False) and log traceback."""
        wf = DataCleaningWorkflow()
        ctx = WorkflowContext(dataset_contexts={"default": DatasetContext(name="default", dataset=MagicMock())})
        result = wf.execute(ctx, self._make_exec_params())
        assert not result.success
        assert "Workflow execution failed" in result.errors[0]
        assert "model crashed" in result.errors[0]
        # Verify traceback is logged
        assert any("Workflow" in r.message and r.levelno == logging.ERROR for r in caplog.records)
        assert "model crashed" in caplog.text

    @patch("dataeval_flow.workflows.cleaning.workflow._run_cleaning")
    @patch("dataeval_flow.metadata.Metadata")
    def test_advisory_mode(self, mock_meta_cls: MagicMock, mock_run_clean: MagicMock):
        wf = DataCleaningWorkflow()
        mock_dataset = MagicMock()
        ctx = WorkflowContext(dataset_contexts={"default": DatasetContext(name="default", dataset=mock_dataset)})

        mock_meta_cls.return_value = MagicMock()
        mock_run_clean.return_value = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={
                "count": 2,
                "issues": [
                    {"item_index": 0, "metric_name": "m", "metric_value": 0.0},
                    {"item_index": 1, "metric_name": "m", "metric_value": 0.0},
                ],
            },
        )

        result = wf.execute(ctx, self._make_exec_params("advisory"))
        assert result.success
        assert result.metadata.mode == "advisory"
        assert not result.metadata.flagged_indices

    @patch("dataeval_flow.workflows.cleaning.workflow._run_cleaning")
    @patch("dataeval_flow.metadata.Metadata")
    def test_preparatory_mode(self, mock_meta_cls: MagicMock, mock_run_clean: MagicMock):
        wf = DataCleaningWorkflow()
        mock_dataset = MagicMock()
        ctx = WorkflowContext(dataset_contexts={"default": DatasetContext(name="default", dataset=mock_dataset)})

        mock_meta_cls.return_value = MagicMock()
        mock_run_clean.return_value = DataCleaningRawOutputs(
            dataset_size=10,
            img_outliers={
                "count": 2,
                "issues": [
                    {"item_index": 0, "metric_name": "m", "metric_value": 0.0},
                    {"item_index": 1, "metric_name": "m", "metric_value": 0.0},
                ],
            },
            duplicates={"items": {"exact": [[3, 4]], "near": []}, "targets": {}},
        )

        result = wf.execute(ctx, self._make_exec_params("preparatory"))
        assert result.success
        meta = result.metadata.model_dump()
        assert meta["mode"] == "preparatory"
        assert meta["flagged_indices"] == [0, 1, 4]
        assert meta["removed_count"] == 3
        assert len(meta["clean_indices"]) == 7
        # Preparatory Mode finding has data-driven brief key
        assert isinstance(result.data, DataCleaningOutputs)
        prep_finding = next(f for f in result.data.report.findings if f.title == "Preparatory Mode")
        assert prep_finding.data["brief"] == "3 flagged, 7 retained"  # type: ignore[index]

    @patch("dataeval_flow.workflows.cleaning.workflow._run_cleaning")
    @patch("dataeval_flow.metadata.Metadata")
    @patch("dataeval_flow.selection.Select")
    @patch("dataeval_flow.selection.sel")
    def test_with_selection(
        self,
        mock_sel_module: MagicMock,
        mock_select_cls: MagicMock,
        mock_meta_cls: MagicMock,
        mock_run_clean: MagicMock,
    ):
        wf = DataCleaningWorkflow()
        mock_dataset = MagicMock()
        selected_dataset = MagicMock()
        mock_select_cls.return_value = selected_dataset

        ctx = WorkflowContext(
            dataset_contexts={
                "default": DatasetContext(
                    name="default",
                    dataset=mock_dataset,
                    selection_steps=[SelectionStep(type="Limit", params={"size": 50})],
                )
            }
        )

        mock_meta_cls.return_value = MagicMock()
        mock_run_clean.return_value = DataCleaningRawOutputs(dataset_size=50, img_outliers={"count": 0, "issues": []})

        result = wf.execute(ctx, self._make_exec_params())
        assert result.success
        mock_select_cls.assert_called_once()
        # _run_cleaning should receive the selected dataset
        assert mock_run_clean.call_args[0][0] is selected_dataset

    @patch("dataeval_flow.workflows.cleaning.workflow._run_cleaning")
    @patch("dataeval_flow.metadata.Metadata")
    @patch("dataeval_flow.embeddings.OnnxExtractor")
    def test_with_embeddings(self, mock_extractor_cls: MagicMock, mock_meta_cls: MagicMock, mock_run_clean: MagicMock):
        wf = DataCleaningWorkflow()
        mock_dataset = MagicMock()

        ctx = WorkflowContext(
            dataset_contexts={
                "default": DatasetContext(
                    name="default",
                    dataset=mock_dataset,
                    extractor=OnnxExtractorConfig(name="test_ext", model_path="./model.onnx", output_name="layer4"),
                )
            }
        )

        mock_meta_cls.return_value = MagicMock()
        mock_run_clean.return_value = DataCleaningRawOutputs(dataset_size=100, img_outliers={"count": 0, "issues": []})

        result = wf.execute(ctx, self._make_exec_params())
        assert result.success
        mock_extractor_cls.assert_called_once()


# ---------------------------------------------------------------------------
# _run_cleaning (mocked evaluators)
# ---------------------------------------------------------------------------


class TestValidateClusterParams:
    def test_outlier_cluster_without_extractor_raises(self):
        params = _make_params(outlier_cluster_threshold=2.5)
        with pytest.raises(ValueError, match="Cluster-based outlier detection requires an extractor"):
            _validate_cluster_params(params, extractor=None)

    def test_duplicate_cluster_without_extractor_raises(self):
        params = _make_params(duplicate_cluster_sensitivity=2.5)
        with pytest.raises(ValueError, match="Cluster-based duplicate detection requires an extractor"):
            _validate_cluster_params(params, extractor=None)

    def test_no_cluster_params_ok(self):
        params = _make_params()
        _validate_cluster_params(params, extractor=None)  # should not raise

    def test_cluster_with_extractor_ok(self):
        params = _make_params(outlier_cluster_threshold=2.5)
        _validate_cluster_params(params, extractor=MagicMock())  # should not raise


class TestRunCleaning:
    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
    @patch("dataeval_flow.workflows.cleaning.workflow.Outliers")
    @patch("dataeval_flow.cache.get_or_compute_stats")
    def test_run_cleaning_basic(self, mock_get_stats: MagicMock, mock_outliers_cls: MagicMock, mock_dup_cls: MagicMock):
        from dataeval_flow.workflows.cleaning.workflow import _run_cleaning

        params = _make_params()

        # Mock centralized stats
        mock_get_stats.return_value = {
            "stats": {},
            "source_index": [],
            "object_count": [],
            "invalid_box_count": [],
            "image_count": 0,
        }

        # Mock Outliers.from_stats — return object with .data() returning DataFrame
        mock_outliers = MagicMock()
        issues_df = pl.DataFrame({"item_index": [0], "metric_name": ["brightness"], "metric_value": [0.1]})
        mock_outlier_output = MagicMock()
        mock_outlier_output.data.return_value = issues_df
        mock_outliers.from_stats.return_value = mock_outlier_output
        mock_outliers_cls.return_value = mock_outliers

        # Mock Duplicates.from_stats — return object with .data() returning empty DataFrame
        mock_dups = MagicMock()
        empty_dup_df = pl.DataFrame(
            {
                "group_id": [],
                "level": [],
                "dup_type": [],
                "item_indices": [],
                "target_indices": [],
                "methods": [],
                "orientation": [],
            }
        )
        mock_dup_result = MagicMock()
        mock_dup_result.data.return_value = empty_dup_df
        mock_dups.from_stats.return_value = mock_dup_result
        mock_dup_cls.return_value = mock_dups

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        raw = _run_cleaning(mock_dataset, params)
        assert raw.dataset_size == 100
        assert raw.img_outliers["count"] == 1
        mock_get_stats.assert_called_once()

    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
    @patch("dataeval_flow.workflows.cleaning.workflow.Outliers")
    @patch("dataeval_flow.cache.get_or_compute_stats")
    def test_run_cleaning_with_target_index(
        self, mock_get_stats: MagicMock, mock_outliers_cls: MagicMock, mock_dup_cls: MagicMock
    ):
        from dataeval_flow.workflows.cleaning.workflow import _run_cleaning

        params = _make_params(outlier_method="iqr", outlier_flags=["dimension"])

        mock_get_stats.return_value = {
            "stats": {},
            "source_index": [],
            "object_count": [],
            "invalid_box_count": [],
            "image_count": 0,
        }

        # Issues DF with target_index column (OD dataset)
        issues_df = pl.DataFrame(
            {
                "item_index": [0, 0, 1],
                "metric_name": ["brightness", "brightness", "size"],
                "metric_value": [0.1, 0.2, 0.3],
                "target_index": [None, 5, None],
            }
        )
        mock_outliers = MagicMock()
        mock_outlier_output = MagicMock()
        mock_outlier_output.data.return_value = issues_df
        mock_outliers.from_stats.return_value = mock_outlier_output
        mock_outliers_cls.return_value = mock_outliers

        # Mock Duplicates.from_stats
        mock_dups = MagicMock()
        empty_dup_df = pl.DataFrame(
            {
                "group_id": [],
                "level": [],
                "dup_type": [],
                "item_indices": [],
                "target_indices": [],
                "methods": [],
                "orientation": [],
            }
        )
        mock_dup_result = MagicMock()
        mock_dup_result.data.return_value = empty_dup_df
        mock_dups.from_stats.return_value = mock_dup_result
        mock_dup_cls.return_value = mock_dups

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=50)

        raw = _run_cleaning(mock_dataset, params)
        assert raw.dataset_size == 50
        # 2 image-level outliers (target_id is null)
        assert raw.img_outliers["count"] == 2
        # 1 target-level outlier
        assert raw.target_outliers is not None
        assert raw.target_outliers["count"] == 1

    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
    @patch("dataeval_flow.workflows.cleaning.workflow.Outliers")
    @patch("dataeval_flow.cache.get_or_compute_stats")
    def test_run_cleaning_with_metadata(
        self, mock_get_stats: MagicMock, mock_outliers_cls: MagicMock, mock_dup_cls: MagicMock
    ):
        """_run_cleaning computes label_stats when metadata is provided."""
        from dataeval_flow.workflows.cleaning.workflow import _run_cleaning

        params = _make_params()

        mock_get_stats.return_value = {
            "stats": {},
            "source_index": [],
            "object_count": [],
            "invalid_box_count": [],
            "image_count": 0,
        }

        # Mock Outliers.from_stats — return object with .data() returning DataFrame
        mock_outliers = MagicMock()
        issues_df = pl.DataFrame({"item_index": [], "metric_name": [], "metric_value": []})
        mock_outlier_output = MagicMock()
        mock_outlier_output.data.return_value = issues_df
        mock_outliers.from_stats.return_value = mock_outlier_output
        mock_outliers_cls.return_value = mock_outliers

        # Mock Duplicates.from_stats — return object with .data() returning empty DataFrame
        mock_dups = MagicMock()
        empty_dup_df = pl.DataFrame(
            {
                "group_id": [],
                "level": [],
                "dup_type": [],
                "item_indices": [],
                "target_indices": [],
                "methods": [],
                "orientation": [],
            }
        )
        mock_dup_result = MagicMock()
        mock_dup_result.data.return_value = empty_dup_df
        mock_dups.from_stats.return_value = mock_dup_result
        mock_dup_cls.return_value = mock_dups

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        # Provide real-ish metadata
        mock_metadata = MagicMock()
        mock_metadata.class_labels = [0, 0, 1]
        mock_metadata.index2label = {0: "cat", 1: "dog"}
        mock_metadata.item_count = 3

        raw = _run_cleaning(mock_dataset, params, metadata=mock_metadata)
        assert raw.label_stats != {}
        assert raw.label_stats["item_count"] == 3  # type: ignore[typeddict-item]
        assert raw.label_stats["class_count"] == 2  # type: ignore[typeddict-item]
        assert raw.label_stats["index2label"] == {0: "cat", 1: "dog"}  # type: ignore[typeddict-item]
        assert raw.label_stats["label_counts_per_class"]["cat"] == 2  # type: ignore[typeddict-item]
        assert raw.label_stats["label_counts_per_class"]["dog"] == 1  # type: ignore[typeddict-item]


# ---------------------------------------------------------------------------
# _resolve_flags
# ---------------------------------------------------------------------------


class TestResolveFlags:
    def test_default_hash_flags(self):
        """No duplicate_flags → default HASH_DUPLICATES_BASIC."""
        from dataeval.flags import ImageStats

        params = _make_params()
        _, hash_flags = _resolve_flags(params)
        assert hash_flags == ImageStats.HASH_DUPLICATES_BASIC

    def test_explicit_duplicate_flags(self):
        """Explicit duplicate_flags are resolved via HASH_FLAG_MAP."""
        from dataeval.flags import ImageStats

        params = _make_params(duplicate_flags=["hash_basic", "hash_d4"])
        _, hash_flags = _resolve_flags(params)
        assert hash_flags & ImageStats.HASH_DUPLICATES_BASIC
        assert hash_flags & ImageStats.HASH_DUPLICATES_D4


# ---------------------------------------------------------------------------
# _split_outlier_issues
# ---------------------------------------------------------------------------


class TestSplitOutlierIssues:
    def test_with_target_index_column(self):
        df = pl.DataFrame(
            {
                "item_index": [0, 1, 2],
                "metric_name": ["a", "b", "c"],
                "metric_value": [0.1, 0.2, 0.3],
                "target_index": [None, 5, None],
            }
        )
        img, tgt = _split_outlier_issues(df)
        assert img.shape[0] == 2
        assert tgt is not None
        assert tgt.shape[0] == 1

    def test_without_target_index_column(self):
        df = pl.DataFrame(
            {
                "item_index": [0, 1],
                "metric_name": ["a", "b"],
                "metric_value": [0.1, 0.2],
            }
        )
        img, tgt = _split_outlier_issues(df)
        assert img.shape[0] == 2
        assert tgt is None


# ---------------------------------------------------------------------------
# _build_class_labels_df
# ---------------------------------------------------------------------------


class TestBuildClassLabelsDf:
    def test_classification_dataset(self):
        """Non-target dataset builds labels_df with item_index only."""
        metadata = MagicMock()
        metadata.class_labels = [0, 0, 1, 2]
        metadata.index2label = {0: "cat", 1: "dog", 2: "bird"}
        metadata.has_targets.return_value = False
        metadata.item_indices = [0, 1, 2, 3]

        labels_df, id_cols, label_counts = _build_class_labels_df(metadata)
        assert id_cols == ["item_index"]
        assert labels_df.shape[0] == 4
        assert "class_name" in labels_df.columns
        assert label_counts == {"cat": 2, "dog": 1, "bird": 1}

    def test_od_dataset_with_targets(self):
        """Object-detection dataset with target_data uses target_index."""
        metadata = MagicMock()
        metadata.class_labels = [0, 1]
        metadata.index2label = {0: "cat", 1: "dog"}
        metadata.has_targets.return_value = True
        metadata.target_data = pl.DataFrame(
            {
                "item_index": [0, 0, 1],
                "target_index": [0, 1, 0],
                "class_label": [0, 1, 0],
            }
        )

        labels_df, id_cols, label_counts = _build_class_labels_df(metadata)
        assert id_cols == ["item_index", "target_index"]
        assert "class_name" in labels_df.columns
        assert labels_df.shape[0] == 3


# ---------------------------------------------------------------------------
# _compute_classwise_pivot
# ---------------------------------------------------------------------------


class TestComputeClasswisePivot:
    def test_returns_none_when_no_metadata(self):
        img_issues = pl.DataFrame({"item_index": [0], "metric_name": ["a"], "metric_value": [0.1]})
        assert _compute_classwise_pivot(None, img_issues, metadata=None) is None

    def test_returns_none_when_no_issues(self):
        metadata = MagicMock()
        metadata.has_targets.return_value = False
        empty = pl.DataFrame({"item_index": [], "metric_name": [], "metric_value": []})
        assert _compute_classwise_pivot(None, empty, metadata=metadata) is None

    def test_classification_pivot(self):
        """Image-level issues are grouped by class."""
        metadata = MagicMock()
        metadata.has_targets.return_value = False
        metadata.class_labels = [0, 0, 1, 1]
        metadata.index2label = {0: "cat", 1: "dog"}
        metadata.item_indices = [0, 1, 2, 3]

        img_issues = pl.DataFrame(
            {
                "item_index": [0, 2],
                "metric_name": ["brightness", "brightness"],
                "metric_value": [0.1, 0.2],
            }
        )
        result = _compute_classwise_pivot(None, img_issues, metadata=metadata)
        assert result is not None
        assert "level" in result
        assert "rows" in result
        assert result["level"] == "image"
        rows = result["rows"]
        # Last row is Total
        assert rows[-1]["class_name"] == "Total"
        assert rows[-1]["count"] == 2

    def test_od_pivot_uses_target_issues(self):
        """Object-detection datasets use target_issues for pivot."""
        metadata = MagicMock()
        metadata.has_targets.return_value = True
        metadata.class_labels = [0, 1]
        metadata.index2label = {0: "cat", 1: "dog"}
        metadata.target_data = pl.DataFrame(
            {
                "item_index": [0, 0, 1],
                "target_index": [0, 1, 0],
                "class_label": [0, 1, 0],
            }
        )

        target_issues = pl.DataFrame(
            {
                "item_index": [0, 0],
                "target_index": [0, 1],
                "metric_name": ["area", "area"],
                "metric_value": [0.1, 0.2],
            }
        )
        img_issues = pl.DataFrame({"item_index": [], "metric_name": [], "metric_value": []})
        result = _compute_classwise_pivot(target_issues, img_issues, metadata=metadata)
        assert result is not None
        assert "level" in result
        assert "rows" in result
        assert result["level"] == "target"
        assert result["rows"][-1]["class_name"] == "Total"


# ---------------------------------------------------------------------------
# _compute_embeddings
# ---------------------------------------------------------------------------


class TestComputeEmbeddings:
    @patch("dataeval_flow.cache.get_or_compute_embeddings")
    def test_with_extractor_config(self, mock_get_emb: MagicMock):
        """Uses cached embeddings when extractor_config is available."""
        mock_get_emb.return_value = "cached_embeddings"
        dataset = MagicMock()
        run_ctx = CleaningRunContext(extractor_config=MagicMock(), transforms=None, batch_size=32)

        result = _compute_embeddings(dataset, MagicMock(), run_ctx)
        assert result == "cached_embeddings"
        mock_get_emb.assert_called_once()

    def test_without_extractor_config(self):
        """Falls back to direct extractor call when no extractor_config."""
        import sys
        import types

        import numpy as np

        # Stub the lazy-imported module
        arrays_mod = types.ModuleType("dataeval.utils.arrays")
        arrays_mod.flatten_samples = lambda x: x  # type: ignore[attr-defined]
        arrays_mod.to_numpy = lambda x: x  # type: ignore[attr-defined]
        sys.modules["dataeval.utils.arrays"] = arrays_mod
        try:
            dataset = [(np.zeros((3, 32, 32))), (np.zeros((3, 32, 32)))]
            extractor = MagicMock(return_value=np.zeros((2, 64)))

            result = _compute_embeddings(dataset, extractor, run_ctx=None)  # type: ignore
            assert result is not None
            extractor.assert_called_once()
        finally:
            sys.modules.pop("dataeval.utils.arrays", None)


# ---------------------------------------------------------------------------
# _merge_outlier_outputs
# ---------------------------------------------------------------------------


class TestMergeOutlierOutputs:
    @patch("dataeval_flow.cache.get_or_compute_cluster_result")
    def test_merges_stats_and_cluster(self, mock_cluster: MagicMock):
        """Stats-based and cluster-based outlier issues are concatenated."""
        import numpy as np

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
        embeddings = np.zeros((10, 64))

        result = _merge_outlier_outputs(outliers_eval, stats_output, embeddings, params, run_ctx=None)
        # Result is an OutliersOutput wrapping the merged DataFrame
        merged = result.data()
        assert merged.shape[0] == 2
        assert set(merged["item_index"].to_list()) == {0, 1}


# ---------------------------------------------------------------------------
# _run_duplicate_detection
# ---------------------------------------------------------------------------


class TestRunDuplicateDetection:
    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
    def test_hash_only(self, mock_dup_cls: MagicMock):
        """Hash-only detection (no cluster) returns hash result directly."""
        from dataeval.flags import ImageStats

        hash_result = MagicMock()
        mock_dup_instance = MagicMock()
        mock_dup_instance.from_stats.return_value = hash_result
        mock_dup_cls.return_value = mock_dup_instance

        params = _make_params()
        result = _run_duplicate_detection(
            params, ImageStats.HASH_DUPLICATES_BASIC, MagicMock(), embeddings_array=None, run_ctx=None
        )
        assert result is hash_result

    @patch("dataeval_flow.workflows.cleaning.workflow._merge_duplicate_results")
    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
    def test_with_cluster(self, mock_dup_cls: MagicMock, mock_merge: MagicMock):
        """Cluster-based detection triggers merge."""
        import numpy as np
        from dataeval.flags import ImageStats

        hash_result = MagicMock()
        mock_dup_instance = MagicMock()
        mock_dup_instance.from_stats.return_value = hash_result
        mock_dup_cls.return_value = mock_dup_instance
        mock_merge.return_value = MagicMock()

        params = _make_params(duplicate_cluster_sensitivity=0.5)
        embeddings = np.zeros((10, 64))
        _run_duplicate_detection(params, ImageStats.HASH_DUPLICATES_BASIC, MagicMock(), embeddings, run_ctx=None)
        mock_merge.assert_called_once()


# ---------------------------------------------------------------------------
# _merge_duplicate_results
# ---------------------------------------------------------------------------


class TestMergeDuplicateResults:
    @patch("dataeval_flow.cache.get_or_compute_cluster_result")
    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
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
        result = _merge_duplicate_results(hash_result, "embeddings", params, run_ctx=None)
        merged = result.data()
        # Cluster group IDs should be re-numbered to avoid collision
        assert merged.shape[0] == 4
        group_ids = set(merged["group_id"].to_list())
        assert len(group_ids) == 2  # original 0 + re-numbered 1

    @patch("dataeval_flow.cache.get_or_compute_cluster_result")
    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
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
        result = _merge_duplicate_results(hash_result, "embeddings", params, run_ctx=None)
        assert result is hash_result


# ---------------------------------------------------------------------------
# is_cleaning_result type guard
# ---------------------------------------------------------------------------


class TestIsCleaningResult:
    def test_true_for_cleaning_metadata(self):
        from dataeval_flow.workflow import WorkflowResult

        result = WorkflowResult(
            name="data-cleaning",
            success=True,
            data=DataCleaningOutputs(
                raw=DataCleaningRawOutputs(dataset_size=10), report=DataCleaningReport(summary="s")
            ),
            metadata=DataCleaningMetadata(),
        )
        assert is_cleaning_result(result) is True

    def test_false_for_other_metadata(self):
        from dataeval_flow.config import ResultMetadata
        from dataeval_flow.workflow import WorkflowResult

        result = WorkflowResult(
            name="other",
            success=True,
            data=MagicMock(),
            metadata=ResultMetadata(),
        )
        assert is_cleaning_result(result) is False


# ---------------------------------------------------------------------------
# _compute_classwise_pivot — exception handler (lines 669-671)
# ---------------------------------------------------------------------------


class TestComputeClasswisePivotException:
    def test_exception_returns_none(self):
        metadata = MagicMock()
        metadata.has_targets.return_value = False
        metadata.class_labels = None  # will cause iteration to fail

        img_issues = pl.DataFrame({"item_index": [0], "metric_name": ["brightness"], "metric_value": [0.5]})
        result = _compute_classwise_pivot(None, img_issues, metadata=metadata)
        assert result is None


# ---------------------------------------------------------------------------
# _merge_outlier_outputs — missing target_index added (lines 742-748)
# ---------------------------------------------------------------------------


class TestMergeOutlierOutputsMissingTargetIndex:
    @patch("dataeval_flow.cache.get_or_compute_cluster_result")
    def test_adds_target_index_when_missing(self, mock_cluster: MagicMock):
        import numpy as np

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
        embeddings = np.zeros((10, 64))

        result = _merge_outlier_outputs(outliers_eval, stats_output, embeddings, params, run_ctx=None)
        merged = result.data()
        assert merged.shape[0] == 2
        assert "target_index" not in merged.columns  # all null → dropped


# ---------------------------------------------------------------------------
# _run_duplicate_detection — custom flags (line 770)
# ---------------------------------------------------------------------------


class TestRunDuplicateDetectionCustomFlags:
    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
    def test_custom_flags_passed(self, mock_dup_cls: MagicMock):
        from dataeval.flags import ImageStats

        mock_dup_instance = MagicMock()
        mock_dup_cls.return_value = mock_dup_instance

        params = _make_params(duplicate_flags=["hash_basic", "hash_d4"])
        hash_flags = ImageStats.HASH_DUPLICATES_BASIC | ImageStats.HASH_DUPLICATES_D4
        _run_duplicate_detection(params, hash_flags, MagicMock(), embeddings_array=None, run_ctx=None)

        call_kwargs = mock_dup_cls.call_args[1]
        assert "flags" in call_kwargs
        assert call_kwargs["flags"] == hash_flags


# ---------------------------------------------------------------------------
# _merge_duplicate_results — column alignment (lines 817, 820)
# ---------------------------------------------------------------------------


class TestMergeDuplicateResultsColumnAlignment:
    @patch("dataeval_flow.cache.get_or_compute_cluster_result")
    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
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
        result = _merge_duplicate_results(hash_result, "embeddings", params, run_ctx=None)
        merged = result.data()
        assert merged.shape[0] == 2
        assert "orientation" in merged.columns


# ---------------------------------------------------------------------------
# _run_cleaning — cluster branches (lines 872, 876)
# ---------------------------------------------------------------------------


class TestRunCleaningClusterBranches:
    @patch("dataeval_flow.workflows.cleaning.workflow._merge_outlier_outputs")
    @patch("dataeval_flow.workflows.cleaning.workflow._compute_embeddings")
    @patch("dataeval_flow.workflows.cleaning.workflow.Duplicates")
    @patch("dataeval_flow.workflows.cleaning.workflow.Outliers")
    @patch("dataeval_flow.cache.get_or_compute_stats")
    def test_outlier_cluster_triggers_merge(
        self,
        mock_get_stats: MagicMock,
        mock_outliers_cls: MagicMock,
        mock_dup_cls: MagicMock,
        mock_compute_emb: MagicMock,
        mock_merge_outlier: MagicMock,
    ):
        import numpy as np

        mock_get_stats.return_value = {}

        issues_df = pl.DataFrame({"item_index": [0], "metric_name": ["brightness"], "metric_value": [0.1]})
        mock_outlier_output = MagicMock()
        mock_outlier_output.data.return_value = issues_df
        mock_outliers = MagicMock()
        mock_outliers.from_stats.return_value = mock_outlier_output
        mock_outliers_cls.return_value = mock_outliers

        merged_df = pl.DataFrame(
            {"item_index": [0, 1], "metric_name": ["brightness", "cluster"], "metric_value": [0.1, 0.9]}
        )
        merged_output = MagicMock()
        merged_output.data.return_value = merged_df
        mock_merge_outlier.return_value = merged_output

        mock_compute_emb.return_value = np.zeros((10, 64))

        empty_dup_df = pl.DataFrame(
            {
                "group_id": pl.Series([], dtype=pl.Int64),
                "level": pl.Series([], dtype=pl.Utf8),
                "dup_type": pl.Series([], dtype=pl.Utf8),
                "item_indices": pl.Series([], dtype=pl.List(pl.Int64)),
                "target_indices": pl.Series([], dtype=pl.List(pl.Int64)),
                "methods": pl.Series([], dtype=pl.List(pl.Utf8)),
                "orientation": pl.Series([], dtype=pl.Utf8),
            }
        )
        mock_dup_result = MagicMock()
        mock_dup_result.data.return_value = empty_dup_df
        mock_dups = MagicMock()
        mock_dups.from_stats.return_value = mock_dup_result
        mock_dup_cls.return_value = mock_dups

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        params = _make_params(outlier_cluster_threshold=2.5)
        raw = _run_cleaning(mock_dataset, params, extractor=MagicMock(), run_ctx=None)
        mock_compute_emb.assert_called_once()
        mock_merge_outlier.assert_called_once()
        assert raw.dataset_size == 10
