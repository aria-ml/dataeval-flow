"""Tests for cleaning workflow — serializers, findings, flagged indices, execute."""

import logging
from typing import Literal
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import dataeval_app.embeddings
import dataeval_app.metadata
import dataeval_app.selection  # noqa: F401
from dataeval_app.config.models import OnnxExtractorConfig
from dataeval_app.config.schemas.selection import SelectionStep
from dataeval_app.workflow import DatasetContext, WorkflowContext
from dataeval_app.workflows.cleaning.outputs import (
    DataCleaningOutputs,
    DataCleaningRawOutputs,
)
from dataeval_app.workflows.cleaning.params import DataCleaningParameters
from dataeval_app.workflows.cleaning.workflow import (
    DataCleaningWorkflow,
    _build_duplicates,
    _build_findings,
    _build_outliers,
    _collect_flagged_indices,
    _compute_label_stats,
    _serialize_duplicates,
    _serialize_outlier_issues,
)


def _make_params(**overrides: object) -> DataCleaningParameters:
    """Build DataCleaningParameters with defaults for testing."""
    defaults: dict[str, object] = {
        "outlier_method": "zscore",
        "outlier_flags": ["dimension", "pixel"],
        "outlier_threshold": None,
    }
    defaults.update(overrides)
    return DataCleaningParameters(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _build_outliers / _build_duplicates
# ---------------------------------------------------------------------------


class TestBuildOutliers:
    @patch("dataeval_app.workflows.cleaning.workflow.Outliers")
    def test_basic_flags(self, mock_outliers_cls: MagicMock):
        params = _make_params(outlier_flags=["dimension", "visual"])
        _build_outliers(params)
        mock_outliers_cls.assert_called_once()
        call_kwargs = mock_outliers_cls.call_args[1]
        assert call_kwargs["outlier_threshold"] == ("zscore", None)

    @patch("dataeval_app.workflows.cleaning.workflow.Outliers")
    def test_with_extractor(self, mock_outliers_cls: MagicMock):
        mock_fe = MagicMock()
        params = _make_params()
        _build_outliers(params, extractor=mock_fe)
        call_kwargs = mock_outliers_cls.call_args[1]
        assert call_kwargs["extractor"] is mock_fe

    @patch("dataeval_app.workflows.cleaning.workflow.Outliers")
    def test_with_threshold(self, mock_outliers_cls: MagicMock):
        params = _make_params(outlier_flags=["dimension"], outlier_threshold=2.5)
        _build_outliers(params)
        call_kwargs = mock_outliers_cls.call_args[1]
        assert call_kwargs["outlier_threshold"] == ("zscore", 2.5)

    @patch("dataeval_app.workflows.cleaning.workflow.Outliers")
    def test_cluster_params_passed(self, mock_outliers_cls: MagicMock):
        """Cluster params are passed through to Outliers when extractor provided."""
        mock_fe = MagicMock()
        params = _make_params(
            outlier_cluster_threshold=3.0,
            outlier_cluster_algorithm="kmeans",
            outlier_n_clusters=5,
        )
        _build_outliers(params, extractor=mock_fe)
        call_kwargs = mock_outliers_cls.call_args[1]
        assert call_kwargs["cluster_threshold"] == 3.0
        assert call_kwargs["cluster_algorithm"] == "kmeans"
        assert call_kwargs["n_clusters"] == 5

    @patch("dataeval_app.workflows.cleaning.workflow.Outliers")
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
    @patch("dataeval_app.workflows.cleaning.workflow.Duplicates")
    def test_basic(self, mock_dup_cls: MagicMock):
        params = _make_params()
        _build_duplicates(params)
        mock_dup_cls.assert_called_once()

    @patch("dataeval_app.workflows.cleaning.workflow.Duplicates")
    def test_with_extractor(self, mock_dup_cls: MagicMock):
        mock_fe = MagicMock()
        params = _make_params()
        _build_duplicates(params, extractor=mock_fe)
        call_kwargs = mock_dup_cls.call_args[1]
        assert call_kwargs["extractor"] is mock_fe

    @patch("dataeval_app.workflows.cleaning.workflow.Duplicates")
    def test_with_flags(self, mock_dup_cls: MagicMock):
        """Explicit duplicate_flags are passed through."""
        params = _make_params(duplicate_flags=["hash_basic", "hash_d4"])
        _build_duplicates(params)
        call_kwargs = mock_dup_cls.call_args[1]
        assert "flags" in call_kwargs

    @patch("dataeval_app.workflows.cleaning.workflow.Duplicates")
    def test_merge_near_false(self, mock_dup_cls: MagicMock):
        params = _make_params(duplicate_merge_near=False)
        _build_duplicates(params)
        call_kwargs = mock_dup_cls.call_args[1]
        assert call_kwargs["merge_near_duplicates"] is False

    def test_cluster_without_extractor_raises(self):
        import pytest

        params = _make_params(duplicate_cluster_threshold=2.5)
        with pytest.raises(ValueError, match="requires an extractor"):
            _build_duplicates(params)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


class TestSerializeOutlierIssues:
    def test_serializes_polars_df(self):
        df = pl.DataFrame({"item_id": [0, 1, 2], "metric_name": ["brightness"] * 3, "metric_value": [0.1, 0.2, 0.3]})
        out = _serialize_outlier_issues(df)
        assert out["count"] == 3
        assert len(out["issues"]) == 3
        assert out["issues"][0]["item_id"] == 0

    def test_empty_df(self):
        df = pl.DataFrame({"item_id": [], "metric_name": [], "metric_value": []})
        out = _serialize_outlier_issues(df)
        assert out["count"] == 0
        assert out["issues"] == []


class TestSerializeDuplicates:
    def test_exact_and_near(self):
        result = MagicMock()
        # items
        result.items.exact = [[0, 1], [2, 3]]
        result.items.near = [MagicMock(indices=[4, 5], methods=frozenset({"hash"}), orientation="same")]
        # targets
        result.targets.exact = None
        result.targets.near = None

        out = _serialize_duplicates(result)
        assert len(out["items"]["exact"]) == 2  # type: ignore[typeddict-item]
        assert out["items"]["near"][0]["orientation"] == "same"  # type: ignore[typeddict-item]
        assert "exact" not in out["targets"]

    def test_no_duplicates(self):
        result = MagicMock()
        result.items.exact = None
        result.items.near = None
        result.targets.exact = None
        result.targets.near = None

        out = _serialize_duplicates(result)
        assert out["items"] == {}
        assert out["targets"] == {}


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
# _build_findings
# ---------------------------------------------------------------------------


class TestBuildFindings:
    def test_outlier_finding(self):
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={
                "count": 5,
                "issues": [{"item_id": i, "metric_name": "m", "metric_value": 0.0} for i in range(5)],
            },
        )
        findings = _build_findings(raw, None)
        titles = [f.title for f in findings]
        assert "Image Outliers" in titles

    def test_target_outlier_finding(self):
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            target_outliers={"count": 3, "issues": []},
        )
        findings = _build_findings(raw, None)
        titles = [f.title for f in findings]
        assert "Target Outliers" in titles

    def test_duplicate_finding(self):
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            duplicates={
                "items": {
                    "exact": [[0, 1]],
                    "near": [{"indices": [2, 3], "methods": ["hash"], "orientation": None}],
                },
                "targets": {},
            },
        )
        findings = _build_findings(raw, None)
        titles = [f.title for f in findings]
        assert "Duplicates" in titles

    def test_label_stats_finding(self):
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            label_stats={
                "item_count": 100,
                "class_count": 2,
                "label_counts_per_class": {"cat": 50, "dog": 50},
            },
        )
        findings = _build_findings(raw, None)
        titles = [f.title for f in findings]
        assert "Label Distribution" in titles

    def test_no_findings_when_clean(self):
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
        )
        findings = _build_findings(raw, None)
        assert findings == []


# ---------------------------------------------------------------------------
# _collect_flagged_indices
# ---------------------------------------------------------------------------


class TestCollectFlaggedIndices:
    def test_outlier_indices(self):
        raw = DataCleaningRawOutputs(
            dataset_size=10,
            img_outliers={
                "issues": [
                    {"item_id": 2, "metric_name": "m", "metric_value": 0.0},
                    {"item_id": 5, "metric_name": "m", "metric_value": 0.0},
                ],
                "count": 2,
            },
        )
        flagged = _collect_flagged_indices(raw)
        assert flagged == {2, 5}

    def test_exact_duplicate_indices(self):
        raw = DataCleaningRawOutputs(
            dataset_size=10,
            img_outliers={"issues": [], "count": 0},
            duplicates={"items": {"exact": [[0, 1, 2]], "near": []}, "targets": {}},
        )
        flagged = _collect_flagged_indices(raw)
        # Keep first (0), flag rest (1, 2)
        assert flagged == {1, 2}

    def test_near_duplicate_indices(self):
        raw = DataCleaningRawOutputs(
            dataset_size=10,
            img_outliers={"issues": [], "count": 0},
            duplicates={
                "items": {
                    "exact": [],
                    "near": [{"indices": [3, 4, 5], "methods": ["hash"], "orientation": None}],
                },
                "targets": {},
            },
        )
        flagged = _collect_flagged_indices(raw)
        assert flagged == {4, 5}

    def test_combined_indices(self):
        raw = DataCleaningRawOutputs(
            dataset_size=10,
            img_outliers={
                "issues": [{"item_id": 0, "metric_name": "m", "metric_value": 0.0}],
                "count": 1,
            },
            duplicates={
                "items": {
                    "exact": [[1, 2]],
                    "near": [{"indices": [3, 4], "methods": ["hash"], "orientation": None}],
                },
                "targets": {},
            },
        )
        flagged = _collect_flagged_indices(raw)
        assert flagged == {0, 2, 4}


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

    @patch("dataeval_app.metadata.Metadata", side_effect=RuntimeError("model crashed"))
    def test_execution_error_returns_failed_result(
        self,
        mock_meta_cls: MagicMock,  # noqa: ARG002
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

    @patch("dataeval_app.workflows.cleaning.workflow._run_cleaning")
    @patch("dataeval_app.metadata.Metadata")
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
                    {"item_id": 0, "metric_name": "m", "metric_value": 0.0},
                    {"item_id": 1, "metric_name": "m", "metric_value": 0.0},
                ],
            },
        )

        result = wf.execute(ctx, self._make_exec_params("advisory"))
        assert result.success
        assert result.metadata.mode == "advisory"
        assert not result.metadata.flagged_indices

    @patch("dataeval_app.workflows.cleaning.workflow._run_cleaning")
    @patch("dataeval_app.metadata.Metadata")
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
                    {"item_id": 0, "metric_name": "m", "metric_value": 0.0},
                    {"item_id": 1, "metric_name": "m", "metric_value": 0.0},
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

    @patch("dataeval_app.workflows.cleaning.workflow._run_cleaning")
    @patch("dataeval_app.metadata.Metadata")
    @patch("dataeval_app.selection.Select")
    @patch("dataeval_app.selection.sel")
    def test_with_selection(
        self,
        mock_sel_module: MagicMock,  # noqa: ARG002
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
            },
        )

        mock_meta_cls.return_value = MagicMock()
        mock_run_clean.return_value = DataCleaningRawOutputs(
            dataset_size=50,
            img_outliers={"count": 0, "issues": []},
        )

        result = wf.execute(ctx, self._make_exec_params())
        assert result.success
        mock_select_cls.assert_called_once()
        # _run_cleaning should receive the selected dataset
        assert mock_run_clean.call_args[0][0] is selected_dataset

    @patch("dataeval_app.workflows.cleaning.workflow._run_cleaning")
    @patch("dataeval_app.metadata.Metadata")
    @patch("dataeval_app.embeddings.Embeddings")
    @patch("dataeval_app.embeddings.OnnxExtractor")
    def test_with_embeddings(
        self,
        mock_encoder_cls: MagicMock,
        mock_embed_cls: MagicMock,
        mock_meta_cls: MagicMock,
        mock_run_clean: MagicMock,
    ):
        wf = DataCleaningWorkflow()
        mock_dataset = MagicMock()
        mock_embeddings = MagicMock()
        mock_embed_cls.return_value = mock_embeddings

        ctx = WorkflowContext(
            dataset_contexts={
                "default": DatasetContext(
                    name="default",
                    dataset=mock_dataset,
                    extractor=OnnxExtractorConfig(model_path="/model.onnx", output_name="layer4"),
                )
            },
        )

        mock_meta_cls.return_value = MagicMock()
        mock_run_clean.return_value = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
        )

        result = wf.execute(ctx, self._make_exec_params())
        assert result.success
        mock_encoder_cls.assert_called_once()
        # embeddings passed to _run_cleaning
        assert mock_run_clean.call_args[0][2] is mock_embeddings


# ---------------------------------------------------------------------------
# _run_cleaning (mocked evaluators)
# ---------------------------------------------------------------------------


class TestRunCleaning:
    @patch("dataeval_app.workflows.cleaning.workflow._build_duplicates")
    @patch("dataeval_app.workflows.cleaning.workflow._build_outliers")
    def test_run_cleaning_basic(self, mock_build_out: MagicMock, mock_build_dup: MagicMock):
        from dataeval_app.workflows.cleaning.workflow import _run_cleaning

        params = _make_params()

        # Mock Outliers evaluator
        mock_outliers = MagicMock()
        issues_df = pl.DataFrame({"item_id": [0], "metric_name": ["brightness"], "metric_value": [0.1]})
        mock_outliers.evaluate.return_value = MagicMock(issues=issues_df)
        mock_build_out.return_value = mock_outliers

        # Mock Duplicates evaluator
        mock_dups = MagicMock()
        mock_dup_result = MagicMock()
        mock_dup_result.items.exact = None
        mock_dup_result.items.near = None
        mock_dup_result.targets.exact = None
        mock_dup_result.targets.near = None
        mock_dups.evaluate.return_value = mock_dup_result
        mock_build_dup.return_value = mock_dups

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        raw = _run_cleaning(mock_dataset, params)
        assert raw.dataset_size == 100
        assert raw.img_outliers["count"] == 1

    @patch("dataeval_app.workflows.cleaning.workflow._build_duplicates")
    @patch("dataeval_app.workflows.cleaning.workflow._build_outliers")
    def test_run_cleaning_with_target_id(self, mock_build_out: MagicMock, mock_build_dup: MagicMock):
        from dataeval_app.workflows.cleaning.workflow import _run_cleaning

        params = _make_params(outlier_method="iqr", outlier_flags=["dimension"])

        # Issues DF with target_id column (OD dataset)
        issues_df = pl.DataFrame(
            {
                "item_id": [0, 0, 1],
                "metric_name": ["brightness", "brightness", "size"],
                "metric_value": [0.1, 0.2, 0.3],
                "target_id": [None, 5, None],
            }
        )
        mock_outliers = MagicMock()
        mock_outliers.evaluate.return_value = MagicMock(issues=issues_df)
        mock_build_out.return_value = mock_outliers

        mock_dups = MagicMock()
        mock_dup_result = MagicMock()
        mock_dup_result.items.exact = None
        mock_dup_result.items.near = None
        mock_dup_result.targets.exact = None
        mock_dup_result.targets.near = None
        mock_dups.evaluate.return_value = mock_dup_result
        mock_build_dup.return_value = mock_dups

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=50)

        raw = _run_cleaning(mock_dataset, params)
        assert raw.dataset_size == 50
        # 2 image-level outliers (target_id is null)
        assert raw.img_outliers["count"] == 2
        # 1 target-level outlier
        assert raw.target_outliers is not None
        assert raw.target_outliers["count"] == 1

    @patch("dataeval_app.workflows.cleaning.workflow._build_duplicates")
    @patch("dataeval_app.workflows.cleaning.workflow._build_outliers")
    def test_run_cleaning_with_metadata(self, mock_build_out: MagicMock, mock_build_dup: MagicMock):
        """_run_cleaning computes label_stats when metadata is provided."""
        from dataeval_app.workflows.cleaning.workflow import _run_cleaning

        params = _make_params()

        # Mock Outliers evaluator
        mock_outliers = MagicMock()
        issues_df = pl.DataFrame({"item_id": [], "metric_name": [], "metric_value": []})
        mock_outliers.evaluate.return_value = MagicMock(issues=issues_df)
        mock_build_out.return_value = mock_outliers

        # Mock Duplicates evaluator
        mock_dups = MagicMock()
        mock_dup_result = MagicMock()
        mock_dup_result.items.exact = None
        mock_dup_result.items.near = None
        mock_dup_result.targets.exact = None
        mock_dup_result.targets.near = None
        mock_dups.evaluate.return_value = mock_dup_result
        mock_build_dup.return_value = mock_dups

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
