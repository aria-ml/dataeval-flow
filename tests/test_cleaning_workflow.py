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
from dataeval_app.workflows.cleaning.params import DataCleaningHealthThresholds, DataCleaningParameters
from dataeval_app.workflows.cleaning.workflow import (
    CleaningRunContext,
    DataCleaningWorkflow,
    _build_class_labels_df,
    _build_duplicates,
    _build_findings,
    _build_outliers,
    _classwise_finding,
    _collect_flagged_indices,
    _compute_classwise_pivot,
    _compute_embeddings,
    _compute_label_stats,
    _item_id_of,
    _merge_duplicate_results,
    _merge_outlier_outputs,
    _resolve_flags,
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
    @patch("dataeval_app.workflows.cleaning.workflow.Outliers")
    def test_basic_flags(self, mock_outliers_cls: MagicMock):
        params = _make_params(outlier_flags=["dimension", "visual"])
        _build_outliers(params)
        mock_outliers_cls.assert_called_once()
        call_kwargs = mock_outliers_cls.call_args[1]
        assert call_kwargs["outlier_threshold"] == ("adaptive", None)

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
        assert call_kwargs["outlier_threshold"] == ("adaptive", 2.5)

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
# _item_id_of
# ---------------------------------------------------------------------------


class TestItemIdOf:
    def test_plain_int(self):
        assert _item_id_of(5) == 5

    def test_source_index_dict(self):
        assert _item_id_of({"item": 3, "target": 1, "channel": None}) == 3

    def test_source_index_dict_no_target(self):
        assert _item_id_of({"item": 7, "target": None, "channel": None}) == 7


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
                "issues": [{"item_index": i, "metric_name": "m", "metric_value": 0.0} for i in range(5)],
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds())
        titles = [f.title for f in findings]
        assert "Image Outliers" in titles

    def test_outlier_finding_counts_distinct_images(self):
        """Finding counts distinct images, not total flags (one image can trigger multiple metrics)."""
        raw = DataCleaningRawOutputs(
            dataset_size=29,
            img_outliers={
                "count": 6,
                "issues": [
                    {"item_index": 0, "metric_name": "brightness", "metric_value": 0.1},
                    {"item_index": 0, "metric_name": "entropy", "metric_value": 0.2},
                    {"item_index": 0, "metric_name": "contrast", "metric_value": 0.3},
                    {"item_index": 5, "metric_name": "brightness", "metric_value": 0.4},
                    {"item_index": 5, "metric_name": "entropy", "metric_value": 0.5},
                    {"item_index": 10, "metric_name": "contrast", "metric_value": 0.6},
                ],
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds())
        img_finding = next(f for f in findings if f.title == "Image Outliers")
        # 3 distinct images, not 6 total flags
        assert img_finding.data["count"] == 3  # type: ignore[index]
        assert img_finding.data["percentage"] == round(3 / 29 * 100, 1)  # type: ignore[index]
        # Enriched per_metric breakdown
        data = img_finding.data
        assert isinstance(data, dict)
        per_metric = data["per_metric"]
        assert per_metric["brightness"] == 2  # images 0 and 5
        assert per_metric["entropy"] == 2  # images 0 and 5
        assert per_metric["contrast"] == 2  # images 0 and 10
        assert data["total_flags"] == 6
        assert data["dataset_size"] == 29
        # Data-driven renderer keys
        assert data["brief"] == f"3 images ({round(3 / 29 * 100, 1)}%)"
        assert data["multi_metric_subject"] == "images"

    def test_target_outlier_finding(self):
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            target_outliers={  # type: ignore[typeddict-item]  # target records include target_id
                "count": 4,
                "issues": [
                    {"item_index": 0, "target_index": 0, "metric_name": "brightness", "metric_value": 0.1},
                    {"item_index": 0, "target_index": 0, "metric_name": "contrast", "metric_value": 0.2},
                    {"item_index": 0, "target_index": 1, "metric_name": "brightness", "metric_value": 0.3},
                    {"item_index": 1, "target_index": 0, "metric_name": "brightness", "metric_value": 0.4},
                ],
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds())
        titles = [f.title for f in findings]
        assert "Target Outliers" in titles
        target_finding = next(f for f in findings if f.title == "Target Outliers")
        # 3 distinct (item_id, target_id) pairs, not 4 total flags
        assert target_finding.data["count"] == 3  # type: ignore[index]
        # Enriched per_metric and total_flags
        data = target_finding.data
        assert isinstance(data, dict)
        assert data["total_flags"] == 4
        per_metric = data["per_metric"]
        assert per_metric["brightness"] == 3  # (0,0), (0,1), (1,0)
        assert per_metric["contrast"] == 1  # (0,0)
        # Data-driven renderer keys
        assert data["brief"] == "3 targets (0.0%)"
        assert data["multi_metric_subject"] == "targets"

    def test_duplicate_finding(self):
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            duplicates={
                "items": {
                    "exact": [[0, 1]],
                    "near": [{"indices": [2, 3], "methods": ["hash"], "orientation": "same"}],
                },
                "targets": {},
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds())
        titles = [f.title for f in findings]
        assert "Duplicates" in titles
        dup_finding = next(f for f in findings if f.title == "Duplicates")
        assert dup_finding.data["exact_affected"] == 2  # type: ignore[index]
        assert dup_finding.data["near_affected"] == 2  # type: ignore[index]
        assert dup_finding.data["near_methods"] == ["hash"]  # type: ignore[index]
        assert dup_finding.data["near_orientations"] == {"same": 1}  # type: ignore[index]
        # Data-driven renderer keys
        data = dup_finding.data
        assert isinstance(data, dict)
        assert data["brief"] == "2 exact (2.0%), 2 near (2.0%)"
        assert "detail_lines" in data
        assert any("exact-duplicate" in line for line in data["detail_lines"])
        assert any("near-duplicate" in line for line in data["detail_lines"])

    def test_duplicate_finding_exact_only(self):
        """Exact-only duplicates: near fields are empty."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            duplicates={"items": {"exact": [[0, 1, 2]], "near": []}, "targets": {}},
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds())
        dup_finding = next(f for f in findings if f.title == "Duplicates")
        data = dup_finding.data
        assert isinstance(data, dict)
        assert data["exact_groups"] == 1
        assert data["exact_affected"] == 3
        assert data["near_groups"] == 0
        assert data["near_affected"] == 0
        assert data["near_methods"] == []
        assert data["near_orientations"] == {}
        assert data["brief"] == "3 exact (3.0%), 0 near (0.0%)"
        assert any("exact-duplicate" in line for line in data["detail_lines"])
        assert not any("near-duplicate" in line for line in data["detail_lines"])

    def test_duplicate_finding_null_orientation_skipped(self):
        """Near groups with orientation=None are not counted in orientations."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            duplicates={
                "items": {
                    "exact": [],
                    "near": [
                        {"indices": [0, 1], "methods": ["hash"], "orientation": None},
                        {"indices": [2, 3], "methods": ["hash"], "orientation": "same"},
                    ],
                },
                "targets": {},
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds())
        dup_finding = next(f for f in findings if f.title == "Duplicates")
        data = dup_finding.data
        assert isinstance(data, dict)
        assert data["near_orientations"] == {"same": 1}
        # Data-driven renderer keys
        assert data["brief"] == "0 exact (0.0%), 4 near (4.0%)"
        assert any("near-duplicate" in line for line in data["detail_lines"])

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
        findings = _build_findings(raw, None, DataCleaningHealthThresholds())
        titles = [f.title for f in findings]
        assert "Label Distribution" in titles
        label_finding = next(f for f in findings if f.title == "Label Distribution")
        assert label_finding.data["label_counts"] == {"cat": 50, "dog": 50}  # type: ignore[index]
        assert label_finding.data["class_count"] == 2  # type: ignore[index]
        assert label_finding.data["item_count"] == 100  # type: ignore[index]
        assert label_finding.data["imbalance_ratio"] == 1.0  # type: ignore[index]
        # Data-driven renderer keys
        data = label_finding.data
        assert isinstance(data, dict)
        assert data["brief"] == "2 classes, 100 items, imbalance 1.0:1"
        assert data["table_data"] == {"cat": 50, "dog": 50}
        assert data["table_headers"] == ("Class", "Count")
        assert data["footer_lines"] == ["Balanced: all classes have equal counts"]

    def test_label_stats_finding_imbalanced(self):
        """Imbalanced labels produce non-empty footer_lines."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            label_stats={
                "item_count": 100,
                "class_count": 2,
                "label_counts_per_class": {"cat": 80, "dog": 20},
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds())
        label_finding = next(f for f in findings if f.title == "Label Distribution")
        data = label_finding.data
        assert isinstance(data, dict)
        assert data["imbalance_ratio"] == 4.0
        assert data["footer_lines"] == ["Imbalance ratio: 4.0 (max/min)"]

    def test_label_distribution_suppressed_when_no_classes(self):
        """Label distribution finding is suppressed when class_count == 0 (unlabeled dataset)."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            label_stats={
                "item_count": 100,
                "class_count": 0,
                "label_counts_per_class": {},
                "index2label": {},
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds())
        titles = [f.title for f in findings]
        assert "Label Distribution" not in titles

    def test_label_distribution_warning_when_empty_class(self):
        """A class with zero items triggers a warning even if imbalance_ratio is 0."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            label_stats={
                "item_count": 100,
                "class_count": 3,
                "label_counts_per_class": {"cat": 50, "dog": 50, "bird": 0},
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds())
        label_finding = next(f for f in findings if "Distribution" in f.title)
        assert label_finding.severity == "warning"
        assert any("zero items" in line for line in label_finding.data["footer_lines"])  # type: ignore[union-attr]

    def test_label_title_with_inferred_directory_labels(self):
        """image_folder with inferred labels uses 'Label/Directory_Name Distribution' title."""
        raw = DataCleaningRawOutputs(
            dataset_size=10,
            img_outliers={"count": 0, "issues": []},
            label_stats={"item_count": 10, "class_count": 2, "label_counts_per_class": {"a": 5, "b": 5}},
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds(), label_source="filepath")
        label_finding = next(f for f in findings if "Distribution" in f.title)
        assert label_finding.title == "Label/Directory_Name Distribution"

    def test_label_title_with_annotation_labels(self):
        """COCO/YOLO annotation labels use 'Label Distribution' title (not directory-name variant)."""
        raw = DataCleaningRawOutputs(
            dataset_size=10,
            img_outliers={"count": 0, "issues": []},
            label_stats={"item_count": 10, "class_count": 2, "label_counts_per_class": {"a": 5, "b": 5}},
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds(), label_source="annotations")
        label_finding = next(f for f in findings if "Distribution" in f.title)
        assert label_finding.title == "Label Distribution"
        # Footer should still show the label_source annotation
        assert any("annotations" in line for line in label_finding.data["footer_lines"])  # type: ignore[union-attr]

    def test_clean_data_shows_ok_findings(self):
        """Clean data still produces Image Outliers and Classwise Outliers with severity='ok'."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds())
        titles = [f.title for f in findings]
        assert "Image Outliers" in titles
        assert "Classwise Outliers" in titles
        for f in findings:
            assert f.severity == "ok"


# ---------------------------------------------------------------------------
# Health threshold severity tests
# ---------------------------------------------------------------------------


class TestHealthThresholdSeverity:
    """Verify findings are elevated to warning when thresholds are exceeded."""

    def test_image_outliers_info_within_threshold(self):
        """3% outliers with 5% threshold → info."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={
                "count": 3,
                "issues": [{"item_index": i, "metric_name": "brightness", "metric_value": 0.1} for i in range(3)],
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds(image_outliers=5.0))
        f = next(f for f in findings if f.title == "Image Outliers")
        assert f.severity == "info"

    def test_image_outliers_warning_exceeds_threshold(self):
        """10% outliers with 5% threshold → warning."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={
                "count": 10,
                "issues": [{"item_index": i, "metric_name": "brightness", "metric_value": 0.1} for i in range(10)],
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds(image_outliers=5.0))
        f = next(f for f in findings if f.title == "Image Outliers")
        assert f.severity == "warning"

    def test_target_outliers_warning_exceeds_threshold(self):
        """Target outlier % exceeds threshold → warning."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            target_outliers={
                "count": 6,
                "issues": [
                    {"item_index": i, "target_index": 0, "metric_name": "area", "metric_value": 0.1} for i in range(6)
                ],
            },
            label_stats={"item_count": 100, "class_count": 1, "label_counts_per_class": {"a": 100}},
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds(target_outliers=5.0))
        f = next(f for f in findings if f.title == "Target Outliers")
        assert f.severity == "warning"

    def test_exact_duplicates_warning_at_zero_threshold(self):
        """Any exact duplicates with 0% threshold → warning."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            duplicates={
                "items": {"exact": [[0, 1]], "near": []},
                "targets": {},
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds(exact_duplicates=0.0))
        f = next(f for f in findings if f.title == "Duplicates")
        assert f.severity == "warning"

    def test_near_duplicates_info_within_threshold(self):
        """2% near duplicates with 5% threshold → info."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            duplicates={
                "items": {
                    "exact": [],
                    "near": [{"indices": [0, 1], "methods": ["phash"], "orientation": None}],
                },
                "targets": {},
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds(near_duplicates=5.0))
        f = next(f for f in findings if f.title == "Duplicates")
        assert f.severity == "info"

    def test_near_duplicates_warning_exceeds_threshold(self):
        """10% near duplicates with 5% threshold → warning."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            duplicates={
                "items": {
                    "exact": [],
                    "near": [
                        {"indices": list(range(10)), "methods": ["phash"], "orientation": None},
                    ],
                },
                "targets": {},
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds(near_duplicates=5.0))
        f = next(f for f in findings if f.title == "Duplicates")
        assert f.severity == "warning"

    def test_label_imbalance_info_within_threshold(self):
        """Imbalance ratio 2.0 with threshold 10.0 → info."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            label_stats={"item_count": 30, "class_count": 2, "label_counts_per_class": {"a": 20, "b": 10}},
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds(class_label_imbalance=10.0))
        f = next(f for f in findings if "Distribution" in f.title)
        assert f.severity == "info"

    def test_label_imbalance_warning_exceeds_threshold(self):
        """Imbalance ratio 10.0 with threshold 5.0 → warning."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            label_stats={"item_count": 110, "class_count": 2, "label_counts_per_class": {"a": 100, "b": 10}},
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds(class_label_imbalance=5.0))
        f = next(f for f in findings if "Distribution" in f.title)
        assert f.severity == "warning"

    def test_default_thresholds_exact_dup_always_warns(self):
        """Default exact_duplicates=0.0 means any exact dups trigger warning."""
        raw = DataCleaningRawOutputs(
            dataset_size=1000,
            duplicates={
                "items": {"exact": [[0, 1]], "near": []},
                "targets": {},
            },
        )
        findings = _build_findings(raw, None, DataCleaningHealthThresholds())
        f = next(f for f in findings if f.title == "Duplicates")
        assert f.severity == "warning"

    def test_relaxed_thresholds_all_info(self):
        """Very high thresholds → everything stays info."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={
                "count": 50,
                "issues": [{"item_index": i, "metric_name": "brightness", "metric_value": 0.1} for i in range(50)],
            },
            duplicates={
                "items": {"exact": [[0, 1, 2]], "near": []},
                "targets": {},
            },
            label_stats={"item_count": 100, "class_count": 2, "label_counts_per_class": {"a": 90, "b": 10}},
        )
        thresholds = DataCleaningHealthThresholds(
            exact_duplicates=100.0,
            near_duplicates=100.0,
            image_outliers=100.0,
            target_outliers=100.0,
            classwise_outliers=100.0,
            class_label_imbalance=100.0,
        )
        findings = _build_findings(raw, None, thresholds)
        assert all(f.severity in ("ok", "info") for f in findings)


# ---------------------------------------------------------------------------
# _collect_flagged_indices
# ---------------------------------------------------------------------------


class TestCollectFlaggedIndices:
    def test_outlier_indices(self):
        raw = DataCleaningRawOutputs(
            dataset_size=10,
            img_outliers={
                "issues": [
                    {"item_index": 2, "metric_name": "m", "metric_value": 0.0},
                    {"item_index": 5, "metric_name": "m", "metric_value": 0.0},
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
                "issues": [{"item_index": 0, "metric_name": "m", "metric_value": 0.0}],
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
                    {"item_index": 0, "metric_name": "m", "metric_value": 0.0},
                    {"item_index": 1, "metric_name": "m", "metric_value": 0.0},
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
    @patch("dataeval_app.embeddings.OnnxExtractor")
    def test_with_embeddings(
        self,
        mock_extractor_cls: MagicMock,
        mock_meta_cls: MagicMock,
        mock_run_clean: MagicMock,
    ):
        wf = DataCleaningWorkflow()
        mock_dataset = MagicMock()

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
    @patch("dataeval_app.workflows.cleaning.workflow.Duplicates")
    @patch("dataeval_app.workflows.cleaning.workflow.Outliers")
    @patch("dataeval_app.cache.get_or_compute_stats")
    def test_run_cleaning_basic(self, mock_get_stats: MagicMock, mock_outliers_cls: MagicMock, mock_dup_cls: MagicMock):
        from dataeval_app.workflows.cleaning.workflow import _run_cleaning

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

    @patch("dataeval_app.workflows.cleaning.workflow.Duplicates")
    @patch("dataeval_app.workflows.cleaning.workflow.Outliers")
    @patch("dataeval_app.cache.get_or_compute_stats")
    def test_run_cleaning_with_target_index(
        self, mock_get_stats: MagicMock, mock_outliers_cls: MagicMock, mock_dup_cls: MagicMock
    ):
        from dataeval_app.workflows.cleaning.workflow import _run_cleaning

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

    @patch("dataeval_app.workflows.cleaning.workflow.Duplicates")
    @patch("dataeval_app.workflows.cleaning.workflow.Outliers")
    @patch("dataeval_app.cache.get_or_compute_stats")
    def test_run_cleaning_with_metadata(
        self, mock_get_stats: MagicMock, mock_outliers_cls: MagicMock, mock_dup_cls: MagicMock
    ):
        """_run_cleaning computes label_stats when metadata is provided."""
        from dataeval_app.workflows.cleaning.workflow import _run_cleaning

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
# _classwise_finding (non-empty rows path)
# ---------------------------------------------------------------------------


class TestClasswiseFinding:
    def test_with_rows(self):
        """Classwise finding with rows produces worst-class summary."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            classwise_outliers={
                "level": "image",
                "rows": [
                    {"class_name": "cat", "count": 5, "pct": 10.0},
                    {"class_name": "dog", "count": 2, "pct": 4.0},
                    {"class_name": "Total", "count": 7, "pct": 7.0},
                ],
            },
        )
        finding = _classwise_finding(raw, DataCleaningHealthThresholds())
        assert finding.title == "Classwise Outliers"
        assert finding.data["worst_class"] == "cat"  # type: ignore[index]
        assert finding.data["worst_pct"] == 10.0  # type: ignore[index]
        assert finding.data["level"] == "image"  # type: ignore[index]

    def test_warning_when_total_exceeds_threshold(self):
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            classwise_outliers={
                "level": "target",
                "rows": [
                    {"class_name": "a", "count": 20, "pct": 40.0},
                    {"class_name": "Total", "count": 20, "pct": 40.0},
                ],
            },
        )
        finding = _classwise_finding(raw, DataCleaningHealthThresholds(classwise_outliers=5.0))
        assert finding.severity == "warning"
        assert finding.data["classes_over_threshold"] == 1  # type: ignore[index]


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
    @patch("dataeval_app.cache.get_or_compute_embeddings")
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
            dataset = [(np.zeros((3, 32, 32)),), (np.zeros((3, 32, 32)),)]
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
    @patch("dataeval_app.cache.get_or_compute_cluster_result")
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
    @patch("dataeval_app.workflows.cleaning.workflow.Duplicates")
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

    @patch("dataeval_app.workflows.cleaning.workflow._merge_duplicate_results")
    @patch("dataeval_app.workflows.cleaning.workflow.Duplicates")
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
    @patch("dataeval_app.cache.get_or_compute_cluster_result")
    @patch("dataeval_app.workflows.cleaning.workflow.Duplicates")
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

    @patch("dataeval_app.cache.get_or_compute_cluster_result")
    @patch("dataeval_app.workflows.cleaning.workflow.Duplicates")
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
# _collect_flagged_indices — target duplicates
# ---------------------------------------------------------------------------


class TestCollectFlaggedIndicesTargetDups:
    def test_target_exact_duplicates(self):
        """Target-level exact duplicates with SourceIndexDict entries."""
        raw = DataCleaningRawOutputs(
            dataset_size=10,
            img_outliers={"issues": [], "count": 0},
            duplicates={
                "items": {
                    "exact": [[{"item": 0, "target": 0, "channel": None}, {"item": 1, "target": 0, "channel": None}]],
                    "near": [],
                },
                "targets": {},
            },
        )
        flagged = _collect_flagged_indices(raw)
        # Keep first (item 0), flag rest (item 1)
        assert flagged == {1}
