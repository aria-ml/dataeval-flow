"""Tests for cleaning report."""

from collections.abc import Sequence

import pytest

from dataeval_flow._blocks import Fields
from dataeval_flow.workflows import Finding
from dataeval_flow.workflows.data_cleaning import DataCleaningHealthThresholds
from dataeval_flow.workflows.data_cleaning._outputs import DataCleaningRawOutput
from dataeval_flow.workflows.data_cleaning._report import (
    _classwise_finding,
    _duplicate_finding,
    _item_id_of,
    _label_distribution_finding,
    build_findings,
    collect_flagged_indices,
)
from tests.finding_blocks import column, fields, paragraphs, rendered, sections, tables

pytestmark = pytest.mark.required

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
# build_findings
# ---------------------------------------------------------------------------


class TestBuildFindings:
    def test_outlier_finding(self):
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={
                "count": 5,
                "issues": [{"item_index": i, "metric_name": "m", "metric_value": 0.0} for i in range(5)],
            },
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
        titles = [f.title for f in findings]
        assert "Image Outliers" in titles

    def test_outlier_finding_counts_distinct_images(self):
        """Finding counts distinct images, not total flags (one image can trigger multiple metrics)."""
        raw = DataCleaningRawOutput(
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
        img_finding = next(f for f in findings if f.title == "Image Outliers")
        # 3 distinct images, not 6 total flags
        assert img_finding.brief == f"3 images ({round(3 / 29 * 100, 1)}%)"
        assert fields(img_finding)["Percentage"] == round(3 / 29 * 100, 1)
        # Enriched per_metric breakdown
        (table,) = tables(img_finding)
        per_metric = dict(zip(column(table, "metric"), column(table, "count"), strict=True))
        assert per_metric["brightness"] == 2  # images 0 and 5
        assert per_metric["entropy"] == 2  # images 0 and 5
        assert per_metric["contrast"] == 2  # images 0 and 10
        # 6 flags over 3 images: the raw issues hold the flags, and the note says some images repeat
        assert len(raw.img_outliers["issues"]) == 6
        assert paragraphs(img_finding) == ["(Some images trigger multiple metrics.)"]
        assert fields(img_finding)["Dataset size"] == 29

    def test_target_outlier_finding(self):
        raw = DataCleaningRawOutput(
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
        titles = [f.title for f in findings]
        assert "Target Outliers" in titles
        target_finding = next(f for f in findings if f.title == "Target Outliers")
        # 3 distinct (item_id, target_id) pairs, not 4 total flags
        assert target_finding.brief == "3 targets (0.0%)"
        # Enriched per_metric and total_flags
        assert raw.target_outliers is not None
        assert len(raw.target_outliers["issues"]) == 4
        assert paragraphs(target_finding) == ["(Some targets trigger multiple metrics.)"]
        (table,) = tables(target_finding)
        per_metric = dict(zip(column(table, "metric"), column(table, "count"), strict=True))
        assert per_metric["brightness"] == 3  # (0,0), (0,1), (1,0)
        assert per_metric["contrast"] == 1  # (0,0)
        assert fields(target_finding) == {"Percentage": 0.0, "Total targets": 0}

    def test_duplicate_finding(self):
        raw = DataCleaningRawOutput(
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
        titles = [f.title for f in findings]
        assert "Duplicates" in titles
        dup_finding = next(f for f in findings if f.title == "Duplicates")
        assert fields(dup_finding)["Exact affected"] == 2
        assert fields(dup_finding)["Near affected"] == 2
        assert fields(dup_finding)["Methods"] == "hash"
        assert fields(dup_finding)["Orientations"] == "1 same"
        assert dup_finding.brief == "2 exact (2.0%), 2 near (2.0%)"
        assert paragraphs(dup_finding) == ["1 exact-duplicate groups (2 images)"]
        assert [section.title for section in sections(dup_finding)] == ["1 near-duplicate groups (2 images)"]

    def test_duplicate_finding_exact_only(self):
        """Exact-only duplicates: near fields are empty."""
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            duplicates={"items": {"exact": [[0, 1, 2]], "near": []}, "targets": {}},
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
        dup_finding = next(f for f in findings if f.title == "Duplicates")
        assert fields(dup_finding) == {"Exact groups": 1, "Near groups": 0, "Exact affected": 3, "Near affected": 0}
        assert dup_finding.brief == "3 exact (3.0%), 0 near (0.0%)"
        assert paragraphs(dup_finding) == ["1 exact-duplicate groups (3 images)"]
        assert sections(dup_finding) == []

    def test_duplicate_finding_null_orientation_skipped(self):
        """Near groups with orientation=None are not counted in orientations."""
        raw = DataCleaningRawOutput(
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
        dup_finding = next(f for f in findings if f.title == "Duplicates")
        assert fields(dup_finding)["Orientations"] == "1 same"
        assert dup_finding.brief == "0 exact (0.0%), 4 near (4.0%)"
        assert [section.title for section in sections(dup_finding)] == ["2 near-duplicate groups (4 images)"]

    def test_label_stats_finding(self):
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            label_stats={
                "item_count": 100,
                "class_count": 2,
                "label_counts_per_class": {"cat": 50, "dog": 50},
            },
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
        titles = [f.title for f in findings]
        assert "Label Distribution" in titles
        label_finding = next(f for f in findings if f.title == "Label Distribution")
        # The class count, item count and imbalance ratio are in the brief
        assert label_finding.brief == "2 classes, 100 items, imbalance 1.0:1"
        (table,) = tables(label_finding)
        assert dict(zip(column(table, "name"), column(table, "value"), strict=True)) == {"cat": 50, "dog": 50}
        assert [col.header for col in table.columns] == ["Class", "Count", ""]
        assert paragraphs(label_finding) == ["Balanced: all classes have equal counts"]

    def test_label_stats_finding_imbalanced(self):
        """Imbalanced labels produce a non-empty footer."""
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            label_stats={
                "item_count": 100,
                "class_count": 2,
                "label_counts_per_class": {"cat": 80, "dog": 20},
            },
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
        label_finding = next(f for f in findings if f.title == "Label Distribution")
        assert label_finding.brief == "2 classes, 100 items, imbalance 4.0:1"
        assert paragraphs(label_finding) == ["Imbalance ratio: 4.0 (max/min)"]

    def test_label_distribution_suppressed_when_no_classes(self):
        """Label distribution finding is suppressed when class_count == 0 (unlabeled dataset)."""
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            label_stats={
                "item_count": 100,
                "class_count": 0,
                "label_counts_per_class": {},
                "index2label": {},
            },
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
        titles = [f.title for f in findings]
        assert "Label Distribution" not in titles

    def test_label_distribution_warning_when_empty_class(self):
        """A class with zero items triggers a warning even if imbalance_ratio is 0."""
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            label_stats={
                "item_count": 100,
                "class_count": 3,
                "label_counts_per_class": {"cat": 50, "dog": 50, "bird": 0},
            },
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
        label_finding = next(f for f in findings if "Distribution" in f.title)
        assert label_finding.severity == "warning"
        assert paragraphs(label_finding) == ["Warning: one or more classes have zero items"]

    def test_label_title_with_inferred_directory_labels(self):
        """image_folder with inferred labels uses 'Label/Directory_Name Distribution' title."""
        raw = DataCleaningRawOutput(
            dataset_size=10,
            img_outliers={"count": 0, "issues": []},
            label_stats={"item_count": 10, "class_count": 2, "label_counts_per_class": {"a": 5, "b": 5}},
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds(), label_source="filepath")
        label_finding = next(f for f in findings if "Distribution" in f.title)
        assert label_finding.title == "Label/Directory_Name Distribution"

    def test_label_title_with_annotation_labels(self):
        """COCO/YOLO annotation labels use 'Label Distribution' title (not directory-name variant)."""
        raw = DataCleaningRawOutput(
            dataset_size=10,
            img_outliers={"count": 0, "issues": []},
            label_stats={"item_count": 10, "class_count": 2, "label_counts_per_class": {"a": 5, "b": 5}},
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds(), label_source="annotations")
        label_finding = next(f for f in findings if "Distribution" in f.title)
        assert label_finding.title == "Label Distribution"
        # Footer should still show the label_source annotation
        assert paragraphs(label_finding) == ["Labels annotations", "Balanced: all classes have equal counts"]

    def test_clean_data_shows_ok_findings(self):
        """Clean data still produces Image Outliers and Classwise Outliers with severity='ok'."""
        raw = DataCleaningRawOutput(dataset_size=100, img_outliers={"count": 0, "issues": []})
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
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
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={
                "count": 3,
                "issues": [{"item_index": i, "metric_name": "brightness", "metric_value": 0.1} for i in range(3)],
            },
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds(image_outliers=5.0))
        f = next(f for f in findings if f.title == "Image Outliers")
        assert f.severity == "info"

    def test_image_outliers_warning_exceeds_threshold(self):
        """10% outliers with 5% threshold → warning."""
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={
                "count": 10,
                "issues": [{"item_index": i, "metric_name": "brightness", "metric_value": 0.1} for i in range(10)],
            },
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds(image_outliers=5.0))
        f = next(f for f in findings if f.title == "Image Outliers")
        assert f.severity == "warning"

    def test_target_outliers_warning_exceeds_threshold(self):
        """Target outlier % exceeds threshold → warning."""
        raw = DataCleaningRawOutput(
            dataset_size=100,
            target_outliers={
                "count": 6,
                "issues": [
                    {"item_index": i, "target_index": 0, "metric_name": "area", "metric_value": 0.1} for i in range(6)
                ],
            },
            label_stats={"item_count": 100, "class_count": 1, "label_counts_per_class": {"a": 100}},
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds(target_outliers=5.0))
        f = next(f for f in findings if f.title == "Target Outliers")
        assert f.severity == "warning"

    def test_exact_duplicates_warning_at_zero_threshold(self):
        """Any exact duplicates with 0% threshold → warning."""
        raw = DataCleaningRawOutput(
            dataset_size=100,
            duplicates={
                "items": {"exact": [[0, 1]], "near": []},
                "targets": {},
            },
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds(exact_duplicates=0.0))
        f = next(f for f in findings if f.title == "Duplicates")
        assert f.severity == "warning"

    def test_near_duplicates_info_within_threshold(self):
        """2% near duplicates with 5% threshold → info."""
        raw = DataCleaningRawOutput(
            dataset_size=100,
            duplicates={
                "items": {
                    "exact": [],
                    "near": [{"indices": [0, 1], "methods": ["phash"], "orientation": None}],
                },
                "targets": {},
            },
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds(near_duplicates=5.0))
        f = next(f for f in findings if f.title == "Duplicates")
        assert f.severity == "info"

    def test_near_duplicates_warning_exceeds_threshold(self):
        """10% near duplicates with 5% threshold → warning."""
        raw = DataCleaningRawOutput(
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds(near_duplicates=5.0))
        f = next(f for f in findings if f.title == "Duplicates")
        assert f.severity == "warning"

    def test_label_imbalance_info_within_threshold(self):
        """Imbalance ratio 2.0 with threshold 10.0 → info."""
        raw = DataCleaningRawOutput(
            dataset_size=100,
            label_stats={"item_count": 30, "class_count": 2, "label_counts_per_class": {"a": 20, "b": 10}},
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds(class_label_imbalance=10.0))
        f = next(f for f in findings if "Distribution" in f.title)
        assert f.severity == "info"

    def test_label_imbalance_warning_exceeds_threshold(self):
        """Imbalance ratio 10.0 with threshold 5.0 → warning."""
        raw = DataCleaningRawOutput(
            dataset_size=100,
            label_stats={"item_count": 110, "class_count": 2, "label_counts_per_class": {"a": 100, "b": 10}},
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds(class_label_imbalance=5.0))
        f = next(f for f in findings if "Distribution" in f.title)
        assert f.severity == "warning"

    def test_default_thresholds_exact_dup_always_warns(self):
        """Default exact_duplicates=0.0 means any exact duplicates trigger warning."""
        raw = DataCleaningRawOutput(
            dataset_size=1000,
            duplicates={
                "items": {"exact": [[0, 1]], "near": []},
                "targets": {},
            },
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
        f = next(f for f in findings if f.title == "Duplicates")
        assert f.severity == "warning"

    def test_relaxed_thresholds_all_info(self):
        """Very high thresholds → everything stays info."""
        raw = DataCleaningRawOutput(
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
        findings = build_findings(raw, None, thresholds)
        assert all(f.severity in ("ok", "info") for f in findings)


# ---------------------------------------------------------------------------
# collect_flagged_indices
# ---------------------------------------------------------------------------


class TestCollectFlaggedIndices:
    def test_outlier_indices(self):
        raw = DataCleaningRawOutput(
            dataset_size=10,
            img_outliers={
                "issues": [
                    {"item_index": 2, "metric_name": "m", "metric_value": 0.0},
                    {"item_index": 5, "metric_name": "m", "metric_value": 0.0},
                ],
                "count": 2,
            },
        )
        flagged = collect_flagged_indices(raw)
        assert flagged == {2, 5}

    def test_exact_duplicate_indices(self):
        raw = DataCleaningRawOutput(
            dataset_size=10,
            img_outliers={"issues": [], "count": 0},
            duplicates={"items": {"exact": [[0, 1, 2]], "near": []}, "targets": {}},
        )
        flagged = collect_flagged_indices(raw)
        # Keep first (0), flag rest (1, 2)
        assert flagged == {1, 2}

    def test_near_duplicate_indices(self):
        raw = DataCleaningRawOutput(
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
        flagged = collect_flagged_indices(raw)
        assert flagged == {4, 5}

    def test_combined_indices(self):
        raw = DataCleaningRawOutput(
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
        flagged = collect_flagged_indices(raw)
        assert flagged == {0, 2, 4}


# ---------------------------------------------------------------------------
# _classwise_finding (non-empty rows path)
# ---------------------------------------------------------------------------


class TestClasswiseFinding:
    def test_with_rows(self):
        """Classwise finding with rows produces worst-class summary."""
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            classwise_outliers={
                "count_basis": "image",
                "rows": [
                    {"class_name": "cat", "count": 5, "pct": 10.0},
                    {"class_name": "dog", "count": 2, "pct": 4.0},
                    {"class_name": "Total", "count": 7, "pct": 7.0},
                ],
            },
        )
        finding = _classwise_finding(raw, DataCleaningHealthThresholds())
        assert finding.title == "Classwise Outliers"
        assert finding.brief == "worst: cat (10.0%), 2/2 classes over 3.0%"
        assert raw.classwise_outliers is not None
        assert raw.classwise_outliers.get("count_basis") == "image"

    def test_warning_when_total_exceeds_threshold(self):
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            classwise_outliers={
                "count_basis": "annotation",
                "rows": [
                    {"class_name": "a", "count": 20, "pct": 40.0},
                    {"class_name": "Total", "count": 20, "pct": 40.0},
                ],
            },
        )
        finding = _classwise_finding(raw, DataCleaningHealthThresholds(classwise_outliers=5.0))
        assert finding.severity == "warning"
        assert finding.brief == "worst: a (40.0%), 1/1 classes over 5.0%"


# ---------------------------------------------------------------------------
# collect_flagged_indices — target duplicates
# ---------------------------------------------------------------------------


class TestCollectFlaggedIndicesTargetDups:
    def test_target_exact_duplicates(self):
        """Target-level exact duplicates with SourceIndexDict entries."""
        raw = DataCleaningRawOutput(
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
        flagged = collect_flagged_indices(raw)
        # Keep first (item 0), flag rest (item 1)
        assert flagged == {1}


# ---------------------------------------------------------------------------
# _duplicate_finding — near groups with methods and orientations (lines 233-235)
# ---------------------------------------------------------------------------


class TestDuplicateFindingNearGroupDetail:
    def test_near_groups_methods_and_orientations(self):
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            duplicates={
                "items": {
                    "exact": [],
                    "near": [
                        {"indices": [0, 1], "methods": ["phash", "dhash"], "orientation": "same"},
                        {"indices": [2, 3], "methods": ["phash"], "orientation": "flipped"},
                    ],
                },
                "targets": {},
            },
        )
        finding = _duplicate_finding(raw, DataCleaningHealthThresholds())
        assert finding is not None
        (near,) = sections(finding)
        assert near.title == "2 near-duplicate groups (4 images)"
        assert near.blocks == [Fields(items=[("Methods", "dhash, phash"), ("Orientations", "1 flipped, 1 same")])]


# ---------------------------------------------------------------------------
# _label_distribution_finding — imbalance footer (lines 288-290)
# ---------------------------------------------------------------------------


class TestLabelDistributionFindingImbalanceFooter:
    def test_imbalance_ratio_nonzero_not_one(self):
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            label_stats={
                "item_count": 100,
                "class_count": 2,
                "label_counts_per_class": {"cat": 75, "dog": 25},
            },
        )
        finding = _label_distribution_finding(raw, DataCleaningHealthThresholds())
        assert finding is not None
        assert paragraphs(finding) == ["Imbalance ratio: 3.0 (max/min)"]


# ---------------------------------------------------------------------------
# _classwise_finding — threshold warning + all-classes-within brief (lines 344-348, 359)
# ---------------------------------------------------------------------------


class TestClasswiseFindingThresholdAndBrief:
    def test_total_pct_exceeds_threshold_warning(self):
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            classwise_outliers={
                "count_basis": "image",
                "rows": [
                    {"class_name": "cat", "count": 8, "pct": 8.0},
                    {"class_name": "dog", "count": 3, "pct": 3.0},
                    {"class_name": "Total", "count": 11, "pct": 5.5},
                ],
            },
        )
        finding = _classwise_finding(raw, DataCleaningHealthThresholds(classwise_outliers=5.0))
        assert finding.severity == "warning"

    def test_classes_over_zero_all_within_brief(self):
        raw = DataCleaningRawOutput(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            classwise_outliers={
                "count_basis": "image",
                "rows": [
                    {"class_name": "cat", "count": 2, "pct": 2.0},
                    {"class_name": "dog", "count": 1, "pct": 1.0},
                    {"class_name": "Total", "count": 3, "pct": 1.5},
                ],
            },
        )
        finding = _classwise_finding(raw, DataCleaningHealthThresholds(classwise_outliers=5.0))
        assert finding.brief == "worst: cat (2.0%), all classes within 5.0%"


# ---------------------------------------------------------------------------
# Findings as report blocks
# ---------------------------------------------------------------------------

_RULE = "=" * 80


def _detection_raw(**overrides: object) -> DataCleaningRawOutput:
    """Multi-metric image and target outliers, both kinds of duplicate, imbalanced labels, a classwise pivot."""
    values: dict[str, object] = {
        "dataset_size": 29,
        "img_outliers": {
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
        "target_outliers": {
            "count": 4,
            "issues": [
                {"item_index": 0, "target_index": 0, "metric_name": "brightness", "metric_value": 0.1},
                {"item_index": 0, "target_index": 0, "metric_name": "contrast", "metric_value": 0.2},
                {"item_index": 0, "target_index": 1, "metric_name": "brightness", "metric_value": 0.3},
                {"item_index": 1, "target_index": 0, "metric_name": "brightness", "metric_value": 0.4},
            ],
        },
        "duplicates": {
            "items": {
                "exact": [[0, 1]],
                "near": [
                    {"indices": [2, 3], "methods": ["phash", "dhash"], "orientation": "same"},
                    {"indices": [4, 6], "methods": ["phash"], "orientation": "flipped"},
                ],
            },
            "targets": {},
        },
        "label_stats": {"item_count": 29, "class_count": 2, "label_counts_per_class": {"cat": 21, "dog": 8}},
        "classwise_outliers": {
            "count_basis": "annotation",
            "rows": [
                {"class_name": "cat", "count": 2, "pct": 9.5},
                {"class_name": "dog", "count": 1, "pct": 12.0},
                {"class_name": "Total", "count": 3, "pct": 10.3},
            ],
        },
    }
    values.update(overrides)
    return DataCleaningRawOutput.model_validate(values)


def _finding(title: str, raw: DataCleaningRawOutput, label_source: str | Sequence[str] | None = None) -> Finding:
    """The finding titled *title* that default thresholds draw from *raw*."""
    findings = build_findings(raw, None, DataCleaningHealthThresholds(), label_source=label_source)
    return next(f for f in findings if f.title == title)


class TestFindingBlocks:
    def test_detection_findings_come_in_order(self):
        """A detection dataset with outliers, duplicates and labels reports all five findings, in order."""
        findings = build_findings(_detection_raw(), None, DataCleaningHealthThresholds(), label_source="annotations")
        titles = [f.title for f in findings]
        assert titles == ["Image Outliers", "Target Outliers", "Classwise Outliers", "Duplicates", "Label Distribution"]

    def test_image_outliers_draw_the_per_metric_table_note_and_fields(self):
        """The Count column fits its header without the adapter's `{:>5}`, so the table reads as it did."""
        assert rendered(_finding("Image Outliers", _detection_raw())).splitlines() == [
            _RULE,
            "  IMAGE OUTLIERS" + "3 images (10.3%)".rjust(64),
            _RULE,
            "  3 images (10.3%) flagged as outliers.",
            "",
            "  Metric      Count",
            "  ----------  -----",
            "  brightness      2",
            "  entropy         2",
            "  contrast        2",
            "",
            "  (Some images trigger multiple metrics.)",
            "",
            "  Percentage:   10.3",
            "  Dataset size: 29",
        ]

    def test_per_metric_counts_rank_largest_first(self):
        finding = _finding("Target Outliers", _detection_raw())
        (table,) = tables(finding)
        assert column(table, "metric") == ["brightness", "contrast"]
        assert column(table, "count") == [3, 1]
        assert [(col.header, col.format) for col in table.columns] == [("Metric", None), ("Count", None)]

    def test_target_outliers_count_targets_from_the_label_stats(self):
        finding = _finding("Target Outliers", _detection_raw())
        assert finding.brief == "3 targets (10.3%)"
        assert fields(finding) == {"Percentage": 10.3, "Total targets": 29}

    def test_one_metric_per_image_draws_no_note(self):
        raw = _detection_raw(
            img_outliers={
                "count": 2,
                "issues": [{"item_index": i, "metric_name": "brightness", "metric_value": 0.1} for i in range(2)],
            }
        )
        finding = _finding("Image Outliers", raw)
        assert paragraphs(finding) == []
        assert fields(finding) == {"Percentage": 6.9, "Dataset size": 29}

    def test_no_image_outliers_draw_only_the_fields(self):
        raw = _detection_raw(img_outliers={"count": 0, "issues": []})
        finding = _finding("Image Outliers", raw)
        assert finding.blocks == [Fields(items=[("Percentage", 0.0), ("Dataset size", 29)])]

    def test_classwise_table_keeps_raw_percentages_and_formats_them_as_before(self):
        finding = _finding("Classwise Outliers", _detection_raw())
        (table,) = tables(finding)
        assert column(table, "class_name") == ["cat", "dog", "Total"]
        assert column(table, "count") == [2, 1, 3]
        assert column(table, "pct") == [9.5, 12.0, 10.3]
        assert rendered(finding).splitlines()[3:] == [
            "  Most outliers in dog (12.0%). 2/2 classes exceed 3.0% threshold.",
            "",
            "  Class Name  Count      %",
            "  ----------  -----  -----",
            "  cat             2   9.5%",
            "  dog             1  12.0%",
            "  Total           3  10.3%",
        ]

    def test_classwise_without_outliers_has_no_blocks(self):
        finding = _finding("Classwise Outliers", _detection_raw(classwise_outliers=None))
        assert finding.brief == "no outliers detected"
        assert finding.description == "No outliers detected — classwise breakdown not applicable."
        assert finding.blocks == []

    def test_near_duplicates_draw_as_a_section_holding_methods_and_orientations(self):
        assert rendered(_finding("Duplicates", _detection_raw())).splitlines()[3:] == [
            "  1 exact duplicate groups, 2 near-duplicate groups found.",
            "",
            "  1 exact-duplicate groups (2 images)",
            "",
            "  2 near-duplicate groups (4 images)",
            "    Methods:      dhash, phash",
            "    Orientations: 1 flipped, 1 same",
            "",
            "  Exact groups:   1",
            "  Near groups:    2",
            "  Exact affected: 2",
            "  Near affected:  4",
        ]

    def test_near_duplicates_without_methods_or_orientations_are_a_paragraph(self):
        raw = _detection_raw(
            duplicates={
                "items": {"exact": [], "near": [{"indices": [7, 8], "methods": [], "orientation": None}]},
                "targets": {},
            }
        )
        finding = _finding("Duplicates", raw)
        assert sections(finding) == []
        assert paragraphs(finding) == ["1 near-duplicate groups (2 images)"]

    def test_label_distribution_draws_a_ranked_table_and_one_paragraph_per_footer_line(self):
        finding = _finding("Label Distribution", _detection_raw(), label_source=["annotations", "filepath"])
        assert rendered(finding).splitlines()[3:] == [
            "  2 classes, 29 items.",
            "",
            "  Class  Count",
            "  -----  -----  ------------------------------",
            "  cat       21  ██████████████████████████████",
            "  dog        8  ███████████▍",
            "",
            "  Labels annotations, filepath",
            "",
            "  Imbalance ratio: 2.6 (max/min)",
        ]

    def test_label_distribution_without_counts_draws_nothing(self):
        """Classes with no counted labels leave nothing to rank, so neither the table nor its footer shows."""
        raw = _detection_raw(label_stats={"item_count": 4, "class_count": 2, "label_counts_per_class": {}})
        finding = _finding("Label Distribution", raw, label_source="annotations")
        assert finding.brief == "2 classes, 4 items, imbalance 0.0:1"
        assert finding.blocks == []
