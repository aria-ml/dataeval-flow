"""Tests for cleaning report."""

from dataeval_flow.workflows.cleaning.outputs import DataCleaningRawOutputs
from dataeval_flow.workflows.cleaning.params import DataCleaningHealthThresholds
from dataeval_flow.workflows.cleaning.report import (
    _classwise_finding,
    _duplicate_finding,
    _item_id_of,
    _label_distribution_finding,
    build_findings,
    collect_flagged_indices,
)

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
        raw = DataCleaningRawOutputs(
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds(), label_source="filepath")
        label_finding = next(f for f in findings if "Distribution" in f.title)
        assert label_finding.title == "Label/Directory_Name Distribution"

    def test_label_title_with_annotation_labels(self):
        """COCO/YOLO annotation labels use 'Label Distribution' title (not directory-name variant)."""
        raw = DataCleaningRawOutputs(
            dataset_size=10,
            img_outliers={"count": 0, "issues": []},
            label_stats={"item_count": 10, "class_count": 2, "label_counts_per_class": {"a": 5, "b": 5}},
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds(), label_source="annotations")
        label_finding = next(f for f in findings if "Distribution" in f.title)
        assert label_finding.title == "Label Distribution"
        # Footer should still show the label_source annotation
        assert any("annotations" in line for line in label_finding.data["footer_lines"])  # type: ignore[union-attr]

    def test_clean_data_shows_ok_findings(self):
        """Clean data still produces Image Outliers and Classwise Outliers with severity='ok'."""
        raw = DataCleaningRawOutputs(dataset_size=100, img_outliers={"count": 0, "issues": []})
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
        raw = DataCleaningRawOutputs(
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
        raw = DataCleaningRawOutputs(
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds(target_outliers=5.0))
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds(exact_duplicates=0.0))
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds(near_duplicates=5.0))
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds(near_duplicates=5.0))
        f = next(f for f in findings if f.title == "Duplicates")
        assert f.severity == "warning"

    def test_label_imbalance_info_within_threshold(self):
        """Imbalance ratio 2.0 with threshold 10.0 → info."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            label_stats={"item_count": 30, "class_count": 2, "label_counts_per_class": {"a": 20, "b": 10}},
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds(class_label_imbalance=10.0))
        f = next(f for f in findings if "Distribution" in f.title)
        assert f.severity == "info"

    def test_label_imbalance_warning_exceeds_threshold(self):
        """Imbalance ratio 10.0 with threshold 5.0 → warning."""
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            label_stats={"item_count": 110, "class_count": 2, "label_counts_per_class": {"a": 100, "b": 10}},
        )
        findings = build_findings(raw, None, DataCleaningHealthThresholds(class_label_imbalance=5.0))
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
        findings = build_findings(raw, None, DataCleaningHealthThresholds())
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
        findings = build_findings(raw, None, thresholds)
        assert all(f.severity in ("ok", "info") for f in findings)


# ---------------------------------------------------------------------------
# collect_flagged_indices
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
        flagged = collect_flagged_indices(raw)
        assert flagged == {2, 5}

    def test_exact_duplicate_indices(self):
        raw = DataCleaningRawOutputs(
            dataset_size=10,
            img_outliers={"issues": [], "count": 0},
            duplicates={"items": {"exact": [[0, 1, 2]], "near": []}, "targets": {}},
        )
        flagged = collect_flagged_indices(raw)
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
        flagged = collect_flagged_indices(raw)
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
        flagged = collect_flagged_indices(raw)
        assert flagged == {0, 2, 4}


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
# collect_flagged_indices — target duplicates
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
        flagged = collect_flagged_indices(raw)
        # Keep first (item 0), flag rest (item 1)
        assert flagged == {1}


# ---------------------------------------------------------------------------
# _duplicate_finding — near groups with methods and orientations (lines 233-235)
# ---------------------------------------------------------------------------


class TestDuplicateFindingDetailLines:
    def test_near_groups_methods_and_orientations(self):
        raw = DataCleaningRawOutputs(
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
        detail_lines = finding.data["detail_lines"]  # type: ignore[index]
        assert any("Methods:" in line for line in detail_lines)
        assert any("Orientations:" in line for line in detail_lines)


# ---------------------------------------------------------------------------
# _label_distribution_finding — imbalance footer (lines 288-290)
# ---------------------------------------------------------------------------


class TestLabelDistributionFindingImbalanceFooter:
    def test_imbalance_ratio_nonzero_not_one(self):
        raw = DataCleaningRawOutputs(
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
        footer_lines = finding.data["footer_lines"]  # type: ignore[index]
        assert any("Imbalance ratio:" in line for line in footer_lines)


# ---------------------------------------------------------------------------
# _classwise_finding — threshold warning + all-classes-within brief (lines 344-348, 359)
# ---------------------------------------------------------------------------


class TestClasswiseFindingThresholdAndBrief:
    def test_total_pct_exceeds_threshold_warning(self):
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            classwise_outliers={
                "level": "image",
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
        raw = DataCleaningRawOutputs(
            dataset_size=100,
            img_outliers={"count": 0, "issues": []},
            classwise_outliers={
                "level": "image",
                "rows": [
                    {"class_name": "cat", "count": 2, "pct": 2.0},
                    {"class_name": "dog", "count": 1, "pct": 1.0},
                    {"class_name": "Total", "count": 3, "pct": 1.5},
                ],
            },
        )
        finding = _classwise_finding(raw, DataCleaningHealthThresholds(classwise_outliers=5.0))
        brief = finding.data["brief"]  # type: ignore[index]
        assert "all classes within 5.0%" in brief
