"""Tests for the dataset splitting report."""

from dataeval_flow.workflows.splitting.outputs import DataSplittingRawOutputs, SplitInfo
from dataeval_flow.workflows.splitting.report import (
    _format_factor_table,
    build_findings,
)

# ---------------------------------------------------------------------------
# TestBuildFindings
# ---------------------------------------------------------------------------


class TestBuildFindings:
    def test_label_distribution_info(self) -> None:
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            label_stats_full={"label_counts_per_class": {"a": 50, "b": 45}},
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        label_finding = next(f for f in findings if "Class distribution" in f.title)
        assert label_finding.severity == "info"

    def test_label_distribution_from_list(self) -> None:
        """label_counts_per_class is a list after NDArray.tolist() in real usage."""
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            label_stats_full={"label_counts_per_class": [55, 40, 5]},
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        label_finding = next(f for f in findings if "Class distribution" in f.title)
        assert label_finding.severity == "warning"  # 55/5 = 11:1 > 10

    def test_label_distribution_warning(self) -> None:
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            label_stats_full={"label_counts_per_class": {"a": 95, "b": 5}},
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        label_finding = next(f for f in findings if "Class distribution" in f.title)
        assert label_finding.severity == "warning"

    def test_split_sizes_finding(self) -> None:
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        size_finding = next(f for f in findings if "split sizes" in f.title)
        assert isinstance(size_finding.data, dict)
        assert size_finding.data["train"] == 70
        assert size_finding.data["val"] == 10
        assert size_finding.data["test"] == 20

    def test_coverage_warning(self) -> None:
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            test_indices=list(range(20)),
            folds=[
                SplitInfo(
                    fold=0,
                    train_indices=list(range(70)),
                    val_indices=list(range(10)),
                    coverage_train={"uncovered_indices": list(range(10)), "coverage_radius": 0.5},
                )
            ],
        )
        findings = build_findings(raw)
        cov_finding = next(f for f in findings if "Coverage" in f.title)
        assert cov_finding.severity == "warning"  # 10/70 = 14.3% > 5%


# ---------------------------------------------------------------------------
# TestFormatFactorTable
# ---------------------------------------------------------------------------


class TestFormatFactorTable:
    def test_empty_rows(self) -> None:
        assert _format_factor_table([], "mi_value", "MI Score") == []

    def test_basic_rows(self) -> None:
        rows = [
            {"factor_name": "weather", "mi_value": 0.5, "is_imbalanced": True},
            {"factor_name": "lighting", "mi_value": 0.1, "is_imbalanced": False},
        ]
        lines = _format_factor_table(rows, "mi_value", "MI Score")
        text = "\n".join(lines)
        assert "weather" in text
        assert "[!!]" in text
        assert "lighting" in text


# ---------------------------------------------------------------------------
# TestBuildFindingsCoverageTest
# ---------------------------------------------------------------------------


class TestBuildFindingsCoverageTest:
    def test_coverage_test_info(self) -> None:
        """Lines 209-213: coverage_test present with low uncovered %."""
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            test_indices=list(range(20)),
            coverage_test={"uncovered_indices": [0], "coverage_radius": 0.4},
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        cov_finding = next(f for f in findings if f.title == "Coverage: test")
        assert cov_finding.severity == "info"  # 1/20 = 5%
        assert isinstance(cov_finding.data, dict)
        assert cov_finding.data["uncovered_count"] == 1

    def test_coverage_test_warning(self) -> None:
        """Lines 209-213: coverage_test present with high uncovered %."""
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            test_indices=list(range(20)),
            coverage_test={"uncovered_indices": list(range(5)), "coverage_radius": 0.4},
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        cov_finding = next(f for f in findings if f.title == "Coverage: test")
        assert cov_finding.severity == "warning"  # 5/20 = 25% > 5%

    def test_coverage_test_empty_indices(self) -> None:
        """Lines 209-213: coverage_test with empty test_indices."""
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            test_indices=[],
            coverage_test={"uncovered_indices": [], "coverage_radius": 0.4},
            folds=[SplitInfo(fold=0, train_indices=list(range(80)), val_indices=list(range(20)))],
        )
        findings = build_findings(raw)
        cov_finding = next(f for f in findings if f.title == "Coverage: test")
        assert isinstance(cov_finding.data, dict)
        assert cov_finding.data["uncovered_pct"] == 0
