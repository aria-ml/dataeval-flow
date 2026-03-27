"""Tests for the dataset splitting report."""

from dataeval_flow.workflows.splitting.outputs import DataSplittingRawOutputs, SplitInfo
from dataeval_flow.workflows.splitting.report import (
    _format_factor_table,
    _normalize_label_counts,
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


# ---------------------------------------------------------------------------
# TestNormalizeLabelCounts
# ---------------------------------------------------------------------------


class TestNormalizeLabelCounts:
    def test_none(self) -> None:
        assert _normalize_label_counts(None) == {}

    def test_empty_dict(self) -> None:
        assert _normalize_label_counts({}) == {}

    def test_dict(self) -> None:
        assert _normalize_label_counts({"cat": 10, "dog": 5}) == {"cat": 10, "dog": 5}

    def test_list(self) -> None:
        assert _normalize_label_counts([30, 20, 10]) == {"0": 30, "1": 20, "2": 10}


# ---------------------------------------------------------------------------
# Helpers for building test data with per-split label stats
# ---------------------------------------------------------------------------


def _make_raw(
    full_counts: dict[str, int],
    folds: list[tuple[dict[str, int], dict[str, int]]],
    test_counts: dict[str, int] | None = None,
    test_size: int = 20,
) -> DataSplittingRawOutputs:
    """Build a DataSplittingRawOutputs with per-split label stats populated."""
    total = sum(full_counts.values())
    fold_infos: list[SplitInfo] = []
    for i, (train_c, val_c) in enumerate(folds):
        train_total = sum(train_c.values())
        val_total = sum(val_c.values())
        fold_infos.append(
            SplitInfo(
                fold=i,
                train_indices=list(range(train_total)),
                val_indices=list(range(val_total)),
                label_stats_train={"label_counts_per_class": train_c},
                label_stats_val={"label_counts_per_class": val_c},
            )
        )
    test_indices = list(range(test_size)) if test_counts else []
    return DataSplittingRawOutputs(
        dataset_size=total,
        label_stats_full={"label_counts_per_class": full_counts},
        test_indices=test_indices,
        label_stats_test={"label_counts_per_class": test_counts} if test_counts else {},
        folds=fold_infos,
    )


# ---------------------------------------------------------------------------
# TestConsolidatedSplitSizes
# ---------------------------------------------------------------------------


class TestConsolidatedSplitSizes:
    def test_single_fold_keeps_key_value(self) -> None:
        """Single fold preserves existing key_value format."""
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        size_finding = next(f for f in findings if "split sizes" in f.title.lower())
        assert size_finding.report_type == "key_value"
        assert isinstance(size_finding.data, dict)
        assert size_finding.data["train"] == 70

    def test_multi_fold_uses_pivot_table(self) -> None:
        """Multi-fold emits a single pivot_table instead of N key_values."""
        raw = DataSplittingRawOutputs(
            dataset_size=300,
            test_indices=list(range(60)),
            folds=[
                SplitInfo(fold=0, train_indices=list(range(160)), val_indices=list(range(80))),
                SplitInfo(fold=1, train_indices=list(range(155)), val_indices=list(range(85))),
                SplitInfo(fold=2, train_indices=list(range(160)), val_indices=list(range(80))),
            ],
        )
        findings = build_findings(raw)
        size_findings = [f for f in findings if "split sizes" in f.title.lower()]
        assert len(size_findings) == 1
        f = size_findings[0]
        assert f.report_type == "pivot_table"
        assert isinstance(f.data, dict)
        assert len(f.data["table_data"]) == 3  # 3 folds
        assert f.data["table_data"][0]["Train"] == 160
        assert f.data["table_data"][1]["Train"] == 155


# ---------------------------------------------------------------------------
# TestCrossSplitDistribution
# ---------------------------------------------------------------------------


class TestCrossSplitDistribution:
    def test_single_fold(self) -> None:
        """Single fold emits cross-split distribution pivot table."""
        raw = _make_raw(
            full_counts={"a": 50, "b": 50},
            folds=[({"a": 35, "b": 35}, {"a": 10, "b": 10})],
            test_counts={"a": 5, "b": 5},
            test_size=10,
        )
        findings = build_findings(raw)
        cross = next(f for f in findings if "across splits" in f.title)
        assert cross.report_type == "pivot_table"
        assert isinstance(cross.data, dict)
        headers = cross.data["table_headers"]
        assert "Train" in headers
        assert "Val" in headers
        assert "Test" in headers
        assert "Full" in headers
        # Should have 2 class rows
        assert len(cross.data["table_data"]) == 2
        # Title should NOT mention "of N" for single fold
        assert "of" not in cross.title

    def test_no_test_split(self) -> None:
        """Test column omitted when no test split."""
        raw = _make_raw(
            full_counts={"a": 60, "b": 40},
            folds=[({"a": 36, "b": 24}, {"a": 24, "b": 16})],
            test_counts=None,
        )
        findings = build_findings(raw)
        cross = next(f for f in findings if "across splits" in f.title)
        assert isinstance(cross.data, dict)
        assert "Test" not in cross.data["table_headers"]

    def test_multi_fold_shows_fold_0_only(self) -> None:
        """Multi-fold only shows fold 0, title includes 'of N'."""
        raw = _make_raw(
            full_counts={"a": 50, "b": 50},
            folds=[
                ({"a": 17, "b": 17}, {"a": 8, "b": 8}),
                ({"a": 17, "b": 17}, {"a": 8, "b": 8}),
                ({"a": 16, "b": 16}, {"a": 9, "b": 9}),
            ],
            test_counts={"a": 5, "b": 5},
            test_size=10,
        )
        findings = build_findings(raw)
        cross_findings = [f for f in findings if "across splits" in f.title]
        assert len(cross_findings) == 1  # Only fold 0
        assert "of 3" in cross_findings[0].title

    def test_many_classes_truncation(self) -> None:
        """More than 20 classes triggers truncation."""
        # 25 classes
        full = {f"cls_{i:02d}": 100 - i for i in range(25)}
        train = {k: int(v * 0.7) for k, v in full.items()}
        val = {k: v - int(v * 0.7) for k, v in full.items()}
        raw = _make_raw(full_counts=full, folds=[(train, val)])
        findings = build_findings(raw)
        cross = next(f for f in findings if "across splits" in f.title)
        assert isinstance(cross.data, dict)
        rows = cross.data["table_data"]
        # 10 top + 1 placeholder + 5 bottom = 16
        assert len(rows) == 16
        # Check placeholder row exists
        placeholder = rows[10]
        assert "more" in placeholder["Class"]

    def test_deduplicates_percentages(self) -> None:
        """When all splits have the same %, show counts only + % on Full."""
        raw = _make_raw(
            full_counts={"a": 50, "b": 50},
            folds=[({"a": 35, "b": 35}, {"a": 10, "b": 10})],
            test_counts={"a": 5, "b": 5},
            test_size=10,
        )
        findings = build_findings(raw)
        cross = next(f for f in findings if "across splits" in f.title)
        assert isinstance(cross.data, dict)
        rows = cross.data["table_data"]
        row_a = next(r for r in rows if r["Class"] == "a")
        # All splits have 50% — show count only for splits, count (%) for Full
        assert row_a["Train"] == "35"
        assert row_a["Val"] == "10"
        assert row_a["Test"] == "5"
        assert "%" in row_a["Full"]

    def test_shows_percentages_when_different(self) -> None:
        """When splits have different %, show count (%) for each."""
        raw = _make_raw(
            full_counts={"a": 50, "b": 50},
            folds=[({"a": 56, "b": 14}, {"a": 10, "b": 10})],
            test_counts={"a": 5, "b": 5},
            test_size=10,
        )
        findings = build_findings(raw)
        cross = next(f for f in findings if "across splits" in f.title)
        assert isinstance(cross.data, dict)
        rows = cross.data["table_data"]
        row_a = next(r for r in rows if r["Class"] == "a")
        # Train has 80%, Val has 50%, Test has 50% — different, so show all %
        assert "%" in row_a["Train"]
        assert "%" in row_a["Val"]

    def test_empty_label_stats_skips(self) -> None:
        """No cross-split finding when per-split stats are missing."""
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            label_stats_full={"label_counts_per_class": {"a": 50, "b": 50}},
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        cross_findings = [f for f in findings if "across splits" in f.title]
        assert len(cross_findings) == 0


# ---------------------------------------------------------------------------
# TestStratificationQuality
# ---------------------------------------------------------------------------


class TestStratificationQuality:
    def test_ok_severity(self) -> None:
        """Near-identical proportions → severity 'ok'."""
        raw = _make_raw(
            full_counts={"a": 500, "b": 500},
            folds=[({"a": 350, "b": 350}, {"a": 100, "b": 100})],
            test_counts={"a": 50, "b": 50},
            test_size=100,
        )
        findings = build_findings(raw)
        stratification = next(f for f in findings if "Stratification" in f.title)
        assert stratification.severity == "ok"

    def test_warning_severity(self) -> None:
        """Large deviation → severity 'warning'."""
        # train has 80% a, but full has 50% a → deviation = 30pp
        raw = _make_raw(
            full_counts={"a": 50, "b": 50},
            folds=[({"a": 56, "b": 14}, {"a": 10, "b": 10})],
            test_counts={"a": 5, "b": 5},
            test_size=10,
        )
        findings = build_findings(raw)
        stratification = next(f for f in findings if "Stratification" in f.title)
        assert stratification.severity == "warning"
        assert isinstance(stratification.data, dict)
        assert "WARNING" in stratification.data["brief"]

    def test_info_severity(self) -> None:
        """Moderate deviation → severity 'info'."""
        # train: a=39/70=55.7%, full: a=50% → dev=5.7pp
        raw = _make_raw(
            full_counts={"a": 50, "b": 50},
            folds=[({"a": 39, "b": 31}, {"a": 10, "b": 10})],
            test_counts={"a": 5, "b": 5},
            test_size=10,
        )
        findings = build_findings(raw)
        stratification = next(f for f in findings if "Stratification" in f.title)
        assert stratification.severity == "info"

    def test_empty_label_stats_skips(self) -> None:
        """No stratification finding when per-split stats are missing."""
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            label_stats_full={"label_counts_per_class": {"a": 50, "b": 50}},
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        strat_findings = [f for f in findings if "Stratification" in f.title]
        assert len(strat_findings) == 0

    def test_multi_fold_checks_all_folds(self) -> None:
        """Stratification check considers all folds, not just fold 0."""
        raw = _make_raw(
            full_counts={"a": 50, "b": 50},
            folds=[
                ({"a": 17, "b": 17}, {"a": 8, "b": 8}),  # fold 0: balanced
                ({"a": 17, "b": 17}, {"a": 8, "b": 8}),  # fold 1: balanced
                ({"a": 28, "b": 6}, {"a": 8, "b": 8}),  # fold 2: imbalanced train
            ],
            test_counts={"a": 5, "b": 5},
            test_size=10,
        )
        findings = build_findings(raw)
        stratification = next(f for f in findings if "Stratification" in f.title)
        # fold 2 train: a=28/34=82.4%, full a=50% → dev=32.4pp
        assert stratification.severity == "warning"
