"""Tests for the dataset splitting report."""

import pytest

from dataeval_flow._blocks import Fields
from dataeval_flow.workflows import Finding
from dataeval_flow.workflows.data_splitting._outputs import DataSplittingRawOutput, SplitInfo
from dataeval_flow.workflows.data_splitting._report import (
    _factor_table,
    _normalize_label_counts,
    build_findings,
)
from tests.finding_blocks import blocks_of, column, fields, paragraphs, rendered, tables

pytestmark = pytest.mark.required


def _body(finding: Finding) -> list[str]:
    """The finding's rendered detail section below its heading's closing rule."""
    return rendered(finding).splitlines()[3:]


# ---------------------------------------------------------------------------
# TestBuildFindings
# ---------------------------------------------------------------------------


class TestBuildFindings:
    def test_label_distribution_info(self) -> None:
        raw = DataSplittingRawOutput(
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
        raw = DataSplittingRawOutput(
            dataset_size=100,
            label_stats_full={"label_counts_per_class": [55, 40, 5]},
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        label_finding = next(f for f in findings if "Class distribution" in f.title)
        assert label_finding.severity == "warning"  # 55/5 = 11:1 > 10

    def test_label_distribution_uses_class_names(self) -> None:
        """A dataset carrying a class-name table reports names, not class indices."""
        raw = DataSplittingRawOutput(
            dataset_size=100,
            label_stats_full={
                "label_counts_per_class": {0: 50, 1: 45},
                "index2label": {0: "Coverall", 1: "Mask"},
            },
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        label_finding = next(f for f in findings if "Class distribution" in f.title)
        (table,) = tables(label_finding)
        assert column(table, "name") == ["Coverall", "Mask"]
        assert column(table, "value") == [50, 45]

    def test_label_distribution_warning(self) -> None:
        raw = DataSplittingRawOutput(
            dataset_size=100,
            label_stats_full={"label_counts_per_class": {"a": 95, "b": 5}},
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        label_finding = next(f for f in findings if "Class distribution" in f.title)
        assert label_finding.severity == "warning"

    def test_split_sizes_finding(self) -> None:
        raw = DataSplittingRawOutput(
            dataset_size=100,
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        size_finding = next(f for f in findings if "split sizes" in f.title)
        assert fields(size_finding)["Train"] == 70
        assert fields(size_finding)["Val"] == 10
        assert fields(size_finding)["Test"] == 20

    def test_coverage_warning(self) -> None:
        raw = DataSplittingRawOutput(
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

    def test_coverage_reads_as_labelled_values(self) -> None:
        """A split's coverage is one set of labelled values, in the order the report has always listed them."""
        raw = DataSplittingRawOutput(
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
        cov_finding = next(f for f in build_findings(raw) if f.title == "Coverage: fold 0 train")
        (block,) = blocks_of(cov_finding, Fields)
        assert block.items == [
            ("Uncovered count", 10),
            ("Split size", 70),
            ("Uncovered %", 14.29),
            ("Coverage radius", 0.5),
        ]

    def test_every_finding_carries_blocks(self) -> None:
        """Every finding has evidence to show beyond its brief and description."""
        raw = _make_raw(
            full_counts={"a": 50, "b": 50},
            folds=[({"a": 35, "b": 35}, {"a": 10, "b": 10}), ({"a": 36, "b": 34}, {"a": 9, "b": 11})],
            test_counts={"a": 5, "b": 5},
            test_size=10,
        ).model_copy(
            update={
                "pre_split_balance": {"balance": [{"factor_name": "weather", "mi_value": 0.5}]},
                "pre_split_diversity": {"factors": [{"factor_name": "weather", "diversity_value": 0.9}]},
                "coverage_test": {"uncovered_indices": [0], "coverage_radius": 0.4},
            }
        )
        findings = build_findings(raw)
        assert len(findings) == 7
        for finding in findings:
            assert finding.blocks, finding.title


# ---------------------------------------------------------------------------
# TestFactorTable
# ---------------------------------------------------------------------------


class TestFactorTable:
    def test_empty_rows(self) -> None:
        assert _factor_table([], "mi_value", "MI Score", "is_imbalanced").rows == []

    def test_basic_rows(self) -> None:
        rows = [
            {"factor_name": "weather", "mi_value": 0.5, "is_imbalanced": True},
            {"factor_name": "lighting", "mi_value": 0.1, "is_imbalanced": False},
        ]
        table = _factor_table(rows, "mi_value", "MI Score", "is_imbalanced")
        assert column(table, "factor") == ["weather", "lighting"]
        assert column(table, "value") == [0.5, 0.1]
        assert column(table, "flag") == ["[!!]", ""]

    def test_balance_renders_scores_to_four_places(self) -> None:
        """The raw MI score is drawn to four decimals, with the flag under its own left-aligned heading."""
        raw = DataSplittingRawOutput(
            dataset_size=100,
            pre_split_balance={
                "balance": [
                    {"factor_name": "weather", "mi_value": 0.5, "is_imbalanced": True},
                    {"factor_name": "lighting", "mi_value": 0.1, "is_imbalanced": False},
                ]
            },
        )
        balance = next(f for f in build_findings(raw) if f.title == "Pre-split balance (mutual information)")
        assert balance.description == "Higher MI = stronger correlation between factor and class label."
        assert _body(balance) == [
            "  Higher MI = stronger correlation between factor and class label.",
            "",
            "  Factor    MI Score  Flag",
            "  --------  --------  ----",
            "  weather     0.5000  [!!]",
            "  lighting    0.1000",
        ]

    def test_diversity_flags_low_diversity(self) -> None:
        """Diversity reads its own flag, and its value heading sets the column's width."""
        raw = DataSplittingRawOutput(
            dataset_size=100,
            pre_split_diversity={
                "factors": [
                    {"factor_name": "weather", "diversity_value": 0.18731, "is_low_diversity": True},
                    {"factor_name": "time_of_day", "diversity_value": 0.812, "is_low_diversity": False},
                ]
            },
        )
        diversity = next(f for f in build_findings(raw) if f.title == "Pre-split diversity")
        assert _body(diversity) == [
            "  Values near 1.0 = high diversity. Low diversity factors are flagged.",
            "",
            "  Factor       Diversity  Flag",
            "  -----------  ---------  ----",
            "  weather         0.1873  [!!]",
            "  time_of_day     0.8120",
        ]


# ---------------------------------------------------------------------------
# TestBuildFindingsCoverageTest
# ---------------------------------------------------------------------------


class TestBuildFindingsCoverageTest:
    def test_coverage_test_info(self) -> None:
        """Lines 209-213: coverage_test present with low uncovered %."""
        raw = DataSplittingRawOutput(
            dataset_size=100,
            test_indices=list(range(20)),
            coverage_test={"uncovered_indices": [0], "coverage_radius": 0.4},
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        cov_finding = next(f for f in findings if f.title == "Coverage: test")
        assert cov_finding.severity == "info"  # 1/20 = 5%
        assert fields(cov_finding)["Uncovered count"] == 1

    def test_coverage_test_warning(self) -> None:
        """Lines 209-213: coverage_test present with high uncovered %."""
        raw = DataSplittingRawOutput(
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
        raw = DataSplittingRawOutput(
            dataset_size=100,
            test_indices=[],
            coverage_test={"uncovered_indices": [], "coverage_radius": 0.4},
            folds=[SplitInfo(fold=0, train_indices=list(range(80)), val_indices=list(range(20)))],
        )
        findings = build_findings(raw)
        cov_finding = next(f for f in findings if f.title == "Coverage: test")
        assert fields(cov_finding)["Uncovered %"] == 0


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

    def test_dict_named_via_index2label(self) -> None:
        counts = _normalize_label_counts({0: 10, 1: 5}, {0: "Coverall", 1: "Mask"})
        assert counts == {"Coverall": 10, "Mask": 5}

    def test_list_named_via_index2label(self) -> None:
        counts = _normalize_label_counts([30, 20], {0: "Coverall", 1: "Mask"})
        assert counts == {"Coverall": 30, "Mask": 20}

    def test_index2label_with_string_keys(self) -> None:
        """A JSON round-trip stringifies both the count keys and the name-table keys."""
        counts = _normalize_label_counts({"0": 10, "1": 5}, {"0": "Coverall", "1": "Mask"})
        assert counts == {"Coverall": 10, "Mask": 5}

    def test_unnamed_indices_keep_their_index(self) -> None:
        counts = _normalize_label_counts({0: 10, 7: 5}, {0: "Coverall"})
        assert counts == {"Coverall": 10, "7": 5}


# ---------------------------------------------------------------------------
# Helpers for building test data with per-split label stats
# ---------------------------------------------------------------------------


def _make_raw(
    full_counts: dict[str, int],
    folds: list[tuple[dict[str, int], dict[str, int]]],
    test_counts: dict[str, int] | None = None,
    test_size: int = 20,
) -> DataSplittingRawOutput:
    """Build a DataSplittingRawOutput with per-split label stats populated."""
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
    return DataSplittingRawOutput(
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
    def test_single_fold_keeps_labelled_sizes(self) -> None:
        """Single fold lists its sizes as labelled values rather than a table."""
        raw = DataSplittingRawOutput(
            dataset_size=100,
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = build_findings(raw)
        size_finding = next(f for f in findings if "split sizes" in f.title.lower())
        assert tables(size_finding) == []
        (block,) = blocks_of(size_finding, Fields)
        assert block.items == [("Train", 70), ("Val", 10), ("Test", 20)]

    def test_multi_fold_uses_one_table(self) -> None:
        """Multi-fold emits a single table instead of one finding per fold."""
        raw = DataSplittingRawOutput(
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
        (table,) = tables(f)
        assert len(table.rows) == 3  # 3 folds
        assert column(table, "train")[:2] == [160, 155]

    def test_multi_fold_table_and_ranges(self) -> None:
        """The table holds each fold's sizes as numbers; the ranges below it are labelled values."""
        raw = DataSplittingRawOutput(
            dataset_size=300,
            test_indices=list(range(60)),
            folds=[
                SplitInfo(fold=0, train_indices=list(range(160)), val_indices=list(range(80))),
                SplitInfo(fold=1, train_indices=list(range(155)), val_indices=list(range(85))),
                SplitInfo(fold=2, train_indices=list(range(160)), val_indices=list(range(80))),
            ],
        )
        (f,) = [f for f in build_findings(raw) if f.title == "Split sizes across folds"]
        assert f.brief == "3 folds, test=60"
        (table,) = tables(f)
        assert [(c.key, c.header) for c in table.columns] == [
            ("fold", "Fold"),
            ("train", "Train"),
            ("val", "Val"),
            ("test", "Test"),
        ]
        assert column(table, "fold") == [0, 1, 2]
        assert column(table, "val") == [80, 85, 80]
        assert column(table, "test") == [60, 60, 60]
        assert _body(f) == [
            "  Split sizes per fold. Test set is shared across folds.",
            "",
            "  Fold  Train  Val  Test",
            "  ----  -----  ---  ----",
            "  0       160   80    60",
            "  1       155   85    60",
            "  2       160   80    60",
            "",
            "  Train: 155-160 (range 5)",
            "  Val:   80-85 (range 5)",
            "  Test:  60 (shared across folds)",
        ]

    def test_equal_fold_sizes_read_as_one_number(self) -> None:
        """A split every fold sizes alike shows that size alone, as a number."""
        raw = DataSplittingRawOutput(
            dataset_size=300,
            test_indices=list(range(60)),
            folds=[
                SplitInfo(fold=0, train_indices=list(range(160)), val_indices=list(range(80))),
                SplitInfo(fold=1, train_indices=list(range(155)), val_indices=list(range(80))),
            ],
        )
        (f,) = [f for f in build_findings(raw) if f.title == "Split sizes across folds"]
        assert fields(f) == {"Train": "155-160 (range 5)", "Val": 80, "Test": "60 (shared across folds)"}


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
        (table,) = tables(cross)
        headers = [c.header for c in table.columns]
        assert "Train" in headers
        assert "Val" in headers
        assert "Test" in headers
        assert "Full" in headers
        # Should have 2 class rows
        assert len(table.rows) == 2
        # Title should NOT mention "of N" for single fold
        assert "of" not in cross.title

    def test_columns_read_lowercase_split_keys(self) -> None:
        """Each split's column reads its own row key, under the split's name."""
        raw = _make_raw(
            full_counts={"a": 50, "b": 50},
            folds=[({"a": 35, "b": 35}, {"a": 10, "b": 10})],
            test_counts={"a": 5, "b": 5},
            test_size=10,
        )
        cross = next(f for f in build_findings(raw) if "across splits" in f.title)
        (table,) = tables(cross)
        assert [(c.key, c.header) for c in table.columns] == [
            ("class", "Class"),
            ("train", "Train"),
            ("val", "Val"),
            ("test", "Test"),
            ("full", "Full"),
        ]

    def test_no_test_split(self) -> None:
        """Test column omitted when no test split."""
        raw = _make_raw(
            full_counts={"a": 60, "b": 40},
            folds=[({"a": 36, "b": 24}, {"a": 24, "b": 16})],
            test_counts=None,
        )
        findings = build_findings(raw)
        cross = next(f for f in findings if "across splits" in f.title)
        (table,) = tables(cross)
        assert "Test" not in [c.header for c in table.columns]

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
        (table,) = tables(cross)
        rows = table.rows
        # 10 top + 1 placeholder + 5 bottom = 16
        assert len(rows) == 16
        # Check placeholder row exists
        placeholder = rows[10]
        assert placeholder == {"class": "... 10 more ...", "train": "", "val": "", "full": ""}

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
        (table,) = tables(cross)
        row_a = next(r for r in table.rows if r["class"] == "a")
        # All splits have 50% — show count only for splits, count (%) for Full
        assert row_a["train"] == 35
        assert row_a["val"] == 10
        assert row_a["test"] == 5
        assert row_a["full"] == "50 (50%)"
        # No split deviates from the full dataset, so there is no deviation line.
        assert paragraphs(cross) == []

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
        (table,) = tables(cross)
        row_a = next(r for r in table.rows if r["class"] == "a")
        # Train has 80%, Val has 50%, Test has 50% — different, so show all %
        assert row_a["train"] == "56 (80%)"
        assert row_a["val"] == "10 (50%)"

    def test_deviation_line_follows_the_table(self) -> None:
        """The largest deviation from the full dataset reads as a sentence after the table."""
        raw = _make_raw(
            full_counts={"a": 50, "b": 50},
            folds=[({"a": 56, "b": 14}, {"a": 10, "b": 10})],
            test_counts={"a": 5, "b": 5},
            test_size=10,
        )
        cross = next(f for f in build_findings(raw) if "across splits" in f.title)
        assert paragraphs(cross) == ["Max proportion deviation from full dataset: 30.0pp (a in Train)"]
        assert _body(cross) == [
            "  Per-class counts and proportions across splits.",
            "",
            "  Class     Train       Val     Test      Full",
            "  -----  --------  --------  -------  --------",
            "  a      56 (80%)  10 (50%)  5 (50%)  50 (50%)",
            "  b      14 (20%)  10 (50%)  5 (50%)  50 (50%)",
            "",
            "  Max proportion deviation from full dataset: 30.0pp (a in Train)",
        ]

    def test_empty_label_stats_skips(self) -> None:
        """No cross-split finding when per-split stats are missing."""
        raw = DataSplittingRawOutput(
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
        assert stratification.brief == "WARNING - max deviation 30.0pp"

    def test_detail_reads_as_two_groups_of_labelled_values(self) -> None:
        """The deviation and its worst case, then what was checked; the lede is one wrapped paragraph."""
        raw = _make_raw(
            full_counts={"a": 50, "b": 50},
            folds=[({"a": 56, "b": 14}, {"a": 10, "b": 10})],
            test_counts={"a": 5, "b": 5},
            test_size=10,
        )
        stratification = next(f for f in build_findings(raw) if "Stratification" in f.title)
        assert stratification.description == (
            "Checks whether class proportions in each split match the full dataset. "
            "Train proportions may differ if rebalancing was applied."
        )
        deviation, checked = blocks_of(stratification, Fields)
        assert deviation.items == [
            ("Max proportion deviation", "30.0pp"),
            ("Worst", "class 'a' in train (deviation 30.0pp from 50.0% in full)"),
        ]
        assert checked.items == [("Folds checked", 1), ("Classes checked", 2)]
        assert _body(stratification) == [
            "  Checks whether class proportions in each split match the full dataset. Train",
            "  proportions may differ if rebalancing was applied.",
            "",
            "  Max proportion deviation: 30.0pp",
            "  Worst:                    class 'a' in train (deviation 30.0pp from 50.0% in",
            "                            full)",
            "",
            "  Folds checked:   1",
            "  Classes checked: 2",
        ]

    def test_no_worst_case_without_deviation(self) -> None:
        """Matching proportions have no worst case to name."""
        raw = _make_raw(
            full_counts={"a": 500, "b": 500},
            folds=[({"a": 350, "b": 350}, {"a": 100, "b": 100})],
            test_counts={"a": 50, "b": 50},
            test_size=100,
        )
        stratification = next(f for f in build_findings(raw) if "Stratification" in f.title)
        assert fields(stratification) == {"Max proportion deviation": "0.0pp", "Folds checked": 1, "Classes checked": 2}

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
        raw = DataSplittingRawOutput(
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
        # Both classes deviate by 32.4pp there, so only the fold and split are pinned.
        worst = fields(stratification)["Worst"]
        assert isinstance(worst, str)
        assert worst.endswith(" in fold 2 train (deviation 32.4pp from 50.0% in full)")
        assert fields(stratification)["Folds checked"] == 3
