"""Tests for OOD detection report."""

import pytest

from dataeval_flow._blocks import Column, Fields, ItemRef, Paragraph, Table
from dataeval_flow.workflows.ood_detection import OODDetectionHealthThresholds
from dataeval_flow.workflows.ood_detection._outputs import FactorDeviationDict, OODDetectionRawOutput, OODSampleDict
from dataeval_flow.workflows.ood_detection._report import (
    _build_aggregate_finding,
    _build_detector_finding,
    _build_factor_deviations_finding,
    _build_factor_predictors_finding,
    _build_unique_ood_finding,
    _compute_normalized_scores,
    _score_histogram_blocks,
    _severity_for_ood,
    build_findings,
    locator,
)
from tests.finding_blocks import blocks_of, column, fields, paragraphs, rendered, sections, tables
from tests.test_ood_workflow import _make_detector_result, _make_params

pytestmark = pytest.mark.required

# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


class TestSeverityForOOD:
    def test_above_warning_threshold(self):
        t = OODDetectionHealthThresholds(ood_pct_warning=10.0, ood_pct_info=1.0)
        assert _severity_for_ood(15.0, t) == "warning"

    def test_between_info_and_warning(self):
        t = OODDetectionHealthThresholds(ood_pct_warning=10.0, ood_pct_info=1.0)
        assert _severity_for_ood(5.0, t) == "info"

    def test_below_info_threshold(self):
        t = OODDetectionHealthThresholds(ood_pct_warning=10.0, ood_pct_info=1.0)
        assert _severity_for_ood(0.5, t) == "ok"

    def test_zero_ood(self):
        t = OODDetectionHealthThresholds()
        assert _severity_for_ood(0.0, t) == "ok"

    def test_exactly_at_warning(self):
        t = OODDetectionHealthThresholds(ood_pct_warning=10.0)
        assert _severity_for_ood(10.0, t) == "warning"

    def test_exactly_at_info(self):
        t = OODDetectionHealthThresholds(ood_pct_info=1.0)
        assert _severity_for_ood(1.0, t) == "info"


# ---------------------------------------------------------------------------
# Findings builders
# ---------------------------------------------------------------------------


class TestBuildDetectorFinding:
    def test_ood_finding(self):
        result = _make_detector_result(ood_count=15, total_count=100)
        t = OODDetectionHealthThresholds()
        finding = _build_detector_finding("K-Neighbors", result, t)

        assert finding.title == "K-Neighbors"
        assert finding.severity == "warning"  # 15% > default 10%
        assert fields(finding)["OOD count"] == 15

    def test_no_ood_finding(self):
        result = _make_detector_result(ood_count=0, total_count=100)
        t = OODDetectionHealthThresholds()
        finding = _build_detector_finding("K-Neighbors", result, t)

        assert finding.severity == "ok"

    def test_score_histogram_in_a_table(self):
        result = _make_detector_result(
            ood_count=2,
            total_count=5,
            samples=[
                OODSampleDict(index=0, score=0.1, is_ood=False),
                OODSampleDict(index=1, score=0.2, is_ood=False),
                OODSampleDict(index=2, score=0.3, is_ood=False),
                OODSampleDict(index=3, score=0.8, is_ood=True),
                OODSampleDict(index=4, score=0.9, is_ood=True),
            ],
        )
        t = OODDetectionHealthThresholds()
        finding = _build_detector_finding("K-Neighbors", result, t)
        assert len(tables(finding)) == 1
        assert len(tables(finding)[0].rows) > 0

    def test_generic_pairs_are_labelled_fields_after_the_histogram(self):
        result = _make_detector_result(
            ood_count=1,
            total_count=20,
            threshold_score=0.4567891,
            samples=[OODSampleDict(index=i, score=0.1 * i, is_ood=i == 5) for i in range(6)],
        )
        finding = _build_detector_finding("K-Neighbors", result, OODDetectionHealthThresholds())
        assert finding.description == "K-Neighbors: 1/20 samples OOD (5.0%)"
        assert [type(block) for block in finding.blocks] == [Table, Fields]
        assert blocks_of(finding, Fields)[0].items == [
            ("OOD count", 1),
            ("Total count", 20),
            ("OOD percentage", "5.0%"),
            ("Threshold score", 0.456789),
        ]

    def test_rendered_histogram(self):
        """A bin holding one sample against a peak of 38 draws a bar: the old truncation drew nothing."""
        samples = [OODSampleDict(index=0, score=0.0, is_ood=False)]
        samples += [OODSampleDict(index=i, score=0.05, is_ood=False) for i in range(1, 38)]
        samples += [
            OODSampleDict(index=38, score=0.35, is_ood=False),
            OODSampleDict(index=39, score=0.75, is_ood=True),
            OODSampleDict(index=40, score=0.75, is_ood=True),
            OODSampleDict(index=41, score=1.0, is_ood=True),
        ]
        result = _make_detector_result(ood_count=3, total_count=42, threshold_score=0.35, samples=samples)
        finding = _build_detector_finding("K-Neighbors", result, OODDetectionHealthThresholds())
        assert rendered(finding).splitlines() == [
            "=" * 80,
            "  K-NEIGHBORS",
            "=" * 80,
            "  K-Neighbors: 3/42 samples OOD (7.1%)",
            "",
            "        Range  In  OOD  █ in-dist  ░ OOD",
            "  -----------  --  ---  ------------------------------  -----------",
            "  0.000-0.100  38    0  ██████████████████████████████",
            "  0.100-0.200   0    0",
            "  0.200-0.300   0    0",
            "  0.300-0.400   1    0  █                               ← threshold",
            "  0.400-0.500   0    0",
            "  0.500-0.600   0    0",
            "  0.600-0.700   0    0",
            "  0.700-0.800   0    2  ░░",
            "  0.800-0.900   0    0",
            "  0.900-1.000   0    1  ░",
            "",
            "  OOD count:       3",
            "  Total count:     42",
            "  OOD percentage:  7.1%",
            "  Threshold score: 0.35",
        ]


class TestBuildFactorPredictorsFinding:
    def test_basic(self):
        predictors = {"altitude": 0.84, "temperature": 0.12}
        finding = _build_factor_predictors_finding(predictors)
        assert finding.title == "OOD Factor Predictors"
        (table,) = tables(finding)
        assert [c.header for c in table.columns[:2]] == ["Factor", "MI (bits)"]
        assert column(table, "name") == ["altitude", "temperature"]
        assert column(table, "value") == [0.84, 0.12]


def _one_source(index: int) -> ItemRef:
    return ItemRef(source="test", index=index)


class TestLocator:
    """An index into the test sources joined end to end names an item in one of them."""

    def test_each_index_falls_in_its_own_source(self):
        locate = locator([("day", 3), ("night", 2)])
        assert [locate(i) for i in range(5)] == [
            ItemRef(source="day", index=0),
            ItemRef(source="day", index=1),
            ItemRef(source="day", index=2),
            ItemRef(source="night", index=0),
            ItemRef(source="night", index=1),
        ]

    def test_an_empty_source_takes_no_index(self):
        assert locator([("empty", 0), ("full", 2)])(0) == ItemRef(source="full", index=0)


class TestBuildFactorDeviationsFinding:
    def test_each_agreed_sample_is_a_row_with_its_top_factors_most_out_of_distribution_first(self):
        devs = [
            FactorDeviationDict(index=5, deviations={"altitude": 3.0}),
            FactorDeviationDict(index=2, deviations={"altitude": 5.0, "temp": 2.0, "hour": 1.5, "lat": 0.2}),
        ]
        finding = _build_factor_deviations_finding(devs, {2: 1.5, 5: 1.2}, {2, 5}, _one_source)
        (table,) = tables(finding)
        assert [column.header for column in table.columns] == ["", "Item", "Source", "Score", "Top factors"]
        assert [(row["item"], row["factors"]) for row in table.rows] == [
            (2, "altitude=5.00, temp=2.00, hour=1.50"),
            (5, "altitude=3.00"),
        ]
        assert table.rows[0]["image"] == ItemRef(source="test", index=2)

    def test_filters_to_mutual_ood(self):
        devs = [
            FactorDeviationDict(index=2, deviations={"altitude": 5.0}),
            FactorDeviationDict(index=5, deviations={"altitude": 3.0}),
        ]
        finding = _build_factor_deviations_finding(devs, {2: 1.5, 5: 1.2}, {2}, _one_source)
        assert column(tables(finding)[0], "item") == [2]
        assert finding.description == "1/2 OOD samples agreed by all detectors, most out of distribution first"


class TestScoreHistogramBlocks:
    def test_no_samples(self):
        result = _make_detector_result(samples=[])
        blocks = _score_histogram_blocks(result)
        assert blocks == []

    def test_all_same_score(self):
        samples = [OODSampleDict(index=i, score=0.5, is_ood=False) for i in range(5)]
        result = _make_detector_result(samples=samples)
        blocks = _score_histogram_blocks(result)
        assert blocks == [Paragraph(text="All scores = 0.5000")]

    def test_normal_histogram(self):
        samples = [
            OODSampleDict(index=0, score=0.1, is_ood=False),
            OODSampleDict(index=1, score=0.2, is_ood=False),
            OODSampleDict(index=2, score=0.9, is_ood=True),
        ]
        result = _make_detector_result(samples=samples)
        (table,) = _score_histogram_blocks(result)
        assert isinstance(table, Table)
        # One row per bin
        assert len(table.rows) == 10
        assert "\u2190 threshold" in column(table, "marker")
        assert [1.0, 0.0] in column(table, "bar")  # an in-dist segment

    def test_columns(self):
        """Range right-aligned, the two counts, a stacked bar whose header is its legend, then the marker."""
        samples = [OODSampleDict(index=0, score=0.1, is_ood=False), OODSampleDict(index=1, score=0.9, is_ood=True)]
        (table,) = _score_histogram_blocks(_make_detector_result(samples=samples))
        assert isinstance(table, Table)
        assert table.columns == [
            Column(key="range", header="Range", align="right"),
            Column(key="in", header="In"),
            Column(key="ood", header="OOD"),
            Column(key="bar", kind="stacked", series=["in-dist", "OOD"]),
            Column(key="marker", align="left"),
        ]

    def test_stacked_cells_hold_each_bins_in_dist_and_ood_counts(self):
        samples = [
            OODSampleDict(index=0, score=0.0, is_ood=False),
            OODSampleDict(index=1, score=0.05, is_ood=False),
            OODSampleDict(index=2, score=0.55, is_ood=False),
            OODSampleDict(index=3, score=0.58, is_ood=True),
            OODSampleDict(index=4, score=1.0, is_ood=True),
        ]
        (table,) = _score_histogram_blocks(_make_detector_result(threshold_score=0.55, samples=samples))
        assert isinstance(table, Table)
        assert column(table, "range") == [f"{i / 10:.3f}-{(i + 1) / 10:.3f}" for i in range(10)]
        assert column(table, "in") == [2, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        assert column(table, "ood") == [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        assert column(table, "bar") == [[i, o] for i, o in zip(column(table, "in"), column(table, "ood"), strict=True)]
        assert column(table, "marker") == ["", "", "", "", "", "\u2190 threshold", "", "", "", ""]


class TestComputeNormalizedScores:
    def test_basic_normalization(self):
        detectors = {
            "kneighbors": _make_detector_result(
                threshold_score=0.5,
                samples=[
                    OODSampleDict(index=0, score=0.2, is_ood=False),
                    OODSampleDict(index=1, score=0.8, is_ood=True),
                ],
            ),
        }
        norm_scores, mutual_ood, unique_ood = _compute_normalized_scores(detectors)
        # score / threshold: 0.2/0.5=0.4, 0.8/0.5=1.6
        assert abs(norm_scores[0] - 0.4) < 0.01
        assert abs(norm_scores[1] - 1.6) < 0.01
        assert mutual_ood == {1}

    def test_skips_zero_threshold(self):
        detectors = {
            "bad": _make_detector_result(
                threshold_score=0.0,
                samples=[OODSampleDict(index=0, score=0.5, is_ood=True)],
            ),
        }
        norm_scores, mutual_ood, unique_ood = _compute_normalized_scores(detectors)
        assert norm_scores == {}
        assert mutual_ood == set()

    def test_multi_detector_mutual_and_unique(self):
        detectors = {
            "det_a": _make_detector_result(
                threshold_score=1.0,
                samples=[
                    OODSampleDict(index=0, score=1.5, is_ood=True),
                    OODSampleDict(index=1, score=1.2, is_ood=True),
                    OODSampleDict(index=2, score=0.3, is_ood=False),
                ],
            ),
            "det_b": _make_detector_result(
                threshold_score=1.0,
                samples=[
                    OODSampleDict(index=0, score=1.8, is_ood=True),
                    OODSampleDict(index=1, score=0.5, is_ood=False),
                    OODSampleDict(index=2, score=0.4, is_ood=False),
                ],
            ),
        }
        norm_scores, mutual_ood, unique_ood = _compute_normalized_scores(detectors)
        # Index 0 flagged by both, index 1 only by det_a
        assert mutual_ood == {0}
        assert unique_ood["det_a"] == {1}
        assert unique_ood["det_b"] == set()


class TestBuildAggregateFinding:
    def test_basic(self):
        thresholds = OODDetectionHealthThresholds(ood_pct_warning=5.0)
        finding = _build_aggregate_finding({0, 1, 2}, {0: 2.0, 1: 1.5, 2: 1.2}, 5, 100, thresholds, _one_source)
        assert finding.title == "Aggregate OOD (all detectors agree)"
        assert finding.severity == "info"  # 3% between 1% info and 5% warning
        assert finding.description is not None
        assert finding.description.startswith("3/5 OOD samples agreed by all detectors (3.0%), most out of")

    def test_warning_severity(self):
        thresholds = OODDetectionHealthThresholds(ood_pct_warning=1.0)
        finding = _build_aggregate_finding({0, 1}, {0: 2.0, 1: 1.5}, 2, 10, thresholds, _one_source)
        assert finding.severity == "warning"  # 20% > 1%

    def test_samples_are_a_table_most_out_of_distribution_first(self):
        finding = _build_aggregate_finding(
            {0, 1}, {0: 1.5, 1: 2.25}, 4, 100, OODDetectionHealthThresholds(), _one_source
        )
        (table,) = tables(finding)
        assert table.rows == [
            {"image": ItemRef(source="test", index=1), "item": 1, "source": "test", "score": 2.25},
            {"image": ItemRef(source="test", index=0), "item": 0, "source": "test", "score": 1.5},
        ]
        assert table.preview == 10
        assert rendered(finding).splitlines()[-4:] == [
            "  Item  Source  Score",
            "  ----  ------  -----",
            "  1     test    2.25x",
            "  0     test    1.50x",
        ]

    def test_past_500_samples_a_paragraph_counts_the_rest(self):
        scores = {i: 1.0 + i / 1000 for i in range(600)}
        finding = _build_aggregate_finding(set(scores), scores, 600, 600, OODDetectionHealthThresholds(), _one_source)
        (table,) = tables(finding)
        assert len(table.rows) == 500
        assert table.rows[0]["item"] == 599
        assert paragraphs(finding) == [
            "600 samples; the 500 most out of distribution are listed, and every one is in `output.raw`."
        ]

    def test_no_agreed_samples_lists_nothing(self):
        finding = _build_aggregate_finding(set(), {}, 4, 100, OODDetectionHealthThresholds(), _one_source)
        assert finding.description is not None
        assert finding.description.startswith("0/4 OOD samples agreed by all detectors")
        assert finding.blocks == []


class TestBuildUniqueOODFinding:
    _NAMES = {"kneighbors": "K-Neighbors", "domain_classifier": "Domain Classifier"}

    def test_basic(self):
        unique_ood = {"kneighbors": {3, 4}, "domain_classifier": {5}}
        finding = _build_unique_ood_finding(unique_ood, {3: 1.8, 4: 1.3, 5: 1.1}, self._NAMES, _one_source)
        assert finding.title == "Unique OOD Samples (single-detector only)"
        assert finding.description is not None
        assert "3 sample(s)" in finding.description
        titles = [section.title for section in sections(finding)]
        assert "K-Neighbors" in titles
        assert "Domain Classifier" in titles

    def test_skips_empty_sets(self):
        unique_ood = {"kneighbors": set(), "domain_classifier": {5}}
        finding = _build_unique_ood_finding(unique_ood, {5: 1.1}, self._NAMES, _one_source)
        assert "K-Neighbors" not in [section.title for section in sections(finding)]

    def test_each_detector_is_a_section_of_its_samples(self):
        unique_ood = {"kneighbors": {3, 4}, "domain_classifier": {5}}
        finding = _build_unique_ood_finding(unique_ood, {3: 1.8, 4: 1.3, 5: 1.1}, self._NAMES, _one_source)
        knn, dc = sections(finding)
        assert (knn.title, knn.brief) == ("K-Neighbors", "2 unique sample(s)")
        (knn_table,) = [block for block in knn.blocks if isinstance(block, Table)]
        assert column(knn_table, "item") == [3, 4]
        assert (dc.title, dc.brief) == ("Domain Classifier", "1 unique sample(s)")
        assert rendered(finding).splitlines()[3:] == [
            "  3 sample(s) flagged by only one detector",
            "",
            "  K-Neighbors — 2 unique sample(s)",
            "    Item  Source  Score",
            "    ----  ------  -----",
            "    3     test    1.80x",
            "    4     test    1.30x",
            "",
            "  Domain Classifier — 1 unique sample(s)",
            "    Item  Source  Score",
            "    ----  ------  -----",
            "    5     test    1.10x",
        ]


class TestBuildFindings:
    def test_basic_findings(self):
        raw = OODDetectionRawOutput(
            dataset_size=300,
            reference_size=200,
            test_size=100,
            detectors={
                "kneighbors": _make_detector_result(ood_count=5, total_count=100),
            },
            ood_indices=[2, 5, 8, 12, 15],
        )
        params = _make_params()
        names = {"kneighbors": "K-Neighbors"}
        findings = build_findings(raw, params, names, parts=[("test", 100)])

        # Should have detector finding only (no metadata insights, single detector)
        assert len(findings) == 1
        assert any(f.title == "K-Neighbors" for f in findings)

    def test_with_metadata_insights(self):
        raw = OODDetectionRawOutput(
            dataset_size=300,
            reference_size=200,
            test_size=100,
            detectors={"kneighbors": _make_detector_result()},
            ood_indices=[2, 5],
            factor_predictors={"altitude": 0.84, "temp": 0.12},
            factor_deviations=[
                FactorDeviationDict(index=2, deviations={"altitude": 5.0}),
            ],
        )
        params = _make_params()
        names = {"kneighbors": "K-Neighbors"}
        findings = build_findings(raw, params, names, parts=[("test", 100)])

        # detector + factor predictors + factor deviations
        assert len(findings) == 3

    def test_no_ood_no_samples_finding(self):
        raw = OODDetectionRawOutput(
            dataset_size=300,
            reference_size=200,
            test_size=100,
            detectors={"kneighbors": _make_detector_result(ood_count=0, total_count=100)},
            ood_indices=[],
        )
        params = _make_params()
        names = {"kneighbors": "K-Neighbors"}
        findings = build_findings(raw, params, names, parts=[("test", 100)])

        # Only detector finding, no OOD samples table
        assert len(findings) == 1

    def test_multi_detector_includes_aggregate_and_unique(self):
        """Multi-detector should produce aggregate + unique findings."""
        raw = OODDetectionRawOutput(
            dataset_size=400,
            reference_size=200,
            test_size=200,
            detectors={
                "kneighbors": _make_detector_result(
                    ood_count=2,
                    total_count=200,
                    threshold_score=0.5,
                    samples=[
                        OODSampleDict(index=0, score=0.8, is_ood=True),
                        OODSampleDict(index=1, score=0.7, is_ood=True),
                        OODSampleDict(index=2, score=0.1, is_ood=False),
                    ],
                ),
                "domain_classifier": _make_detector_result(
                    method="domain_classifier",
                    ood_count=1,
                    total_count=200,
                    threshold_score=0.6,
                    samples=[
                        OODSampleDict(index=0, score=0.9, is_ood=True),
                        OODSampleDict(index=1, score=0.3, is_ood=False),
                        OODSampleDict(index=2, score=0.2, is_ood=False),
                    ],
                ),
            },
            ood_indices=[0, 1],
        )
        params = _make_params()
        names = {"kneighbors": "K-Neighbors", "domain_classifier": "Domain Classifier"}
        findings = build_findings(raw, params, names, parts=[("test", 100)])

        titles = [f.title for f in findings]
        # 2 detectors + aggregate + unique = 4
        assert "K-Neighbors" in titles
        assert "Domain Classifier" in titles
        assert "Aggregate OOD (all detectors agree)" in titles
        assert "Unique OOD Samples (single-detector only)" in titles

    def test_multi_detector_no_unique_omits_finding(self):
        """When all OOD samples are mutual, unique finding is omitted."""
        raw = OODDetectionRawOutput(
            dataset_size=400,
            reference_size=200,
            test_size=200,
            detectors={
                "kneighbors": _make_detector_result(
                    ood_count=1,
                    total_count=200,
                    threshold_score=0.5,
                    samples=[
                        OODSampleDict(index=0, score=0.8, is_ood=True),
                        OODSampleDict(index=1, score=0.1, is_ood=False),
                    ],
                ),
                "domain_classifier": _make_detector_result(
                    method="domain_classifier",
                    ood_count=1,
                    total_count=200,
                    threshold_score=0.6,
                    samples=[
                        OODSampleDict(index=0, score=0.9, is_ood=True),
                        OODSampleDict(index=1, score=0.2, is_ood=False),
                    ],
                ),
            },
            ood_indices=[0],
        )
        params = _make_params()
        names = {"kneighbors": "K-Neighbors", "domain_classifier": "Domain Classifier"}
        findings = build_findings(raw, params, names, parts=[("test", 100)])

        titles = [f.title for f in findings]
        assert "Aggregate OOD (all detectors agree)" in titles
        assert "Unique OOD Samples (single-detector only)" not in titles

    def test_each_sample_is_named_in_the_test_source_it_came_from(self):
        """Test sources are scored joined end to end; each sample's thumbnail reads from its own source."""
        samples = [OODSampleDict(index=i, score=0.9 if i in (1, 3) else 0.1, is_ood=i in (1, 3)) for i in range(4)]
        detector = _make_detector_result(ood_count=2, total_count=4, threshold_score=0.5, samples=samples)
        raw = OODDetectionRawOutput(
            dataset_size=8,
            reference_size=4,
            test_size=4,
            detectors={"kneighbors": detector, "domain_classifier": {**detector, "method": "domain_classifier"}},
            ood_indices=[1, 3],
        )
        names = {"kneighbors": "K-Neighbors", "domain_classifier": "Domain Classifier"}
        findings = build_findings(raw, _make_params(), names, parts=[("day", 2), ("night", 2)])
        (aggregate,) = [f for f in findings if f.title.startswith("Aggregate")]
        assert [row["image"] for row in tables(aggregate)[0].rows] == [
            ItemRef(source="day", index=1),
            ItemRef(source="night", index=1),
        ]
