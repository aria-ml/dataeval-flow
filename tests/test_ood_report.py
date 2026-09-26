"""Tests for OOD detection report."""

import pytest

from dataeval_flow._blocks import BulletList, Column, Fields, Paragraph, Table
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
)
from tests.finding_blocks import blocks_of, bullets, column, fields, rendered, sections, tables
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


class TestBuildFactorDeviationsFinding:
    def test_basic(self):
        devs = [
            FactorDeviationDict(index=2, deviations={"altitude": 5.0, "temp": 2.0}),
            FactorDeviationDict(index=5, deviations={"altitude": 3.0}),
        ]
        normalized = {2: 1.5, 5: 1.2}
        mutual_ood = {2, 5}
        finding = _build_factor_deviations_finding(devs, normalized, mutual_ood)
        assert len(bullets(finding)) == 2
        # Should be sorted by normalized score descending (index 2 first)
        assert "Sample    2" in bullets(finding)[0]

    def test_filters_to_mutual_ood(self):
        devs = [
            FactorDeviationDict(index=2, deviations={"altitude": 5.0}),
            FactorDeviationDict(index=5, deviations={"altitude": 3.0}),
        ]
        normalized = {2: 1.5, 5: 1.2}
        mutual_ood = {2}  # Only index 2 agreed by all detectors
        finding = _build_factor_deviations_finding(devs, normalized, mutual_ood)
        assert len(bullets(finding)) == 1


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
        mutual_ood = {0, 1, 2}
        normalized_scores = {0: 2.0, 1: 1.5, 2: 1.2}
        thresholds = OODDetectionHealthThresholds(ood_pct_warning=5.0)
        finding = _build_aggregate_finding(mutual_ood, normalized_scores, 5, 100, thresholds)
        assert finding.title == "Aggregate OOD (all detectors agree)"
        assert finding.severity == "info"  # 3% between 1% info and 5% warning
        assert finding.description is not None
        assert "3/5" in finding.description
        # Sorted by score descending
        assert "Sample    0" in bullets(finding)[0]

    def test_warning_severity(self):
        mutual_ood = {0, 1}
        normalized_scores = {0: 2.0, 1: 1.5}
        thresholds = OODDetectionHealthThresholds(ood_pct_warning=1.0)
        finding = _build_aggregate_finding(mutual_ood, normalized_scores, 2, 10, thresholds)
        assert finding.severity == "warning"  # 20% > 1%

    def test_samples_are_a_bullet_list(self):
        finding = _build_aggregate_finding({0, 1}, {0: 1.5, 1: 2.25}, 4, 100, OODDetectionHealthThresholds())
        assert finding.blocks == [BulletList(items=["Sample    1 (score=2.25x)", "Sample    0 (score=1.50x)"])]

    def test_no_agreed_samples_lists_nothing(self):
        finding = _build_aggregate_finding(set(), {}, 4, 100, OODDetectionHealthThresholds())
        assert finding.description is not None
        assert finding.description.startswith("0/4 OOD samples agreed by all detectors")
        assert finding.blocks == []


class TestBuildUniqueOODFinding:
    def test_basic(self):
        unique_ood = {
            "kneighbors": {3, 4},
            "domain_classifier": {5},
        }
        normalized_scores = {3: 1.8, 4: 1.3, 5: 1.1}
        names = {"kneighbors": "K-Neighbors", "domain_classifier": "Domain Classifier"}
        finding = _build_unique_ood_finding(unique_ood, normalized_scores, names)
        assert finding.title == "Unique OOD Samples (single-detector only)"
        assert finding.description is not None
        assert "3 sample(s)" in finding.description
        titles = [section.title for section in sections(finding)]
        assert "K-Neighbors" in titles
        assert "Domain Classifier" in titles

    def test_skips_empty_sets(self):
        unique_ood = {"kneighbors": set(), "domain_classifier": {5}}
        normalized_scores = {5: 1.1}
        names = {"kneighbors": "K-Neighbors", "domain_classifier": "Domain Classifier"}
        finding = _build_unique_ood_finding(unique_ood, normalized_scores, names)
        assert "K-Neighbors" not in [section.title for section in sections(finding)]

    def test_each_detector_is_a_section_of_its_samples(self):
        unique_ood = {"kneighbors": {3, 4}, "domain_classifier": {5}}
        normalized_scores = {3: 1.8, 4: 1.3, 5: 1.1}
        names = {"kneighbors": "K-Neighbors", "domain_classifier": "Domain Classifier"}
        finding = _build_unique_ood_finding(unique_ood, normalized_scores, names)
        knn, dc = sections(finding)
        assert (knn.title, knn.brief) == ("K-Neighbors", "2 unique sample(s)")
        assert knn.blocks == [BulletList(items=["Sample    3 (score=1.80x)", "Sample    4 (score=1.30x)"])]
        assert (dc.title, dc.brief) == ("Domain Classifier", "1 unique sample(s)")
        assert dc.blocks == [BulletList(items=["Sample    5 (score=1.10x)"])]
        assert rendered(finding).splitlines()[3:] == [
            "  3 sample(s) flagged by only one detector",
            "",
            "  K-Neighbors — 2 unique sample(s)",
            "    - Sample    3 (score=1.80x)",
            "    - Sample    4 (score=1.30x)",
            "",
            "  Domain Classifier — 1 unique sample(s)",
            "    - Sample    5 (score=1.10x)",
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
        findings = build_findings(raw, params, names)

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
        findings = build_findings(raw, params, names)

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
        findings = build_findings(raw, params, names)

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
        findings = build_findings(raw, params, names)

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
        findings = build_findings(raw, params, names)

        titles = [f.title for f in findings]
        assert "Aggregate OOD (all detectors agree)" in titles
        assert "Unique OOD Samples (single-detector only)" not in titles
