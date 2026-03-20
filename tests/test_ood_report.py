"""Tests for OOD detection report."""

from dataeval_flow.workflows.ood.outputs import FactorDeviationDict, OODDetectionRawOutputs, OODSampleDict
from dataeval_flow.workflows.ood.params import OODHealthThresholds
from dataeval_flow.workflows.ood.report import (
    _build_aggregate_finding,
    _build_detector_finding,
    _build_factor_deviations_finding,
    _build_factor_predictors_finding,
    _build_unique_ood_finding,
    _compute_normalized_scores,
    _score_histogram_lines,
    _severity_for_ood,
    build_findings,
)
from tests.test_ood_workflow import _make_detector_result, _make_params

# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


class TestSeverityForOOD:
    def test_above_warning_threshold(self):
        t = OODHealthThresholds(ood_pct_warning=10.0, ood_pct_info=1.0)
        assert _severity_for_ood(15.0, t) == "warning"

    def test_between_info_and_warning(self):
        t = OODHealthThresholds(ood_pct_warning=10.0, ood_pct_info=1.0)
        assert _severity_for_ood(5.0, t) == "info"

    def test_below_info_threshold(self):
        t = OODHealthThresholds(ood_pct_warning=10.0, ood_pct_info=1.0)
        assert _severity_for_ood(0.5, t) == "ok"

    def test_zero_ood(self):
        t = OODHealthThresholds()
        assert _severity_for_ood(0.0, t) == "ok"

    def test_exactly_at_warning(self):
        t = OODHealthThresholds(ood_pct_warning=10.0)
        assert _severity_for_ood(10.0, t) == "warning"

    def test_exactly_at_info(self):
        t = OODHealthThresholds(ood_pct_info=1.0)
        assert _severity_for_ood(1.0, t) == "info"


# ---------------------------------------------------------------------------
# Findings builders
# ---------------------------------------------------------------------------


class TestBuildDetectorFinding:
    def test_ood_finding(self):
        result = _make_detector_result(ood_count=15, total_count=100)
        t = OODHealthThresholds()
        finding = _build_detector_finding("K-Neighbors", result, t)

        assert finding.title == "K-Neighbors"
        assert finding.severity == "warning"  # 15% > default 10%
        assert finding.report_type == "key_value"
        assert isinstance(finding.data, dict)
        assert finding.data["ood_count"] == 15

    def test_no_ood_finding(self):
        result = _make_detector_result(ood_count=0, total_count=100)
        t = OODHealthThresholds()
        finding = _build_detector_finding("K-Neighbors", result, t)

        assert finding.severity == "ok"

    def test_score_histogram_in_detail_lines(self):
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
        t = OODHealthThresholds()
        finding = _build_detector_finding("K-Neighbors", result, t)
        assert isinstance(finding.data, dict)
        assert len(finding.data["detail_lines"]) > 0


class TestBuildFactorPredictorsFinding:
    def test_basic(self):
        predictors = {"altitude": 0.84, "temperature": 0.12}
        finding = _build_factor_predictors_finding(predictors)
        assert finding.report_type == "table"
        assert finding.title == "OOD Factor Predictors"
        assert isinstance(finding.data, dict)
        assert "table_data" in finding.data
        assert finding.data["table_data"]["altitude"] == 0.84


class TestBuildFactorDeviationsFinding:
    def test_basic(self):
        devs = [
            FactorDeviationDict(index=2, deviations={"altitude": 5.0, "temp": 2.0}),
            FactorDeviationDict(index=5, deviations={"altitude": 3.0}),
        ]
        normalized = {2: 1.5, 5: 1.2}
        mutual_ood = {2, 5}
        finding = _build_factor_deviations_finding(devs, normalized, mutual_ood)
        assert finding.report_type == "key_value"
        assert isinstance(finding.data, dict)
        assert "detail_lines" in finding.data
        assert len(finding.data["detail_lines"]) == 2
        # Should be sorted by normalized score descending (index 2 first)
        assert "Sample    2" in finding.data["detail_lines"][0]

    def test_filters_to_mutual_ood(self):
        devs = [
            FactorDeviationDict(index=2, deviations={"altitude": 5.0}),
            FactorDeviationDict(index=5, deviations={"altitude": 3.0}),
        ]
        normalized = {2: 1.5, 5: 1.2}
        mutual_ood = {2}  # Only index 2 agreed by all detectors
        finding = _build_factor_deviations_finding(devs, normalized, mutual_ood)
        assert isinstance(finding.data, dict)
        assert len(finding.data["detail_lines"]) == 1


class TestScoreHistogramLines:
    def test_no_samples(self):
        result = _make_detector_result(samples=[])
        lines = _score_histogram_lines(result)
        assert lines == []

    def test_all_same_score(self):
        samples = [OODSampleDict(index=i, score=0.5, is_ood=False) for i in range(5)]
        result = _make_detector_result(samples=samples)
        lines = _score_histogram_lines(result)
        assert len(lines) == 1
        assert "0.5000" in lines[0]

    def test_normal_histogram(self):
        samples = [
            OODSampleDict(index=0, score=0.1, is_ood=False),
            OODSampleDict(index=1, score=0.2, is_ood=False),
            OODSampleDict(index=2, score=0.9, is_ood=True),
        ]
        result = _make_detector_result(samples=samples)
        lines = _score_histogram_lines(result)
        # Header + separator + bins + legend
        assert len(lines) > 5
        assert any("threshold" in line for line in lines)
        assert any("\u2588" in line for line in lines)  # in-dist bar


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
        thresholds = OODHealthThresholds(ood_pct_warning=5.0)
        finding = _build_aggregate_finding(mutual_ood, normalized_scores, 5, 100, thresholds)
        assert finding.title == "Aggregate OOD (all detectors agree)"
        assert finding.severity == "info"  # 3% between 1% info and 5% warning
        assert finding.description is not None
        assert "3/5" in finding.description
        assert isinstance(finding.data, dict)
        # Sorted by score descending
        assert "Sample    0" in finding.data["detail_lines"][0]

    def test_warning_severity(self):
        mutual_ood = {0, 1}
        normalized_scores = {0: 2.0, 1: 1.5}
        thresholds = OODHealthThresholds(ood_pct_warning=1.0)
        finding = _build_aggregate_finding(mutual_ood, normalized_scores, 2, 10, thresholds)
        assert finding.severity == "warning"  # 20% > 1%


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
        assert isinstance(finding.data, dict)
        lines = finding.data["detail_lines"]
        assert any("K-Neighbors" in line for line in lines)
        assert any("Domain Classifier" in line for line in lines)

    def test_skips_empty_sets(self):
        unique_ood = {"kneighbors": set(), "domain_classifier": {5}}
        normalized_scores = {5: 1.1}
        names = {"kneighbors": "K-Neighbors", "domain_classifier": "Domain Classifier"}
        finding = _build_unique_ood_finding(unique_ood, normalized_scores, names)
        assert isinstance(finding.data, dict)
        lines = finding.data["detail_lines"]
        assert not any("K-Neighbors" in line for line in lines)


class TestBuildFindings:
    def test_basic_findings(self):
        raw = OODDetectionRawOutputs(
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
        raw = OODDetectionRawOutputs(
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
        raw = OODDetectionRawOutputs(
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
        raw = OODDetectionRawOutputs(
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
        raw = OODDetectionRawOutputs(
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
