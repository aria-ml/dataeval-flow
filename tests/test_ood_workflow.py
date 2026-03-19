"""Tests for OOD detection workflow."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from dataeval_flow.workflow import DatasetContext, WorkflowContext, WorkflowResult
from dataeval_flow.workflows.ood.outputs import (
    DetectorOODResultDict,
    FactorDeviationDict,
    OODDetectionMetadata,
    OODDetectionOutputs,
    OODDetectionRawOutputs,
    OODDetectionReport,
    OODSampleDict,
    is_ood_result,
)
from dataeval_flow.workflows.ood.params import (
    OODDetectionParameters,
    OODDetectorDomainClassifier,
    OODDetectorKNeighbors,
    OODHealthThresholds,
)
from dataeval_flow.workflows.ood.workflow import (
    OODDetectionWorkflow,
    _build_aggregate_finding,
    _build_detector_finding,
    _build_factor_deviations_finding,
    _build_factor_predictors_finding,
    _build_findings,
    _build_ood_detector,
    _build_unique_ood_finding,
    _compute_normalized_scores,
    _intersect_numeric_factors,
    _merge_factor_parts,
    _ood_detector_display_name,
    _score_histogram_lines,
    _serialize_ood_result,
    _severity_for_ood,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(**overrides: object) -> OODDetectionParameters:
    """Build OODDetectionParameters with minimal defaults for testing."""
    defaults: dict[str, object] = {
        "detectors": [{"method": "kneighbors"}],
    }
    defaults.update(overrides)
    return OODDetectionParameters.model_validate(defaults)


def _make_embeddings(n: int, d: int = 10, seed: int = 42) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def _make_ood_output(
    n: int,
    ood_indices: set[int] | None = None,
    seed: int = 42,
):
    """Build a mock OODOutput."""
    from dataeval.shift import OODOutput

    ood_indices = ood_indices or set()
    rng = np.random.default_rng(seed)
    scores = rng.random(n).astype(np.float32)
    # Make OOD samples have higher scores
    for idx in ood_indices:
        scores[idx] = scores[idx] + 10.0
    is_ood = np.zeros(n, dtype=bool)
    for idx in ood_indices:
        is_ood[idx] = True

    return OODOutput(is_ood=is_ood, instance_score=scores, feature_score=None)


def _make_detector_result(
    method: str = "kneighbors",
    ood_count: int = 5,
    total_count: int = 100,
    **kwargs: Any,
) -> DetectorOODResultDict:
    defaults: dict[str, Any] = {
        "method": method,
        "ood_count": ood_count,
        "total_count": total_count,
        "ood_percentage": round(100.0 * ood_count / total_count, 2) if total_count > 0 else 0.0,
        "threshold_score": 0.85,
    }
    defaults.update(kwargs)
    return DetectorOODResultDict(**defaults)


class _FakeDataset:
    """Minimal dataset for testing."""

    def __init__(self, targets: list[Any]):
        self._targets = targets

    @property
    def metadata(self) -> Any:
        return {"id": "fake"}

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, Any, dict[str, Any]]:
        img = np.zeros((3, 32, 32), dtype=np.float32)
        return img, self._targets[idx], {"id": idx}


# ---------------------------------------------------------------------------
# _build_ood_detector
# ---------------------------------------------------------------------------


class TestBuildOODDetector:
    def test_kneighbors(self):
        from dataeval.shift import OODKNeighbors

        det = _build_ood_detector(OODDetectorKNeighbors(k=5, distance_metric="euclidean"))
        assert isinstance(det, OODKNeighbors)

    def test_domain_classifier(self):
        from dataeval.shift import OODDomainClassifier

        det = _build_ood_detector(OODDetectorDomainClassifier(n_folds=3))
        assert isinstance(det, OODDomainClassifier)

    def test_unknown_config_type(self):
        with pytest.raises(ValueError, match="Unknown OOD detector"):
            _build_ood_detector("not_a_config")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _ood_detector_display_name
# ---------------------------------------------------------------------------


class TestOODDetectorDisplayName:
    def test_kneighbors_default(self):
        assert _ood_detector_display_name(OODDetectorKNeighbors()) == "K-Neighbors"

    def test_kneighbors_non_default(self):
        name = _ood_detector_display_name(OODDetectorKNeighbors(k=20, distance_metric="euclidean"))
        assert "K-Neighbors" in name
        assert "k=20" in name
        assert "distance_metric=euclidean" in name

    def test_domain_classifier_default(self):
        assert _ood_detector_display_name(OODDetectorDomainClassifier()) == "Domain Classifier"

    def test_domain_classifier_non_default(self):
        name = _ood_detector_display_name(OODDetectorDomainClassifier(n_folds=3))
        assert "Domain Classifier" in name
        assert "n_folds=3" in name


# ---------------------------------------------------------------------------
# _serialize_ood_result
# ---------------------------------------------------------------------------


class TestSerializeOODResult:
    def test_basic_serialization(self):
        output = _make_ood_output(10, ood_indices={2, 5, 8})
        config = OODDetectorKNeighbors()
        result = _serialize_ood_result(output, config, 10)

        assert result["method"] == "kneighbors"
        assert result["ood_count"] == 3
        assert result["total_count"] == 10
        assert result["ood_percentage"] == 30.0
        samples = result.get("samples", [])
        assert len(samples) == 10
        assert samples[2]["is_ood"] is True
        assert samples[0]["is_ood"] is False

    def test_no_ood_samples(self):
        output = _make_ood_output(10, ood_indices=set())
        config = OODDetectorKNeighbors()
        result = _serialize_ood_result(output, config, 10)

        assert result["ood_count"] == 0
        assert result["ood_percentage"] == 0.0

    def test_all_ood_samples(self):
        output = _make_ood_output(5, ood_indices={0, 1, 2, 3, 4})
        config = OODDetectorKNeighbors()
        result = _serialize_ood_result(output, config, 5)

        assert result["ood_count"] == 5
        assert result["ood_percentage"] == 100.0


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


class TestMergeFactorParts:
    def test_meta_only(self):
        meta = [{"a": np.array([1, 2]), "b": np.array([3, 4])}]
        result = _merge_factor_parts(meta, [])
        assert len(result) == 1
        assert "a" in result[0]

    def test_stats_only(self):
        stats = [{"f_mean": np.array([0.1, 0.2])}]
        result = _merge_factor_parts([], stats)
        assert len(result) == 1
        assert "f_mean" in result[0]

    def test_merged(self):
        meta = [{"a": np.array([1])}]
        stats = [{"f_mean": np.array([0.1])}]
        result = _merge_factor_parts(meta, stats)
        assert len(result) == 1
        assert "a" in result[0]
        assert "f_mean" in result[0]

    def test_extra_stats(self):
        meta = [{"a": np.array([1])}]
        stats = [{"f_mean": np.array([0.1])}, {"f_std": np.array([0.2])}]
        result = _merge_factor_parts(meta, stats)
        assert len(result) == 2
        assert "a" in result[0]
        assert "f_std" in result[1]


class TestIntersectNumericFactors:
    def test_basic_intersection(self):
        ref = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        test_parts = [{"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])}]
        result = _intersect_numeric_factors(ref, test_parts)
        assert result is not None
        ref_common, test_common = result
        assert "a" in ref_common
        assert "b" in ref_common

    def test_no_common_keys(self):
        ref = {"a": np.array([1.0])}
        test_parts = [{"b": np.array([2.0])}]
        result = _intersect_numeric_factors(ref, test_parts)
        assert result is None

    def test_filters_non_numeric(self):
        ref = {"a": np.array(["x", "y"]), "b": np.array([1.0, 2.0])}
        test_parts = [{"a": np.array(["z", "w"]), "b": np.array([3.0, 4.0])}]
        result = _intersect_numeric_factors(ref, test_parts)
        assert result is not None
        ref_common, _ = result
        assert "a" not in ref_common
        assert "b" in ref_common

    def test_filters_2d_arrays(self):
        ref = {"a": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": np.array([5.0, 6.0])}
        test_parts = [{"a": np.array([[7.0, 8.0], [9.0, 10.0]]), "b": np.array([11.0, 12.0])}]
        result = _intersect_numeric_factors(ref, test_parts)
        assert result is not None
        ref_common, _ = result
        assert "a" not in ref_common
        assert "b" in ref_common

    def test_filters_nan(self):
        ref = {"a": np.array([1.0, np.nan]), "b": np.array([1.0, 2.0])}
        test_parts = [{"a": np.array([3.0, 4.0]), "b": np.array([5.0, 6.0])}]
        result = _intersect_numeric_factors(ref, test_parts)
        assert result is not None
        ref_common, _ = result
        assert "a" not in ref_common

    def test_filters_zero_variance(self):
        ref = {"a": np.array([1.0, 2.0]), "b": np.array([1.0, 2.0])}
        test_parts = [{"a": np.array([5.0, 5.0]), "b": np.array([3.0, 4.0])}]  # a has zero variance in test
        result = _intersect_numeric_factors(ref, test_parts)
        assert result is not None
        ref_common, _ = result
        assert "a" not in ref_common
        assert "b" in ref_common

    def test_all_filtered_returns_none(self):
        ref = {"a": np.array([1.0, np.nan])}
        test_parts = [{"a": np.array([3.0, 4.0])}]
        result = _intersect_numeric_factors(ref, test_parts)
        assert result is None

    def test_concatenates_test_parts(self):
        ref = {"a": np.array([1.0, 2.0])}
        test_parts = [{"a": np.array([3.0])}, {"a": np.array([4.0])}]
        result = _intersect_numeric_factors(ref, test_parts)
        assert result is not None
        _, test_common = result
        np.testing.assert_array_equal(test_common["a"], [3.0, 4.0])


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
        findings = _build_findings(raw, params, names)

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
        findings = _build_findings(raw, params, names)

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
        findings = _build_findings(raw, params, names)

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
        findings = _build_findings(raw, params, names)

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
        findings = _build_findings(raw, params, names)

        titles = [f.title for f in findings]
        assert "Aggregate OOD (all detectors agree)" in titles
        assert "Unique OOD Samples (single-detector only)" not in titles


# ---------------------------------------------------------------------------
# Metadata factor extraction
# ---------------------------------------------------------------------------


class TestExtractMetadataFactors:
    @patch("dataeval_flow.workflows.ood.workflow.get_or_compute_metadata")
    def test_basic_extraction(self, mock_metadata: MagicMock):
        from dataeval_flow.workflows.ood.workflow import _extract_metadata_factors

        # Build a mock metadata result with a dataframe
        mock_df = MagicMock()
        mock_df.columns = ["brightness", "contrast"]
        mock_df.__len__ = lambda _self: 2
        mock_df.__getitem__ = lambda _self, _key: MagicMock(to_numpy=lambda: np.array([1.0, 2.0]))
        mock_meta = MagicMock()
        mock_meta.factor_names = ["brightness", "contrast"]
        mock_meta.dataframe = mock_df
        mock_meta.class_labels = np.array([0, 1])
        mock_metadata.return_value = mock_meta

        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        result = _extract_metadata_factors(dc, dc.dataset)

        assert result is not None
        assert "brightness" in result
        assert "class_label" in result

    @patch("dataeval_flow.workflows.ood.workflow.get_or_compute_metadata")
    def test_returns_none_on_exception(self, mock_metadata: MagicMock):
        from dataeval_flow.workflows.ood.workflow import _extract_metadata_factors

        mock_metadata.side_effect = RuntimeError("fail")
        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        result = _extract_metadata_factors(dc, dc.dataset)
        assert result is None


class TestExtractStatsFactors:
    @patch("dataeval_flow.workflows.ood.workflow.get_or_compute_stats")
    def test_basic_extraction(self, mock_stats: MagicMock):
        from dataeval_flow.workflows.ood.workflow import _extract_stats_factors

        mock_stats.return_value = {
            "stats": {"mean": np.array([0.1, 0.2, 0.3]), "std": np.array([0.01, 0.02, 0.03])},
            "image_count": 3,
        }
        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        result = _extract_stats_factors(dc, dc.dataset)

        assert result is not None
        assert "f_mean" in result
        assert "f_std" in result
        assert len(result["f_mean"]) == 3

    @patch("dataeval_flow.workflows.ood.workflow.get_or_compute_stats")
    def test_empty_stats_returns_none(self, mock_stats: MagicMock):
        from dataeval_flow.workflows.ood.workflow import _extract_stats_factors

        mock_stats.return_value = {"stats": {}, "image_count": 0}
        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        result = _extract_stats_factors(dc, dc.dataset)
        assert result is None

    @patch("dataeval_flow.workflows.ood.workflow.get_or_compute_stats")
    def test_returns_none_on_exception(self, mock_stats: MagicMock):
        from dataeval_flow.workflows.ood.workflow import _extract_stats_factors

        mock_stats.side_effect = RuntimeError("fail")
        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        result = _extract_stats_factors(dc, dc.dataset)
        assert result is None

    @patch("dataeval_flow.workflows.ood.workflow.get_or_compute_stats")
    def test_filters_wrong_length(self, mock_stats: MagicMock):
        from dataeval_flow.workflows.ood.workflow import _extract_stats_factors

        mock_stats.return_value = {
            "stats": {"mean": np.array([0.1, 0.2, 0.3]), "bad": np.array([0.1])},
            "image_count": 3,
        }
        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        result = _extract_stats_factors(dc, dc.dataset)
        assert result is not None
        assert "f_mean" in result
        assert "f_bad" not in result


class TestCollectNumericFactors:
    @patch("dataeval_flow.workflows.ood.workflow._extract_stats_factors")
    @patch("dataeval_flow.workflows.ood.workflow._extract_metadata_factors")
    def test_returns_none_when_no_ref_factors(self, mock_meta: MagicMock, mock_stats: MagicMock):
        from dataeval_flow.workflows.ood.workflow import _collect_numeric_factors

        mock_meta.return_value = None
        mock_stats.return_value = None

        ref_dc = DatasetContext(name="ref", dataset=MagicMock(), extractor=None)
        result = _collect_numeric_factors(ref_dc, ref_dc.dataset, [])
        assert result is None

    @patch("dataeval_flow.workflows.ood.workflow._extract_stats_factors")
    @patch("dataeval_flow.workflows.ood.workflow._extract_metadata_factors")
    def test_returns_none_when_no_test_factors(self, mock_meta: MagicMock, mock_stats: MagicMock):
        from dataeval_flow.workflows.ood.workflow import _collect_numeric_factors

        mock_meta.side_effect = [{"a": np.array([1.0])}, None]
        mock_stats.side_effect = [None, None]

        ref_dc = DatasetContext(name="ref", dataset=MagicMock(), extractor=None)
        test_dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        result = _collect_numeric_factors(ref_dc, ref_dc.dataset, [("test", test_dc, test_dc.dataset)])
        assert result is None


class TestComputeMetadataInsights:
    def test_returns_none_when_no_ood(self):
        from dataeval_flow.workflows.ood.workflow import _compute_metadata_insights

        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        devs, preds = _compute_metadata_insights(dc, dc.dataset, [], [], 50)
        assert devs is None
        assert preds is None

    @patch("dataeval_flow.workflows.ood.workflow._collect_numeric_factors")
    def test_returns_none_when_no_factors(self, mock_collect: MagicMock):
        from dataeval_flow.workflows.ood.workflow import _compute_metadata_insights

        mock_collect.return_value = None
        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        devs, preds = _compute_metadata_insights(dc, dc.dataset, [], [0, 1], 50)
        assert devs is None
        assert preds is None

    @patch("dataeval_flow.workflows.ood.workflow.factor_predictors")
    @patch("dataeval_flow.workflows.ood.workflow.factor_deviation")
    @patch("dataeval_flow.workflows.ood.workflow._collect_numeric_factors")
    def test_handles_deviation_exception(self, mock_collect: MagicMock, mock_dev: MagicMock, mock_pred: MagicMock):
        from dataeval_flow.workflows.ood.workflow import _compute_metadata_insights

        ref = {"a": np.array([1.0, 2.0, 3.0])}
        test = {"a": np.array([4.0, 5.0, 6.0])}
        mock_collect.return_value = (ref, test)
        mock_dev.side_effect = RuntimeError("deviation failed")
        mock_pred.return_value = {"a": 0.5}

        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        devs, preds = _compute_metadata_insights(dc, dc.dataset, [], [0], 50)
        assert devs is None  # failed
        assert preds is not None  # still succeeded

    @patch("dataeval_flow.workflows.ood.workflow.factor_predictors")
    @patch("dataeval_flow.workflows.ood.workflow.factor_deviation")
    @patch("dataeval_flow.workflows.ood.workflow._collect_numeric_factors")
    def test_handles_predictors_exception(self, mock_collect: MagicMock, mock_dev: MagicMock, mock_pred: MagicMock):
        from dataeval_flow.workflows.ood.workflow import _compute_metadata_insights

        ref = {"a": np.array([1.0, 2.0, 3.0])}
        test = {"a": np.array([4.0, 5.0, 6.0])}
        mock_collect.return_value = (ref, test)
        mock_dev.return_value = [{"a": 1.0}]
        mock_pred.side_effect = RuntimeError("predictors failed")

        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        devs, preds = _compute_metadata_insights(dc, dc.dataset, [], [0], 50)
        assert devs is not None  # succeeded
        assert preds is None  # failed


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


class TestGetEmbeddingsForContext:
    def test_raises_without_extractor(self):
        from dataeval_flow.workflows.ood.workflow import _get_embeddings_for_context

        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        with pytest.raises(ValueError, match="requires a model/extractor"):
            _get_embeddings_for_context(dc, MagicMock())

    @patch("dataeval_flow.workflows.ood.workflow.get_or_compute_embeddings")
    @patch("dataeval_flow.workflows.ood.workflow.selection_repr", return_value="sel:all")
    def test_calls_get_or_compute(self, mock_sel: MagicMock, mock_emb: MagicMock):  # noqa: ARG002
        from dataeval_flow.workflows.ood.workflow import _get_embeddings_for_context

        expected = _make_embeddings(10)
        mock_emb.return_value = expected
        dc = DatasetContext(
            name="test",
            dataset=MagicMock(),
            extractor=MagicMock(),
            batch_size=16,
        )
        result = _get_embeddings_for_context(dc, dc.dataset)
        np.testing.assert_array_equal(result, expected)
        mock_emb.assert_called_once()

    @patch("dataeval_flow.workflows.ood.workflow.get_or_compute_embeddings")
    @patch("dataeval_flow.workflows.ood.workflow.active_cache")
    @patch("dataeval_flow.workflows.ood.workflow.selection_repr", return_value="sel:all")
    def test_uses_cache_when_available(
        self,
        mock_sel: MagicMock,  # noqa: ARG002
        mock_cache: MagicMock,
        mock_emb: MagicMock,
    ):
        from dataeval_flow.workflows.ood.workflow import _get_embeddings_for_context

        expected = _make_embeddings(10)
        mock_emb.return_value = expected
        dc = DatasetContext(
            name="test",
            dataset=MagicMock(),
            extractor=MagicMock(),
            batch_size=16,
            cache=MagicMock(),
        )
        result = _get_embeddings_for_context(dc, dc.dataset)
        np.testing.assert_array_equal(result, expected)
        mock_cache.assert_called_once()


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class TestOODDetectionOutputs:
    def test_raw_defaults(self):
        raw = OODDetectionRawOutputs(dataset_size=0)
        assert raw.reference_size == 0
        assert raw.test_size == 0
        assert raw.detectors == {}
        assert raw.ood_indices == []
        assert raw.factor_deviations is None
        assert raw.factor_predictors is None

    def test_metadata_defaults(self):
        meta = OODDetectionMetadata()
        assert meta.detectors_used == []
        assert meta.metadata_insights_enabled is False
        assert meta.mode == "advisory"

    def test_is_ood_result_guard(self):
        meta = OODDetectionMetadata()
        data = OODDetectionOutputs(
            raw=OODDetectionRawOutputs(dataset_size=0),
            report=OODDetectionReport(summary="test", findings=[]),
        )
        result = WorkflowResult(name="ood-detection", success=True, data=data, metadata=meta)
        assert is_ood_result(result)

    def test_is_ood_result_false_for_other(self):
        from dataeval_flow.workflows.drift.outputs import DriftMonitoringMetadata

        meta = DriftMonitoringMetadata()
        result = WorkflowResult(name="drift-monitoring", success=True, data=MagicMock(), metadata=meta)
        assert not is_ood_result(result)

    def test_json_serialization(self):
        raw = OODDetectionRawOutputs(
            dataset_size=300,
            reference_size=200,
            test_size=100,
            detectors={"kneighbors": _make_detector_result()},
            ood_indices=[2, 5, 8],
        )
        report = OODDetectionReport(summary="test", findings=[])
        outputs = OODDetectionOutputs(raw=raw, report=report)
        data = outputs.model_dump(mode="json")
        assert data["raw"]["reference_size"] == 200
        assert data["raw"]["ood_indices"] == [2, 5, 8]


# ---------------------------------------------------------------------------
# Workflow execute
# ---------------------------------------------------------------------------


class TestOODDetectionWorkflowExecute:
    """Test the workflow's execute() method with mocked embedding extraction."""

    def _make_workflow(self) -> OODDetectionWorkflow:
        return OODDetectionWorkflow()

    def _make_context(
        self,
        n_datasets: int = 2,
        with_extractor: bool = True,
    ) -> WorkflowContext:
        contexts: dict[str, DatasetContext] = {}
        for i in range(n_datasets):
            name = f"ds{i}"
            ds = _FakeDataset([0] * 50 + [1] * 50)
            dc = DatasetContext(
                name=name,
                dataset=ds,
                extractor=MagicMock() if with_extractor else None,
                batch_size=32 if with_extractor else None,
            )
            contexts[name] = dc
        return WorkflowContext(dataset_contexts=contexts)

    def test_properties(self):
        wf = self._make_workflow()
        assert wf.name == "ood-detection"
        assert wf.params_schema is OODDetectionParameters
        assert wf.output_schema is OODDetectionOutputs
        assert "ood" in wf.description.lower() or "out-of-distribution" in wf.description.lower()

    def test_rejects_non_workflow_context(self):
        wf = self._make_workflow()
        result = wf.execute("not_a_context", _make_params())  # type: ignore[arg-type]
        assert not result.success
        assert "WorkflowContext" in result.errors[0]

    def test_rejects_none_params(self):
        wf = self._make_workflow()
        result = wf.execute(self._make_context(), None)
        assert not result.success
        assert "required" in result.errors[0].lower()

    def test_rejects_wrong_params_type(self):
        wf = self._make_workflow()
        result = wf.execute(self._make_context(), MagicMock(spec=[]))
        assert not result.success
        assert "OODDetectionParameters" in result.errors[0]

    def test_rejects_single_dataset(self):
        wf = self._make_workflow()
        ctx = self._make_context(n_datasets=1)
        result = wf.execute(ctx, _make_params())
        assert not result.success
        assert "at least 2" in result.errors[0]

    @patch("dataeval_flow.workflows.ood.workflow._get_embeddings_for_context")
    def test_successful_execution(self, mock_get_emb: MagicMock):
        ref_emb = _make_embeddings(100, seed=1)
        test_emb = _make_embeddings(50, seed=2)
        mock_get_emb.side_effect = [ref_emb, test_emb]

        wf = self._make_workflow()
        ctx = self._make_context(n_datasets=2)
        params = _make_params(
            detectors=[{"method": "kneighbors", "k": 5}],
            metadata_insights=False,
        )
        result = wf.execute(ctx, params)

        assert result.success
        assert isinstance(result.data, OODDetectionOutputs)
        assert result.data.raw.reference_size == 100
        assert result.data.raw.test_size == 50
        assert "kneighbors" in result.data.raw.detectors
        assert isinstance(result.metadata, OODDetectionMetadata)
        assert result.metadata.detectors_used == ["kneighbors"]

    @patch("dataeval_flow.workflows.ood.workflow._get_embeddings_for_context")
    def test_multiple_test_datasets_concatenated(self, mock_get_emb: MagicMock):
        ref_emb = _make_embeddings(100, seed=1)
        test_emb1 = _make_embeddings(30, seed=2)
        test_emb2 = _make_embeddings(20, seed=3)
        mock_get_emb.side_effect = [ref_emb, test_emb1, test_emb2]

        wf = self._make_workflow()
        ctx = self._make_context(n_datasets=3)
        params = _make_params(metadata_insights=False)
        result = wf.execute(ctx, params)

        assert result.success
        assert result.data.raw.reference_size == 100
        assert result.data.raw.test_size == 50  # 30 + 20

    @patch("dataeval_flow.workflows.ood.workflow._get_embeddings_for_context")
    def test_detects_ood_with_shifted_data(self, mock_get_emb: MagicMock):
        """Test that OOD samples are detected when test data is shifted."""
        ref_emb = _make_embeddings(100, seed=1)
        test_emb = _make_embeddings(50, seed=2)
        # Shift some test samples far from reference
        test_emb[40:] += 20.0
        mock_get_emb.side_effect = [ref_emb, test_emb]

        wf = self._make_workflow()
        ctx = self._make_context()
        params = _make_params(
            detectors=[{"method": "kneighbors", "k": 5, "threshold_perc": 95.0}],
            metadata_insights=False,
        )
        result = wf.execute(ctx, params)

        assert result.success
        assert len(result.data.raw.ood_indices) > 0
        # The shifted samples should be detected
        det = result.data.raw.detectors["kneighbors"]
        assert det["ood_count"] > 0

    @patch("dataeval_flow.workflows.ood.workflow._get_embeddings_for_context")
    def test_exception_in_run_returns_error_result(self, mock_get_emb: MagicMock):
        mock_get_emb.side_effect = RuntimeError("unexpected")

        wf = self._make_workflow()
        ctx = self._make_context()
        result = wf.execute(ctx, _make_params(metadata_insights=False))
        assert not result.success
        assert "unexpected" in result.errors[0]

    @patch("dataeval_flow.workflows.ood.workflow._get_embeddings_for_context")
    def test_detector_error_isolation(self, mock_get_emb: MagicMock):
        """One detector failing should not prevent others from running."""
        ref_emb = _make_embeddings(100, seed=1)
        test_emb = _make_embeddings(50, seed=2)
        mock_get_emb.side_effect = [ref_emb, test_emb]

        wf = self._make_workflow()
        ctx = self._make_context()
        params = _make_params(
            detectors=[
                {"method": "kneighbors", "k": 5},
                {"method": "kneighbors", "k": 10},
            ],
            metadata_insights=False,
        )

        with patch("dataeval_flow.workflows.ood.workflow._build_ood_detector") as mock_build:
            good_detector = MagicMock()
            good_detector.fit.return_value = good_detector
            good_detector.predict.return_value = _make_ood_output(50, ood_indices=set())

            bad_detector = MagicMock()
            bad_detector.fit.side_effect = RuntimeError("OOM")

            mock_build.side_effect = [bad_detector, good_detector]

            result = wf.execute(ctx, params)

        assert result.success
        assert len(result.data.raw.detectors) == 1
        assert len(result.errors) == 1

    def test_empty_outputs(self):
        wf = self._make_workflow()
        outputs = wf._empty_outputs()
        assert outputs.raw.dataset_size == 0
        assert outputs.report.summary == "Workflow failed"

    @patch("dataeval_flow.workflows.ood.workflow._get_embeddings_for_context")
    def test_summary_includes_counts(self, mock_get_emb: MagicMock):
        ref_emb = _make_embeddings(100, seed=1)
        test_emb = _make_embeddings(50, seed=2)
        mock_get_emb.side_effect = [ref_emb, test_emb]

        wf = self._make_workflow()
        ctx = self._make_context()
        params = _make_params(metadata_insights=False)
        result = wf.execute(ctx, params)

        assert result.success
        assert "Reference: 100" in result.data.report.summary
        assert "Test: 50" in result.data.report.summary

    @patch("dataeval_flow.workflows.ood.workflow._compute_metadata_insights")
    @patch("dataeval_flow.workflows.ood.workflow._get_embeddings_for_context")
    def test_ood_union_across_detectors(self, mock_get_emb: MagicMock, mock_insights: MagicMock):
        """Union of OOD indices is computed across all detectors."""
        ref_emb = _make_embeddings(100, seed=1)
        test_emb = _make_embeddings(50, seed=2)
        test_emb[40:] += 20.0  # shift last 10 samples
        mock_get_emb.side_effect = [ref_emb, test_emb]
        mock_insights.return_value = (None, None)

        wf = self._make_workflow()
        ctx = self._make_context()
        params = _make_params(
            detectors=[
                {"method": "kneighbors", "k": 5, "threshold_perc": 95.0},
                {"method": "kneighbors", "k": 10, "threshold_perc": 95.0},
            ],
            metadata_insights=True,
        )
        result = wf.execute(ctx, params)

        assert result.success
        assert len(result.data.raw.ood_indices) > 0
        # metadata insights should have been called since we have OOD indices
        mock_insights.assert_called_once()

    @patch("dataeval_flow.workflows.ood.workflow._get_embeddings_for_context")
    def test_no_ood_skips_metadata_insights(self, mock_get_emb: MagicMock):
        """When no OOD samples detected, metadata insights are skipped."""
        ref_emb = _make_embeddings(100, seed=1)
        test_emb = _make_embeddings(50, seed=1)  # same distribution
        mock_get_emb.side_effect = [ref_emb, test_emb]

        wf = self._make_workflow()
        ctx = self._make_context()
        params = _make_params(
            detectors=[{"method": "kneighbors", "k": 5, "threshold_perc": 99.9}],
            metadata_insights=True,
        )

        with patch("dataeval_flow.workflows.ood.workflow._compute_metadata_insights") as mock_insights:
            result = wf.execute(ctx, params)
            if len(result.data.raw.ood_indices) == 0:
                mock_insights.assert_not_called()

    @patch("dataeval_flow.workflows.ood.workflow._get_embeddings_for_context")
    def test_metadata_insights_disabled(self, mock_get_emb: MagicMock):
        """When metadata_insights=False, insights are not computed."""
        ref_emb = _make_embeddings(100, seed=1)
        test_emb = _make_embeddings(50, seed=2)
        test_emb[40:] += 20.0
        mock_get_emb.side_effect = [ref_emb, test_emb]

        wf = self._make_workflow()
        ctx = self._make_context()
        params = _make_params(
            detectors=[{"method": "kneighbors", "k": 5, "threshold_perc": 95.0}],
            metadata_insights=False,
        )
        result = wf.execute(ctx, params)

        assert result.success
        assert result.data.raw.factor_deviations is None
        assert result.data.raw.factor_predictors is None
        assert result.metadata.metadata_insights_enabled is False


# ---------------------------------------------------------------------------
# Workflow registration
# ---------------------------------------------------------------------------


class TestOODWorkflowRegistration:
    def test_get_workflow_returns_ood(self):
        from dataeval_flow.workflow import get_workflow

        wf = get_workflow("ood-detection")
        assert wf.name == "ood-detection"

    def test_list_workflows_includes_ood(self):
        from dataeval_flow.workflow import list_workflows

        names = [w["name"] for w in list_workflows()]
        assert "ood-detection" in names


# ---------------------------------------------------------------------------
# Params validation
# ---------------------------------------------------------------------------


class TestOODParams:
    def test_kneighbors_defaults(self):
        cfg = OODDetectorKNeighbors()
        assert cfg.method == "kneighbors"
        assert cfg.k == 10
        assert cfg.distance_metric == "cosine"
        assert cfg.threshold_perc == 95.0

    def test_domain_classifier_defaults(self):
        cfg = OODDetectorDomainClassifier()
        assert cfg.method == "domain_classifier"
        assert cfg.n_folds == 5
        assert cfg.n_repeats == 5
        assert cfg.n_std == 2.0
        assert cfg.threshold_perc == 95.0

    def test_health_thresholds_defaults(self):
        t = OODHealthThresholds()
        assert t.ood_pct_warning == 10.0
        assert t.ood_pct_info == 1.0

    def test_parameters_requires_detectors(self):
        with pytest.raises(ValidationError):
            OODDetectionParameters.model_validate({"detectors": []})

    def test_parameters_discriminated_union(self):
        params = OODDetectionParameters.model_validate(
            {
                "detectors": [
                    {"method": "kneighbors", "k": 20},
                    {"method": "domain_classifier", "n_folds": 3},
                ]
            }
        )
        assert len(params.detectors) == 2
        assert isinstance(params.detectors[0], OODDetectorKNeighbors)
        assert isinstance(params.detectors[1], OODDetectorDomainClassifier)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            OODDetectorKNeighbors(method="kneighbors", bogus="field")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------


class TestOODConfigSchema:
    def test_workflow_config_parses(self):
        from dataeval_flow.config.schemas import OODDetectionWorkflowConfig

        cfg = OODDetectionWorkflowConfig.model_validate(
            {
                "name": "ood_knn",
                "type": "ood-detection",
                "detectors": [{"method": "kneighbors", "k": 10}],
            }
        )
        assert cfg.name == "ood_knn"
        assert cfg.type == "ood-detection"

    def test_workflow_config_in_discriminated_union(self):
        from pydantic import TypeAdapter

        from dataeval_flow.config.schemas import WorkflowConfig

        adapter = TypeAdapter(WorkflowConfig)
        cfg = adapter.validate_python(
            {
                "name": "ood_test",
                "type": "ood-detection",
                "detectors": [{"method": "kneighbors"}],
            }
        )
        assert cfg.type == "ood-detection"
