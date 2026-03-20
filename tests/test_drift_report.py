"""Tests for drift monitoring report."""

import numpy as np
import pytest

from dataeval_flow.workflows.drift.outputs import (
    ClasswiseDriftDict,
    ClasswiseDriftRowDict,
    DetectorResultDict,
    DriftMonitoringRawOutputs,
)
from dataeval_flow.workflows.drift.params import DriftDetectorMMD, DriftHealthThresholds
from dataeval_flow.workflows.drift.report import (
    _build_chunked_finding,
    _build_detector_finding,
    _max_consecutive_drifted,
    _severity_for_chunks,
    _severity_for_detector,
    build_findings,
)
from dataeval_flow.workflows.drift.workflow import _serialize_result
from tests.test_drift_workflow import _make_chunk_results, _make_detector_result, _make_params

# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


class TestSeverityForDetector:
    def test_drifted_with_warning_enabled(self):
        t = DriftHealthThresholds(any_drift_is_warning=True)
        assert _severity_for_detector(True, t) == "warning"

    def test_drifted_with_warning_disabled(self):
        t = DriftHealthThresholds(any_drift_is_warning=False)
        assert _severity_for_detector(True, t) == "info"

    def test_not_drifted(self):
        t = DriftHealthThresholds()
        assert _severity_for_detector(False, t) == "ok"


class TestMaxConsecutiveDrifted:
    def test_no_drift(self):
        chunks = _make_chunk_results(5, drifted_indices=set())
        assert _max_consecutive_drifted(chunks) == 0

    def test_all_drifted(self):
        chunks = _make_chunk_results(4, drifted_indices={0, 1, 2, 3})
        assert _max_consecutive_drifted(chunks) == 4

    def test_gap_in_middle(self):
        chunks = _make_chunk_results(5, drifted_indices={0, 1, 3, 4})
        assert _max_consecutive_drifted(chunks) == 2

    def test_single_drift(self):
        chunks = _make_chunk_results(5, drifted_indices={2})
        assert _max_consecutive_drifted(chunks) == 1

    def test_empty_list(self):
        assert _max_consecutive_drifted([]) == 0


class TestSeverityForChunks:
    def test_no_chunks(self):
        t = DriftHealthThresholds()
        assert _severity_for_chunks([], t) == "ok"

    def test_below_pct_threshold(self):
        # 1/10 = 10%, threshold is 10 -> not exceeded
        chunks = _make_chunk_results(10, drifted_indices={5})
        t = DriftHealthThresholds(chunk_drift_pct_warning=11.0, consecutive_chunks_warning=3)
        assert _severity_for_chunks(chunks, t) == "info"

    def test_above_pct_threshold(self):
        # 3/10 = 30% > 10%
        chunks = _make_chunk_results(10, drifted_indices={2, 5, 8})
        t = DriftHealthThresholds(chunk_drift_pct_warning=10.0)
        assert _severity_for_chunks(chunks, t) == "warning"

    def test_consecutive_trigger(self):
        # 3 consecutive: indices 2,3,4
        chunks = _make_chunk_results(10, drifted_indices={2, 3, 4})
        t = DriftHealthThresholds(chunk_drift_pct_warning=50.0, consecutive_chunks_warning=3)
        assert _severity_for_chunks(chunks, t) == "warning"

    def test_no_drift_at_all(self):
        chunks = _make_chunk_results(5, drifted_indices=set())
        t = DriftHealthThresholds()
        assert _severity_for_chunks(chunks, t) == "ok"


# ---------------------------------------------------------------------------
# Findings builders
# ---------------------------------------------------------------------------


class TestBuildDetectorFinding:
    def test_drifted_finding(self):
        result = _make_detector_result(drifted=True, distance=0.34)
        t = DriftHealthThresholds()
        finding = _build_detector_finding("KS Univariate", result, t)

        assert finding.title == "KS Univariate"
        assert finding.severity == "warning"
        assert finding.report_type == "key_value"
        assert isinstance(finding.data, dict)
        assert finding.data["distance"] == pytest.approx(0.34)

    def test_no_drift_finding(self):
        result = _make_detector_result(drifted=False, distance=0.01)
        t = DriftHealthThresholds()
        finding = _build_detector_finding("MMD", result, t)

        assert finding.severity == "ok"
        assert isinstance(finding.data, dict)

    def test_includes_p_val(self):
        result = _make_detector_result(details={"p_val": 0.001})
        t = DriftHealthThresholds()
        finding = _build_detector_finding("Test", result, t)
        assert isinstance(finding.data, dict)
        assert "p_val" in finding.data

    def test_includes_feature_drift_summary(self):
        result = _make_detector_result(details={"p_val": 0.01, "feature_drift": [True, False, True, False, False]})
        t = DriftHealthThresholds()
        finding = _build_detector_finding("Test", result, t)
        assert isinstance(finding.data, dict)
        assert finding.data["features_drifted"] == "2 / 5"


class TestBuildChunkedFinding:
    def test_chunked_table_type(self):
        chunks = _make_chunk_results(5, drifted_indices={2, 3})
        result = _make_detector_result(chunks=chunks)
        t = DriftHealthThresholds()
        finding = _build_chunked_finding("KS Univariate", result, t)

        assert finding.report_type == "chunk_table"
        assert finding.title == "KS Univariate"
        assert isinstance(finding.data, dict)
        rows = finding.data["table_rows"]
        assert len(rows) == 5
        assert rows[2]["Status"] == "DRIFT"
        assert rows[0]["Status"] == "ok"
        assert finding.data["drift_flags"] == [False, False, True, True, False]

    def test_no_chunks_falls_back_to_detector_finding(self):
        result = _make_detector_result()
        t = DriftHealthThresholds()
        finding = _build_chunked_finding("MMD", result, t)
        assert finding.report_type == "key_value"  # fell back

    def test_description_includes_stats(self):
        chunks = _make_chunk_results(10, drifted_indices={3, 4, 5})
        result = _make_detector_result(chunks=chunks)
        t = DriftHealthThresholds()
        finding = _build_chunked_finding("Test", result, t)
        assert finding.description is not None
        assert "3/10" in finding.description
        assert "max consecutive: 3" in finding.description


class TestBuildFindings:
    def test_non_chunked_findings(self):
        raw = DriftMonitoringRawOutputs(
            dataset_size=300,
            reference_size=200,
            test_size=100,
            detectors={
                "univariate": _make_detector_result(drifted=True),
                "mmd": _make_detector_result(method="mmd", drifted=False, distance=0.01),
            },
        )
        params = _make_params(detectors=[{"method": "univariate"}, {"method": "mmd"}])
        names = {"univariate": "KS Univariate", "mmd": "MMD"}
        findings = build_findings(raw, params, names)

        assert len(findings) == 2
        assert any(f.title == "KS Univariate" for f in findings)
        assert any(f.title == "MMD" for f in findings)

    def test_chunked_findings(self):
        chunks = _make_chunk_results(3, drifted_indices={1})
        raw = DriftMonitoringRawOutputs(
            dataset_size=400,
            reference_size=100,
            test_size=300,
            detectors={"univariate": _make_detector_result(chunks=chunks)},
        )
        params = _make_params(
            detectors=[{"method": "univariate", "chunking": {"chunk_size": 100}}],
        )
        names = {"univariate": "KS Univariate"}
        findings = build_findings(raw, params, names)
        assert findings[0].report_type == "chunk_table"

    def test_classwise_finding_appended(self):
        raw = DriftMonitoringRawOutputs(
            dataset_size=300,
            reference_size=200,
            test_size=100,
            detectors={"univariate": _make_detector_result()},
            classwise=[
                ClasswiseDriftDict(
                    detector="KS Univariate",
                    rows=[ClasswiseDriftRowDict(class_name="0", drifted=False, distance=0.1, p_val=0.5)],
                )
            ],
        )
        params = _make_params(detectors=[{"method": "univariate", "classwise": True}])
        names = {"univariate": "KS Univariate"}
        findings = build_findings(raw, params, names)
        # Classwise data is rendered as a classwise_table within the detector finding
        assert len(findings) == 1
        assert findings[0].title == "KS Univariate"
        assert findings[0].report_type == "classwise_table"
        assert "table_rows" in findings[0].data


# ---------------------------------------------------------------------------
# _serialize_result edge cases
# ---------------------------------------------------------------------------


class TestSerializeResultEdgeCases:
    def test_feature_drift_as_numpy_array(self):
        """Cover the numpy array branch in _build_detector_finding."""
        result = _make_detector_result(details={"p_val": 0.01, "feature_drift": np.array([True, False, True])})
        t = DriftHealthThresholds()
        finding = _build_detector_finding("Test", result, t)
        assert isinstance(finding.data, dict)
        assert finding.data["features_drifted"] == "2 / 3"

    def test_details_not_dict(self):
        """When details is not a dict (e.g. polars DataFrame), serialize handles it."""
        from dataeval.shift import DriftOutput

        output = DriftOutput(
            drifted=False,
            threshold=0.05,
            distance=0.01,
            metric_name="test",
            details="not_a_dict",  # type: ignore[arg-type]
        )
        result = _serialize_result(output, DriftDetectorMMD())
        assert result["details"] == {}  # type: ignore[reportTypedDictNotRequiredAccess]


# ---------------------------------------------------------------------------
# _build_detector_finding with classwise data
# ---------------------------------------------------------------------------


class TestBuildDetectorFindingClasswise:
    def test_classwise_table_structure(self):
        """Classwise rows produce a classwise_table finding with table_rows."""
        result = _make_detector_result(drifted=True)
        cw_rows = [
            ClasswiseDriftRowDict(class_name="cat", drifted=True, distance=0.5, p_val=0.001),
            ClasswiseDriftRowDict(class_name="dog", drifted=False, distance=0.1, p_val=0.4),
        ]
        t = DriftHealthThresholds()
        finding = _build_detector_finding("KS Univariate", result, t, classwise_rows=cw_rows)

        assert finding.report_type == "classwise_table"
        assert isinstance(finding.data, dict)
        table_rows = finding.data.get("table_rows")
        assert isinstance(table_rows, list)
        assert len(table_rows) == 2
        assert table_rows[0]["Class"] == "cat"
        assert table_rows[0]["Status"] == "DRIFT"
        assert finding.description == "Classes drifted: cat"

    def test_classwise_warning_severity(self):
        """When classwise drift is detected, severity is elevated to warning."""
        result = _make_detector_result(drifted=False)
        cw_rows = [
            ClasswiseDriftRowDict(class_name="a", drifted=True, distance=0.5, p_val=0.01),
        ]
        t = DriftHealthThresholds(classwise_any_drift_is_warning=True)
        finding = _build_detector_finding("MMD", result, t, classwise_rows=cw_rows)
        assert finding.severity == "warning"

    def test_classwise_warning_disabled(self):
        """When classwise_any_drift_is_warning is False, severity is not elevated."""
        result = _make_detector_result(drifted=False)
        cw_rows = [
            ClasswiseDriftRowDict(class_name="a", drifted=True, distance=0.5, p_val=0.01),
        ]
        t = DriftHealthThresholds(classwise_any_drift_is_warning=False)
        finding = _build_detector_finding("MMD", result, t, classwise_rows=cw_rows)
        # Overall detector didn't drift, classwise warning disabled → stays "ok"
        assert finding.severity == "ok"

    def test_no_classwise_rows(self):
        """Without classwise rows, finding is key_value with no table_rows."""
        result = _make_detector_result()
        t = DriftHealthThresholds()
        finding = _build_detector_finding("MMD", result, t)
        assert finding.report_type == "key_value"
        assert "table_rows" not in finding.data


# ---------------------------------------------------------------------------
# _build_detector_finding — p_val and feature_drift branches
# ---------------------------------------------------------------------------


class TestBuildDetectorFindingBranches:
    def test_p_val_in_details(self):
        """Lines 294->298: p_val from details dict appears in finding."""
        result: DetectorResultDict = {  # type: ignore[typeddict-item]
            "method": "univariate",
            "drifted": True,
            "distance": 0.5,
            "threshold": 0.3,
            "metric_name": "KS",
            "details": {"p_val": 0.001},
        }
        finding = _build_detector_finding("Univariate", result, DriftHealthThresholds(), classwise_rows=None)
        assert isinstance(finding.data, dict)
        assert finding.data["p_val"] == 0.001
        assert finding.description is not None
        assert "p=0.001" in finding.description

    def test_feature_drift_as_list(self):
        """Lines 309->313: feature_drift as a list populates features_drifted."""
        result: DetectorResultDict = {  # type: ignore[typeddict-item]
            "method": "univariate",
            "drifted": True,
            "distance": 0.5,
            "threshold": 0.3,
            "metric_name": "KS",
            "details": {"feature_drift": [True, False, True, False, True]},
        }
        finding = _build_detector_finding("Univariate", result, DriftHealthThresholds(), classwise_rows=None)
        assert isinstance(finding.data, dict)
        assert finding.data["features_drifted"] == "3 / 5"
