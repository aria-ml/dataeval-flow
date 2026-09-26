"""Tests for drift monitoring report."""

import numpy as np
import pytest

from dataeval_flow._blocks import Fields
from dataeval_flow.workflows import Finding
from dataeval_flow.workflows.drift_monitoring import DriftDetectorMMD, DriftMonitoringHealthThresholds
from dataeval_flow.workflows.drift_monitoring._outputs import (
    ChunkResultDict,
    ClasswiseDriftDict,
    ClasswiseDriftRowDict,
    DetectorResultDict,
    DriftMonitoringRawOutput,
)
from dataeval_flow.workflows.drift_monitoring._report import (
    _build_chunked_finding,
    _build_detector_finding,
    _max_consecutive_drifted,
    _severity_for_chunks,
    _severity_for_detector,
    build_findings,
)
from dataeval_flow.workflows.drift_monitoring._workflow import _serialize_result
from tests.finding_blocks import blocks_of, column, fields, rendered, tables
from tests.test_drift_workflow import _make_chunk_results, _make_detector_result, _make_params

pytestmark = pytest.mark.required

# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


class TestSeverityForDetector:
    def test_drifted_with_warning_enabled(self):
        t = DriftMonitoringHealthThresholds(any_drift_is_warning=True)
        assert _severity_for_detector(True, t) == "warning"

    def test_drifted_with_warning_disabled(self):
        t = DriftMonitoringHealthThresholds(any_drift_is_warning=False)
        assert _severity_for_detector(True, t) == "info"

    def test_not_drifted(self):
        t = DriftMonitoringHealthThresholds()
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
        t = DriftMonitoringHealthThresholds()
        assert _severity_for_chunks([], t) == "ok"

    def test_below_pct_threshold(self):
        # 1/10 = 10%, threshold is 10 -> not exceeded
        chunks = _make_chunk_results(10, drifted_indices={5})
        t = DriftMonitoringHealthThresholds(chunk_drift_pct_warning=11.0, consecutive_chunks_warning=3)
        assert _severity_for_chunks(chunks, t) == "info"

    def test_above_pct_threshold(self):
        # 3/10 = 30% > 10%
        chunks = _make_chunk_results(10, drifted_indices={2, 5, 8})
        t = DriftMonitoringHealthThresholds(chunk_drift_pct_warning=10.0)
        assert _severity_for_chunks(chunks, t) == "warning"

    def test_consecutive_trigger(self):
        # 3 consecutive: indices 2,3,4
        chunks = _make_chunk_results(10, drifted_indices={2, 3, 4})
        t = DriftMonitoringHealthThresholds(chunk_drift_pct_warning=50.0, consecutive_chunks_warning=3)
        assert _severity_for_chunks(chunks, t) == "warning"

    def test_no_drift_at_all(self):
        chunks = _make_chunk_results(5, drifted_indices=set())
        t = DriftMonitoringHealthThresholds()
        assert _severity_for_chunks(chunks, t) == "ok"


# ---------------------------------------------------------------------------
# Findings builders
# ---------------------------------------------------------------------------


class TestBuildDetectorFinding:
    def test_drifted_finding(self):
        result = _make_detector_result(drifted=True, distance=0.34)
        t = DriftMonitoringHealthThresholds()
        finding = _build_detector_finding("KS Univariate", result, t)

        assert finding.title == "KS Univariate"
        assert finding.severity == "warning"
        assert [block.type for block in finding.blocks] == ["fields"]
        assert fields(finding)["Distance"] == pytest.approx(0.34)

    def test_no_drift_finding(self):
        result = _make_detector_result(drifted=False, distance=0.01)
        t = DriftMonitoringHealthThresholds()
        finding = _build_detector_finding("MMD", result, t)

        assert finding.severity == "ok"
        assert fields(finding)["Distance"] == pytest.approx(0.01)

    def test_includes_p_val(self):
        result = _make_detector_result(details={"p_val": 0.001})
        t = DriftMonitoringHealthThresholds()
        finding = _build_detector_finding("Test", result, t)
        assert "p-value" in fields(finding)

    def test_includes_feature_drift_summary(self):
        result = _make_detector_result(details={"p_val": 0.01, "feature_drift": [True, False, True, False, False]})
        t = DriftMonitoringHealthThresholds()
        finding = _build_detector_finding("Test", result, t)
        assert fields(finding)["Features drifted"] == "2 / 5"


class TestBuildChunkedFinding:
    def test_chunked_table_type(self):
        chunks = _make_chunk_results(5, drifted_indices={2, 3})
        result = _make_detector_result(chunks=chunks)
        t = DriftMonitoringHealthThresholds()
        finding = _build_chunked_finding("KS Univariate", result, t)

        (table,) = tables(finding)
        assert finding.title == "KS Univariate"
        assert len(table.rows) == 5
        assert column(table, "status") == ["ok", "ok", "DRIFT", "DRIFT", "ok"]

    def test_no_chunks_falls_back_to_detector_finding(self):
        result = _make_detector_result()
        t = DriftMonitoringHealthThresholds()
        finding = _build_chunked_finding("MMD", result, t)
        assert tables(finding) == []  # fell back
        assert fields(finding)["Metric"] == "ks_distance"

    def test_description_includes_stats(self):
        chunks = _make_chunk_results(10, drifted_indices={3, 4, 5})
        result = _make_detector_result(chunks=chunks)
        t = DriftMonitoringHealthThresholds()
        finding = _build_chunked_finding("Test", result, t)
        assert finding.description is not None
        assert "3/10" in finding.description
        assert "max consecutive: 3" in finding.description


class TestBuildFindings:
    def test_non_chunked_findings(self):
        raw = DriftMonitoringRawOutput(
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
        raw = DriftMonitoringRawOutput(
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
        (table,) = tables(findings[0])
        assert [c.header for c in table.columns] == ["Chunk", "Distance", "", "Status"]

    def test_classwise_finding_appended(self):
        raw = DriftMonitoringRawOutput(
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
        # Classwise data is rendered as a per-class table within the detector finding
        assert len(findings) == 1
        assert findings[0].title == "KS Univariate"
        assert findings[0].brief == "0/1 classes drifted"
        (table,) = tables(findings[0])
        assert column(table, "class_name") == ["0"]


# ---------------------------------------------------------------------------
# _serialize_result edge cases
# ---------------------------------------------------------------------------


class TestSerializeResultEdgeCases:
    def test_feature_drift_as_numpy_array(self):
        """Cover the numpy array branch in _build_detector_finding."""
        result = _make_detector_result(details={"p_val": 0.01, "feature_drift": np.array([True, False, True])})
        t = DriftMonitoringHealthThresholds()
        finding = _build_detector_finding("Test", result, t)
        assert fields(finding)["Features drifted"] == "2 / 3"

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
        """Classwise rows produce a finding with a per-class table."""
        result = _make_detector_result(drifted=True)
        cw_rows = [
            ClasswiseDriftRowDict(class_name="cat", drifted=True, distance=0.5, p_val=0.001),
            ClasswiseDriftRowDict(class_name="dog", drifted=False, distance=0.1, p_val=0.4),
        ]
        t = DriftMonitoringHealthThresholds()
        finding = _build_detector_finding("KS Univariate", result, t, classwise_rows=cw_rows)

        (table,) = tables(finding)
        assert len(table.rows) == 2
        assert column(table, "class_name")[0] == "cat"
        assert column(table, "status")[0] == "DRIFT"
        assert finding.description == "Classes drifted: cat"

    def test_classwise_warning_severity(self):
        """When classwise drift is detected, severity is elevated to warning."""
        result = _make_detector_result(drifted=False)
        cw_rows = [
            ClasswiseDriftRowDict(class_name="a", drifted=True, distance=0.5, p_val=0.01),
        ]
        t = DriftMonitoringHealthThresholds(classwise_any_drift_is_warning=True)
        finding = _build_detector_finding("MMD", result, t, classwise_rows=cw_rows)
        assert finding.severity == "warning"

    def test_classwise_warning_disabled(self):
        """When classwise_any_drift_is_warning is False, severity is not elevated."""
        result = _make_detector_result(drifted=False)
        cw_rows = [
            ClasswiseDriftRowDict(class_name="a", drifted=True, distance=0.5, p_val=0.01),
        ]
        t = DriftMonitoringHealthThresholds(classwise_any_drift_is_warning=False)
        finding = _build_detector_finding("MMD", result, t, classwise_rows=cw_rows)
        # Overall detector didn't drift, classwise warning disabled → stays "ok"
        assert finding.severity == "ok"

    def test_no_classwise_rows(self):
        """Without classwise rows, finding is labelled fields with no table."""
        result = _make_detector_result()
        t = DriftMonitoringHealthThresholds()
        finding = _build_detector_finding("MMD", result, t)
        assert blocks_of(finding, Fields)
        assert tables(finding) == []


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
        finding = _build_detector_finding("Univariate", result, DriftMonitoringHealthThresholds(), classwise_rows=None)
        assert fields(finding)["p-value"] == 0.001
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
        finding = _build_detector_finding("Univariate", result, DriftMonitoringHealthThresholds(), classwise_rows=None)
        assert fields(finding)["Features drifted"] == "3 / 5"


# ---------------------------------------------------------------------------
# Report blocks: what each kind of finding draws
# ---------------------------------------------------------------------------


def _chunks(*rows: tuple[float, float | None, float | None, bool]) -> list[ChunkResultDict]:
    """Chunks of 100 items from ``(value, lower_threshold, upper_threshold, drifted)`` rows."""
    return [
        ChunkResultDict(
            key=f"[{i * 100}:{(i + 1) * 100 - 1}]",
            index=i,
            start_index=i * 100,
            end_index=(i + 1) * 100 - 1,
            value=value,
            upper_threshold=upper,
            lower_threshold=lower,
            drifted=drifted,
        )
        for i, (value, lower, upper, drifted) in enumerate(rows)
    ]


def _chunked(chunks: list[ChunkResultDict]) -> Finding:
    return _build_chunked_finding("MMD", _make_detector_result(chunks=chunks), DriftMonitoringHealthThresholds())


def _classwise(*rows: ClasswiseDriftRowDict) -> Finding:
    result = _make_detector_result(drifted=True)
    return _build_detector_finding("KS Univariate", result, DriftMonitoringHealthThresholds(), list(rows))


class TestDetectorFields:
    def test_labelled_values_in_order(self):
        details = {
            "p_val": 0.0012345,
            "feature_drift": [True, False, True, False, False, True, False, False, True, False],
        }
        result = _make_detector_result(distance=0.341276, details=details)
        finding = _build_detector_finding("KS Univariate", result, DriftMonitoringHealthThresholds())
        (block,) = blocks_of(finding, Fields)
        assert block.items == [
            ("Distance", 0.3413),
            ("Threshold", 0.05),
            ("Metric", "ks_distance"),
            ("p-value", 0.001234),
            ("Features drifted", "4 / 10"),
        ]
        assert finding.brief is None
        assert finding.description == "KS Univariate: distance=0.3413, threshold=0.05, p=0.001234"

    def test_without_a_p_value_or_feature_flags(self):
        result = _make_detector_result(method="kneighbors", distance=0.2, metric_name="knn", details={})
        finding = _build_detector_finding("K-Neighbors", result, DriftMonitoringHealthThresholds())
        (block,) = blocks_of(finding, Fields)
        assert block.items == [("Distance", 0.2), ("Threshold", 0.05), ("Metric", "knn")]
        assert finding.description == "K-Neighbors: distance=0.2, threshold=0.05"

    def test_renders_as_aligned_fields(self):
        result = _make_detector_result(method="mmd", drifted=False, distance=0.0123, metric_name="mmd2")
        finding = _build_detector_finding("MMD", result, DriftMonitoringHealthThresholds())
        assert rendered(finding).splitlines() == [
            "=" * 80,
            "  MMD",
            "=" * 80,
            "  MMD: distance=0.0123, threshold=0.05, p=0.001",
            "",
            "  Distance:  0.0123",
            "  Threshold: 0.05",
            "  Metric:    mmd2",
            "  p-value:   0.001",
        ]


class TestClasswiseTable:
    def test_columns(self):
        finding = _classwise(ClasswiseDriftRowDict(class_name="cat", drifted=True, distance=0.5, p_val=0.01))
        (table,) = tables(finding)
        assert [(c.key, c.header, c.kind, c.format, c.align) for c in table.columns] == [
            ("class_name", "Class", "text", None, None),
            ("distance", "Distance", "text", "{:.4f}", None),
            ("p_val", "PVal", "text", "{:.2f}", None),
            ("abs_distance", "", "bar", None, None),
            ("status", "Status", "text", None, "left"),
        ]

    def test_cells_keep_todays_rounding(self):
        finding = _classwise(
            ClasswiseDriftRowDict(class_name="cat", drifted=True, distance=0.51234, p_val=0.00012345),
            ClasswiseDriftRowDict(class_name="dog", drifted=False, distance=0.1, p_val=None),
        )
        (table,) = tables(finding)
        assert column(table, "class_name") == ["cat", "dog"]
        assert column(table, "distance") == [0.5123, 0.1]
        assert column(table, "p_val") == [0.000123, None]
        assert column(table, "status") == ["DRIFT", "ok"]

    def test_detector_fields_are_not_shown(self):
        finding = _classwise(ClasswiseDriftRowDict(class_name="cat", drifted=True, distance=0.5, p_val=0.01))
        assert fields(finding) == {}
        assert finding.brief == "1/1 classes drifted"

    def test_pval_column_appears_when_some_row_has_one(self):
        finding = _classwise(
            ClasswiseDriftRowDict(class_name="cat", drifted=True, distance=0.30, p_val=0.02),
            ClasswiseDriftRowDict(class_name="dog", drifted=False, distance=0.10, p_val=None),
        )
        text = rendered(finding)
        assert "PVal" in text
        assert "0.02" in text

    def test_no_pval_column_when_no_row_has_one(self):
        finding = _classwise(
            ClasswiseDriftRowDict(class_name="cat", drifted=True, distance=0.30, p_val=None),
            ClasswiseDriftRowDict(class_name="dog", drifted=False, distance=0.10, p_val=None),
        )
        (table,) = tables(finding)
        assert "PVal" not in [c.header for c in table.columns]
        assert "PVal" not in rendered(finding)

    def test_negative_distance_draws_its_size(self):
        """The Distance column shows the signed value; the bar shows its size."""
        finding = _classwise(ClasswiseDriftRowDict(class_name="neg", drifted=True, distance=-0.40, p_val=None))
        (table,) = tables(finding)
        assert column(table, "abs_distance") == [0.4]
        text = rendered(finding)
        assert "-0.4000" in text
        assert "█" * 30 in text

    def test_drifted_and_steady_rows_draw_the_same_bar(self):
        """The Status column says DRIFT; the bar only shows the distance, so equal distances draw alike."""
        finding = _classwise(
            ClasswiseDriftRowDict(class_name="a", drifted=True, distance=0.5, p_val=None),
            ClasswiseDriftRowDict(class_name="b", drifted=False, distance=0.5, p_val=None),
        )
        lines = rendered(finding).splitlines()
        a_line = next(ln for ln in lines if ln.strip().startswith("a "))
        b_line = next(ln for ln in lines if ln.strip().startswith("b "))
        assert "█" * 30 in a_line
        assert "█" * 30 in b_line

    def test_renders_the_table(self):
        finding = _classwise(
            ClasswiseDriftRowDict(class_name="cat", drifted=True, distance=0.5123, p_val=0.0004),
            ClasswiseDriftRowDict(class_name="dog", drifted=False, distance=0.1041, p_val=0.4),
            ClasswiseDriftRowDict(class_name="horse", drifted=True, distance=-0.31, p_val=None),
        )
        assert rendered(finding).splitlines() == [
            "=" * 80,
            "  KS UNIVARIATE" + "2/3 classes drifted".rjust(65),
            "=" * 80,
            "  Classes drifted: cat, horse",
            "",
            "  Class  Distance  PVal                                  Status",
            "  -----  --------  ----  ------------------------------  ------",
            "  cat      0.5123  0.00  ██████████████████████████████  DRIFT",
            "  dog      0.1041  0.40  ██████▏                         ok",
            "  horse   -0.3100        ██████████████████▏             DRIFT",
        ]


class TestChunkTable:
    def test_columns(self):
        (table,) = tables(_chunked(_chunks((0.1, 0.05, 0.25, False), (0.3, 0.05, 0.25, True))))
        assert [(c.key, c.header, c.kind, c.format, c.align) for c in table.columns] == [
            ("chunk", "Chunk", "text", None, None),
            ("distance", "Distance", "text", "{:.4f}", None),
            ("distance", "", "bar", "{:.4f}", None),
            ("status", "Status", "text", None, "left"),
        ]
        assert column(table, "chunk") == ["[0:99]", "[100:199]"]
        assert column(table, "distance") == [0.1, 0.3]
        assert column(table, "status") == ["ok", "DRIFT"]

    def test_markers_come_from_the_first_row_that_has_each(self):
        chunks = _chunks((0.1, None, 0.25, False), (0.1, 0.05, 0.3, False), (0.1, 0.07, 0.35, False))
        (table,) = tables(_chunked(chunks))
        assert table.columns[2].markers == [("Threshold", 0.05), ("Threshold", 0.25)]

    def test_both_thresholds_draw_a_scale_line_under_the_bar(self):
        chunks = _chunks(
            (0.45, 0.05, 0.35, True), (0.15, 0.05, 0.35, False), (0.15, 0.05, 0.35, False), (0.15, 0.05, 0.35, False)
        )
        assert rendered(_chunked(chunks)).splitlines() == [
            "=" * 80,
            "  MMD",
            "=" * 80,
            "  1/4 chunks drifted (25%) | max consecutive: 1",
            "",
            "  Chunk      Distance                                  Status",
            "  ---------  --------  ------------------------------  ------",
            "  [0:99]       0.4500  ██████████████████████████████  DRIFT",
            "  [100:199]    0.1500  ██████████                      ok",
            "  [200:299]    0.1500  ██████████                      ok",
            "  [300:399]    0.1500  ██████████                      ok",
            "  Threshold       (0.0500)|-------------------|(0.3500)",
        ]

    def test_equal_thresholds_share_one_pipe(self):
        lines = rendered(_chunked(_chunks((0.45, 0.2, 0.2, True), (0.15, 0.2, 0.2, False)))).splitlines()
        assert lines[-1] == "  Threshold                 (0.2000)|(0.2000)"

    def test_a_lone_upper_threshold_draws_from_the_bar_start(self):
        lines = rendered(_chunked(_make_chunk_results(3, drifted_indices={1}))).splitlines()
        assert lines[-1] == "  Threshold            -------------------------|(0.2500)"

    def test_a_lone_lower_threshold_draws_from_the_bar_start(self):
        lines = rendered(_chunked(_chunks((0.1, 0.05, None, False), (0.1, 0.05, None, False)))).splitlines()
        assert lines[-1] == "  Threshold            ---------------|(0.0500)"

    def test_no_thresholds_skip_the_scale_line(self):
        text = rendered(_chunked(_chunks((0.15, None, None, False), (0.16, None, None, False))))
        assert "Distance" in text
        assert "Threshold" not in text

    def test_bars_are_full_blocks_with_no_track(self):
        text = rendered(_chunked(_make_chunk_results(3, drifted_indices={0})))
        assert "█" in text
        assert "░" not in text
