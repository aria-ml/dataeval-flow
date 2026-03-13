"""Tests for drift monitoring workflow."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from dataeval_app.workflow import DatasetContext, WorkflowContext, WorkflowResult
from dataeval_app.workflows.drift.outputs import (
    ChunkResultDict,
    ClasswiseDriftDict,
    ClasswiseDriftRowDict,
    DetectorResultDict,
    DriftMonitoringMetadata,
    DriftMonitoringOutputs,
    DriftMonitoringRawOutputs,
    DriftMonitoringReport,
    is_drift_result,
)
from dataeval_app.workflows.drift.params import (
    DriftDetectorDomainClassifier,
    DriftDetectorKNeighbors,
    DriftDetectorMMD,
    DriftDetectorUnivariate,
    DriftHealthThresholds,
    DriftMonitoringParameters,
)
from dataeval_app.workflows.drift.workflow import (
    DriftMonitoringWorkflow,
    _build_chunked_finding,
    _build_classwise_finding,
    _build_detector,
    _build_detector_finding,
    _build_findings,
    _detector_display_name,
    _extract_labels,
    _max_consecutive_drifted,
    _run_classwise_drift,
    _serialize_chunked_result,
    _serialize_result,
    _severity_for_chunks,
    _severity_for_detector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(**overrides: object) -> DriftMonitoringParameters:
    """Build DriftMonitoringParameters with minimal defaults for testing."""
    defaults: dict[str, object] = {
        "detectors": [{"method": "univariate"}],
    }
    defaults.update(overrides)
    return DriftMonitoringParameters.model_validate(defaults)


def _make_embeddings(n: int, d: int = 10, seed: int = 42) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def _make_chunk_results(n: int, drifted_indices: set[int] | None = None) -> list[ChunkResultDict]:
    """Build a list of chunk result dicts."""
    drifted_indices = drifted_indices or set()
    return [
        ChunkResultDict(
            key=f"[{i * 100}:{(i + 1) * 100 - 1}]",
            index=i,
            start_index=i * 100,
            end_index=(i + 1) * 100 - 1,
            value=0.3 if i in drifted_indices else 0.1,
            upper_threshold=0.25,
            lower_threshold=None,
            drifted=i in drifted_indices,
        )
        for i in range(n)
    ]


def _make_detector_result(
    method: str = "univariate",
    drifted: bool = True,
    distance: float = 0.34,
    **kwargs: Any,
) -> DetectorResultDict:
    defaults: dict[str, Any] = {
        "method": method,
        "drifted": drifted,
        "distance": distance,
        "threshold": 0.05,
        "metric_name": "ks_distance",
        "details": {"p_val": 0.001},
    }
    defaults.update(kwargs)
    return DetectorResultDict(**defaults)


class _FakeDataset:
    """Minimal dataset for testing label extraction."""

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
# _build_detector
# ---------------------------------------------------------------------------


class TestBuildDetector:
    def test_univariate(self):
        from dataeval.shift import DriftUnivariate

        det = _build_detector(DriftDetectorUnivariate(test="cvm", p_val=0.01))
        assert isinstance(det, DriftUnivariate)

    def test_mmd(self):
        from dataeval.shift import DriftMMD

        det = _build_detector(DriftDetectorMMD(n_permutations=50))
        assert isinstance(det, DriftMMD)

    def test_domain_classifier(self):
        from dataeval.shift import DriftDomainClassifier

        det = _build_detector(DriftDetectorDomainClassifier())
        assert isinstance(det, DriftDomainClassifier)

    def test_kneighbors(self):
        from dataeval.shift import DriftKNeighbors

        det = _build_detector(DriftDetectorKNeighbors(k=5, distance_metric="cosine"))
        assert isinstance(det, DriftKNeighbors)

    def test_unknown_config_type(self):
        with pytest.raises(ValueError, match="Unknown detector"):
            _build_detector("not_a_config")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _detector_display_name
# ---------------------------------------------------------------------------


class TestDetectorDisplayName:
    def test_univariate_includes_test(self):
        assert _detector_display_name(DriftDetectorUnivariate(test="ks")) == "KS Univariate"
        assert _detector_display_name(DriftDetectorUnivariate(test="cvm")) == "CVM Univariate"

    def test_mmd(self):
        assert _detector_display_name(DriftDetectorMMD()) == "MMD"

    def test_domain_classifier(self):
        assert _detector_display_name(DriftDetectorDomainClassifier()) == "Domain Classifier"

    def test_kneighbors(self):
        assert _detector_display_name(DriftDetectorKNeighbors()) == "K-Neighbors"


# ---------------------------------------------------------------------------
# _serialize_result / _serialize_chunked_result
# ---------------------------------------------------------------------------


class TestSerializeResult:
    def test_non_chunked_basic_fields(self):
        from dataeval.shift import DriftOutput

        output = DriftOutput(
            drifted=True,
            threshold=0.05,
            distance=0.34,
            metric_name="ks_distance",
            details={"p_val": 0.001, "feature_drift": np.array([True, False])},
        )
        cfg = DriftDetectorUnivariate()
        result = _serialize_result(output, cfg)

        assert result["method"] == "univariate"
        assert result["drifted"] is True
        assert result["distance"] == pytest.approx(0.34)
        assert result["threshold"] == pytest.approx(0.05)
        assert result["metric_name"] == "ks_distance"
        # numpy arrays should be converted to lists
        assert result["details"]["feature_drift"] == [True, False]  # type: ignore[reportTypedDictNotRequiredAccess]
        assert result["details"]["p_val"] == 0.001  # type: ignore[reportTypedDictNotRequiredAccess]

    def test_non_chunked_no_chunks_key(self):
        from dataeval.shift import DriftOutput

        output = DriftOutput(
            drifted=False,
            threshold=0.05,
            distance=0.01,
            metric_name="mmd2",
            details={"p_val": 0.42},
        )
        result = _serialize_result(output, DriftDetectorMMD())
        assert "chunks" not in result


class TestSerializeChunkedResult:
    def test_chunked_dataframe(self):
        from dataeval.shift import DriftOutput

        df = pl.DataFrame(
            {
                "key": ["[0:99]", "[100:199]"],
                "index": [0, 1],
                "start_index": [0, 100],
                "end_index": [99, 199],
                "value": [0.12, 0.31],
                "upper_threshold": [0.25, 0.25],
                "lower_threshold": [None, None],
                "drifted": [False, True],
            }
        )
        output = DriftOutput(
            drifted=True,
            threshold=0.25,
            distance=0.215,
            metric_name="ks_distance",
            details=df,
        )
        result = _serialize_chunked_result(output, DriftDetectorUnivariate())

        assert result["method"] == "univariate"
        assert result["drifted"] is True
        chunks = result["chunks"]  # type: ignore[reportTypedDictNotRequiredAccess]
        assert len(chunks) == 2
        assert chunks[0]["key"] == "[0:99]"
        assert chunks[0]["drifted"] is False
        assert chunks[1]["drifted"] is True
        assert chunks[1]["value"] == pytest.approx(0.31)

    def test_chunked_empty_details(self):
        from dataeval.shift import DriftOutput

        output = DriftOutput(
            drifted=False,
            threshold=0.25,
            distance=0.1,
            metric_name="mmd2",
            details={},
        )
        result = _serialize_chunked_result(output, DriftDetectorMMD())  # type: ignore[arg-type]
        assert result["chunks"] == []  # type: ignore[reportTypedDictNotRequiredAccess]


# ---------------------------------------------------------------------------
# _extract_labels
# ---------------------------------------------------------------------------


class TestExtractLabels:
    def test_scalar_targets(self):
        ds = _FakeDataset([0, 1, 2, 0, 1])
        labels = _extract_labels(ds)
        assert labels is not None
        np.testing.assert_array_equal(labels, [0, 1, 2, 0, 1])

    def test_one_hot_targets(self):
        ds = _FakeDataset(
            [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ]
        )
        labels = _extract_labels(ds)
        assert labels is not None
        np.testing.assert_array_equal(labels, [0, 1, 2])

    def test_single_element_array(self):
        ds = _FakeDataset([np.array([3]), np.array([5])])
        labels = _extract_labels(ds)
        assert labels is not None
        np.testing.assert_array_equal(labels, [3, 5])

    def test_2d_target_returns_none(self):
        ds = _FakeDataset([np.array([[1, 2], [3, 4]])])
        labels = _extract_labels(ds)
        assert labels is None

    def test_error_returns_none(self):
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=2)
        ds.__getitem__ = MagicMock(side_effect=RuntimeError("broken"))
        labels = _extract_labels(ds)
        assert labels is None


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
        assert finding.title == "KS Univariate — Chunks"
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


class TestBuildClasswiseFinding:
    def test_pivot_table_structure(self):
        classwise = [
            ClasswiseDriftDict(
                detector="KS Univariate",
                rows=[
                    ClasswiseDriftRowDict(class_name="cat", drifted=True, distance=0.5, p_val=0.001),
                    ClasswiseDriftRowDict(class_name="dog", drifted=False, distance=0.1, p_val=0.4),
                ],
            ),
            ClasswiseDriftDict(
                detector="MMD",
                rows=[
                    ClasswiseDriftRowDict(class_name="cat", drifted=False, distance=0.02, p_val=0.3),
                    ClasswiseDriftRowDict(class_name="dog", drifted=False, distance=0.01, p_val=0.8),
                ],
            ),
        ]
        t = DriftHealthThresholds()
        finding = _build_classwise_finding(classwise, t)

        assert finding.report_type == "pivot_table"
        assert finding.title == "Class-wise Drift Summary"
        assert finding.severity == "warning"  # cat drifted in univariate
        assert isinstance(finding.data, list)
        assert len(finding.data) == 2  # cat, dog
        assert finding.data[0]["Class"] == "cat"
        assert finding.data[0]["KS Univariate"] == "DRIFT"
        assert finding.data[0]["MMD"] == "ok"

    def test_no_drift_severity_ok(self):
        classwise = [
            ClasswiseDriftDict(
                detector="MMD",
                rows=[
                    ClasswiseDriftRowDict(class_name="a", drifted=False, distance=0.01, p_val=0.5),
                ],
            ),
        ]
        t = DriftHealthThresholds()
        finding = _build_classwise_finding(classwise, t)
        assert finding.severity == "ok"


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
        findings = _build_findings(raw, params, names)

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
        findings = _build_findings(raw, params, names)
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
        params = _make_params(classwise=True)
        names = {"univariate": "KS Univariate"}
        findings = _build_findings(raw, params, names)
        assert any(f.report_type == "pivot_table" for f in findings)


# ---------------------------------------------------------------------------
# _run_classwise_drift (with real detectors on synthetic data)
# ---------------------------------------------------------------------------


class TestRunClasswiseDrift:
    def test_detects_per_class_drift(self):
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((200, 10)).astype(np.float32)
        test = rng.standard_normal((100, 10)).astype(np.float32)
        # Shift class 1 significantly
        test[50:, :] += 3.0

        ref_labels = np.array([0] * 100 + [1] * 100, dtype=np.intp)
        test_labels = np.array([0] * 50 + [1] * 50, dtype=np.intp)

        params = _make_params(detectors=[{"method": "univariate", "test": "ks"}])
        names = {"univariate": "KS Univariate"}
        results = _run_classwise_drift(ref, test, ref_labels, test_labels, params, names)

        assert len(results) == 1
        rows = results[0]["rows"]
        class_map = {r["class_name"]: r for r in rows}
        assert class_map["1"]["drifted"] is True
        assert class_map["0"]["drifted"] is False

    def test_skips_class_with_too_few_samples(self):
        ref = _make_embeddings(100)
        test = _make_embeddings(10, seed=99)
        # Class 2 has only 1 sample in test
        ref_labels = np.array([0] * 50 + [1] * 49 + [2], dtype=np.intp)
        test_labels = np.array([0] * 5 + [1] * 4 + [2], dtype=np.intp)

        params = _make_params(detectors=[{"method": "kneighbors", "k": 5}])
        names = {"kneighbors": "K-Neighbors"}
        results = _run_classwise_drift(ref, test, ref_labels, test_labels, params, names)

        # Class 2 should be skipped (1 sample in test)
        row_names = [r["class_name"] for r in results[0]["rows"]]
        assert "2" not in row_names

    def test_multiple_detectors(self):
        ref = _make_embeddings(100)
        test = _make_embeddings(50, seed=99)
        ref_labels = np.array([0] * 50 + [1] * 50, dtype=np.intp)
        test_labels = np.array([0] * 25 + [1] * 25, dtype=np.intp)

        params = _make_params(detectors=[{"method": "univariate"}, {"method": "kneighbors", "k": 5}])
        names = {"univariate": "KS Univariate", "kneighbors": "K-Neighbors"}
        results = _run_classwise_drift(ref, test, ref_labels, test_labels, params, names)
        assert len(results) == 2
        assert results[0]["detector"] == "KS Univariate"
        assert results[1]["detector"] == "K-Neighbors"


# ---------------------------------------------------------------------------
# _get_embeddings_for_context
# ---------------------------------------------------------------------------


class TestGetEmbeddingsForContext:
    def test_raises_without_extractor(self):
        from dataeval_app.workflows.drift.workflow import _get_embeddings_for_context

        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        with pytest.raises(ValueError, match="requires a model/extractor"):
            _get_embeddings_for_context(dc, MagicMock())

    @patch("dataeval_app.workflows.drift.workflow.get_or_compute_embeddings")
    @patch("dataeval_app.workflows.drift.workflow.selection_repr", return_value="sel:all")
    def test_calls_get_or_compute(self, mock_sel: MagicMock, mock_emb: MagicMock):  # noqa: ARG002
        from dataeval_app.workflows.drift.workflow import _get_embeddings_for_context

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
# _build_classwise_finding edge cases
# ---------------------------------------------------------------------------


class TestBuildClasswiseFindingEdgeCases:
    def test_missing_class_in_detector_shows_dash(self):
        """Cover the '—' branch when a class is missing from a detector."""
        classwise = [
            ClasswiseDriftDict(
                detector="D1",
                rows=[ClasswiseDriftRowDict(class_name="a", drifted=False, distance=0.1, p_val=0.5)],
            ),
            ClasswiseDriftDict(
                detector="D2",
                rows=[
                    ClasswiseDriftRowDict(class_name="a", drifted=False, distance=0.1, p_val=0.5),
                    ClasswiseDriftRowDict(class_name="b", drifted=True, distance=0.8, p_val=0.001),
                ],
            ),
        ]
        t = DriftHealthThresholds()
        finding = _build_classwise_finding(classwise, t)
        # Class "b" should have "—" for D1
        assert isinstance(finding.data, list)
        b_row = next(r for r in finding.data if r["Class"] == "b")
        assert b_row["D1"] == "—"

    def test_classwise_warning_disabled(self):
        classwise = [
            ClasswiseDriftDict(
                detector="D1",
                rows=[ClasswiseDriftRowDict(class_name="a", drifted=True, distance=0.5, p_val=0.01)],
            ),
        ]
        t = DriftHealthThresholds(classwise_any_drift_is_warning=False)
        finding = _build_classwise_finding(classwise, t)
        assert finding.severity == "ok"


# ---------------------------------------------------------------------------
# Workflow execute edge cases
# ---------------------------------------------------------------------------


class TestDriftWorkflowExecuteEdgeCases:
    @patch("dataeval_app.workflows.drift.workflow._get_embeddings_for_context")
    @patch("dataeval_app.workflows.drift.workflow._extract_labels")
    def test_classwise_skipped_when_no_labels(self, mock_labels: MagicMock, mock_get_emb: MagicMock):
        """Cover the 'labels not available' warning branch."""
        mock_get_emb.side_effect = [_make_embeddings(100, seed=1), _make_embeddings(50, seed=2)]
        mock_labels.return_value = None  # no labels available

        wf = DriftMonitoringWorkflow()
        ds = _FakeDataset([0] * 100)
        ctx = WorkflowContext(
            dataset_contexts={
                "ref": DatasetContext(name="ref", dataset=ds, extractor=MagicMock(), batch_size=32),  # type: ignore[call-arg]
                "test": DatasetContext(name="test", dataset=ds, extractor=MagicMock(), batch_size=32),  # type: ignore[call-arg]
            }
        )
        params = _make_params(detectors=[{"method": "univariate"}], classwise=True)
        result = wf.execute(ctx, params)
        assert result.success
        assert result.data.raw.classwise is None

    @patch("dataeval_app.workflows.drift.workflow._get_embeddings_for_context")
    def test_exception_in_run_returns_error_result(self, mock_get_emb: MagicMock):
        """Cover the top-level except in execute()."""
        mock_get_emb.side_effect = RuntimeError("unexpected")

        wf = DriftMonitoringWorkflow()
        ds = _FakeDataset([0] * 10)
        ctx = WorkflowContext(
            dataset_contexts={
                "ref": DatasetContext(name="ref", dataset=ds, extractor=MagicMock(), batch_size=32),  # type: ignore[call-arg]
                "test": DatasetContext(name="test", dataset=ds, extractor=MagicMock(), batch_size=32),  # type: ignore[call-arg]
            }
        )
        result = wf.execute(ctx, _make_params())
        assert not result.success
        assert "unexpected" in result.errors[0]


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class TestDriftMonitoringOutputs:
    def test_raw_defaults(self):
        raw = DriftMonitoringRawOutputs(dataset_size=0)
        assert raw.reference_size == 0
        assert raw.test_size == 0
        assert raw.detectors == {}
        assert raw.classwise is None

    def test_metadata_defaults(self):
        meta = DriftMonitoringMetadata()
        assert meta.detectors_used == []
        assert meta.chunking_enabled is False
        assert meta.classwise_enabled is False
        assert meta.mode == "advisory"

    def test_is_drift_result_guard(self):
        meta = DriftMonitoringMetadata()
        data = DriftMonitoringOutputs(
            raw=DriftMonitoringRawOutputs(dataset_size=0),
            report=DriftMonitoringReport(summary="test", findings=[]),
        )
        result = WorkflowResult(name="drift-monitoring", success=True, data=data, metadata=meta)
        assert is_drift_result(result)

    def test_is_drift_result_false_for_other(self):
        from dataeval_app.workflows.cleaning.outputs import DataCleaningMetadata

        meta = DataCleaningMetadata()
        result = WorkflowResult(name="data-cleaning", success=True, data=MagicMock(), metadata=meta)
        assert not is_drift_result(result)

    def test_json_serialization(self):
        raw = DriftMonitoringRawOutputs(
            dataset_size=300,
            reference_size=200,
            test_size=100,
            detectors={
                "univariate": _make_detector_result(),
            },
        )
        report = DriftMonitoringReport(summary="test", findings=[])
        outputs = DriftMonitoringOutputs(raw=raw, report=report)
        data = outputs.model_dump(mode="json")
        assert data["raw"]["reference_size"] == 200
        assert data["raw"]["detectors"]["univariate"]["drifted"] is True


# ---------------------------------------------------------------------------
# DriftMonitoringWorkflow — execute()
# ---------------------------------------------------------------------------


class TestDriftMonitoringWorkflowExecute:
    """Test the workflow's execute() method with mocked embedding extraction."""

    def _make_workflow(self) -> DriftMonitoringWorkflow:
        return DriftMonitoringWorkflow()

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
        assert wf.name == "drift-monitoring"
        assert wf.params_schema is DriftMonitoringParameters
        assert wf.output_schema is DriftMonitoringOutputs
        assert "drift" in wf.description.lower()

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
        assert "DriftMonitoringParameters" in result.errors[0]

    def test_rejects_single_dataset(self):
        wf = self._make_workflow()
        ctx = self._make_context(n_datasets=1)
        result = wf.execute(ctx, _make_params())
        assert not result.success
        assert "at least 2" in result.errors[0]

    @patch("dataeval_app.workflows.drift.workflow._get_embeddings_for_context")
    def test_successful_execution(self, mock_get_emb: MagicMock):
        ref_emb = _make_embeddings(100, seed=1)
        test_emb = _make_embeddings(50, seed=2)
        mock_get_emb.side_effect = [ref_emb, test_emb]

        wf = self._make_workflow()
        ctx = self._make_context(n_datasets=2)
        params = _make_params(detectors=[{"method": "univariate", "test": "ks"}])
        result = wf.execute(ctx, params)

        assert result.success
        assert isinstance(result.data, DriftMonitoringOutputs)
        assert result.data.raw.reference_size == 100
        assert result.data.raw.test_size == 50
        assert "univariate" in result.data.raw.detectors
        assert isinstance(result.metadata, DriftMonitoringMetadata)
        assert result.metadata.detectors_used == ["univariate"]

    @patch("dataeval_app.workflows.drift.workflow._get_embeddings_for_context")
    def test_multiple_test_datasets_concatenated(self, mock_get_emb: MagicMock):
        ref_emb = _make_embeddings(100, seed=1)
        test_emb1 = _make_embeddings(30, seed=2)
        test_emb2 = _make_embeddings(20, seed=3)
        mock_get_emb.side_effect = [ref_emb, test_emb1, test_emb2]

        wf = self._make_workflow()
        ctx = self._make_context(n_datasets=3)
        result = wf.execute(ctx, _make_params())

        assert result.success
        assert result.data.raw.reference_size == 100
        assert result.data.raw.test_size == 50  # 30 + 20

    @patch("dataeval_app.workflows.drift.workflow._get_embeddings_for_context")
    def test_chunked_execution(self, mock_get_emb: MagicMock):
        ref_emb = _make_embeddings(200, seed=1)
        test_emb = _make_embeddings(100, seed=2)
        mock_get_emb.side_effect = [ref_emb, test_emb]

        wf = self._make_workflow()
        ctx = self._make_context()
        params = _make_params(
            detectors=[{"method": "univariate", "chunking": {"chunk_size": 25}}],
        )
        result = wf.execute(ctx, params)

        assert result.success
        assert result.metadata.chunking_enabled is True
        det_result = result.data.raw.detectors["univariate"]
        assert "chunks" in det_result
        assert len(det_result["chunks"]) == 4  # type: ignore[reportTypedDictNotRequiredAccess]  # 100 / 25

    @patch("dataeval_app.workflows.drift.workflow._get_embeddings_for_context")
    @patch("dataeval_app.workflows.drift.workflow._extract_labels")
    def test_classwise_execution(self, mock_labels: MagicMock, mock_get_emb: MagicMock):
        ref_emb = _make_embeddings(100, seed=1)
        test_emb = _make_embeddings(50, seed=2)
        mock_get_emb.side_effect = [ref_emb, test_emb]

        ref_labels = np.array([0] * 50 + [1] * 50, dtype=np.intp)
        test_labels = np.array([0] * 25 + [1] * 25, dtype=np.intp)
        mock_labels.side_effect = [test_labels, ref_labels]  # test first, then ref

        wf = self._make_workflow()
        ctx = self._make_context()
        params = _make_params(
            detectors=[{"method": "kneighbors", "k": 5}],
            classwise=True,
        )
        result = wf.execute(ctx, params)

        assert result.success
        assert result.metadata.classwise_enabled is True
        assert result.data.raw.classwise is not None
        assert len(result.data.raw.classwise) == 1

    @patch("dataeval_app.workflows.drift.workflow._get_embeddings_for_context")
    def test_detector_error_isolation(self, mock_get_emb: MagicMock):
        """One detector failing should not prevent others from running."""
        ref_emb = _make_embeddings(100, seed=1)
        test_emb = _make_embeddings(50, seed=2)
        mock_get_emb.side_effect = [ref_emb, test_emb]

        wf = self._make_workflow()
        ctx = self._make_context()
        # domain_classifier needs enough data; use a very small dataset to potentially cause issues
        # But more reliably, we'll patch one detector to fail
        params = _make_params(
            detectors=[
                {"method": "univariate"},
                {"method": "kneighbors", "k": 5},
            ]
        )

        with patch("dataeval_app.workflows.drift.workflow._build_detector") as mock_build:
            good_detector = MagicMock()
            good_detector.fit.return_value = good_detector
            from dataeval.shift import DriftOutput

            good_detector.predict.return_value = DriftOutput(
                drifted=False,
                threshold=0.05,
                distance=0.01,
                metric_name="test",
                details={"p_val": 0.5},
            )

            bad_detector = MagicMock()
            bad_detector.fit.side_effect = RuntimeError("OOM")

            mock_build.side_effect = [bad_detector, good_detector]

            result = wf.execute(ctx, params)

        assert result.success
        # One detector failed, one succeeded
        assert len(result.data.raw.detectors) == 1
        assert result.errors is not None
        assert len(result.errors) == 1

    @patch("dataeval_app.workflows.drift.workflow._get_embeddings_for_context")
    def test_update_strategy_logged_but_ignored(self, mock_get_emb: MagicMock):
        ref_emb = _make_embeddings(100, seed=1)
        test_emb = _make_embeddings(50, seed=2)
        mock_get_emb.side_effect = [ref_emb, test_emb]

        wf = self._make_workflow()
        ctx = self._make_context()
        params = _make_params(
            detectors=[{"method": "univariate"}],
            update_strategy={"type": "last_seen", "n": 500},
        )
        result = wf.execute(ctx, params)
        assert result.success  # should succeed despite stubbed strategy

    def test_empty_outputs(self):
        wf = self._make_workflow()
        outputs = wf._empty_outputs()
        assert outputs.raw.dataset_size == 0
        assert outputs.report.summary == "Workflow failed"

    @patch("dataeval_app.workflows.drift.workflow._get_embeddings_for_context")
    def test_summary_line(self, mock_get_emb: MagicMock):
        ref_emb = _make_embeddings(100, seed=1)
        test_emb = _make_embeddings(50, seed=2) + 5.0  # large shift
        mock_get_emb.side_effect = [ref_emb, test_emb]

        wf = self._make_workflow()
        ctx = self._make_context()
        params = _make_params(detectors=[{"method": "univariate"}])
        result = wf.execute(ctx, params)

        assert result.success
        assert "Reference: 100" in result.data.report.summary
        assert "Test: 50" in result.data.report.summary


# ---------------------------------------------------------------------------
# Workflow registration
# ---------------------------------------------------------------------------


class TestWorkflowRegistration:
    def test_get_workflow_returns_drift(self):
        from dataeval_app.workflow import get_workflow

        wf = get_workflow("drift-monitoring")
        assert wf.name == "drift-monitoring"

    def test_list_workflows_includes_drift(self):
        from dataeval_app.workflow import list_workflows

        names = [w["name"] for w in list_workflows()]
        assert "drift-monitoring" in names

    def test_unknown_workflow_raises(self):
        from dataeval_app.workflow import get_workflow

        with pytest.raises(ValueError, match="Unknown workflow"):
            get_workflow("nonexistent")
