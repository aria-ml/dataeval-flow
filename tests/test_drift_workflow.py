"""Tests for drift monitoring workflow."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from dataeval_flow.workflow import DatasetContext, WorkflowContext, WorkflowResult
from dataeval_flow.workflows.drift.outputs import (
    ChunkResultDict,
    DetectorResultDict,
    DriftMonitoringMetadata,
    DriftMonitoringOutputs,
    DriftMonitoringRawOutputs,
    DriftMonitoringReport,
    is_drift_result,
)
from dataeval_flow.workflows.drift.params import (
    DriftDetectorDomainClassifier,
    DriftDetectorKNeighbors,
    DriftDetectorMMD,
    DriftDetectorUnivariate,
    DriftMonitoringParameters,
)
from dataeval_flow.workflows.drift.workflow import (
    DriftMonitoringWorkflow,
    _build_detector,
    _detector_display_name,
    _extract_labels,
    _run_classwise_drift,
    _serialize_chunked_result,
    _serialize_result,
    _unique_method_keys,
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

        params = _make_params(detectors=[{"method": "univariate", "test": "ks", "classwise": True}])
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

        params = _make_params(detectors=[{"method": "kneighbors", "k": 5, "classwise": True}])
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

        params = _make_params(
            detectors=[
                {"method": "univariate", "classwise": True},
                {"method": "kneighbors", "k": 5, "classwise": True},
            ]
        )
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
        from dataeval_flow.workflows.drift.workflow import _get_embeddings_for_context

        dc = DatasetContext(name="test", dataset=MagicMock(), extractor=None)
        with pytest.raises(ValueError, match="requires a model/extractor"):
            _get_embeddings_for_context(dc, MagicMock())

    @patch("dataeval_flow.workflows.drift.workflow.get_or_compute_embeddings")
    @patch("dataeval_flow.workflows.drift.workflow.selection_repr", return_value="sel:all")
    def test_calls_get_or_compute(self, mock_sel: MagicMock, mock_emb: MagicMock):
        from dataeval_flow.workflows.drift.workflow import _get_embeddings_for_context

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
# Workflow execute edge cases
# ---------------------------------------------------------------------------


class TestDriftWorkflowExecuteEdgeCases:
    @patch("dataeval_flow.workflows.drift.workflow._get_embeddings_for_context")
    @patch("dataeval_flow.workflows.drift.workflow._extract_labels")
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
        params = _make_params(detectors=[{"method": "univariate", "classwise": True}])
        result = wf.execute(ctx, params)
        assert result.success
        assert result.data.raw.classwise is None

    @patch("dataeval_flow.workflows.drift.workflow._get_embeddings_for_context")
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
        from dataeval_flow.workflows.cleaning.outputs import DataCleaningMetadata

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

    @patch("dataeval_flow.workflows.drift.workflow._get_embeddings_for_context")
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

    @patch("dataeval_flow.workflows.drift.workflow._get_embeddings_for_context")
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

    @patch("dataeval_flow.workflows.drift.workflow._get_embeddings_for_context")
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

    @patch("dataeval_flow.workflows.drift.workflow._get_embeddings_for_context")
    @patch("dataeval_flow.workflows.drift.workflow._extract_labels")
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
            detectors=[{"method": "kneighbors", "k": 5, "classwise": True}],
        )
        result = wf.execute(ctx, params)

        assert result.success
        assert result.metadata.classwise_enabled is True
        assert result.data.raw.classwise is not None
        assert len(result.data.raw.classwise) == 1

    @patch("dataeval_flow.workflows.drift.workflow._get_embeddings_for_context")
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

        with patch("dataeval_flow.workflows.drift.workflow._build_detector") as mock_build:
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

    @patch("dataeval_flow.workflows.drift.workflow._get_embeddings_for_context")
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

    @patch("dataeval_flow.workflows.drift.workflow._get_embeddings_for_context")
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
        from dataeval_flow.workflow import get_workflow

        wf = get_workflow("drift-monitoring")
        assert wf.name == "drift-monitoring"

    def test_list_workflows_includes_drift(self):
        from dataeval_flow.workflow import list_workflows

        names = [w["name"] for w in list_workflows()]
        assert "drift-monitoring" in names

    def test_unknown_workflow_raises(self):
        from dataeval_flow.workflow import get_workflow

        with pytest.raises(ValueError, match="Unknown workflow"):
            get_workflow("nonexistent")


# ---------------------------------------------------------------------------
# _unique_method_keys — duplicate method suffix assignment
# ---------------------------------------------------------------------------


class TestUniqueMethodKeys:
    def test_no_duplicates(self):
        dets = [DriftDetectorMMD(), DriftDetectorUnivariate()]
        keys = _unique_method_keys(dets)
        assert keys == ["mmd", "univariate"]

    def test_duplicates_get_suffix(self):
        dets = [DriftDetectorMMD(), DriftDetectorMMD(), DriftDetectorUnivariate()]
        keys = _unique_method_keys(dets)
        assert keys == ["mmd_1", "mmd_2", "univariate"]

    def test_all_same(self):
        dets = [DriftDetectorMMD(), DriftDetectorMMD(), DriftDetectorMMD()]
        keys = _unique_method_keys(dets)
        assert keys == ["mmd_1", "mmd_2", "mmd_3"]


# ---------------------------------------------------------------------------
# _run_classwise_drift — branches
# ---------------------------------------------------------------------------


class TestRunClasswiseDriftBranches:
    def test_skips_non_classwise_detectors(self):
        """Line 442: detectors without classwise=True are skipped."""
        params = _make_params(
            detectors=[
                {"method": "mmd", "classwise": False},
                {"method": "mmd", "classwise": True},
            ]
        )
        ref_emb = np.random.default_rng(42).standard_normal((20, 4)).astype(np.float32)
        test_emb = np.random.default_rng(99).standard_normal((20, 4)).astype(np.float32)
        ref_labels = np.array([0] * 10 + [1] * 10, dtype=np.intp)
        test_labels = np.array([0] * 10 + [1] * 10, dtype=np.intp)

        with patch("dataeval_flow.workflows.drift.workflow._build_detector") as mock_build:
            mock_det = MagicMock()
            mock_det.predict.return_value = MagicMock(drifted=False, distance=0.1, details={"p_val": 0.5})
            mock_build.return_value = mock_det

            results = _run_classwise_drift(
                ref_emb, test_emb, ref_labels, test_labels, params, {"mmd_1": "MMD(1)", "mmd_2": "MMD(2)"}
            )

        # Only one detector had classwise=True
        assert len(results) == 1

    def test_exception_continues_other_classes(self):
        """Lines 480-482: one class failure doesn't stop others."""
        params = _make_params(detectors=[{"method": "mmd", "classwise": True}])
        ref_emb = np.random.default_rng(42).standard_normal((20, 4)).astype(np.float32)
        test_emb = np.random.default_rng(99).standard_normal((20, 4)).astype(np.float32)
        ref_labels = np.array([0] * 10 + [1] * 10, dtype=np.intp)
        test_labels = np.array([0] * 10 + [1] * 10, dtype=np.intp)

        call_count = 0

        def mock_build_side_effect(det_config):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            det = MagicMock()
            if call_count == 1:
                det.fit.side_effect = RuntimeError("boom")
            else:
                det.predict.return_value = MagicMock(drifted=False, distance=0.1, details={"p_val": 0.5})
            return det

        with patch("dataeval_flow.workflows.drift.workflow._build_detector", side_effect=mock_build_side_effect):
            results = _run_classwise_drift(ref_emb, test_emb, ref_labels, test_labels, params, {"mmd": "MMD"})

        # One class failed, one succeeded — should still have 1 row
        assert len(results) == 1
        assert len(results[0]["rows"]) == 1

    def test_p_val_from_output_details(self):
        """Lines 469->472: classwise detection picks up p_val from output.details."""
        params = _make_params(detectors=[{"method": "mmd", "classwise": True}])
        ref_emb = np.random.default_rng(42).standard_normal((20, 4)).astype(np.float32)
        test_emb = np.random.default_rng(99).standard_normal((20, 4)).astype(np.float32)
        ref_labels = np.array([0] * 10 + [1] * 10, dtype=np.intp)
        test_labels = np.array([0] * 10 + [1] * 10, dtype=np.intp)

        with patch("dataeval_flow.workflows.drift.workflow._build_detector") as mock_build:
            mock_det = MagicMock()
            mock_det.predict.return_value = MagicMock(drifted=True, distance=0.8, details={"p_val": 0.002})
            mock_build.return_value = mock_det

            results = _run_classwise_drift(ref_emb, test_emb, ref_labels, test_labels, params, {"mmd": "MMD"})

        assert len(results[0]["rows"]) == 2
        assert results[0]["rows"][0]["p_val"] == 0.002


# ---------------------------------------------------------------------------
# _get_embeddings_for_context — cache branch
# ---------------------------------------------------------------------------


class TestGetEmbeddingsWithCache:
    def test_cache_context_entered(self):
        """Line 220: active_cache context manager is entered when cache is present."""
        from dataeval_flow.workflows.drift.workflow import _get_embeddings_for_context

        ds = MagicMock()
        dc = DatasetContext(name="ref", dataset=ds, extractor=MagicMock(), cache=MagicMock())

        with (
            patch("dataeval_flow.workflows.drift.workflow.selection_repr", return_value="sel_all"),
            patch("dataeval_flow.workflows.drift.workflow.active_cache") as mock_active,
            patch("dataeval_flow.workflows.drift.workflow.get_or_compute_embeddings", return_value=np.zeros((5, 3))),
        ):
            mock_active.return_value.__enter__ = MagicMock()
            mock_active.return_value.__exit__ = MagicMock(return_value=False)
            _get_embeddings_for_context(dc, ds)

        mock_active.assert_called_once()


# ---------------------------------------------------------------------------
# _execute — selection_steps branches
# ---------------------------------------------------------------------------


class TestDriftExecuteSelections:
    @patch("dataeval_flow.selection.build_selection")
    def test_selection_applied_to_ref_and_test(self, mock_build_sel):
        """Lines 737, 744: build_selection called for ref and test datasets."""
        mock_build_sel.side_effect = lambda ds, _steps: ds  # passthrough

        wf = DriftMonitoringWorkflow()
        ref_ds = MagicMock()
        ref_ds.__len__ = MagicMock(return_value=50)
        test_ds = MagicMock()
        test_ds.__len__ = MagicMock(return_value=50)

        ref_dc = DatasetContext(
            name="ref",
            dataset=ref_ds,
            extractor=MagicMock(),
            selection_steps=[MagicMock()],
        )
        test_dc = DatasetContext(
            name="test",
            dataset=test_ds,
            extractor=MagicMock(),
            selection_steps=[MagicMock()],
        )
        ctx = WorkflowContext(dataset_contexts={"ref": ref_dc, "test": test_dc})
        params = _make_params()

        ref_emb = np.random.default_rng(0).standard_normal((50, 4)).astype(np.float32)
        test_emb = np.random.default_rng(1).standard_normal((50, 4)).astype(np.float32)

        with (
            patch.object(
                wf,
                "_extract_all_embeddings",
                return_value=(
                    ref_emb,
                    test_emb,
                    None,
                    [],
                ),
            ),
            patch("dataeval_flow.workflows.drift.workflow._run_all_detectors", return_value=({}, {}, [])),
            patch("dataeval_flow.workflows.drift.workflow._handle_classwise", return_value=[]),
        ):
            result = wf.execute(ctx, params)

        assert result.success
        assert mock_build_sel.call_count == 2
