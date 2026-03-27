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
    OODDetectionMetadata,
    OODDetectionOutputs,
    OODDetectionRawOutputs,
    OODDetectionReport,
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
    _build_ood_detector,
    _intersect_numeric_factors,
    _merge_factor_parts,
    _ood_detector_display_name,
    _serialize_ood_result,
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
    def test_calls_get_or_compute(self, mock_sel: MagicMock, mock_emb: MagicMock):
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
        mock_sel: MagicMock,
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


# ---------------------------------------------------------------------------
# _extract_metadata_factors — factor population + class_labels branch
# ---------------------------------------------------------------------------


class TestExtractMetadataFactorsBranches:
    def test_factors_populated_from_df_columns(self):
        """Lines 244-245, 253->252: factors populated from df columns, missing ones skipped."""
        from dataeval_flow.workflows.ood.workflow import _extract_metadata_factors

        dc = DatasetContext(name="ds", dataset=MagicMock())
        ds = MagicMock()

        mock_meta = MagicMock()
        mock_meta.factor_names = ["brightness", "missing_col"]
        mock_meta.dataframe = MagicMock()
        mock_meta.dataframe.columns = ["brightness"]
        mock_meta.dataframe.__len__ = MagicMock(return_value=10)
        mock_meta.dataframe.__getitem__ = MagicMock(
            return_value=MagicMock(to_numpy=MagicMock(return_value=np.arange(10, dtype=float)))
        )
        mock_meta.class_labels = None

        with patch("dataeval_flow.workflows.ood.workflow.get_or_compute_metadata", return_value=mock_meta):
            result = _extract_metadata_factors(dc, ds)

        assert result is not None
        assert "brightness" in result
        assert "missing_col" not in result

    def test_class_labels_added_as_factor(self):
        """Lines 260->265, 262->265: numeric class_labels added as 'class_label' factor."""
        from dataeval_flow.workflows.ood.workflow import _extract_metadata_factors

        dc = DatasetContext(name="ds", dataset=MagicMock())
        ds = MagicMock()

        mock_meta = MagicMock()
        mock_meta.factor_names = ["brightness"]
        mock_meta.dataframe = MagicMock()
        mock_meta.dataframe.columns = ["brightness"]
        mock_meta.dataframe.__len__ = MagicMock(return_value=10)
        mock_meta.dataframe.__getitem__ = MagicMock(
            return_value=MagicMock(to_numpy=MagicMock(return_value=np.arange(10, dtype=float)))
        )
        mock_meta.class_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with patch("dataeval_flow.workflows.ood.workflow.get_or_compute_metadata", return_value=mock_meta):
            result = _extract_metadata_factors(dc, ds)

        assert result is not None
        assert "class_label" in result
        assert len(result["class_label"]) == 10


# ---------------------------------------------------------------------------
# _collect_numeric_factors — ref_stats, ref_meta, empty ref/test branches
# ---------------------------------------------------------------------------


class TestCollectNumericFactorsBranches:
    def test_ref_meta_and_stats_merged(self):
        """Lines 411, 420, 424-427: both ref_meta and ref_stats get merged."""
        from dataeval_flow.workflows.ood.workflow import _collect_numeric_factors

        ref_dc = DatasetContext(name="ref", dataset=MagicMock())
        test_dc = DatasetContext(name="test", dataset=MagicMock())
        test_datasets = [("test", test_dc, MagicMock())]

        ref_meta = {"brightness": np.arange(10, dtype=float)}
        ref_stats = {"contrast": np.arange(10, dtype=float)}
        test_meta = {"brightness": np.arange(10, dtype=float)}
        test_stats = {"contrast": np.arange(10, dtype=float)}

        with (
            patch("dataeval_flow.workflows.ood.workflow._extract_metadata_factors", side_effect=[ref_meta, test_meta]),
            patch("dataeval_flow.workflows.ood.workflow._extract_stats_factors", side_effect=[ref_stats, test_stats]),
        ):
            result = _collect_numeric_factors(ref_dc, MagicMock(), test_datasets)  # type: ignore[arg-type]

        assert result is not None

    def test_empty_ref_factors_returns_none(self):
        """Line 427: returns None when ref has no factors."""
        from dataeval_flow.workflows.ood.workflow import _collect_numeric_factors

        ref_dc = DatasetContext(name="ref", dataset=MagicMock())
        test_dc = DatasetContext(name="test", dataset=MagicMock())
        test_datasets = [("test", test_dc, MagicMock())]

        with (
            patch("dataeval_flow.workflows.ood.workflow._extract_metadata_factors", return_value=None),
            patch("dataeval_flow.workflows.ood.workflow._extract_stats_factors", return_value=None),
        ):
            result = _collect_numeric_factors(ref_dc, MagicMock(), test_datasets)  # type: ignore[arg-type]

        assert result is None

    def test_empty_test_factors_returns_none(self):
        """Line 440: returns None when test has no factors."""
        from dataeval_flow.workflows.ood.workflow import _collect_numeric_factors

        ref_dc = DatasetContext(name="ref", dataset=MagicMock())
        test_dc = DatasetContext(name="test", dataset=MagicMock())
        test_datasets = [("test", test_dc, MagicMock())]

        ref_stats = {"contrast": np.arange(10, dtype=float)}

        with (
            patch("dataeval_flow.workflows.ood.workflow._extract_metadata_factors", return_value=None),
            patch("dataeval_flow.workflows.ood.workflow._extract_stats_factors", side_effect=[ref_stats, None]),
        ):
            result = _collect_numeric_factors(ref_dc, MagicMock(), test_datasets)  # type: ignore[arg-type]

        assert result is None


# ---------------------------------------------------------------------------
# OOD execute — selection steps
# ---------------------------------------------------------------------------


class TestOODExecuteSelections:
    @patch("dataeval_flow.selection.build_selection")
    def test_selection_applied_to_ref_and_test(self, mock_build_sel):
        """Lines 938, 945: build_selection called for ref and test datasets."""
        mock_build_sel.side_effect = lambda ds, _steps: ds

        wf = OODDetectionWorkflow()
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
                    [("test", test_dc, test_emb)],
                ),
            ),
            patch("dataeval_flow.workflows.ood.workflow._run_all_ood_detectors", return_value=({}, {}, [], [])),
            patch("dataeval_flow.workflows.ood.workflow._compute_metadata_insights", return_value=(None, None)),
        ):
            result = wf.execute(ctx, params)

        assert result.success
        assert mock_build_sel.call_count == 2
