"""Tests for data analysis workflow — params, findings builders, helpers, execute."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from dataeval.core import LabelStatsResult, StatsResult
from pydantic import ValidationError

from dataeval_flow.config.schemas import SelectionStep
from dataeval_flow.workflow import DatasetContext, WorkflowContext
from dataeval_flow.workflows.analysis.outputs import (
    BiasResult,
    CrossSplitLabelHealth,
    CrossSplitRedundancy,
    CrossSplitResult,
    DataAnalysisOutputs,
    DataAnalysisRawOutputs,
    DataAnalysisReport,
    DistributionShiftResult,
    ImageQualityResult,
    LabelHealthResult,
    RedundancyResult,
    SplitResult,
)
from dataeval_flow.workflows.analysis.params import DataAnalysisHealthThresholds, DataAnalysisParameters
from dataeval_flow.workflows.analysis.workflow import (
    DataAnalysisWorkflow,
    SplitData,
    _assess_bias,
    _assess_cross_label_health,
    _assess_cross_redundancy,
    _assess_distribution_shift,
    _assess_image_quality,
    _assess_label_health,
    _assess_redundancy,
    _build_findings,
    _compute_metadata_summary,
    _compute_split_data,
    _divergence_level,
    _extract_balance_insights,
    _extract_diversity_insights,
    _extract_level_stats,
    _finding_bias,
    _finding_distribution_shift,
    _finding_image_quality,
    _finding_label_balance,
    _finding_label_overlap,
    _finding_label_parity,
    _finding_leakage,
    _finding_redundancy,
    _impute_array,
    _inject_image_stats,
    _labels_from_counts,
    _to_serializable,
)

_WF = "dataeval_flow.workflows.analysis.workflow"
_DEFAULT_THRESHOLDS = DataAnalysisHealthThresholds()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cr(stats: dict[str, Any]) -> StatsResult:
    return StatsResult(stats=stats)  # type: ignore


def _make_ls(values: dict[str, Any]) -> LabelStatsResult:
    return LabelStatsResult(**values)  # type: ignore


def _make_params(**overrides: object) -> DataAnalysisParameters:
    """Build DataAnalysisParameters with required fields for testing."""
    defaults: dict[str, object] = {
        "outlier_method": "modzscore",
        "outlier_flags": ["dimension", "pixel", "visual"],
    }
    defaults.update(overrides)
    return DataAnalysisParameters(**defaults)  # type: ignore[arg-type]


def _make_image_quality(
    outlier_count: int = 0,
    outlier_rate: float = 0.0,
    outlier_summary: dict[str, int] | None = None,
) -> ImageQualityResult:
    return ImageQualityResult(
        outlier_count=outlier_count,
        outlier_rate=outlier_rate,
        outlier_summary=outlier_summary or {},
    )


def _make_redundancy(
    exact_groups: int = 0,
    near_groups: int = 0,
    exact_count: int = 0,
    near_count: int = 0,
) -> RedundancyResult:
    return RedundancyResult(
        exact_duplicate_groups=exact_groups,
        near_duplicate_groups=near_groups,
        exact_duplicates_count=exact_count,
        near_duplicates_count=near_count,
    )


def _make_label_health(
    num_classes: int = 3,
    class_distribution: dict[str, int] | None = None,
    empty_images: list[int] | None = None,
) -> LabelHealthResult:
    return LabelHealthResult(
        num_classes=num_classes,
        class_distribution=class_distribution or {"cat": 10, "dog": 8, "bird": 5},
        empty_images=empty_images or [],
    )


def _make_bias(
    factors: list[str] | None = None,
    balance: dict | None = None,
    diversity: dict | None = None,
) -> BiasResult:
    return BiasResult(
        metadata_factors=factors or ["width", "height"],
        metadata_summary={},
        balance_summary=balance,
        diversity_summary=diversity,
    )


def _make_split_result(**overrides: object) -> SplitResult:
    defaults = {
        "num_samples": 100,
        "image_quality": _make_image_quality(),
        "redundancy": _make_redundancy(),
        "label_health": _make_label_health(),
        "bias": _make_bias(),
    }
    defaults.update(overrides)
    return SplitResult(**defaults)  # type: ignore[arg-type]


def _make_cross_split_result(
    overlap: dict | None = None,
    parity: dict | None = None,
    leakage: dict | None = None,
    divergence: float | None = None,
    divergence_method: str | None = None,
) -> CrossSplitResult:
    return CrossSplitResult(
        redundancy=CrossSplitRedundancy(
            duplicate_leakage=leakage or {"exact_count": 0, "near_count": 0, "exact_groups": [], "near_groups": []},
        ),
        label_health=CrossSplitLabelHealth(
            label_overlap=overlap or {"shared_classes": ["cat", "dog"], "train_only": [], "test_only": []},
            label_parity=parity,
        ),
        distribution_shift=DistributionShiftResult(
            divergence=divergence,
            divergence_method=divergence_method,
        ),
    )


# ===========================================================================
# DataAnalysisParameters
# ===========================================================================


class TestDataAnalysisParameters:
    def test_required_outlier_method(self):
        """outlier_method is required — no default."""
        with pytest.raises(ValidationError, match="outlier_method"):
            DataAnalysisParameters()  # type: ignore[call-arg]

    def test_minimal_valid(self):
        p = _make_params()
        assert p.outlier_method == "modzscore"
        assert p.outlier_threshold is None
        assert p.balance is False
        assert p.diversity_method is None
        assert p.include_image_stats is False
        assert p.divergence_method is None
        assert p.mode == "advisory"

    def test_all_methods_accepted(self):
        for method in ("adaptive", "zscore", "modzscore", "iqr"):
            p = _make_params(outlier_method=method)
            assert p.outlier_method == method

    def test_invalid_outlier_method_rejected(self):
        with pytest.raises(ValidationError, match="outlier_method"):
            _make_params(outlier_method="invalid")

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValidationError, match="outlier_threshold"):
            _make_params(outlier_threshold=-1.0)

    def test_zero_threshold_accepted(self):
        p = _make_params(outlier_threshold=0.0)
        assert p.outlier_threshold == 0.0

    def test_custom_threshold(self):
        p = _make_params(outlier_threshold=3.5)
        assert p.outlier_threshold == 3.5

    def test_balance_and_diversity(self):
        p = _make_params(balance=True, diversity_method="shannon")
        assert p.balance is True
        assert p.diversity_method == "shannon"

    def test_invalid_diversity_method(self):
        with pytest.raises(ValidationError, match="diversity_method"):
            _make_params(diversity_method="entropy")

    def test_divergence_methods(self):
        for method in ("mst", "fnn"):
            p = _make_params(divergence_method=method)
            assert p.divergence_method == method

    def test_invalid_divergence_method(self):
        with pytest.raises(ValidationError, match="divergence_method"):
            _make_params(divergence_method="kl")

    def test_preparatory_mode(self):
        p = _make_params(mode="preparatory")
        assert p.mode == "preparatory"


# ===========================================================================
# _to_serializable
# ===========================================================================


class TestToSerializable:
    def test_plain_types_pass_through(self):
        assert _to_serializable(42) == 42
        assert _to_serializable("hello") == "hello"
        assert _to_serializable(3.14) == 3.14
        assert _to_serializable(None) is None

    def test_numpy_integer(self):
        assert _to_serializable(np.int64(7)) == 7
        assert isinstance(_to_serializable(np.int32(3)), int)

    def test_numpy_float(self):
        result = _to_serializable(np.float64(2.5))
        assert result == 2.5
        assert isinstance(result, float)

    def test_numpy_bool(self):
        assert _to_serializable(np.bool_(True)) is True
        assert _to_serializable(np.bool_(False)) is False

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        assert _to_serializable(arr) == [1, 2, 3]

    def test_nested_dict(self):
        data = {"a": np.int64(1), "b": {"c": np.float64(2.5)}}
        result = _to_serializable(data)
        assert result == {"a": 1, "b": {"c": 2.5}}

    def test_tuple_to_list(self):
        assert _to_serializable((1, 2, 3)) == [1, 2, 3]

    def test_set_sorted_strings(self):
        result = _to_serializable({3, 1, 2})
        assert result == ["1", "2", "3"]

    def test_frozenset_sorted_strings(self):
        result = _to_serializable(frozenset(["b", "a"]))
        assert result == ["a", "b"]

    def test_list_with_numpy(self):
        data = [np.int64(1), np.float64(2.0), "three"]
        result = _to_serializable(data)
        assert result == [1, 2.0, "three"]


# ===========================================================================
# _labels_from_counts
# ===========================================================================


class TestLabelsFromCounts:
    def test_basic(self):
        labels = _labels_from_counts({0: 3, 1: 2})
        assert list(labels) == [0, 0, 0, 1, 1]

    def test_single_class(self):
        labels = _labels_from_counts({5: 4})
        assert list(labels) == [5, 5, 5, 5]

    def test_preserves_class_ids(self):
        labels = _labels_from_counts({10: 1, 20: 2})
        assert 10 in labels
        assert list(labels).count(20) == 2


# ===========================================================================
# Findings builders — Image Quality
# ===========================================================================


class TestFindingImageQuality:
    def _splits(self, **overrides: SplitResult) -> dict[str, SplitResult]:
        return overrides if overrides else {"train": _make_split_result(num_samples=200)}

    def test_no_outliers(self):
        finding = _finding_image_quality(self._splits(), _DEFAULT_THRESHOLDS)
        assert finding.severity == "ok"
        assert finding.description is not None
        assert "0" in finding.description

    def test_with_outliers(self):
        iq = _make_image_quality(outlier_count=10, outlier_rate=0.05, outlier_summary={"brightness": 6, "contrast": 3})
        splits = {"train": _make_split_result(num_samples=200, image_quality=iq)}
        finding = _finding_image_quality(splits, _DEFAULT_THRESHOLDS)
        assert finding.severity == "warning"
        assert finding.description is not None
        assert "10" in finding.description

    def test_cross_split_table(self):
        iq1 = _make_image_quality(outlier_count=10, outlier_rate=0.05, outlier_summary={"brightness": 6})
        iq2 = _make_image_quality(outlier_count=3, outlier_rate=0.1, outlier_summary={"contrast": 2})
        splits = {
            "train": _make_split_result(num_samples=200, image_quality=iq1),
            "test": _make_split_result(num_samples=30, image_quality=iq2),
        }
        finding = _finding_image_quality(splits, _DEFAULT_THRESHOLDS)
        assert finding.report_type == "pivot_table"
        assert isinstance(finding.data, dict)
        assert len(finding.data["table_data"]) == 2


# ===========================================================================
# Findings builders — Redundancy
# ===========================================================================


class TestFindingsRedundancy:
    def test_no_duplicates(self):
        splits = {"train": _make_split_result(num_samples=200)}
        finding = _finding_redundancy(splits, _DEFAULT_THRESHOLDS)
        assert finding.severity == "ok"
        assert finding.description is not None
        assert "No duplicates" in finding.description

    def test_with_duplicates(self):
        rd = _make_redundancy(exact_groups=3, exact_count=9)
        splits = {"train": _make_split_result(num_samples=200, redundancy=rd)}
        finding = _finding_redundancy(splits, _DEFAULT_THRESHOLDS)
        assert finding.severity == "warning"
        assert finding.description is not None
        assert "9 exact" in finding.description

    def test_cross_split_table(self):
        rd1 = _make_redundancy(exact_groups=2, exact_count=6)
        rd2 = _make_redundancy(near_groups=4, near_count=12)
        splits = {
            "train": _make_split_result(num_samples=200, redundancy=rd1),
            "test": _make_split_result(num_samples=100, redundancy=rd2),
        }
        finding = _finding_redundancy(splits, _DEFAULT_THRESHOLDS)
        assert finding.report_type == "pivot_table"
        assert isinstance(finding.data, dict)
        assert len(finding.data["table_data"]) == 2


# ===========================================================================
# Findings builders — Label Balance
# ===========================================================================


class TestFindingsLabelBalance:
    def test_basic(self):
        splits = {"train": _make_split_result(num_samples=100)}
        finding = _finding_label_balance(splits, _DEFAULT_THRESHOLDS)
        assert finding.description is not None
        assert "3 classes" in finding.description

    def test_with_empty_images(self):
        lh = _make_label_health(empty_images=[0, 5, 10])
        splits = {"train": _make_split_result(num_samples=100, label_health=lh)}
        finding = _finding_label_balance(splits, _DEFAULT_THRESHOLDS)
        assert finding.severity == "warning"

    def test_cross_split_table(self):
        lh1 = _make_label_health(class_distribution={"cat": 50, "dog": 30})
        lh2 = _make_label_health(class_distribution={"cat": 10, "dog": 5})
        splits = {
            "train": _make_split_result(num_samples=80, label_health=lh1),
            "test": _make_split_result(num_samples=15, label_health=lh2),
        }
        finding = _finding_label_balance(splits, _DEFAULT_THRESHOLDS)
        assert finding.report_type == "pivot_table"
        assert isinstance(finding.data, dict)
        rows = finding.data["table_data"]
        assert len(rows) == 2
        assert "train" in finding.data["table_headers"]


# ===========================================================================
# Findings builders — Bias
# ===========================================================================


class TestFindingsBias:
    def test_no_analysis(self):
        splits = {"train": _make_split_result()}
        finding = _finding_bias(splits)
        assert finding.severity == "info"
        assert finding.description is not None
        assert "2" in finding.description

    def test_with_balance(self):
        bias = _make_bias(balance={"factors": [{"factor": "width", "score": 0.5}]})
        splits = {"train": _make_split_result(bias=bias)}
        finding = _finding_bias(splits)
        assert finding.severity == "warning"

    def test_with_diversity(self):
        bias = _make_bias(diversity={"factors": [{"factor": "height", "score": 0.1}]})
        splits = {"train": _make_split_result(bias=bias)}
        finding = _finding_bias(splits)
        assert finding.description is not None


# ===========================================================================
# Findings builders — Cross-split
# ===========================================================================


class TestFindingLabelOverlap:
    def _cross(self, **kwargs: Any) -> dict[str, CrossSplitResult]:
        return {"train_vs_test": _make_cross_split_result(**kwargs)}

    def test_all_shared(self):
        cross = self._cross(overlap={"shared_classes": ["cat", "dog"], "train_only": [], "test_only": []})
        finding = _finding_label_overlap(cross)
        assert finding.severity == "ok"
        assert "All 2 classes" in (finding.description or "")

    def test_exclusive_warning(self):
        cross = self._cross(overlap={"shared_classes": ["cat"], "train_only": ["dog"], "test_only": []})
        finding = _finding_label_overlap(cross)
        assert finding.severity == "warning"
        assert "1 exclusive" in (finding.description or "")

    def test_multi_pair_table(self):
        cross = {
            "a_vs_b": _make_cross_split_result(overlap={"shared_classes": ["cat"], "a_only": [], "b_only": []}),
            "a_vs_c": _make_cross_split_result(overlap={"shared_classes": ["cat"], "a_only": ["dog"], "c_only": []}),
        }
        finding = _finding_label_overlap(cross)
        assert finding.report_type == "pivot_table"
        assert isinstance(finding.data, dict)
        assert len(finding.data["table_data"]) == 2


class TestFindingLabelParity:
    def _cross(self, **kwargs: Any) -> dict[str, CrossSplitResult]:
        return {"train_vs_test": _make_cross_split_result(**kwargs)}

    def test_significant(self):
        cross = self._cross(parity={"significant": True, "p_value": 0.001, "chi_squared": 20.0})
        finding = _finding_label_parity(cross)
        assert finding is not None
        assert finding.severity == "warning"
        assert "significant" in (finding.description or "")

    def test_not_significant(self):
        cross = self._cross(parity={"significant": False, "p_value": 0.75, "chi_squared": 0.5})
        finding = _finding_label_parity(cross)
        assert finding is not None
        assert finding.severity == "ok"

    def test_no_parity_returns_none(self):
        cross = self._cross(parity=None)
        assert _finding_label_parity(cross) is None


class TestFindingLeakage:
    def _cross(self, **kwargs: Any) -> dict[str, CrossSplitResult]:
        return {"train_vs_test": _make_cross_split_result(**kwargs)}

    def test_leakage_detected(self):
        cross = self._cross(leakage={"exact_count": 5, "near_count": 2, "exact_groups": [], "near_groups": []})
        finding = _finding_leakage(cross)
        assert finding.severity == "warning"
        assert "5 exact" in (finding.description or "")
        assert "2 near" in (finding.description or "")
        assert "data leakage" in (finding.description or "")

    def test_no_leakage(self):
        cross = self._cross()
        finding = _finding_leakage(cross)
        assert finding.severity == "ok"
        assert "No cross-split" in (finding.description or "")

    def test_exact_only(self):
        cross = self._cross(leakage={"exact_count": 3, "near_count": 0, "exact_groups": [], "near_groups": []})
        finding = _finding_leakage(cross)
        assert "3 exact" in (finding.description or "")

    def test_multi_pair_table(self):
        cross = {
            "a_vs_b": _make_cross_split_result(
                leakage={"exact_count": 2, "near_count": 5, "exact_groups": [], "near_groups": []}
            ),
            "a_vs_c": _make_cross_split_result(
                leakage={"exact_count": 1, "near_count": 0, "exact_groups": [], "near_groups": []}
            ),
        }
        finding = _finding_leakage(cross)
        assert finding.report_type == "pivot_table"
        assert isinstance(finding.data, dict)
        assert len(finding.data["table_data"]) == 2
        assert "3 exact" in (finding.description or "")


# ===========================================================================
# _build_findings
# ===========================================================================


class TestBuildFindings:
    def test_single_split_clean(self):
        splits = {"train": _make_split_result()}
        findings = _build_findings(splits, {}, _DEFAULT_THRESHOLDS)
        # 4 per-split findings: image_quality, redundancy, label_health, bias
        assert len(findings) == 4
        severities = {f.severity for f in findings}
        assert "warning" not in severities

    def test_multi_split_with_cross(self):
        splits = {
            "train": _make_split_result(),
            "test": _make_split_result(num_samples=50),
        }
        cross = {
            "train_vs_test": _make_cross_split_result(
                parity={"significant": False, "p_value": 0.5, "chi_squared": 1.0},
            ),
        }
        findings = _build_findings(splits, cross, _DEFAULT_THRESHOLDS)
        # 4 cross-split comparison + 3 cross-split pair (overlap, parity, leakage)
        assert len(findings) == 7

    def test_findings_with_warnings(self):
        splits = {
            "train": _make_split_result(
                image_quality=_make_image_quality(outlier_count=5, outlier_rate=0.05),
            ),
        }
        findings = _build_findings(splits, {}, _DEFAULT_THRESHOLDS)
        warnings = [f for f in findings if f.severity == "warning"]
        assert len(warnings) == 1
        assert "Image Quality" in warnings[0].title


# ===========================================================================
# DataAnalysisWorkflow — properties and guard clauses
# ===========================================================================


class TestDataAnalysisWorkflow:
    def test_name(self):
        wf = DataAnalysisWorkflow()
        assert wf.name == "data-analysis"

    def test_description(self):
        wf = DataAnalysisWorkflow()
        assert isinstance(wf.description, str)
        assert len(wf.description) > 0

    def test_params_schema(self):
        wf = DataAnalysisWorkflow()
        assert wf.params_schema is DataAnalysisParameters

    def test_output_schema(self):
        wf = DataAnalysisWorkflow()
        assert wf.output_schema is DataAnalysisOutputs

    def test_execute_wrong_context_type(self):
        wf = DataAnalysisWorkflow()
        result = wf.execute("not a context", _make_params())  # type: ignore[arg-type]
        assert result.success is False
        assert "Expected WorkflowContext" in result.errors[0]

    def test_execute_none_params(self):
        wf = DataAnalysisWorkflow()
        ctx = WorkflowContext(dataset_contexts={"default": DatasetContext(name="default", dataset=MagicMock())})
        result = wf.execute(ctx, params=None)
        assert result.success is False
        assert "required" in result.errors[0].lower()

    def test_execute_wrong_params_type(self):
        wf = DataAnalysisWorkflow()
        ctx = WorkflowContext(dataset_contexts={"default": DatasetContext(name="default", dataset=MagicMock())})
        result = wf.execute(ctx, params=MagicMock())
        assert result.success is False
        assert "Expected DataAnalysisParameters" in result.errors[0]

    def test_execute_catches_runtime_errors(self):
        wf = DataAnalysisWorkflow()
        ctx = WorkflowContext(dataset_contexts={"default": DatasetContext(name="default", dataset=MagicMock())})
        params = _make_params()
        with patch.object(wf, "_run", side_effect=RuntimeError("boom")):
            result = wf.execute(ctx, params)
        assert result.success is False
        assert "boom" in result.errors[0]

    def test_failed_result_has_empty_outputs(self):
        wf = DataAnalysisWorkflow()
        result = wf.execute("bad", _make_params())  # type: ignore[arg-type]
        assert isinstance(result.data, DataAnalysisOutputs)
        assert result.data.raw.dataset_size == 0
        assert result.data.report.findings == []

    def test_registered_in_workflow_discovery(self):
        from dataeval_flow.workflow import list_workflows

        names = [w["name"] for w in list_workflows()]
        assert "data-analysis" in names


# ===========================================================================
# Output models — serialization round-trip
# ===========================================================================


class TestOutputModels:
    def test_split_result_roundtrip(self):
        sr = _make_split_result()
        data = sr.model_dump()
        sr2 = SplitResult.model_validate(data)
        assert sr2.num_samples == sr.num_samples
        assert sr2.image_quality.outlier_count == sr.image_quality.outlier_count

    def test_cross_split_result_roundtrip(self):
        csr = _make_cross_split_result(
            parity={"significant": False, "p_value": 0.5, "chi_squared": 1.0},
            divergence=0.3,
            divergence_method="mst",
        )
        data = csr.model_dump()
        csr2 = CrossSplitResult.model_validate(data)
        assert csr2.distribution_shift.divergence == 0.3
        assert csr2.label_health.label_parity is not None
        assert csr2.label_health.label_parity["p_value"] == 0.5

    def test_full_outputs_roundtrip(self):
        outputs = DataAnalysisOutputs(
            raw=DataAnalysisRawOutputs(
                dataset_size=200,
                splits={"train": _make_split_result()},  # type: ignore
                cross_split={},  # type: ignore
            ),
            report=DataAnalysisReport(summary="Test", findings=[]),
        )
        json_str = outputs.model_dump_json()
        restored = DataAnalysisOutputs.model_validate_json(json_str)
        assert restored.raw.dataset_size == 200
        assert restored.raw.splits["train"].num_samples == 100

    def test_distribution_shift_defaults_to_none(self):
        ds = DistributionShiftResult()
        assert ds.divergence is None
        assert ds.divergence_method is None

    def test_cross_split_redundancy_defaults(self):
        csr = CrossSplitRedundancy()
        assert csr.duplicate_leakage["exact_count"] == 0
        assert csr.duplicate_leakage["near_count"] == 0


# ===========================================================================
# DataAnalysisHealthThresholds
# ===========================================================================


class TestHealthThresholds:
    def test_defaults(self):
        t = DataAnalysisHealthThresholds()
        assert t.image_outliers == 3.0
        assert t.exact_duplicates == 0.0
        assert t.near_duplicates == 5.0
        assert t.class_label_imbalance == 5.0
        assert t.distribution_shift == 0.5

    def test_custom_values(self):
        t = DataAnalysisHealthThresholds(image_outliers=10.0, exact_duplicates=1.0)
        assert t.image_outliers == 10.0
        assert t.exact_duplicates == 1.0


# ===========================================================================
# _impute_array
# ===========================================================================


class TestImputeArray:
    def test_returns_array_unchanged_when_all_finite(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _impute_array(arr)
        assert result is not None
        np.testing.assert_array_equal(result, arr)

    def test_replaces_nan_with_median(self):
        arr = np.array([1.0, np.nan, 3.0])
        result = _impute_array(arr)
        assert result is not None
        assert np.isfinite(result).all()
        assert result[1] == 2.0  # median of [1.0, 3.0]

    def test_replaces_inf_with_median(self):
        arr = np.array([1.0, np.inf, 3.0])
        result = _impute_array(arr)
        assert result is not None
        assert np.isfinite(result).all()

    def test_all_non_finite_returns_none(self):
        arr = np.array([np.nan, np.inf, -np.inf])
        result = _impute_array(arr)
        assert result is None


# ===========================================================================
# _extract_level_stats
# ===========================================================================


class TestExtractLevelStats:
    def test_extracts_numeric_1d(self):
        calc_result = _make_cr(stats={"brightness": np.array([1.0, 2.0, 3.0, 4.0])})
        mask = np.array([True, True, False, False])
        result = _extract_level_stats(calc_result, mask, expected_len=2)
        assert "brightness" in result
        np.testing.assert_array_equal(result["brightness"], [1.0, 2.0])

    def test_skips_non_numeric(self):
        calc_result = _make_cr(stats={"hash": np.array(["abc", "def"])})
        mask = np.array([True, True])
        result = _extract_level_stats(calc_result, mask, expected_len=2)
        assert result == {}

    def test_skips_non_1d(self):
        calc_result = _make_cr(stats={"matrix": np.array([[1, 2], [3, 4]])})
        mask = np.array([True, True])
        result = _extract_level_stats(calc_result, mask, expected_len=2)
        assert result == {}

    def test_skips_wrong_length(self):
        calc_result = _make_cr(stats={"brightness": np.array([1.0, 2.0, 3.0])})
        mask = np.array([True, True, True])
        result = _extract_level_stats(calc_result, mask, expected_len=5)
        assert result == {}

    def test_skips_when_impute_returns_none(self):
        calc_result = _make_cr(stats={"brightness": np.array([1.0, 2.0])})
        mask = np.array([True, True])
        with patch(f"{_WF}._impute_array", return_value=None):
            result = _extract_level_stats(calc_result, mask, expected_len=2)
        assert result == {}


# ===========================================================================
# _inject_image_stats
# ===========================================================================


class TestInjectImageStats:
    def test_classification_path(self):
        metadata = MagicMock()
        metadata.has_targets.return_value = False
        calc_result = _make_cr(stats={"brightness": np.array([1.0, 2.0, 3.0])})
        img_mask = np.array([True, True, True])

        _inject_image_stats(metadata, calc_result, img_mask, n_images=3)
        metadata.add_factors.assert_called_once()
        _, kwargs = metadata.add_factors.call_args
        assert kwargs["level"] == "image"

    def test_od_path_broadcasts(self):
        metadata = MagicMock()
        metadata.has_targets.return_value = True
        metadata.item_indices = np.array([0, 0, 1, 1, 2])
        calc_result = _make_cr(stats={"brightness": np.array([1.0, 2.0, 3.0, 10.0, 20.0])})
        img_mask = np.array([True, True, True, False, False])

        _inject_image_stats(metadata, calc_result, img_mask, n_images=3)
        assert metadata.add_factors.call_count >= 1
        # First call should be target-level broadcast
        _, kwargs = metadata.add_factors.call_args_list[0]
        assert kwargs["level"] == "target"

    def test_od_path_with_target_stats(self):
        metadata = MagicMock()
        metadata.has_targets.return_value = True
        metadata.item_indices = np.array([0, 1])
        calc_result = _make_cr(stats={"brightness": np.array([1.0, 2.0, 10.0, 20.0])})
        img_mask = np.array([True, True, False, False])

        _inject_image_stats(metadata, calc_result, img_mask, n_images=2)
        # Should have calls for image-level broadcast AND target-level stats
        assert metadata.add_factors.call_count == 2

    def test_no_factors_skips(self):
        metadata = MagicMock()
        metadata.has_targets.return_value = False
        calc_result = _make_cr(stats={"hash": np.array(["a", "b"])})
        img_mask = np.array([True, True])

        _inject_image_stats(metadata, calc_result, img_mask, n_images=2)
        metadata.add_factors.assert_not_called()

    def test_od_no_img_factors_skips_broadcast(self):
        metadata = MagicMock()
        metadata.has_targets.return_value = True
        metadata.item_indices = np.array([0])
        # Non-numeric stats — no factors extracted
        calc_result = _make_cr(stats={"hash": np.array(["a", "b"])})
        img_mask = np.array([True, False])

        _inject_image_stats(metadata, calc_result, img_mask, n_images=1)
        metadata.add_factors.assert_not_called()

    def test_od_no_target_rows(self):
        """OD path where tgt_mask is all False (no target-level rows)."""
        metadata = MagicMock()
        metadata.has_targets.return_value = True
        metadata.item_indices = np.array([0, 1])
        calc_result = _make_cr(stats={"brightness": np.array([1.0, 2.0])})
        # All rows are image-level — no target rows
        img_mask = np.array([True, True])

        _inject_image_stats(metadata, calc_result, img_mask, n_images=2)
        # Should broadcast image-level to target-level, but skip target stats
        assert metadata.add_factors.call_count == 1
        _, kwargs = metadata.add_factors.call_args
        assert kwargs["level"] == "target"


# ===========================================================================
# _compute_metadata_summary
# ===========================================================================


class TestComputeMetadataSummary:
    def test_continuous_factor(self):
        metadata = MagicMock()
        metadata.image_data = pl.DataFrame({"width": [100.0, 200.0, 300.0]})
        info = MagicMock()
        info.factor_type = "continuous"
        metadata.factor_info = {"width": info}

        result = _compute_metadata_summary(metadata)
        assert "width" in result
        assert result["width"]["type"] == "continuous"
        assert result["width"]["min"] == 100.0
        assert result["width"]["max"] == 300.0

    def test_categorical_factor(self):
        metadata = MagicMock()
        metadata.image_data = pl.DataFrame({"color": ["red", "blue", "red"]})
        info = MagicMock()
        info.factor_type = "categorical"
        metadata.factor_info = {"color": info}

        result = _compute_metadata_summary(metadata)
        assert "color" in result
        assert result["color"]["type"] == "categorical"
        assert result["color"]["unique_values"] == 2
        assert "top_values" in result["color"]

    def test_factor_not_in_columns(self):
        metadata = MagicMock()
        metadata.image_data = pl.DataFrame({"other": [1]})
        info = MagicMock()
        info.factor_type = "continuous"
        metadata.factor_info = {"missing_col": info}

        result = _compute_metadata_summary(metadata)
        assert result["missing_col"] == {"type": "continuous"}

    def test_categorical_empty_column(self):
        """Categorical factor with empty column — empty value_counts."""
        metadata = MagicMock()
        metadata.image_data = pl.DataFrame({"color": pl.Series([], dtype=pl.Utf8)})
        info = MagicMock()
        info.factor_type = "categorical"
        metadata.factor_info = {"color": info}

        result = _compute_metadata_summary(metadata)
        assert "color" in result
        assert result["color"]["unique_values"] == 0
        assert "top_values" not in result["color"]


# ===========================================================================
# _compute_split_data
# ===========================================================================


def _mock_calc_result(n: int = 10) -> StatsResult:
    """Build a mock StatsResult dict."""
    si = MagicMock()
    si.target = None
    return StatsResult(
        source_index=[si] * n,
        object_count=np.array([1] * n).tolist(),
        invalid_box_count=np.array([0] * n).tolist(),
        image_count=n,
        stats={"brightness": np.arange(n, dtype=float)},
    )


def _mock_label_stats() -> dict:
    return {
        "class_count": 2,
        "label_count": 10,
        "index2label": {0: "cat", 1: "dog"},
        "label_counts_per_class": {0: 5, 1: 5},
        "empty_image_indices": [],
    }


class TestComputeSplitData:
    @patch(f"{_WF}.label_stats")
    @patch(f"{_WF}.get_or_compute_stats")
    @patch(f"{_WF}.get_or_compute_metadata")
    def test_basic_flow(self, mock_get_meta, mock_get_stats, mock_ls):
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=10)
        dataset.metadata = {"index2label": {0: "cat"}}

        mock_meta = MagicMock()
        mock_meta.has_targets.return_value = False
        mock_meta.class_labels = np.array([0, 0, 1])
        mock_meta.item_indices = np.array([0, 1, 2])
        mock_get_meta.return_value = mock_meta

        mock_get_stats.return_value = _mock_calc_result(10)
        mock_ls.return_value = _mock_label_stats()

        params = _make_params()
        result = _compute_split_data(dataset, params=params, split_name="train")

        assert isinstance(result, SplitData)
        assert result.dataset_len == 10
        assert result.embeddings is None

    @patch(f"{_WF}.label_stats")
    @patch(f"{_WF}.get_or_compute_stats")
    @patch(f"{_WF}.get_or_compute_metadata")
    def test_with_embeddings(self, mock_get_meta, mock_get_stats, mock_ls):
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=5)
        dataset.metadata = {"index2label": {0: "cat"}}

        mock_meta = MagicMock()
        mock_meta.has_targets.return_value = False
        mock_meta.class_labels = np.array([0])
        mock_meta.item_indices = np.array([0])
        mock_get_meta.return_value = mock_meta
        mock_get_stats.return_value = _mock_calc_result(5)
        mock_ls.return_value = _mock_label_stats()

        extractor = MagicMock()
        extractor.__array__ = MagicMock(return_value=np.zeros((5, 64), dtype=np.float32))

        params = _make_params()
        result = _compute_split_data(dataset, params=params, extractor=extractor, split_name="train")
        assert result.embeddings is not None

    @patch(f"{_WF}._inject_image_stats")
    @patch(f"{_WF}.label_stats")
    @patch(f"{_WF}.get_or_compute_stats")
    @patch(f"{_WF}.get_or_compute_metadata")
    def test_include_image_stats_calls_inject(
        self,
        mock_get_meta,
        mock_get_stats,
        mock_ls,
        mock_inject,
    ):
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=5)
        dataset.metadata = {"index2label": {}}

        mock_meta = MagicMock()
        mock_meta.has_targets.return_value = False
        mock_meta.class_labels = np.array([0])
        mock_meta.item_indices = np.array([0])
        mock_get_meta.return_value = mock_meta
        mock_get_stats.return_value = _mock_calc_result(5)
        mock_ls.return_value = _mock_label_stats()

        params = _make_params(include_image_stats=True)
        _compute_split_data(dataset, params=params, split_name="train")
        mock_inject.assert_called_once()


# ===========================================================================
# _assess_image_quality
# ===========================================================================


class TestAssessImageQuality:
    def _make_split_data(self, n: int = 3) -> MagicMock:
        data = MagicMock(spec=SplitData)
        si = MagicMock()
        si.target = None
        data.calc_result = {
            "source_index": [si] * n,
            "object_count": np.array([1] * n),
            "invalid_box_count": np.array([0] * n),
            "image_count": n,
            "stats": {"brightness": np.arange(n, dtype=float)},
        }
        data.img_mask = np.ones(n, dtype=bool)
        data.dataset_len = n
        return data

    @patch(f"{_WF}.Outliers")
    def test_no_outliers(self, mock_outliers_cls):
        mock_outliers_cls.return_value.from_stats.return_value.data.return_value = pl.DataFrame(
            schema={"item_index": pl.Int64, "metric_name": pl.Utf8, "metric_value": pl.Float64}
        )
        data = self._make_split_data()
        result = _assess_image_quality(data, "modzscore")
        assert result.outlier_count == 0
        assert result.outlier_rate == 0.0

    @patch(f"{_WF}.Outliers")
    def test_with_outliers(self, mock_outliers_cls):
        mock_outliers_cls.return_value.from_stats.return_value.data.return_value = pl.DataFrame(
            {
                "item_index": [0, 1, 1],
                "metric_name": ["brightness", "brightness", "contrast"],
                "metric_value": [5.0, 6.0, 4.0],
            }
        )
        data = self._make_split_data(10)
        result = _assess_image_quality(data, "modzscore")
        assert result.outlier_count == 2
        assert result.outlier_rate == 0.2
        assert "brightness" in result.outlier_summary


# ===========================================================================
# _assess_redundancy
# ===========================================================================


class TestAssessRedundancy:
    @patch(f"{_WF}.Duplicates")
    def test_no_duplicates(self, mock_dup_cls):
        mock_dup_cls.return_value.from_stats.return_value.items.exact = []
        mock_dup_cls.return_value.from_stats.return_value.items.near = []
        data = MagicMock()
        result = _assess_redundancy(data)
        assert result.exact_duplicate_groups == 0
        assert result.near_duplicate_groups == 0

    @patch(f"{_WF}.Duplicates")
    def test_with_duplicates(self, mock_dup_cls):
        near_group = MagicMock()
        near_group.indices = [5, 6]
        mock_dup_cls.return_value.from_stats.return_value.items.exact = [[0, 1], [2, 3, 4]]
        mock_dup_cls.return_value.from_stats.return_value.items.near = [near_group]
        data = MagicMock()
        result = _assess_redundancy(data)
        assert result.exact_duplicate_groups == 2
        assert result.exact_duplicates_count == 5
        assert result.near_duplicate_groups == 1
        assert result.near_duplicates_count == 2


# ===========================================================================
# _assess_label_health
# ===========================================================================


class TestAssessLabelHealth:
    def test_basic(self):
        data = MagicMock(spec=SplitData)
        data.label_stats = {
            "class_count": 3,
            "label_count": 100,
            "index2label": {0: "cat", 1: "dog", 2: "bird"},
            "label_counts_per_class": {0: 50, 1: 30, 2: 20},
            "empty_image_indices": np.array([5, 10]),
        }
        result = _assess_label_health(data)
        assert result.num_classes == 3
        assert result.class_distribution["cat"] == 50
        assert result.empty_images == [5, 10]


# ===========================================================================
# _assess_bias
# ===========================================================================


class TestAssessBias:
    @patch(f"{_WF}._compute_metadata_summary")
    def test_no_balance_no_diversity(self, mock_summary):
        mock_summary.return_value = {"f1": {"type": "continuous"}}
        data = MagicMock()
        data.metadata.factor_names = ["f1"]

        result = _assess_bias(data, balance=False, diversity_method=None)
        assert result.balance_summary is None
        assert result.diversity_summary is None
        assert result.metadata_factors == ["f1"]

    @patch(f"{_WF}._compute_metadata_summary")
    @patch(f"{_WF}.Balance")
    def test_with_balance(self, mock_bal_cls, mock_summary):
        mock_summary.return_value = {}
        bal = MagicMock()
        bal.balance.to_dicts.return_value = [{"v": 0.5}]
        bal.factors.to_dicts.return_value = []
        bal.classwise.to_dicts.return_value = []
        mock_bal_cls.return_value.evaluate.return_value = bal

        data = MagicMock()
        data.metadata.factor_names = ["f1"]

        result = _assess_bias(data, balance=True, diversity_method=None)
        assert result.balance_summary is not None
        assert result.diversity_summary is None

    @patch(f"{_WF}._compute_metadata_summary")
    @patch(f"{_WF}.Balance")
    def test_with_balance_no_factors(self, mock_bal_cls, mock_summary):
        """Balance requested but no factors — should skip gracefully."""
        mock_summary.return_value = {}

        data = MagicMock()
        data.metadata.factor_names = []

        result = _assess_bias(data, balance=True, diversity_method=None)
        assert result.balance_summary is None
        mock_bal_cls.return_value.evaluate.assert_not_called()

    @patch(f"{_WF}._compute_metadata_summary")
    @patch(f"{_WF}.Diversity")
    def test_with_diversity(self, mock_div_cls, mock_summary):
        mock_summary.return_value = {}
        div = MagicMock()
        div.factors.to_dicts.return_value = []
        div.classwise.to_dicts.return_value = []
        mock_div_cls.return_value.evaluate.return_value = div

        data = MagicMock()
        data.metadata.factor_names = ["f1"]

        result = _assess_bias(data, balance=False, diversity_method="shannon")
        assert result.balance_summary is None
        assert result.diversity_summary is not None

    @patch(f"{_WF}._compute_metadata_summary")
    @patch(f"{_WF}.Diversity")
    def test_with_diversity_no_factors(self, mock_div_cls, mock_summary):
        """Diversity requested but no factors — should skip gracefully."""
        mock_summary.return_value = {}

        data = MagicMock()
        data.metadata.factor_names = []

        result = _assess_bias(data, balance=False, diversity_method="shannon")
        assert result.diversity_summary is None
        mock_div_cls.return_value.evaluate.assert_not_called()


# ===========================================================================
# _assess_cross_redundancy
# ===========================================================================


def _cross_dup_df(rows: list[dict[str, object]]) -> pl.DataFrame:
    """Build a mock DuplicatesOutput-style DataFrame for cross-split tests."""
    if not rows:
        return pl.DataFrame(
            schema={
                "group_id": pl.Int64,
                "level": pl.Utf8,
                "dup_type": pl.Utf8,
                "item_indices": pl.List(pl.Int64),
                "methods": pl.List(pl.Utf8),
                "dataset_index": pl.List(pl.Int64),
            }
        )
    return pl.DataFrame(rows)


class TestAssessCrossRedundancy:
    @patch(f"{_WF}.Duplicates")
    def test_no_duplicates(self, mock_dup_cls):
        mock_items = mock_dup_cls.return_value.from_stats.return_value.items
        mock_items.data.return_value = _cross_dup_df([])

        result = _assess_cross_redundancy({}, {}, "train", "test")  # type: ignore
        assert result.duplicate_leakage["exact_count"] == 0
        assert result.duplicate_leakage["near_count"] == 0

    @patch(f"{_WF}.Duplicates")
    def test_with_cross_split_exact(self, mock_dup_cls):
        # One group with items from both datasets (true cross-split leakage)
        mock_items = mock_dup_cls.return_value.from_stats.return_value.items
        mock_items.data.return_value = _cross_dup_df(
            [
                {
                    "group_id": 0,
                    "level": "item",
                    "dup_type": "exact",
                    "item_indices": [1, 2],
                    "methods": ["xxhash"],
                    "dataset_index": [0, 1],
                },
            ]
        )

        result = _assess_cross_redundancy({}, {}, "train", "test")  # type: ignore
        assert result.duplicate_leakage["exact_count"] == 2
        groups = result.duplicate_leakage["exact_groups"]
        assert len(groups) == 1
        assert groups[0]["train"] == [1]
        assert groups[0]["test"] == [2]

    @patch(f"{_WF}.Duplicates")
    def test_no_cross_split_gives_zero(self, mock_dup_cls):
        # Group with items only from one dataset — NOT cross-split leakage
        mock_items = mock_dup_cls.return_value.from_stats.return_value.items
        mock_items.data.return_value = _cross_dup_df(
            [
                {
                    "group_id": 0,
                    "level": "item",
                    "dup_type": "exact",
                    "item_indices": [1, 2],
                    "methods": ["xxhash"],
                    "dataset_index": [0, 0],
                },
            ]
        )

        result = _assess_cross_redundancy({}, {}, "train", "test")  # type: ignore
        assert result.duplicate_leakage["exact_count"] == 0

    @patch(f"{_WF}.Duplicates")
    def test_with_cross_split_near(self, mock_dup_cls):
        mock_items = mock_dup_cls.return_value.from_stats.return_value.items
        mock_items.data.return_value = _cross_dup_df(
            [
                {
                    "group_id": 0,
                    "level": "item",
                    "dup_type": "near",
                    "item_indices": [1, 3],
                    "methods": ["phash"],
                    "dataset_index": [0, 1],
                },
            ]
        )

        result = _assess_cross_redundancy({}, {}, "train", "test")  # type: ignore
        assert result.duplicate_leakage["near_count"] == 2
        assert result.duplicate_leakage["exact_count"] == 0


# ===========================================================================
# _assess_cross_label_health
# ===========================================================================


class TestAssessCrossLabelHealth:
    @patch(f"{_WF}.label_parity")
    def test_shared_classes(self, mock_lp):
        mock_lp.return_value = {"chi_squared": 5.0, "p_value": 0.03}

        ls_a = _make_ls(
            {
                "label_counts_per_class": {0: 50, 1: 50},
                "label_count": 100,
                "class_count": 2,
                "index2label": {0: "cat", 1: "dog"},
            }
        )
        ls_b = _make_ls(
            {
                "label_counts_per_class": {0: 30, 1: 70},
                "label_count": 100,
                "class_count": 2,
                "index2label": {0: "cat", 1: "dog"},
            }
        )

        result = _assess_cross_label_health(ls_a, ls_b, "train", "test")
        assert "shared_classes" in result.label_overlap
        assert len(result.label_overlap["shared_classes"]) == 2
        assert result.label_parity is not None
        assert result.label_parity["significant"] is True

    @patch(f"{_WF}.label_parity")
    def test_exclusive_classes(self, mock_lp):
        mock_lp.return_value = {"chi_squared": 1.0, "p_value": 0.5}

        ls_a = _make_ls(
            {
                "label_counts_per_class": {0: 50, 1: 50},
                "label_count": 100,
                "class_count": 2,
                "index2label": {0: "cat", 1: "dog"},
            }
        )
        ls_b = _make_ls(
            {
                "label_counts_per_class": {0: 30, 2: 70},
                "label_count": 100,
                "class_count": 2,
                "index2label": {0: "cat", 2: "bird"},
            }
        )

        result = _assess_cross_label_health(ls_a, ls_b, "train", "test")
        assert len(result.label_overlap["train_only"]) == 1
        assert len(result.label_overlap["test_only"]) == 1


# ===========================================================================
# _assess_distribution_shift
# ===========================================================================


class TestAssessDistributionShift:
    def test_no_method_returns_empty(self):
        result = _assess_distribution_shift(np.zeros((5, 10)), np.zeros((5, 10)), None)
        assert result.divergence is None

    def test_no_embeddings_a_returns_empty(self):
        result = _assess_distribution_shift(None, np.zeros((5, 10)), "mst")
        assert result.divergence is None

    def test_no_embeddings_b_returns_empty(self):
        result = _assess_distribution_shift(np.zeros((5, 10)), None, "mst")
        assert result.divergence is None

    @patch(f"{_WF}.divergence_mst")
    def test_mst_method(self, mock_div):
        mock_div.return_value = {"divergence": 0.42}
        result = _assess_distribution_shift(np.zeros((5, 10)), np.ones((5, 10)), "mst")
        assert result.divergence == 0.42
        assert result.divergence_method == "mst"

    @patch(f"{_WF}.divergence_fnn")
    def test_fnn_method(self, mock_div):
        mock_div.return_value = {"divergence": 0.15}
        result = _assess_distribution_shift(np.zeros((5, 10)), np.ones((5, 10)), "fnn")
        assert result.divergence == 0.15
        assert result.divergence_method == "fnn"


# ===========================================================================
# DataAnalysisWorkflow._run (full execution)
# ===========================================================================


class TestWorkflowRun:
    @patch(f"{_WF}._assess_bias")
    @patch(f"{_WF}._assess_label_health")
    @patch(f"{_WF}._assess_redundancy")
    @patch(f"{_WF}._assess_image_quality")
    @patch(f"{_WF}._compute_split_data")
    def test_single_split(self, mock_compute, mock_iq, mock_rd, mock_lh, mock_bias):
        mock_compute.return_value = MagicMock(dataset_len=50)
        mock_iq.return_value = _make_image_quality()
        mock_rd.return_value = _make_redundancy()
        mock_lh.return_value = _make_label_health()
        mock_bias.return_value = _make_bias()

        wf = DataAnalysisWorkflow()
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=50)
        dc = DatasetContext(name="train", dataset=ds)
        ctx = WorkflowContext(dataset_contexts={"train": dc})
        params = _make_params()

        result = wf.execute(ctx, params)
        assert result.success is True
        assert isinstance(result.data, DataAnalysisOutputs)
        assert "train" in result.data.raw.splits
        assert result.data.raw.dataset_size == 50
        assert len(result.data.report.findings) > 0

    @patch(f"{_WF}._assess_distribution_shift")
    @patch(f"{_WF}._assess_cross_label_health")
    @patch(f"{_WF}._assess_cross_redundancy")
    @patch(f"{_WF}._assess_bias")
    @patch(f"{_WF}._assess_label_health")
    @patch(f"{_WF}._assess_redundancy")
    @patch(f"{_WF}._assess_image_quality")
    @patch(f"{_WF}._compute_split_data")
    def test_multi_split_with_cross(
        self, mock_compute, mock_iq, mock_rd, mock_lh, mock_bias, mock_xrd, mock_xlh, mock_xds
    ):
        mock_compute.return_value = MagicMock(dataset_len=100)
        mock_iq.return_value = _make_image_quality()
        mock_rd.return_value = _make_redundancy()
        mock_lh.return_value = _make_label_health()
        mock_bias.return_value = _make_bias()
        mock_xrd.return_value = CrossSplitRedundancy()
        mock_xlh.return_value = CrossSplitLabelHealth(
            label_overlap={"shared_classes": ["cat", "dog"], "train_only": [], "test_only": []},
        )
        mock_xds.return_value = DistributionShiftResult()

        wf = DataAnalysisWorkflow()
        ds_train = MagicMock(__len__=MagicMock(return_value=100))
        ds_test = MagicMock(__len__=MagicMock(return_value=100))
        ctx = WorkflowContext(
            dataset_contexts={
                "train": DatasetContext(name="train", dataset=ds_train),
                "test": DatasetContext(name="test", dataset=ds_test),
            }
        )

        result = wf.execute(ctx, _make_params())
        assert result.success is True
        assert "train" in result.data.raw.splits  # type: ignore
        assert "test" in result.data.raw.splits  # type: ignore
        assert "train_vs_test" in result.data.raw.cross_split  # type: ignore

    @patch(f"{_WF}._assess_bias")
    @patch(f"{_WF}._assess_label_health")
    @patch(f"{_WF}._assess_redundancy")
    @patch(f"{_WF}._assess_image_quality")
    @patch(f"{_WF}._compute_split_data")
    def test_preparatory_mode(self, mock_compute, mock_iq, mock_rd, mock_lh, mock_bias):
        mock_compute.return_value = MagicMock(dataset_len=50)
        mock_iq.return_value = _make_image_quality()
        mock_rd.return_value = _make_redundancy()
        mock_lh.return_value = _make_label_health()
        mock_bias.return_value = _make_bias()

        wf = DataAnalysisWorkflow()
        ds = MagicMock(__len__=MagicMock(return_value=50))
        dc = DatasetContext(name="train", dataset=ds)
        ctx = WorkflowContext(dataset_contexts={"train": dc})

        result = wf.execute(ctx, _make_params(mode="preparatory"))
        assert result.success is True
        assert result.metadata.mode == "preparatory"
        titles = [f.title for f in result.data.report.findings]  # type: ignore
        assert "Preparatory Mode" in titles

    @patch("dataeval_flow.embeddings.build_embeddings")
    @patch(f"{_WF}._assess_bias")
    @patch(f"{_WF}._assess_label_health")
    @patch(f"{_WF}._assess_redundancy")
    @patch(f"{_WF}._assess_image_quality")
    @patch(f"{_WF}._compute_split_data")
    def test_with_extractor(self, mock_compute, mock_iq, mock_rd, mock_lh, mock_bias, mock_build_emb):
        mock_build_emb.return_value = MagicMock()
        mock_compute.return_value = MagicMock(dataset_len=50)
        mock_iq.return_value = _make_image_quality()
        mock_rd.return_value = _make_redundancy()
        mock_lh.return_value = _make_label_health()
        mock_bias.return_value = _make_bias()

        wf = DataAnalysisWorkflow()
        ds = MagicMock(__len__=MagicMock(return_value=50))
        extractor = MagicMock()
        dc = DatasetContext(name="train", dataset=ds, extractor=extractor)
        ctx = WorkflowContext(dataset_contexts={"train": dc})

        result = wf.execute(ctx, _make_params())
        assert result.success is True
        mock_build_emb.assert_called_once()

    @patch("dataeval_flow.selection.build_selection")
    @patch(f"{_WF}._assess_bias")
    @patch(f"{_WF}._assess_label_health")
    @patch(f"{_WF}._assess_redundancy")
    @patch(f"{_WF}._assess_image_quality")
    @patch(f"{_WF}._compute_split_data")
    def test_with_selection_steps(self, mock_compute, mock_iq, mock_rd, mock_lh, mock_bias, mock_build_sel):
        """selection_steps on DatasetContext triggers build_selection (line 1186)."""
        mock_selected = MagicMock(__len__=MagicMock(return_value=50))
        mock_build_sel.return_value = mock_selected
        mock_compute.return_value = MagicMock(dataset_len=50)
        mock_iq.return_value = _make_image_quality()
        mock_rd.return_value = _make_redundancy()
        mock_lh.return_value = _make_label_health()
        mock_bias.return_value = _make_bias()

        wf = DataAnalysisWorkflow()
        ds = MagicMock(__len__=MagicMock(return_value=100))
        steps = [SelectionStep(type="Limit", params={"size": 50})]
        dc = DatasetContext(name="train", dataset=ds, selection_steps=steps)
        ctx = WorkflowContext(dataset_contexts={"train": dc})

        result = wf.execute(ctx, _make_params())
        assert result.success is True
        mock_build_sel.assert_called_once()

    @patch(f"{_WF}.active_cache")
    @patch(f"{_WF}._assess_bias")
    @patch(f"{_WF}._assess_label_health")
    @patch(f"{_WF}._assess_redundancy")
    @patch(f"{_WF}._assess_image_quality")
    @patch(f"{_WF}._compute_split_data")
    def test_with_cache(self, mock_compute, mock_iq, mock_rd, mock_lh, mock_bias, mock_active_cache):
        """cache on DatasetContext triggers active_cache context manager (line 1212)."""
        mock_compute.return_value = MagicMock(dataset_len=50)
        mock_iq.return_value = _make_image_quality()
        mock_rd.return_value = _make_redundancy()
        mock_lh.return_value = _make_label_health()
        mock_bias.return_value = _make_bias()

        wf = DataAnalysisWorkflow()
        ds = MagicMock(__len__=MagicMock(return_value=50))
        dc = DatasetContext(name="train", dataset=ds, cache=MagicMock())
        ctx = WorkflowContext(dataset_contexts={"train": dc})

        result = wf.execute(ctx, _make_params())
        assert result.success is True
        mock_active_cache.assert_called_once()


# ===========================================================================
# _labels_from_counts — empty input
# ===========================================================================


class TestLabelsFromCountsEmpty:
    def test_empty_dict(self):
        """Empty label_counts returns empty array (line 243)."""
        labels = _labels_from_counts({})
        assert len(labels) == 0
        assert labels.dtype == int


# ===========================================================================
# _assess_cross_label_health — fallback branch
# ===========================================================================


class TestAssessCrossLabelHealthFallback:
    def test_no_classes_returns_default_parity(self):
        """When num_classes=0, label parity falls back to defaults (line 557)."""
        ls_a = _make_ls(
            {
                "label_counts_per_class": {},
                "label_count": 0,
                "class_count": 0,
                "index2label": {},
            }
        )
        ls_b = _make_ls(
            {
                "label_counts_per_class": {},
                "label_count": 0,
                "class_count": 0,
                "index2label": {},
            }
        )
        result = _assess_cross_label_health(ls_a, ls_b, "train", "test")
        assert result.label_parity is not None
        assert result.label_parity["chi_squared"] == 0.0
        assert result.label_parity["p_value"] == 1.0
        assert result.label_parity["significant"] is False


# ===========================================================================
# _finding_redundancy — info severity
# ===========================================================================


class TestFindingsRedundancyInfo:
    def test_duplicates_below_threshold_info(self):
        """Duplicates exist but below threshold → severity='info' (line 653)."""
        # exact_duplicates_count=1 out of 1000 → 0.1%, below default threshold of 0%
        # But we need groups to exist and pct <= threshold
        # Default exact_duplicates threshold = 0.0, so any exact pct > 0 → warning
        # We need near_duplicates: default threshold = 5.0
        # Use near_count=1/1000 = 0.1% which is < 5.0 threshold
        rd = _make_redundancy(near_groups=1, near_count=1)
        splits = {"train": _make_split_result(num_samples=1000, redundancy=rd)}
        finding = _finding_redundancy(splits, _DEFAULT_THRESHOLDS)
        assert finding.severity == "info"


# ===========================================================================
# _extract_balance_insights / _extract_diversity_insights
# ===========================================================================


class TestExtractBalanceInsights:
    def test_returns_high_mi_factors(self):
        """Factors with MI > 0.1 are included (lines 760-762)."""
        summary = {"balance": [{"factor": "width", "mi_value": 0.5}, {"factor": "height", "mi_value": 0.3}]}
        result = _extract_balance_insights(summary)
        assert len(result) == 2
        assert "width (MI=0.50)" in result[0]

    def test_filters_low_mi(self):
        """Factors with MI <= 0.1 are excluded."""
        summary = {"balance": [{"factor": "width", "mi_value": 0.05}]}
        result = _extract_balance_insights(summary)
        assert result == []

    def test_none_returns_empty(self):
        result = _extract_balance_insights(None)
        assert result == []

    def test_uses_score_fallback_key(self):
        """Falls back to 'score' key when 'mi_value' is absent."""
        summary = {"factors": [{"factor": "width", "score": 0.8}]}
        result = _extract_balance_insights(summary)
        assert len(result) == 1


class TestExtractDiversityInsights:
    def test_returns_low_diversity_factors(self):
        """Factors with diversity < 0.5 are included (lines 774-776)."""
        summary = {"factors": [{"factor": "color", "diversity_value": 0.2}, {"factor": "size", "diversity_value": 0.1}]}
        result = _extract_diversity_insights(summary)
        assert len(result) == 2
        # Sorted ascending by diversity, so lowest first
        assert "size (0.10)" in result[0]
        assert "color (0.20)" in result[1]

    def test_filters_high_diversity(self):
        """Factors with diversity >= 0.5 are excluded."""
        summary = {"factors": [{"factor": "color", "diversity_value": 0.8}]}
        result = _extract_diversity_insights(summary)
        assert result == []

    def test_none_returns_empty(self):
        result = _extract_diversity_insights(None)
        assert result == []


# ===========================================================================
# _finding_label_parity — multi-pair pivot table
# ===========================================================================


class TestFindingLabelParityMultiPair:
    def test_multi_pair_returns_pivot_table(self):
        """Multiple pairs produce a pivot_table (line 924)."""
        cross = {
            "a_vs_b": _make_cross_split_result(parity={"significant": True, "p_value": 0.001, "chi_squared": 20.0}),
            "a_vs_c": _make_cross_split_result(parity={"significant": False, "p_value": 0.5, "chi_squared": 1.0}),
        }
        finding = _finding_label_parity(cross)
        assert finding is not None
        assert finding.report_type == "pivot_table"
        assert finding.severity == "warning"
        assert isinstance(finding.data, dict)
        assert len(finding.data["table_data"]) == 2


# ===========================================================================
# _finding_leakage — near-only
# ===========================================================================


class TestFindingLeakageNearOnly:
    def test_near_only(self):
        """Near-only leakage: only 'near' part in brief (lines 964-966)."""
        cross = {
            "train_vs_test": _make_cross_split_result(
                leakage={"exact_count": 0, "near_count": 7, "exact_groups": [], "near_groups": []}
            )
        }
        finding = _finding_leakage(cross)
        assert "7 near" in (finding.description or "")
        assert "exact" not in (finding.description or "").lower()


# ===========================================================================
# _divergence_level
# ===========================================================================


class TestDivergenceLevel:
    def test_high(self):
        """Divergence > threshold → high, warning (line 994-995)."""
        level, sev = _divergence_level(0.6, 0.5)
        assert level == "high"
        assert sev == "warning"

    def test_moderate(self):
        """Divergence > threshold*0.4 → moderate, info (line 996-997)."""
        level, sev = _divergence_level(0.25, 0.5)
        assert level == "moderate"
        assert sev == "info"

    def test_low(self):
        """Divergence <= threshold*0.4 → low, ok (line 998)."""
        level, sev = _divergence_level(0.1, 0.5)
        assert level == "low"
        assert sev == "ok"


# ===========================================================================
# _finding_distribution_shift
# ===========================================================================


class TestFindingDistributionShift:
    def test_no_divergence_returns_none(self):
        """All divergences None → returns None."""
        cross = {"train_vs_test": _make_cross_split_result(divergence=None)}
        result = _finding_distribution_shift(cross, _DEFAULT_THRESHOLDS)
        assert result is None

    def test_single_pair_key_value(self):
        """Single pair → key_value report (lines 1031-1040)."""
        cross = {"train_vs_test": _make_cross_split_result(divergence=0.6, divergence_method="mst")}
        finding = _finding_distribution_shift(cross, _DEFAULT_THRESHOLDS)
        assert finding is not None
        assert finding.report_type == "key_value"
        assert finding.severity == "warning"
        assert "mst" in str(finding.data)

    def test_multi_pair_pivot_table(self):
        """Multiple pairs → pivot_table report (lines 1042+)."""
        cross = {
            "a_vs_b": _make_cross_split_result(divergence=0.6, divergence_method="mst"),
            "a_vs_c": _make_cross_split_result(divergence=0.1, divergence_method="mst"),
        }
        finding = _finding_distribution_shift(cross, _DEFAULT_THRESHOLDS)
        assert finding is not None
        assert finding.report_type == "pivot_table"
        assert isinstance(finding.data, dict)
        assert len(finding.data["table_data"]) == 2

    def test_high_divergence_brief(self):
        """High divergence in brief (lines 1023-1025)."""
        cross = {
            "a_vs_b": _make_cross_split_result(divergence=0.8, divergence_method="mst"),
            "a_vs_c": _make_cross_split_result(divergence=0.9, divergence_method="mst"),
        }
        finding = _finding_distribution_shift(cross, _DEFAULT_THRESHOLDS)
        assert finding is not None
        assert isinstance(finding.data, dict)
        assert "high divergence" in finding.data["brief"]

    def test_moderate_divergence_brief(self):
        """Moderate divergence in brief (lines 1026-1027)."""
        cross = {
            "a_vs_b": _make_cross_split_result(divergence=0.25, divergence_method="fnn"),
            "a_vs_c": _make_cross_split_result(divergence=0.25, divergence_method="fnn"),
        }
        finding = _finding_distribution_shift(cross, _DEFAULT_THRESHOLDS)
        assert finding is not None
        assert isinstance(finding.data, dict)
        assert "moderate divergence" in finding.data["brief"]

    def test_low_divergence_brief(self):
        """Low divergence in brief (lines 1028-1029)."""
        cross = {
            "a_vs_b": _make_cross_split_result(divergence=0.05, divergence_method="mst"),
            "a_vs_c": _make_cross_split_result(divergence=0.01, divergence_method="mst"),
        }
        finding = _finding_distribution_shift(cross, _DEFAULT_THRESHOLDS)
        assert finding is not None
        assert isinstance(finding.data, dict)
        assert "Low divergence" in finding.data["brief"]


# ===========================================================================
# _build_findings — with distribution shift
# ===========================================================================


class TestBuildFindingsWithShift:
    def test_distribution_shift_appended(self):
        """_build_findings appends distribution shift when present (line 1078)."""
        splits = {
            "train": _make_split_result(),
            "test": _make_split_result(num_samples=50),
        }
        cross = {
            "train_vs_test": _make_cross_split_result(
                parity={"significant": False, "p_value": 0.5, "chi_squared": 1.0},
                divergence=0.6,
                divergence_method="mst",
            ),
        }
        findings = _build_findings(splits, cross, _DEFAULT_THRESHOLDS)
        titles = [f.title for f in findings]
        assert "Distribution Shift" in titles


# ===========================================================================
# is_analysis_result
# ===========================================================================


class TestIsAnalysisResult:
    def test_true_for_analysis_metadata(self):
        from dataeval_flow.workflows.analysis.outputs import DataAnalysisMetadata, is_analysis_result

        result = MagicMock()
        result.metadata = DataAnalysisMetadata()
        assert is_analysis_result(result) is True

    def test_false_for_other_metadata(self):
        from dataeval_flow.workflows.analysis.outputs import is_analysis_result

        result = MagicMock()
        result.metadata = MagicMock()
        assert is_analysis_result(result) is False
