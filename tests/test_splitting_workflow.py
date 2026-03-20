"""Tests for the dataset splitting workflow."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from pydantic import BaseModel

from dataeval_flow.workflow import DatasetContext, WorkflowContext
from dataeval_flow.workflow.base import WorkflowParametersBase
from dataeval_flow.workflows.splitting.outputs import (
    DataSplittingMetadata,
    DataSplittingOutputs,
    DataSplittingRawOutputs,
    DataSplittingReport,
    SplitInfo,
    is_splitting_result,
)
from dataeval_flow.workflows.splitting.params import DataSplittingParameters
from dataeval_flow.workflows.splitting.workflow import (
    DataSplittingWorkflow,
    _build_findings,
    _format_factor_table,
    _run_coverage,
    _serialize_balance,
    _serialize_coverage,
    _serialize_diversity,
    _serialize_label_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(**overrides: Any) -> DataSplittingParameters:
    defaults = {
        "test_frac": 0.2,
        "val_frac": 0.1,
        "num_folds": 1,
        "stratify": True,
    }
    defaults.update(overrides)
    return DataSplittingParameters(**defaults)


def _make_dataset(n: int = 100) -> MagicMock:
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=n)
    return ds


def _make_metadata(n: int = 100, num_classes: int = 5) -> MagicMock:
    meta = MagicMock()
    meta.class_labels = np.array([i % num_classes for i in range(n)], dtype=np.intp)
    meta.index2label = {i: f"class_{i}" for i in range(num_classes)}
    return meta


def _make_split_result(n: int = 100, test_frac: float = 0.2, val_frac: float = 0.1) -> MagicMock:
    """Create a mock DatasetSplits result."""
    indices = np.arange(n)
    test_size = int(n * test_frac)
    val_size = int(n * val_frac)
    train_size = n - test_size - val_size

    splits = MagicMock()
    splits.test = indices[:test_size]

    fold = MagicMock()
    fold.train = indices[test_size : test_size + train_size]
    fold.val = indices[test_size + train_size :]
    splits.folds = [fold]

    return splits


# ---------------------------------------------------------------------------
# TestDataSplittingParameters
# ---------------------------------------------------------------------------


class TestDataSplittingParameters:
    def test_defaults(self) -> None:
        p = DataSplittingParameters()
        assert p.test_frac == 0.2
        assert p.val_frac == 0.1
        assert p.num_folds == 1
        assert p.stratify is True
        assert p.split_on is None
        assert p.rebalance_method is None
        assert p.coverage_percent == 0.01
        assert p.num_observations == 50

    def test_inherits_workflow_params_base(self) -> None:
        assert issubclass(DataSplittingParameters, WorkflowParametersBase)

    def test_custom_values(self) -> None:
        p = _make_params(
            test_frac=0.3,
            val_frac=0.15,
            num_folds=3,
            stratify=False,
            split_on=["lighting"],
            rebalance_method="global",
        )
        assert p.test_frac == 0.3
        assert p.num_folds == 3
        assert p.split_on == ["lighting"]
        assert p.rebalance_method == "global"


# ---------------------------------------------------------------------------
# TestSerializers
# ---------------------------------------------------------------------------


class TestSerializeLabelStats:
    def test_basic(self) -> None:
        stats = {
            "label_counts_per_class": np.array([10, 20, 30]),
            "class_count": 3,
            "label_count": 60,
            "image_count": 60,
        }
        result = _serialize_label_stats(stats)
        assert result["label_counts_per_class"] == [10, 20, 30]
        assert result["class_count"] == 3

    def test_none(self) -> None:
        assert _serialize_label_stats(None) == {}


class TestSerializeBalance:
    def test_with_dataframes(self) -> None:
        output = MagicMock()
        output.balance = pl.DataFrame({"factor": ["a"], "mi": [0.5]})
        output.factors = pl.DataFrame({"f1": ["a"], "f2": ["b"], "mi_value": [0.1]})
        output.classwise = None
        result = _serialize_balance(output)
        assert "balance" in result
        assert "factors" in result
        assert "classwise" not in result


class TestSerializeDiversity:
    def test_with_dataframes(self) -> None:
        output = MagicMock()
        output.factors = pl.DataFrame({"factor_name": ["a"], "diversity_value": [0.8]})
        output.classwise = None
        result = _serialize_diversity(output)
        assert "factors" in result
        assert "classwise" not in result


class TestSerializeCoverage:
    def test_basic(self) -> None:
        cov = {
            "uncovered_indices": np.array([1, 5, 9]),
            "critical_value_radii": np.array([0.1, 0.2, 0.3]),
            "coverage_radius": 0.15,
        }
        result = _serialize_coverage(cov)
        assert result["uncovered_indices"] == [1, 5, 9]
        assert result["coverage_radius"] == 0.15


# ---------------------------------------------------------------------------
# TestBuildFindings
# ---------------------------------------------------------------------------


class TestBuildFindings:
    def test_label_distribution_info(self) -> None:
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            label_stats_full={"label_counts_per_class": {"a": 50, "b": 45}},
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = _build_findings(raw)
        label_finding = next(f for f in findings if "Class distribution" in f.title)
        assert label_finding.severity == "info"

    def test_label_distribution_from_list(self) -> None:
        """label_counts_per_class is a list after NDArray.tolist() in real usage."""
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            label_stats_full={"label_counts_per_class": [55, 40, 5]},
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = _build_findings(raw)
        label_finding = next(f for f in findings if "Class distribution" in f.title)
        assert label_finding.severity == "warning"  # 55/5 = 11:1 > 10

    def test_label_distribution_warning(self) -> None:
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            label_stats_full={"label_counts_per_class": {"a": 95, "b": 5}},
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = _build_findings(raw)
        label_finding = next(f for f in findings if "Class distribution" in f.title)
        assert label_finding.severity == "warning"

    def test_split_sizes_finding(self) -> None:
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            test_indices=list(range(20)),
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = _build_findings(raw)
        size_finding = next(f for f in findings if "split sizes" in f.title)
        assert isinstance(size_finding.data, dict)
        assert size_finding.data["train"] == 70
        assert size_finding.data["val"] == 10
        assert size_finding.data["test"] == 20

    def test_coverage_warning(self) -> None:
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            test_indices=list(range(20)),
            folds=[
                SplitInfo(
                    fold=0,
                    train_indices=list(range(70)),
                    val_indices=list(range(10)),
                    coverage_train={"uncovered_indices": list(range(10)), "coverage_radius": 0.5},
                )
            ],
        )
        findings = _build_findings(raw)
        cov_finding = next(f for f in findings if "Coverage" in f.title)
        assert cov_finding.severity == "warning"  # 10/70 = 14.3% > 5%


# ---------------------------------------------------------------------------
# TestDataSplittingWorkflow
# ---------------------------------------------------------------------------


class TestDataSplittingWorkflow:
    def test_properties(self) -> None:
        wf = DataSplittingWorkflow()
        assert wf.name == "data-splitting"
        assert "splitting" in wf.description.lower() or "split" in wf.description.lower()
        assert wf.params_schema is DataSplittingParameters
        assert wf.output_schema is DataSplittingOutputs

    def test_rejects_non_context(self) -> None:
        wf = DataSplittingWorkflow()
        with pytest.raises(TypeError, match="WorkflowContext"):
            wf.execute("not a context")  # type: ignore[arg-type]

    def test_rejects_wrong_params(self) -> None:
        class OtherParams(BaseModel):
            x: int = 1

        wf = DataSplittingWorkflow()
        ctx = WorkflowContext()
        with pytest.raises(TypeError, match="DataSplittingParameters"):
            wf.execute(ctx, params=OtherParams())

    def test_error_returns_failed_result(self) -> None:
        wf = DataSplittingWorkflow()
        ctx = WorkflowContext()  # empty — will fail on missing datasets
        result = wf.execute(ctx, _make_params())
        assert result.success is False
        assert result.errors

    @patch("dataeval_flow.metadata.build_metadata")
    @patch("dataeval.utils.data.split_dataset")
    @patch("dataeval.core.label_stats")
    @patch("dataeval.bias.Balance")
    @patch("dataeval.bias.Diversity")
    def test_advisory_mode(
        self,
        mock_diversity: MagicMock,
        mock_balance: MagicMock,
        mock_label_stats: MagicMock,
        mock_split: MagicMock,
        mock_build_meta: MagicMock,
    ) -> None:
        # Setup
        dataset = _make_dataset(100)
        metadata = _make_metadata(100)
        mock_build_meta.return_value = metadata

        mock_balance_inst = MagicMock()
        mock_balance_inst.evaluate.return_value = MagicMock(
            balance=pl.DataFrame({"factor": ["a"], "mi": [0.1]}),
            factors=None,
            classwise=None,
        )
        mock_balance.return_value = mock_balance_inst

        mock_diversity_inst = MagicMock()
        mock_diversity_inst.evaluate.return_value = MagicMock(
            factors=pl.DataFrame({"factor_name": ["a"], "diversity_value": [0.9]}),
            classwise=None,
        )
        mock_diversity.return_value = mock_diversity_inst

        mock_label_stats.return_value = {
            "label_counts_per_class": np.array([20, 20, 20, 20, 20]),
            "class_count": 5,
            "label_count": 100,
            "image_count": 100,
        }

        mock_split.return_value = _make_split_result(100)

        ctx = WorkflowContext(
            dataset_contexts={"test_ds": DatasetContext(name="test_ds", dataset=dataset)},
        )

        # Execute
        result = wf_execute(ctx, _make_params())

        # Verify
        assert result.success is True
        assert result.name == "data-splitting"
        assert isinstance(result.data, DataSplittingOutputs)
        assert result.data.raw.dataset_size == 100
        assert len(result.data.raw.folds) == 1
        assert len(result.data.raw.test_indices) == 20
        assert result.metadata.stratified is True


# Module-level workflow for test_advisory_mode
_wf = DataSplittingWorkflow()
wf_execute = _wf.execute


# ---------------------------------------------------------------------------
# TestOutputTypes
# ---------------------------------------------------------------------------


class TestOutputTypes:
    def test_splitting_metadata(self) -> None:
        meta = DataSplittingMetadata(
            num_folds=3,
            stratified=True,
            split_on=["weather"],
            rebalance_method="global",
            split_sizes={"train": 70, "val": 10, "test": 20},
        )
        assert meta.num_folds == 3
        assert meta.tool == "dataeval-flow"

    def test_is_splitting_result(self) -> None:
        from dataeval_flow.workflow import WorkflowResult

        result = WorkflowResult(
            name="data-splitting",
            success=True,
            data=DataSplittingOutputs(
                raw=DataSplittingRawOutputs(dataset_size=100),
                report=DataSplittingReport(summary="test"),
            ),
            metadata=DataSplittingMetadata(),
        )
        assert is_splitting_result(result)

    def test_split_info(self) -> None:
        info = SplitInfo(
            fold=0,
            train_indices=[0, 1, 2],
            val_indices=[3, 4],
            label_stats_train={"class_count": 2},
            coverage_train={"uncovered_indices": [1]},
        )
        assert info.fold == 0
        assert len(info.train_indices) == 3
        assert info.coverage_val is None


# ---------------------------------------------------------------------------
# TestWorkflowDiscovery
# ---------------------------------------------------------------------------


class TestWorkflowDiscovery:
    def test_registered(self) -> None:
        from dataeval_flow.workflow import get_workflow, list_workflows

        wf = get_workflow("data-splitting")
        assert wf.name == "data-splitting"

        names = [w["name"] for w in list_workflows()]
        assert "data-splitting" in names


# ---------------------------------------------------------------------------
# TestFormatFactorTable
# ---------------------------------------------------------------------------


class TestFormatFactorTable:
    def test_empty_rows(self) -> None:
        assert _format_factor_table([], "mi_value", "MI Score") == []

    def test_basic_rows(self) -> None:
        rows = [
            {"factor_name": "weather", "mi_value": 0.5, "is_imbalanced": True},
            {"factor_name": "lighting", "mi_value": 0.1, "is_imbalanced": False},
        ]
        lines = _format_factor_table(rows, "mi_value", "MI Score")
        text = "\n".join(lines)
        assert "weather" in text
        assert "[!!]" in text
        assert "lighting" in text


# ---------------------------------------------------------------------------
# TestSerializeCoverageBranch
# ---------------------------------------------------------------------------


class TestSerializeCoverageBranch:
    def test_missing_keys_skipped(self) -> None:
        """When a key is None or missing, it should be skipped (branch 77->75)."""
        cov = {
            "uncovered_indices": np.array([1, 2]),
            "critical_value_radii": None,
            "coverage_radius": 0.5,
        }
        result = _serialize_coverage(cov)
        assert "uncovered_indices" in result
        assert "critical_value_radii" not in result
        assert "coverage_radius" in result

    def test_key_not_present(self) -> None:
        """Keys not in the dict at all should be skipped."""
        result = _serialize_coverage({"coverage_radius": 0.3})
        assert result == {"coverage_radius": 0.3}


# ---------------------------------------------------------------------------
# TestBuildFindingsCoverageTest
# ---------------------------------------------------------------------------


class TestBuildFindingsCoverageTest:
    def test_coverage_test_info(self) -> None:
        """Lines 209-213: coverage_test present with low uncovered %."""
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            test_indices=list(range(20)),
            coverage_test={"uncovered_indices": [0], "coverage_radius": 0.4},
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = _build_findings(raw)
        cov_finding = next(f for f in findings if f.title == "Coverage: test")
        assert cov_finding.severity == "info"  # 1/20 = 5%
        assert isinstance(cov_finding.data, dict)
        assert cov_finding.data["uncovered_count"] == 1

    def test_coverage_test_warning(self) -> None:
        """Lines 209-213: coverage_test present with high uncovered %."""
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            test_indices=list(range(20)),
            coverage_test={"uncovered_indices": list(range(5)), "coverage_radius": 0.4},
            folds=[SplitInfo(fold=0, train_indices=list(range(70)), val_indices=list(range(10)))],
        )
        findings = _build_findings(raw)
        cov_finding = next(f for f in findings if f.title == "Coverage: test")
        assert cov_finding.severity == "warning"  # 5/20 = 25% > 5%

    def test_coverage_test_empty_indices(self) -> None:
        """Lines 209-213: coverage_test with empty test_indices."""
        raw = DataSplittingRawOutputs(
            dataset_size=100,
            test_indices=[],
            coverage_test={"uncovered_indices": [], "coverage_radius": 0.4},
            folds=[SplitInfo(fold=0, train_indices=list(range(80)), val_indices=list(range(20)))],
        )
        findings = _build_findings(raw)
        cov_finding = next(f for f in findings if f.title == "Coverage: test")
        assert isinstance(cov_finding.data, dict)
        assert cov_finding.data["uncovered_pct"] == 0


# ---------------------------------------------------------------------------
# TestRunCoverage
# ---------------------------------------------------------------------------


class TestRunCoverage:
    def test_skips_when_no_extractor(self) -> None:
        """Lines 249-285: _run_coverage returns None when extractor is None."""
        ds_ctx = MagicMock()
        ds_ctx.extractor = None
        result = _run_coverage(ds_ctx, MagicMock(), [], [], _make_params())
        assert result is None

    @patch("dataeval_flow.embeddings.build_embeddings")
    @patch("dataeval.core.coverage_adaptive")
    def test_runs_coverage_with_extractor(
        self,
        mock_coverage: MagicMock,
        mock_build_emb: MagicMock,
    ) -> None:
        """Lines 249-285: _run_coverage computes coverage when extractor present."""
        ds_ctx = MagicMock()
        ds_ctx.extractor = MagicMock()
        ds_ctx.transforms = None
        ds_ctx.batch_size = 32

        # Fake embeddings: 10 samples, 4 dims
        embeddings = np.random.default_rng(42).random((10, 4))
        mock_build_emb.return_value = embeddings

        mock_cov_result = MagicMock()
        mock_cov_result.uncovered_indices = np.array([])
        mock_cov_result.critical_value_radii = np.array([0.1])
        mock_cov_result.coverage_radius = 0.2
        del mock_cov_result.get  # force getattr path in _serialize_coverage
        mock_coverage.return_value = mock_cov_result

        fold = SplitInfo(fold=0, train_indices=[0, 1, 2, 3, 4], val_indices=[5, 6, 7])
        test_indices = [8, 9]

        result = _run_coverage(ds_ctx, MagicMock(), [fold], test_indices, _make_params())

        assert result is not None
        assert mock_coverage.call_count == 3  # train + val + test
        assert fold.coverage_train is not None
        assert fold.coverage_val is not None

    @patch("dataeval_flow.embeddings.build_embeddings")
    @patch("dataeval.core.coverage_adaptive")
    def test_no_test_coverage_when_empty(
        self,
        mock_coverage: MagicMock,
        mock_build_emb: MagicMock,
    ) -> None:
        """Lines 279-285: no test coverage when test_indices is empty."""
        ds_ctx = MagicMock()
        ds_ctx.extractor = MagicMock()
        ds_ctx.transforms = None
        ds_ctx.batch_size = 32

        embeddings = np.random.default_rng(42).random((8, 4))
        mock_build_emb.return_value = embeddings

        mock_cov_result = MagicMock()
        mock_cov_result.uncovered_indices = np.array([])
        mock_cov_result.critical_value_radii = np.array([0.1])
        mock_cov_result.coverage_radius = 0.2
        del mock_cov_result.get
        mock_coverage.return_value = mock_cov_result

        fold = SplitInfo(fold=0, train_indices=[0, 1, 2, 3, 4], val_indices=[5, 6, 7])

        result = _run_coverage(ds_ctx, MagicMock(), [fold], [], _make_params())

        assert result is None
        assert mock_coverage.call_count == 2  # train + val only


# ---------------------------------------------------------------------------
# TestExecuteWithSelectionAndRebalance
# ---------------------------------------------------------------------------


class TestExecuteWithSelectionAndRebalance:
    @patch("dataeval_flow.workflows.splitting.workflow._run_coverage", return_value=None)
    @patch("dataeval_flow.metadata.build_metadata")
    @patch("dataeval.utils.data.split_dataset")
    @patch("dataeval.core.label_stats")
    @patch("dataeval.bias.Balance")
    @patch("dataeval.bias.Diversity")
    @patch("dataeval_flow.selection.build_selection")
    def test_selection_applied(
        self,
        mock_build_sel: MagicMock,
        mock_diversity: MagicMock,
        mock_balance: MagicMock,
        mock_label_stats: MagicMock,
        mock_split: MagicMock,
        mock_build_meta: MagicMock,
        mock_run_cov: MagicMock,  # noqa: ARG002
    ) -> None:
        """Line 374: build_selection is called when selection_steps are present."""
        dataset = _make_dataset(100)
        selected_dataset = _make_dataset(80)
        mock_build_sel.return_value = selected_dataset

        metadata = _make_metadata(80)
        mock_build_meta.return_value = metadata

        mock_balance_inst = MagicMock()
        mock_balance_inst.evaluate.return_value = MagicMock(balance=None, factors=None, classwise=None)
        mock_balance.return_value = mock_balance_inst

        mock_diversity_inst = MagicMock()
        mock_diversity_inst.evaluate.return_value = MagicMock(factors=None, classwise=None)
        mock_diversity.return_value = mock_diversity_inst

        mock_label_stats.return_value = {
            "label_counts_per_class": np.array([40, 40]),
            "class_count": 2,
            "label_count": 80,
            "image_count": 80,
        }
        mock_split.return_value = _make_split_result(80)

        ds_ctx = DatasetContext(name="ds", dataset=dataset, selection_steps=[MagicMock()])
        ctx = WorkflowContext(dataset_contexts={"ds": ds_ctx})

        wf = DataSplittingWorkflow()
        result = wf.execute(ctx, _make_params())

        assert result.success is True
        mock_build_sel.assert_called_once()
        # The split should operate on the selected dataset (len 80)
        assert result.data.raw.dataset_size == 80

    @patch("dataeval_flow.workflows.splitting.workflow._run_coverage", return_value=None)
    @patch("dataeval_flow.metadata.build_metadata")
    @patch("dataeval.utils.data.split_dataset")
    @patch("dataeval.core.label_stats")
    @patch("dataeval.bias.Balance")
    @patch("dataeval.bias.Diversity")
    @patch("dataeval.selection.ClassBalance")
    @patch("dataeval.selection.Select")
    @patch("dataeval.selection.Indices")
    def test_rebalance_applied(
        self,
        mock_indices: MagicMock,  # noqa: ARG002
        mock_select: MagicMock,
        mock_class_balance: MagicMock,
        mock_diversity: MagicMock,
        mock_balance: MagicMock,
        mock_label_stats: MagicMock,
        mock_split: MagicMock,
        mock_build_meta: MagicMock,
        mock_run_cov: MagicMock,  # noqa: ARG002
    ) -> None:
        """Lines 427-432: rebalancing is applied when rebalance_method is set."""
        dataset = _make_dataset(100)
        metadata = _make_metadata(100)
        mock_build_meta.return_value = metadata

        mock_balance_inst = MagicMock()
        mock_balance_inst.evaluate.return_value = MagicMock(balance=None, factors=None, classwise=None)
        mock_balance.return_value = mock_balance_inst

        mock_diversity_inst = MagicMock()
        mock_diversity_inst.evaluate.return_value = MagicMock(factors=None, classwise=None)
        mock_diversity.return_value = mock_diversity_inst

        mock_label_stats.return_value = {
            "label_counts_per_class": np.array([20, 20, 20, 20, 20]),
            "class_count": 5,
            "label_count": 100,
            "image_count": 100,
        }
        mock_split.return_value = _make_split_result(100)

        # Mock rebalance chain
        mock_select_inst = MagicMock()
        mock_select_inst.resolve_indices.return_value = list(range(70))
        mock_select.return_value = mock_select_inst

        ctx = WorkflowContext(
            dataset_contexts={"ds": DatasetContext(name="ds", dataset=dataset)},
        )

        wf = DataSplittingWorkflow()
        result = wf.execute(ctx, _make_params(rebalance_method="global"))

        assert result.success is True
        mock_class_balance.assert_called_once_with(method="global")
        mock_select.assert_called_once()
        assert result.metadata.rebalance_method == "global"


# ---------------------------------------------------------------------------
# TestConfigSchemas
# ---------------------------------------------------------------------------


class TestConfigSchemas:
    def test_splitting_workflow_config(self) -> None:
        from dataeval_flow.config.schemas import DataSplittingWorkflowConfig

        config = DataSplittingWorkflowConfig(
            name="split_stratified",
            test_frac=0.2,
            val_frac=0.1,
            stratify=True,
        )
        assert config.type == "data-splitting"
        assert config.name == "split_stratified"
        assert config.test_frac == 0.2

    def test_workflow_config_discriminator(self) -> None:
        from pydantic import TypeAdapter

        from dataeval_flow.config.schemas import WorkflowConfig

        adapter = TypeAdapter(WorkflowConfig)
        config = adapter.validate_python(
            {
                "name": "my_split",
                "type": "data-splitting",
                "test_frac": 0.3,
            }
        )
        assert config.type == "data-splitting"
        assert config.test_frac == 0.3

    def test_splitting_task_config(self) -> None:
        from dataeval_flow.config.schemas import DataSplittingTaskConfig

        task = DataSplittingTaskConfig(
            name="split-cifar",
            workflow="split_stratified",
            sources="cifar10",
        )
        assert task.name == "split-cifar"
