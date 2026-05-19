"""Tests for parameter sweep workflow."""

from unittest.mock import MagicMock, patch

from dataeval_flow.workflow import DatasetContext, WorkflowContext
from dataeval_flow.workflows.parameter_sweep.params import ParameterSweepParameters
from dataeval_flow.workflows.parameter_sweep.workflow import ParameterSweepWorkflow


def _make_params(**overrides: object) -> ParameterSweepParameters:
    """Build ParameterSweepParameters with defaults for testing."""
    defaults: dict[str, object] = {
        "outlier_method": ["adaptive", "zscore"],
        "outlier_threshold": [None, 2.0],
    }
    defaults.update(overrides)
    return ParameterSweepParameters(**defaults)  # type: ignore[arg-type]


class TestParameterSweepWorkflow:
    def test_workflow_properties(self):
        wf = ParameterSweepWorkflow()
        assert wf.name == "parameter-sweep"
        assert wf.params_schema == ParameterSweepParameters

    @patch("dataeval_flow.workflows.parameter_sweep.workflow.get_or_compute_stats")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow.Outliers")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow.Duplicates")
    def test_execute_basic(self, mock_dup_cls, mock_outliers_cls, mock_stats):
        wf = ParameterSweepWorkflow()
        params = _make_params(outlier_method=["adaptive"], outlier_threshold=[None, 3.0])

        # Mock dataset and context
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 10
        dc = DatasetContext(name="test", dataset=mock_ds)
        context = WorkflowContext(dataset_contexts={"test": dc})

        # Mock Outliers and Duplicates outputs
        mock_outlier_eval = MagicMock()
        mock_outliers_cls.return_value = mock_outlier_eval
        mock_outlier_output = MagicMock()
        import polars as pl

        mock_outlier_output.data.return_value = pl.DataFrame({"item_index": [1, 2]})
        mock_outlier_eval.from_stats.return_value = mock_outlier_output

        mock_dup_eval = MagicMock()
        mock_dup_cls.return_value = mock_dup_eval
        mock_dup_output = MagicMock()
        mock_dup_output.data.return_value = pl.DataFrame({"dup_type": ["exact", "near"], "level": ["item", "item"]})
        mock_dup_eval.from_stats.return_value = mock_dup_output

        result = wf.execute(context, params)

        assert result.success is True
        assert len(result.data.raw.results) == 2  # 1 method * 2 thresholds
        assert result.data.raw.results[0].outlier_count == 2
        assert result.data.raw.results[0].exact_duplicate_groups == 1
        assert result.data.raw.results[0].near_duplicate_groups == 1

        # Only outlier_threshold is swept here; Near Duplicates table is omitted.
        assert len(result.data.report.findings) == 1
        finding = result.data.report.findings[0]
        assert finding.title == "Outliers Sweep"
        assert finding.data["table_headers"] == ["outlier_threshold", "Outliers"]
        assert len(finding.data["table_data"]) == 2
        assert "Exact Duplicates" not in finding.data["table_headers"]

    @patch("dataeval_flow.workflows.parameter_sweep.workflow.get_or_compute_stats")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow.Outliers")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow.Duplicates")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow.build_extractor")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow._compute_embeddings")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow._merge_outlier_outputs")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow._merge_duplicate_results")
    def test_execute_cluster(
        self,
        mock_merge_dup,
        mock_merge_outlier,
        mock_comp_emb,
        mock_build_ext,
        mock_dup_cls,
        mock_outliers_cls,
        mock_stats,
    ):
        wf = ParameterSweepWorkflow()
        params = _make_params(outlier_cluster_threshold=[3.0], outlier_cluster_algorithm=["kmeans"])

        # Mock dataset and context with extractor
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 10
        dc = DatasetContext(name="test", dataset=mock_ds, extractor=MagicMock())
        context = WorkflowContext(dataset_contexts={"test": dc})

        # Mock Outliers and Duplicates outputs
        mock_outlier_eval = MagicMock()
        mock_outliers_cls.return_value = mock_outlier_eval
        mock_outlier_output = MagicMock()
        import polars as pl

        mock_outlier_output.data.return_value = pl.DataFrame({"item_index": [1, 2]})
        # Initially from_stats, then merged
        mock_outlier_eval.from_stats.return_value = mock_outlier_output
        mock_merge_outlier.return_value = mock_outlier_output

        mock_dup_eval = MagicMock()
        mock_dup_cls.return_value = mock_dup_eval
        mock_dup_output = MagicMock()
        mock_dup_output.data.return_value = pl.DataFrame({"dup_type": ["exact"], "level": ["item"]})
        mock_dup_eval.from_stats.return_value = mock_dup_output
        # No duplicate cluster params in this test call, so _merge_duplicate_results NOT called here
        # (params defines outlier cluster only)

        result = wf.execute(context, params)

        assert result.success is True
        # 1 method * 2 thresholds * 1 cluster_threshold * 1 cluster_algo = 4 runs
        # wait, _make_params has outlier_method=["adaptive", "zscore"], outlier_threshold=[None, 2.0]
        # so total combinations = 2 * 2 * 1 * 1 = 4
        assert len(result.data.raw.results) == 4
        assert mock_merge_outlier.call_count == 4
        assert mock_merge_dup.call_count == 0

    @patch("dataeval_flow.workflows.parameter_sweep.workflow.get_or_compute_stats")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow.Outliers")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow.Duplicates")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow.build_extractor")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow._compute_embeddings")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow._merge_duplicate_results")
    def test_findings_split_by_outcome(
        self,
        mock_merge_dup,
        mock_comp_emb,
        mock_build_ext,
        mock_dup_cls,
        mock_outliers_cls,
        mock_stats,
    ):
        """Both outlier and near-duplicate inputs swept → two outcome tables."""
        wf = ParameterSweepWorkflow()
        params = ParameterSweepParameters(  # type: ignore[arg-type]
            outlier_method=["adaptive"],
            outlier_threshold=[2.0, 3.0],
            duplicate_cluster_sensitivity=[0.5, 1.5, 2.5],
            duplicate_cluster_algorithm=["hdbscan"],
        )

        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 10
        dc = DatasetContext(name="test", dataset=mock_ds, extractor=MagicMock())
        context = WorkflowContext(dataset_contexts={"test": dc})

        import polars as pl

        mock_outlier_eval = MagicMock()
        mock_outliers_cls.return_value = mock_outlier_eval
        mock_outlier_output = MagicMock()
        mock_outlier_output.data.return_value = pl.DataFrame({"item_index": [1, 2]})
        mock_outlier_eval.from_stats.return_value = mock_outlier_output

        mock_dup_eval = MagicMock()
        mock_dup_cls.return_value = mock_dup_eval
        mock_dup_output = MagicMock()
        mock_dup_output.data.return_value = pl.DataFrame({"dup_type": ["near"], "level": ["item"]})
        mock_dup_eval.from_stats.return_value = mock_dup_output
        mock_merge_dup.return_value = mock_dup_output

        result = wf.execute(context, params)

        assert result.success is True
        # 1 method * 2 thresholds * 3 sensitivities * 1 algo = 6 raw runs
        assert len(result.data.raw.results) == 6

        findings = result.data.report.findings
        assert len(findings) == 2
        titles = [f.title for f in findings]
        assert titles == ["Outliers Sweep", "Near Duplicates Sweep"]

        outliers_finding = findings[0]
        assert outliers_finding.data["table_headers"] == ["outlier_threshold", "Outliers"]
        assert len(outliers_finding.data["table_data"]) == 2  # deduped on threshold

        near_finding = findings[1]
        assert near_finding.data["table_headers"] == ["duplicate_cluster_sensitivity", "Near Duplicates"]
        assert len(near_finding.data["table_data"]) == 3  # deduped on sensitivity

    def test_execute_validation_errors(self):
        wf = ParameterSweepWorkflow()
        # Missing context
        result = wf.execute(None)  # type: ignore
        assert result.success is False
        assert "Expected WorkflowContext" in result.errors[0]

        # Missing params
        context = WorkflowContext()
        result = wf.execute(context, None)
        assert result.success is False
        assert "ParameterSweepParameters required" in result.errors[0]
