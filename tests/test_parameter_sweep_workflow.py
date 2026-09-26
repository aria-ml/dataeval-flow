"""Tests for parameter sweep workflow."""

from unittest.mock import MagicMock, patch

import pytest

from dataeval_flow.workflows import DatasetContext, WorkflowContext
from dataeval_flow.workflows.parameter_sweep import ParameterSweepConfig, ParameterSweepWorkflow
from dataeval_flow.workflows.parameter_sweep._outputs import SweepRunResult
from tests.finding_blocks import column, rendered, tables

pytestmark = pytest.mark.required


def _make_params(**overrides: object) -> ParameterSweepConfig:
    """Build ParameterSweepConfig with defaults for testing."""
    defaults: dict[str, object] = {
        "outlier_method": ["adaptive", "zscore"],
        "outlier_threshold": [None, 2.0],
    }
    defaults.update(overrides)
    return ParameterSweepConfig(**defaults)  # type: ignore[arg-type]


class TestParameterSweepWorkflow:
    def test_workflow_properties(self):
        wf = ParameterSweepWorkflow()
        assert wf.name == "parameter-sweep"
        assert wf.config_type is ParameterSweepConfig

    @patch("dataeval_flow.workflows.parameter_sweep._workflow.get_or_compute_stats")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow.Outliers")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow.Duplicates")
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

        result = wf.run(params, context)

        assert result.success is True
        assert len(result.output.raw.results) == 2  # 1 method * 2 thresholds
        assert result.output.raw.results[0].outlier_count == 2
        assert result.output.raw.results[0].exact_duplicate_groups == 1
        assert result.output.raw.results[0].near_duplicate_groups == 1

        # Only outlier_threshold is swept here; Near Duplicates table is omitted.
        assert len(result.output.report.findings) == 1
        finding = result.output.report.findings[0]
        assert finding.title == "Outliers Sweep"
        (table,) = tables(finding)
        headers = [c.header for c in table.columns]
        assert headers == ["outlier_threshold", "Outliers"]
        assert len(table.rows) == 2
        assert "Exact Duplicates" not in headers

    @patch("dataeval_flow.workflows.parameter_sweep._workflow.get_or_compute_stats")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow.Outliers")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow.Duplicates")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow.build_extractor")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow._compute_embeddings")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow._merge_outlier_outputs")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow._merge_duplicate_results")
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

        result = wf.run(params, context)

        assert result.success is True
        # 1 method * 2 thresholds * 1 cluster_threshold * 1 cluster_algo = 4 runs
        # wait, _make_params has outlier_method=["adaptive", "zscore"], outlier_threshold=[None, 2.0]
        # so total combinations = 2 * 2 * 1 * 1 = 4
        assert len(result.output.raw.results) == 4
        assert mock_merge_outlier.call_count == 4
        assert mock_merge_dup.call_count == 0

    @patch("dataeval_flow.workflows.parameter_sweep._workflow.get_or_compute_stats")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow.Outliers")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow.Duplicates")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow.build_extractor")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow._compute_embeddings")
    @patch("dataeval_flow.workflows.parameter_sweep._workflow._merge_duplicate_results")
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
        params = ParameterSweepConfig(  # type: ignore[arg-type]
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

        result = wf.run(params, context)

        assert result.success is True
        # 1 method * 2 thresholds * 3 sensitivities * 1 algo = 6 raw runs
        assert len(result.output.raw.results) == 6

        findings = result.output.report.findings
        assert len(findings) == 2
        titles = [f.title for f in findings]
        assert titles == ["Outliers Sweep", "Near Duplicates Sweep"]

        (outliers_table,) = tables(findings[0])
        assert [c.header for c in outliers_table.columns] == ["outlier_threshold", "Outliers"]
        assert len(outliers_table.rows) == 2  # deduped on threshold

        (near_table,) = tables(findings[1])
        assert [c.header for c in near_table.columns] == ["duplicate_cluster_sensitivity", "Near Duplicates"]
        assert len(near_table.rows) == 3  # deduped on sensitivity


def _run(method: str, threshold: float | None, outliers: int) -> SweepRunResult:
    params = {
        "outlier_method": method,
        "outlier_threshold": threshold,
        "outlier_cluster_threshold": None,
        "outlier_cluster_algorithm": None,
        "duplicate_cluster_sensitivity": None,
        "duplicate_cluster_algorithm": None,
    }
    return SweepRunResult(params=params, outlier_count=outliers, exact_duplicate_groups=1, near_duplicate_groups=0)


class TestBuildFindings:
    """Each outcome's sweep, as a table of the swept values beside the outcome."""

    _RUNS = [_run("adaptive", None, 12), _run("adaptive", 2.0, 30), _run("zscore", None, 8), _run("zscore", 2.0, 25)]

    def test_one_finding_per_outcome_with_its_brief_and_description(self):
        (finding,) = ParameterSweepWorkflow()._build_findings(self._RUNS, ["outlier_method", "outlier_threshold"])
        assert finding.title == "Outliers Sweep"
        assert finding.brief == "4 unique combinations"
        assert finding.description == "Effect of outlier_method, outlier_threshold on outliers."

    def test_columns_are_keyed_and_headed_by_their_own_names(self):
        (finding,) = ParameterSweepWorkflow()._build_findings(self._RUNS, ["outlier_method", "outlier_threshold"])
        (table,) = tables(finding)
        assert [(c.key, c.header) for c in table.columns] == [
            ("outlier_method", "outlier_method"),
            ("outlier_threshold", "outlier_threshold"),
            ("Outliers", "Outliers"),
        ]

    def test_cells_are_the_raw_values(self):
        (finding,) = ParameterSweepWorkflow()._build_findings(self._RUNS, ["outlier_method", "outlier_threshold"])
        (table,) = tables(finding)
        assert column(table, "outlier_method") == ["adaptive", "adaptive", "zscore", "zscore"]
        assert column(table, "outlier_threshold") == [None, 2.0, None, 2.0]
        assert column(table, "Outliers") == [12, 30, 8, 25]

    def test_an_unset_threshold_draws_blank(self):
        (finding,) = ParameterSweepWorkflow()._build_findings(self._RUNS, ["outlier_method", "outlier_threshold"])
        assert rendered(finding).splitlines() == [
            "=" * 80,
            "  OUTLIERS SWEEP" + "4 unique combinations".rjust(64),
            "=" * 80,
            "  Effect of outlier_method, outlier_threshold on outliers.",
            "",
            "  outlier_method  outlier_threshold  Outliers",
            "  --------------  -----------------  --------",
            "  adaptive                                 12",
            "  adaptive                      2.0        30",
            "  zscore                                    8",
            "  zscore                        2.0        25",
        ]
