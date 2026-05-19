"""End-to-end integration tests for parameter sweep workflow."""

from unittest.mock import MagicMock, patch

import polars as pl

from dataeval_flow.config import PipelineConfig, SourceConfig
from dataeval_flow.config.schemas import (
    HuggingFaceDatasetConfig,
    ParameterSweepTaskConfig,
    ParameterSweepWorkflowConfig,
)
from dataeval_flow.workflow import run_tasks


class TestParameterSweepE2E:
    @patch("dataeval_flow.dataset.resolve_dataset")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow.get_or_compute_stats")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow.Outliers")
    @patch("dataeval_flow.workflows.parameter_sweep.workflow.Duplicates")
    def test_run_tasks_with_parameter_sweep(self, mock_dup_cls, mock_outliers_cls, mock_stats, mock_resolve_ds):
        # 1. Setup mock pipeline config
        config = PipelineConfig(
            datasets=[HuggingFaceDatasetConfig(name="ds1", format="huggingface", path="test", split="train")],
            sources=[SourceConfig(name="src1", dataset="ds1")],
            workflows=[
                ParameterSweepWorkflowConfig(
                    name="sweep1",
                    type="parameter-sweep",
                    outlier_method=["adaptive", "zscore"],
                    outlier_threshold=[None, 2.0],
                )
            ],
            tasks=[ParameterSweepTaskConfig(name="task1", workflow="sweep1", sources="src1")],
        )

        # 2. Mock dataset resolution
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 100
        mock_resolve_ds.return_value = MagicMock(dataset=mock_ds, name="ds1", cache_key="key1", label_source=None)

        # 3. Mock evaluators
        mock_outlier_eval = MagicMock()
        mock_outliers_cls.return_value = mock_outlier_eval
        mock_outlier_output = MagicMock()
        mock_outlier_output.data.return_value = pl.DataFrame({"item_index": [1, 2]})
        mock_outlier_eval.from_stats.return_value = mock_outlier_output

        mock_dup_eval = MagicMock()
        mock_dup_cls.return_value = mock_dup_eval
        mock_dup_output = MagicMock()
        mock_dup_output.data.return_value = pl.DataFrame({"dup_type": ["exact"], "level": ["item"]})
        mock_dup_eval.from_stats.return_value = mock_dup_output

        # 4. Run tasks
        results = run_tasks(config)

        # 5. Verify results
        assert len(results) == 1
        res = results[0]
        assert res.name == "parameter-sweep"
        assert res.success is True
        assert len(res.data.raw.results) == 4  # 2 methods * 2 thresholds
        assert res.metadata.sweep_parameters == ["outlier_method", "outlier_threshold"]

        # Verify findings summary — outlier inputs swept → only Outliers table emitted.
        assert len(res.data.report.findings) == 1
        finding = res.data.report.findings[0]
        assert finding.title == "Outliers Sweep"
        assert finding.data["table_headers"] == ["outlier_method", "outlier_threshold", "Outliers"]
        assert len(finding.data["table_data"]) == 4
