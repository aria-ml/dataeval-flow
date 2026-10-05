"""The ``data-splitting`` preset."""

__all__ = ["DataSplittingConfig", "DataSplittingChecks", "DataSplittingWorkflow", "SplittingCoverage"]

from dataeval_flow.workflows.data_splitting._config import (
    DataSplittingChecks,
    DataSplittingConfig,
    SplittingCoverage,
)
from dataeval_flow.workflows.data_splitting._workflow import DataSplittingWorkflow
