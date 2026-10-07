"""The ``splits`` preset."""

__all__ = ["SplitsChecks", "SplitsConfig", "SplitsWorkflow"]

from dataeval_flow.workflows.splits._config import (
    SplitsChecks,
    SplitsConfig,
)
from dataeval_flow.workflows.splits._workflow import SplitsWorkflow
