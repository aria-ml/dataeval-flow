"""Workflows: the framework for writing one, and the built-in workflows.

A workflow reads one or more sources and returns a verdict — findings judged against health thresholds. Each
built-in lives in its own subpackage (``data_cleaning``, ``drift_monitoring``, …); the names here are what a new
workflow subclasses and what every workflow run hands back.
"""

from dataeval_flow.workflows._base import (
    Finding,
    Workflow,
    WorkflowConfig,
    WorkflowOutput,
    WorkflowRawOutput,
    WorkflowReport,
)
from dataeval_flow.workflows._context import DatasetContext, ResolvedOntology, WorkflowContext
from dataeval_flow.workflows._registry import get_workflow, list_workflows
from dataeval_flow.workflows._result import WorkflowResult

__all__ = [
    "DatasetContext",
    "Finding",
    "ResolvedOntology",
    "Workflow",
    "WorkflowConfig",
    "WorkflowContext",
    "WorkflowOutput",
    "WorkflowRawOutput",
    "WorkflowReport",
    "WorkflowResult",
    "get_workflow",
    "list_workflows",
]
