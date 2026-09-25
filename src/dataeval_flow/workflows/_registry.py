"""The workflow registry: the built-in table and the `dataeval_flow.workflows` entry points."""

from typing import Any

from dataeval_flow._kind import config_type_matches
from dataeval_flow._registry import Registry
from dataeval_flow.workflows._base import Workflow

__all__ = ["WORKFLOWS", "get_workflow", "list_workflows"]

_BUILTINS = {
    "data-analysis": "dataeval_flow.workflows.data_analysis._workflow:DataAnalysisWorkflow",
    "data-cleaning": "dataeval_flow.workflows.data_cleaning._workflow:DataCleaningWorkflow",
    "data-coverage": "dataeval_flow.workflows.data_coverage._workflow:DataCoverageWorkflow",
    "data-prioritization": "dataeval_flow.workflows.data_prioritization._workflow:DataPrioritizationWorkflow",
    "data-splitting": "dataeval_flow.workflows.data_splitting._workflow:DataSplittingWorkflow",
    "drift-monitoring": "dataeval_flow.workflows.drift_monitoring._workflow:DriftMonitoringWorkflow",
    "metadata-triage": "dataeval_flow.workflows.metadata_triage._workflow:MetadataTriageWorkflow",
    "ood-detection": "dataeval_flow.workflows.ood_detection._workflow:OODDetectionWorkflow",
    "parameter-sweep": "dataeval_flow.workflows.parameter_sweep._workflow:ParameterSweepWorkflow",
}


WORKFLOWS: Registry[Workflow[Any, Any]] = Registry(
    kind="workflow",
    group="dataeval_flow.workflows",
    base=lambda: Workflow,
    builtins=_BUILTINS,
    check=config_type_matches,
)


def get_workflow(name: str) -> type[Workflow[Any, Any]]:
    """The workflow registered as `name`, built-in or plugin.

    Parameters
    ----------
    name : str
        The workflow's name: its type id, e.g. ``"data-cleaning"``.

    Returns
    -------
    type[Workflow]
        The workflow class, whose ``name``, ``description`` and ``config_type`` describe it. Flow builds its
        instances.

    Raises
    ------
    ValueError
        When nothing is registered under `name`, or the plugin registered under it failed to load.
    """
    return WORKFLOWS.get(name)


def list_workflows() -> list[type[Workflow[Any, Any]]]:
    """Every installed workflow, built-in or plugin, sorted by name.

    A plugin that failed to load is left out; :func:`get_workflow` raises its error.

    Returns
    -------
    list[type[Workflow]]
        The workflow classes.
    """
    return WORKFLOWS.list()
