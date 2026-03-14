"""Workflow configuration schemas."""

from typing import Annotated, Literal

from pydantic import Field

from dataeval_flow.workflows.cleaning.params import DataCleaningParameters
from dataeval_flow.workflows.drift.params import DriftMonitoringParameters


class DataCleaningWorkflowConfig(DataCleaningParameters):
    """Typed workflow configuration for ``data-cleaning``.

    Inherits all fields from :class:`DataCleaningParameters` — no ``params``
    nesting required.

    Example YAML::

        workflows:
          - name: clean_zscore_stats
            type: data-cleaning
            outlier_method: zscore
            outlier_flags: [pixel, visual]
    """

    name: str = Field(description="Identifier for this workflow")
    type: Literal["data-cleaning"] = "data-cleaning"


class DriftMonitoringWorkflowConfig(DriftMonitoringParameters):
    """Typed workflow configuration for ``drift-monitoring``.

    Inherits all fields from :class:`DriftMonitoringParameters` — no ``params``
    nesting required.

    Example YAML::

        workflows:
          - name: drift_knn
            type: drift-monitoring
            detectors:
              - method: kneighbors
                k: 10
    """

    name: str = Field(description="Identifier for this workflow")
    type: Literal["drift-monitoring"] = "drift-monitoring"


# Discriminated union — Pydantic selects the right config based on ``type``.
# Use this type alias the same way as ``ExtractorConfig``.
WorkflowConfig = Annotated[
    DataCleaningWorkflowConfig | DriftMonitoringWorkflowConfig,
    Field(discriminator="type"),
]
