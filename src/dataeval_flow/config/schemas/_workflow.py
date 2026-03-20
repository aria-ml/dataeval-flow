"""Workflow configuration schemas — one class per workflow type."""

from typing import Literal

from pydantic import Field

from dataeval_flow.workflows.cleaning.params import DataCleaningParameters
from dataeval_flow.workflows.drift.params import DriftMonitoringParameters
from dataeval_flow.workflows.ood.params import OODDetectionParameters
from dataeval_flow.workflows.splitting.params import DataSplittingParameters


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


class DataSplittingWorkflowConfig(DataSplittingParameters):
    """Typed workflow configuration for ``data-splitting``.

    Inherits all fields from :class:`DataSplittingParameters` — no ``params``
    nesting required.

    Example YAML::

        workflows:
          - name: split_stratified
            type: data-splitting
            test_frac: 0.2
            val_frac: 0.1
            stratify: true
    """

    name: str = Field(description="Identifier for this workflow")
    type: Literal["data-splitting"] = "data-splitting"


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


class OODDetectionWorkflowConfig(OODDetectionParameters):
    """Typed workflow configuration for ``ood-detection``.

    Inherits all fields from :class:`OODDetectionParameters` — no ``params``
    nesting required.

    Example YAML::

        workflows:
          - name: ood_knn
            type: ood-detection
            detectors:
              - method: kneighbors
                k: 10
    """

    name: str = Field(description="Identifier for this workflow")
    type: Literal["ood-detection"] = "ood-detection"
