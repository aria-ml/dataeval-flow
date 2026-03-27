"""Workflow configuration schemas — one class per workflow type."""

from typing import Literal

from pydantic import Field

from dataeval_flow.workflows.analysis.params import DataAnalysisParameters
from dataeval_flow.workflows.cleaning.params import DataCleaningParameters
from dataeval_flow.workflows.drift.params import DriftMonitoringParameters
from dataeval_flow.workflows.ood.params import OODDetectionParameters
from dataeval_flow.workflows.prioritization.params import DataPrioritizationParameters
from dataeval_flow.workflows.splitting.params import DataSplittingParameters


class DataAnalysisWorkflowConfig(DataAnalysisParameters):
    """Typed workflow configuration for ``data-analysis``.

    Inherits all fields from :class:`DataAnalysisParameters` — no ``params``
    nesting required.

    Example YAML::

        workflows:
          - name: cppe5_analysis
            type: data-analysis
            outlier_method: zscore
            outlier_flags: [dimension, pixel, visual]
            balance: true
            diversity_method: simpson
    """

    name: str = Field(description="Identifier for this workflow")
    type: Literal["data-analysis"] = "data-analysis"


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


class DataPrioritizationWorkflowConfig(DataPrioritizationParameters):
    """Typed workflow configuration for ``data-prioritization``.

    Inherits all fields from :class:`DataPrioritizationParameters` — no ``params``
    nesting required.

    Example YAML::

        workflows:
          - name: prioritize_knn
            type: data-prioritization
            method: knn
            k: 10
            order: hard_first
            cleaning:
              outlier_method: adaptive
              outlier_flags: [dimension, pixel]
    """

    name: str = Field(description="Identifier for this workflow")
    type: Literal["data-prioritization"] = "data-prioritization"
