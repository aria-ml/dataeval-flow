"""The ``ood-detection`` preset."""

__all__ = ["OODDetectionConfig", "OODDetectionChecks", "OODDetectionWorkflow"]

from dataeval_flow.workflows.ood_detection._config import OODDetectionChecks, OODDetectionConfig
from dataeval_flow.workflows.ood_detection._workflow import OODDetectionWorkflow
