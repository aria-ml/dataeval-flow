"""The ``scope`` preset."""

__all__ = [
    "CoverageSettings",
    "CropParams",
    "RepresentationSettings",
    "ScopeChecks",
    "ScopeConfig",
    "ScopeWorkflow",
    "UncoveredItemsSettings",
    "WrapSettings",
]

from dataeval_flow.workflows.scope._config import (
    CoverageSettings,
    CropParams,
    RepresentationSettings,
    ScopeChecks,
    ScopeConfig,
    UncoveredItemsSettings,
    WrapSettings,
)
from dataeval_flow.workflows.scope._workflow import ScopeWorkflow
