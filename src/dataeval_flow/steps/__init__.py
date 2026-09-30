"""Steps: what a workflow chains, and the vocabulary the engine and the catalog read.

A workflow in ``workflows:`` is either a workflow type (``type:``) or a chain of steps (``steps:``). Each step is
an evaluator, a workflow, a transform, a combine or a check, and names what it reads by address.
"""

from dataeval_flow.steps._address import Address, parse_address
from dataeval_flow.steps._catalog import PortEntry, StepCatalog, StepCatalogEntry, list_steps
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._combine import Combine, CombineConfig, CombineContext
from dataeval_flow.steps._port import DATASET_KINDS, DataType, Port
from dataeval_flow.steps._registry import (
    get_check,
    get_combine,
    get_transform,
    list_checks,
    list_combines,
    list_transforms,
)
from dataeval_flow.steps._result import ChainMetadata, ChainOutput, ChainResult, StepResult
from dataeval_flow.steps._step import (
    Step,
    StepConfig,
    StepKind,
    StepSkipped,
    Transform,
    TransformConfig,
    TransformContext,
)
from dataeval_flow.steps._workflow import CustomWorkflowConfig, InputSlot, StepEntry

__all__ = [
    "DATASET_KINDS",
    "Address",
    "ChainMetadata",
    "ChainOutput",
    "ChainResult",
    "Check",
    "CheckConfig",
    "CheckContext",
    "Combine",
    "CombineConfig",
    "CombineContext",
    "CustomWorkflowConfig",
    "DataType",
    "InputSlot",
    "Port",
    "PortEntry",
    "Step",
    "StepCatalog",
    "StepCatalogEntry",
    "StepConfig",
    "StepEntry",
    "StepKind",
    "StepResult",
    "StepSkipped",
    "Transform",
    "TransformConfig",
    "TransformContext",
    "get_check",
    "get_combine",
    "get_transform",
    "list_checks",
    "list_combines",
    "list_steps",
    "list_transforms",
    "parse_address",
]
