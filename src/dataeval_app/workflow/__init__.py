"""Workflow framework - protocol, context, result, discovery."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from dataeval_app.config.models import ExtractorConfig
    from dataeval_app.config.schemas.dataset import AutoBinMethod
    from dataeval_app.config.schemas.selection import SelectionStep
    from dataeval_app.dataset import MaiteDataset

__all__ = [
    "WorkflowContext",
    "WorkflowProtocol",
    "WorkflowResult",
    "get_workflow",
    "list_workflows",
    "run_task",
]


@dataclass
class WorkflowContext:
    """Runtime context for workflow execution.

    Provides the dataset, model config, and metadata config
    that workflows need beyond just parameters. Fields mirror
    DatasetConfig metadata fields from config schemas.
    """

    dataset: "MaiteDataset"
    extractor: "ExtractorConfig | None" = None
    transforms: Callable | None = None
    selection_steps: list["SelectionStep"] | None = None
    metadata_auto_bin_method: "AutoBinMethod | None" = None
    metadata_exclude: list[str] = field(default_factory=list)
    metadata_continuous_factor_bins: dict[str, list[float]] | None = None


@dataclass
class WorkflowResult:
    """Standardized workflow result."""

    name: str
    success: bool
    data: BaseModel
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class WorkflowProtocol(Protocol):
    """Workflow protocol with schema properties."""

    @property
    def name(self) -> str:
        """Workflow identifier."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    @property
    def params_schema(self) -> type[BaseModel] | None:
        """Pydantic model for workflow parameters, or None."""
        ...

    @property
    def output_schema(self) -> type[BaseModel]:
        """Pydantic model for workflow output."""
        ...

    def execute(self, context: WorkflowContext, params: BaseModel | None = None) -> WorkflowResult:
        """Execute the workflow."""
        ...


# ---------------------------------------------------------------------------
# Workflow discovery (replaces WorkflowRegistry)
# ---------------------------------------------------------------------------

_WORKFLOWS: dict[str, WorkflowProtocol] = {}
_initialized: bool = False


def _ensure_initialized() -> None:
    global _initialized
    if not _initialized:
        from dataeval_app.workflows.cleaning.workflow import DataCleaningWorkflow

        wf = DataCleaningWorkflow()
        _WORKFLOWS[wf.name] = wf
        _initialized = True


def get_workflow(name: str) -> WorkflowProtocol:
    """Look up a workflow by name. Raises ValueError if unknown."""
    _ensure_initialized()
    if name not in _WORKFLOWS:
        raise ValueError(f"Unknown workflow: '{name}'. Available: {list(_WORKFLOWS)}")
    return _WORKFLOWS[name]


def list_workflows() -> list[dict[str, str]]:
    """Return available workflows with name + description (for discovery)."""
    _ensure_initialized()
    return [{"name": w.name, "description": w.description} for w in _WORKFLOWS.values()]


# Re-export orchestrator entry point
from dataeval_app.workflow.orchestrator import run_task  # noqa: E402
