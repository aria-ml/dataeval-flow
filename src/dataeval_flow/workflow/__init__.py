"""Workflow framework - protocol, context, result, discovery."""

__all__ = [
    "DatasetContext",
    "ResolvedOntology",
    "WorkflowContext",
    "WorkflowProtocol",
    "WorkflowResult",
    "get_workflow",
    "list_workflows",
    "run_task",
    "run_tasks",
    "select_tasks",
]

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from dataeval_flow.workflow.orchestrator import run_task, run_tasks, select_tasks
from dataeval_flow.workflow.result import TData, TMetadata, WorkflowResult

if TYPE_CHECKING:
    from dataeval import Ontology
    from dataeval.protocols import AnnotatedDataset

    from dataeval_flow.cache import DatasetCache
    from dataeval_flow.config.schemas import ExtractorConfig, ResultMetadata, ViewOperation
    from dataeval_flow.policy import ResolvedPolicy
    from dataeval_flow.stats import ResolvedStatsPolicy


@dataclass
class DatasetContext:
    """Per-dataset runtime context — groups a loaded dataset with its resolved configs."""

    name: str
    dataset: "AnnotatedDataset[Any]"
    extractor: "ExtractorConfig | None" = None
    transforms: Callable | None = None
    view_operations: "Sequence[ViewOperation] | None" = None
    batch_size: int | None = None
    #: Where the labels came from: one value, or one per operand where a merged corpus
    #: reads more than one provenance.
    label_source: "str | Sequence[str] | None" = None
    value_range: "tuple[float, float] | None" = None
    channel_groups: "Mapping[str, tuple[int, ...]] | None" = None
    """Named band groups this dataset declares, taken from the dataset config.

    Read by the stats policy, which selects the groups it measures from these.
    """
    cache: "DatasetCache | None" = None
    selection_steps: "InitVar[Sequence[ViewOperation] | None]" = None  # deprecated

    def __post_init__(self, selection_steps: "Sequence[ViewOperation] | None" = None) -> None:
        """Validate that the deprecated selection_steps parameter warns on use"""
        if selection_steps is not None:
            warnings.warn(
                "DatasetContext(selection_steps=...) is deprecated; use view_operations instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if self.view_operations is None:
                self.view_operations = selection_steps


@dataclass(frozen=True)
class ResolvedOntology:
    """The label space a task was configured with, resolved before the dataset was read.

    Read :attr:`error` rather than expecting an exception. An ontology problem degrades a
    ``data-coverage`` run to a skip reason and leaves label, metadata and gap analysis
    running, so resolving earlier must not abort the task.
    """

    ontology: "Ontology | None"
    source: str
    error: str | None = None


@dataclass
class WorkflowContext:
    """Runtime context for workflow execution.

    Provides per-dataset bundles and workflow-wide settings.
    """

    dataset_contexts: "Mapping[str, DatasetContext]" = field(default_factory=dict)
    batch_size: int | None = None
    metadata_policy: "ResolvedPolicy | None" = None
    """How factors become codes, resolved and checked before the dataset was read.

    Carried on the context rather than read off the parameters, because resolving it
    needs the pipeline the policy pool lives on and the data root its descriptor is
    relative to — neither of which a workflow has.  None where the caller built a context
    directly, which takes DataEval's defaults.
    """
    ontology: "ResolvedOntology | None" = None
    """The label space this task names, resolved before the dataset was read.

    Set here rather than in the workflow, for the same reason as :attr:`metadata_policy`:
    resolving a name needs the pipeline holding the pool, resolving a path needs the data
    root, and a workflow has neither. ``None`` when the caller built a context directly or
    configured no ontology. The workflow then reads its own parameters.
    """
    stats_policy: "ResolvedStatsPolicy | None" = None
    """What to measure and which views each consumer reads, resolved before the run.

    Set here rather than in the workflow, for the same reason as :attr:`metadata_policy`:
    resolving a name needs the pipeline holding the pool, and resolving its bands needs the
    datasets. ``None`` when the caller built a context directly or named no policy, in which
    case a workflow measures the whole image with its own flags.
    """


@runtime_checkable
class WorkflowProtocol(Protocol[TMetadata, TData]):
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

    def execute(self, context: WorkflowContext, params: BaseModel | None = None) -> "WorkflowResult[TMetadata, TData]":
        """Execute the workflow."""
        ...


# ---------------------------------------------------------------------------
# Workflow discovery (replaces WorkflowRegistry)
# ---------------------------------------------------------------------------

_WORKFLOWS: "dict[str, WorkflowProtocol[ResultMetadata, BaseModel]]" = {}
_initialized: bool = False


def _ensure_initialized() -> None:
    global _initialized
    if not _initialized:
        from dataeval_flow.workflows.analysis.workflow import DataAnalysisWorkflow
        from dataeval_flow.workflows.cleaning.workflow import DataCleaningWorkflow
        from dataeval_flow.workflows.coverage.workflow import DataCoverageWorkflow
        from dataeval_flow.workflows.drift.workflow import DriftMonitoringWorkflow
        from dataeval_flow.workflows.metadata_triage.workflow import MetadataTriageWorkflow
        from dataeval_flow.workflows.ood.workflow import OODDetectionWorkflow
        from dataeval_flow.workflows.parameter_sweep.workflow import ParameterSweepWorkflow
        from dataeval_flow.workflows.prioritization.workflow import DataPrioritizationWorkflow
        from dataeval_flow.workflows.splitting.workflow import DataSplittingWorkflow

        workflows = [
            DataAnalysisWorkflow,
            DataCleaningWorkflow,
            ParameterSweepWorkflow,
            DataCoverageWorkflow,
            DataPrioritizationWorkflow,
            DataSplittingWorkflow,
            DriftMonitoringWorkflow,
            OODDetectionWorkflow,
            MetadataTriageWorkflow,
        ]

        for workflow in workflows:
            wf = workflow()
            _WORKFLOWS[wf.name] = cast("WorkflowProtocol[ResultMetadata, BaseModel]", wf)
        _initialized = True


def get_workflow(name: str) -> "WorkflowProtocol[ResultMetadata, BaseModel]":
    """Look up a workflow by name. Raises ValueError if unknown."""
    _ensure_initialized()
    if name not in _WORKFLOWS:
        raise ValueError(f"Unknown workflow: '{name}'. Available: {list(_WORKFLOWS)}")
    return _WORKFLOWS[name]


def list_workflows() -> list[dict[str, str]]:
    """Return available workflows with name + description (for discovery)."""
    _ensure_initialized()
    return [{"name": w.name, "description": w.description} for w in _WORKFLOWS.values()]
