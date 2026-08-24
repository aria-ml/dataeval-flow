"""Workflow framework - protocol, context, result, discovery."""

__all__ = [
    "DatasetContext",
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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar, cast, overload, runtime_checkable

from pydantic import BaseModel

from dataeval_flow.workflow._text_report import (
    _WIDTH,
    _render_binning_section,
    _render_config_section,
    _render_detail_section,
    _summary_line,
)
from dataeval_flow.workflow.orchestrator import run_task, run_tasks, select_tasks

if TYPE_CHECKING:
    from dataeval.protocols import AnnotatedDataset

    from dataeval_flow.cache import DatasetCache
    from dataeval_flow.config.schemas import ExtractorConfig, ResultMetadata, ViewOperation
    from dataeval_flow.policy import ResolvedPolicy


@dataclass
class DatasetContext:
    """Per-dataset runtime context — groups a loaded dataset with its resolved configs."""

    name: str
    dataset: "AnnotatedDataset[Any]"
    extractor: "ExtractorConfig | None" = None
    transforms: Callable | None = None
    view_operations: "Sequence[ViewOperation] | None" = None
    batch_size: int | None = None
    label_source: str | None = None
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


TMetadata = TypeVar("TMetadata", bound="ResultMetadata")
TData = TypeVar("TData", bound=BaseModel)


def _source_lines(meta: "ResultMetadata") -> list[str]:
    """Render source/dataset lines for the metadata block."""
    lines: list[str] = []
    source_descs = meta.source_descriptions
    if source_descs:
        label = "  Source:       "
        continuation = " " * len(label)
        for i, desc in enumerate(source_descs):
            lines.append(f"{label if i == 0 else continuation}{desc}")
    elif meta.dataset_id or meta.selection_id:
        if meta.dataset_id:
            ds_line = f"  Dataset:      {meta.dataset_id}"
            if meta.label_source:
                ds_line += f"  ({meta.label_source})"
            lines.append(ds_line)
        if meta.selection_id:
            lines.append(f"  Selection:    {meta.selection_id}")
    return lines


@dataclass
class WorkflowResult(Generic[TMetadata, TData]):
    """Standardized workflow result.

    ``metadata`` carries the JATIC-required envelope (timestamp, tool info,
    dataset identifiers) plus any workflow-specific extras.  The orchestrator
    populates timing and dataset fields after execution; workflows construct
    the appropriate ``ResultMetadata`` subclass at creation time.

    Parameterize with metadata and data subclasses for typed access to
    workflow-specific fields, e.g.
    ``WorkflowResult[DataCleaningMetadata, DataCleaningOutputs]``.

    The optional ``dataset`` field holds the resolved, post-selection dataset
    used during workflow execution.  This is *not* serialized by
    :meth:`report`; it is provided purely for downstream programmatic use
    (visualization, filtering, export).

    The optional ``sources`` field maps source names to their resolved,
    post-selection datasets.  Multi-split workflows (e.g. data-analysis)
    populate this so callers can visualize images from any split without
    re-loading.
    """

    name: str
    success: bool
    data: TData
    metadata: TMetadata
    errors: Sequence[str] = field(default_factory=list)
    dataset: "AnnotatedDataset[Any] | None" = None
    sources: "dict[str, AnnotatedDataset[Any]] | None" = None

    @property
    def findings(self) -> list[Any]:
        """The report's findings, or an empty list where the workflow produced none."""
        report_obj = getattr(self.data, "report", None)
        return list(getattr(report_obj, "findings", []) or []) if report_obj is not None else []

    @property
    def warning_count(self) -> int:
        """How many findings breached their health threshold.

        The same count the report's health line renders, reached without parsing text —
        this is what an automated gate reads to decide whether a run is worth stopping on.
        """
        return sum(1 for f in self.findings if getattr(f, "severity", "info") == "warning")

    @property
    def health(self) -> dict[str, Any]:
        """Machine-readable roll-up of the run's findings.

        ``status`` is ``"warning"`` where any finding breached its threshold and ``"ok"``
        otherwise.  A warning is a prompt to look rather than a failure — ``success``
        stays the authority on whether the workflow ran.
        """
        warnings = self.warning_count
        return {
            "status": "warning" if warnings else "ok",
            "warnings": warnings,
            "findings": len(self.findings),
        }

    def report(self, *, detailed: bool = True) -> str:
        """Return a human-readable text report.

        Parameters
        ----------
        detailed : bool
            When ``True`` (default), the report includes detail sections
            for each finding.  When ``False``, only the summary is shown.

        Returns
        -------
        str
            Formatted text report suitable for ``print()``.
        """
        report_obj = getattr(self.data, "report", None)
        if report_obj is None:
            return f"{self.name}: no report available"

        findings = getattr(report_obj, "findings", [])
        lines: list[str] = []
        lines.extend(self._title_lines(report_obj.summary))
        lines.extend(self._metadata_lines())
        lines.extend(self._summary_lines(findings))
        if detailed:
            lines.extend(self._detail_lines(findings))
        lines.extend(_render_binning_section(self.metadata.metadata_binning, self.metadata.diagnostics))
        lines.extend(_render_config_section(self.metadata.resolved_config))
        lines.append("")
        lines.append("=" * _WIDTH)
        return "\n".join(lines)

    def _title_lines(self, summary: str) -> list[str]:
        """Banner with the report summary title."""
        lines = ["", "=" * _WIDTH]
        lines.extend(f"  {part.strip().upper()}" for part in summary.split("\n"))
        lines.append("=" * _WIDTH)
        return lines

    def _metadata_lines(self) -> list[str]:
        """Human-readable metadata block with trailing separator."""
        meta = self.metadata
        lines: list[str] = []
        if meta.timestamp:
            lines.append(f"  Timestamp:    {meta.timestamp.isoformat()}")
        if meta.execution_time_s is not None:
            lines.append(f"  Duration:     {meta.execution_time_s:.2f}s")
        lines.extend(_source_lines(meta))
        if meta.model_id:
            lines.append(f"  Model:        {meta.model_id}")
        if meta.preprocessor_id:
            lines.append(f"  Preprocessor: {meta.preprocessor_id}")
        if lines:
            lines.append("-" * _WIDTH)
        return lines

    def _summary_lines(self, findings: list) -> list[str]:
        """Summary section with per-finding one-liners and health status."""
        if not findings:
            return ["  No findings to report."]

        warnings = self.warning_count
        lines = ["", "  SUMMARY", "  -------"]
        lines.extend(_summary_line(f) for f in findings)
        lines.append("")
        if warnings:
            lines.append(f"  Health: {warnings} warning(s) [!!] — review flagged findings")
        else:
            lines.append("  Health: All checks passed [ok]")
        return lines

    def _detail_lines(self, findings: list) -> list[str]:
        """Expanded detail sections for each finding."""
        lines: list[str] = []
        for finding in findings:
            lines.extend(_render_detail_section(finding))
        return lines

    @overload
    def export(self, path: str | Path, *, fmt: Literal["json", "yaml"] = "json") -> Path: ...
    @overload
    def export(self, path: None = None, *, fmt: Literal["json", "yaml"] = "json") -> str: ...

    def export(
        self,
        path: str | Path | None = None,
        *,
        fmt: Literal["json", "yaml"] = "json",
    ) -> str | Path:
        """Serialize result data to JSON or YAML.

        Parameters
        ----------
        path : str | Path | None
            File or directory path.  If a directory, writes
            ``results.<ext>`` inside it.  If ``None``, returns the
            serialized string.
        fmt : {"json", "yaml"}
            Serialization format.  Defaults to ``"json"``.

        Returns
        -------
        str | Path
            Serialized string when ``path`` is ``None``, otherwise the
            ``Path`` to the written file.
        """
        output = self.to_dict()

        if fmt == "json":
            from json import dumps

            content = dumps(output, indent=2)
            ext = "json"
        else:
            from yaml import safe_dump

            content = safe_dump(output, default_flow_style=False)
            ext = "yaml"

        if path is None:
            return content

        dest = Path(path)
        if dest.is_dir() or not dest.suffix:
            dest.mkdir(parents=True, exist_ok=True)
            dest = dest / f"results.{ext}"
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)

        dest.write_text(content, encoding="utf-8")
        return dest

    def to_dict(self) -> dict[str, object]:
        """Return the result as a plain dictionary (metadata + health + data fields)."""
        return {
            "metadata": self.metadata.model_dump(mode="json"),
            "health": self.health,
            **self.data.model_dump(mode="json"),
        }


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
