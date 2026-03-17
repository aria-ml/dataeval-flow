"""Workflow framework - protocol, context, result, discovery."""

__all__ = [
    "DatasetContext",
    "WorkflowContext",
    "WorkflowProtocol",
    "WorkflowResult",
    "get_workflow",
    "list_workflows",
    "run_tasks",
]

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar, cast, overload, runtime_checkable

from pydantic import BaseModel

from dataeval_flow.workflow._text_report import (
    _WIDTH,
    _render_detail_section,
    _summary_line,
)
from dataeval_flow.workflow.orchestrator import run_tasks

if TYPE_CHECKING:
    from dataeval.protocols import AnnotatedDataset

    from dataeval_flow.cache import DatasetCache
    from dataeval_flow.config.schemas import ExtractorConfig, ResultMetadata, SelectionStep


@dataclass
class DatasetContext:
    """Per-dataset runtime context — groups a loaded dataset with its resolved configs."""

    name: str
    dataset: "AnnotatedDataset[Any]"
    extractor: "ExtractorConfig | None" = None
    transforms: Callable | None = None
    selection_steps: "Sequence[SelectionStep] | None" = None
    batch_size: int | None = None
    label_source: str | None = None
    cache: "DatasetCache | None" = None


@dataclass
class WorkflowContext:
    """Runtime context for workflow execution.

    Provides per-dataset bundles and workflow-wide settings.
    """

    dataset_contexts: "Mapping[str, DatasetContext]" = field(default_factory=dict)
    batch_size: int | None = None


def _default_metadata() -> "ResultMetadata":
    """Lazy factory — avoids circular import at class-definition time."""
    from dataeval_flow.config.schemas import ResultMetadata

    return ResultMetadata()


TMetadata = TypeVar("TMetadata", bound="ResultMetadata")
TData = TypeVar("TData", bound=BaseModel)


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
    """

    name: str
    success: bool
    data: TData
    errors: Sequence[str] = field(default_factory=list)
    metadata: TMetadata = field(default_factory=_default_metadata)  # type: ignore[assignment]
    format: Literal["text", "json", "yaml"] = "text"
    dataset: "AnnotatedDataset[Any] | None" = None

    @overload
    def report(self, *, format: Literal["text", "json", "yaml"] | None = None, path: None = None) -> str: ...
    @overload
    def report(self, *, format: Literal["text", "json", "yaml"] | None = None, path: str | Path) -> Path: ...

    def report(
        self,
        *,
        format: Literal["text", "json", "yaml"] | None = None,  # noqa: A002
        path: str | Path | None = None,
    ) -> str | Path:
        """Generate the workflow report.

        Parameters
        ----------
        format : {"text", "json", "yaml"} | None
            Output format. If ``None``, uses the instance's ``format``.
        path : str | Path | None
            File or directory path for json/yaml output.  If a directory,
            writes ``results.<ext>`` inside it.  If ``None``, returns the
            serialized string.

        Returns
        -------
        str | Path
            - text: formatted string (caller can ``print()`` it)
            - json/yaml without path: serialized string
            - json/yaml with path: ``Path`` to written file
        """
        if format is None:
            format = self.format  # noqa: A001
        if format == "text":
            return self._report_text()
        if format in ("json", "yaml"):
            return self._report_serialized(format, path)
        msg = f"Unknown format: {format!r}. Expected 'text', 'json', or 'yaml'."
        raise ValueError(msg)

    # -- private helpers --------------------------------------------------

    def _metadata_text_lines(self) -> list[str]:
        """Build human-readable metadata lines for the text report."""
        meta = self.metadata
        lines: list[str] = []
        if meta.timestamp:
            lines.append(f"  Timestamp:    {meta.timestamp.isoformat()}")
        if meta.execution_time_s is not None:
            lines.append(f"  Duration:     {meta.execution_time_s:.2f}s")
        if meta.dataset_id:
            ds_line = f"  Dataset:      {meta.dataset_id}"
            if meta.label_source:
                ds_line += f"  ({meta.label_source})"
            lines.append(ds_line)
        if meta.model_id:
            lines.append(f"  Model:        {meta.model_id}")
        if meta.preprocessor_id:
            lines.append(f"  Preprocessor: {meta.preprocessor_id}")
        if meta.selection_id:
            lines.append(f"  Selection:    {meta.selection_id}")
        return lines

    def _report_text(self) -> str:
        """Return executive-summary text report."""
        report = getattr(self.data, "report", None)
        if report is None:
            return f"{self.name}: no report available"

        findings = getattr(report, "findings", [])
        lines: list[str] = []

        # === Title banner ===
        lines.append("")
        lines.append("=" * _WIDTH)
        lines.append(f"  {report.summary.upper()}")
        lines.append("=" * _WIDTH)

        # Metadata block
        meta_lines = self._metadata_text_lines()
        if meta_lines:
            lines.extend(meta_lines)
            lines.append("-" * _WIDTH)

        if not findings:
            lines.append("  No findings to report.")
            lines.append("=" * _WIDTH)
            return "\n".join(lines)

        # === SUMMARY section ===
        warnings = sum(1 for f in findings if getattr(f, "severity", "info") == "warning")
        lines.append("")
        lines.append("  SUMMARY")
        lines.append("  -------")
        lines.extend(_summary_line(f) for f in findings)
        # Health footer — counts findings marked severity="warning".
        # Informational findings (outliers, duplicates, etc.) do not count.
        lines.append("")
        if warnings:
            lines.append(f"  Health: {warnings} warning(s) [!!] — review flagged findings")
        else:
            lines.append("  Health: All checks passed [ok]")

        # === Detail sections ===
        for finding in findings:
            lines.extend(_render_detail_section(finding))

        lines.append("")
        lines.append("=" * _WIDTH)
        return "\n".join(lines)

    def _report_serialized(
        self,
        fmt: Literal["json", "yaml"],
        path: str | Path | None,
    ) -> str | Path:
        """Serialize data + metadata to JSON or YAML, optionally writing to a file."""
        import json as json_mod

        output = {
            "metadata": self.metadata.model_dump(mode="json"),
            **self.data.model_dump(),
        }

        if fmt == "json":
            content = json_mod.dumps(output, indent=2)
            ext = "json"
        else:
            import yaml

            content = yaml.dump(output, default_flow_style=False)
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
        from dataeval_flow.workflows.cleaning.workflow import DataCleaningWorkflow
        from dataeval_flow.workflows.drift.workflow import DriftMonitoringWorkflow

        workflows = [DataCleaningWorkflow, DriftMonitoringWorkflow]

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
