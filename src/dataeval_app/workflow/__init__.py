"""Workflow framework - protocol, context, result, discovery."""

__all__ = [
    "DatasetContext",
    "WorkflowContext",
    "WorkflowProtocol",
    "WorkflowResult",
    "get_workflow",
    "list_workflows",
    "run_task",
]

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar, overload, runtime_checkable

from pydantic import BaseModel

from dataeval_app.workflow.orchestrator import run_task

if TYPE_CHECKING:
    from dataeval_app.config.models import ExtractorConfig
    from dataeval_app.config.schemas.metadata import ResultMetadata
    from dataeval_app.config.schemas.selection import SelectionStep
    from dataeval_app.config.schemas.task import AutoBinMethod
    from dataeval_app.dataset import MaiteDataset


@dataclass
class DatasetContext:
    """Per-dataset runtime context — groups a loaded dataset with its resolved configs."""

    name: str
    dataset: "MaiteDataset"
    extractor: "ExtractorConfig | None" = None
    transforms: Callable | None = None
    selection_steps: "list[SelectionStep] | None" = None
    batch_size: int | None = None


@dataclass
class WorkflowContext:
    """Runtime context for workflow execution.

    Provides per-dataset bundles and workflow-wide metadata config.
    Metadata configuration (binning, exclusions) is set once at the
    workflow level so that all datasets are processed uniformly.
    """

    dataset_contexts: "dict[str, DatasetContext]" = field(default_factory=dict)
    metadata_auto_bin_method: "AutoBinMethod | None" = None
    metadata_exclude: list[str] = field(default_factory=list)
    metadata_continuous_factor_bins: dict[str, list[float]] | None = None
    batch_size: int | None = None


def _default_metadata() -> "ResultMetadata":
    """Lazy factory — avoids circular import at class-definition time."""
    from dataeval_app.config.schemas.metadata import ResultMetadata

    return ResultMetadata()


TMetadata = TypeVar("TMetadata", bound="ResultMetadata")


@dataclass
class WorkflowResult(Generic[TMetadata]):
    """Standardized workflow result.

    ``metadata`` carries the JATIC-required envelope (timestamp, tool info,
    dataset identifiers) plus any workflow-specific extras.  The orchestrator
    populates timing and dataset fields after execution; workflows construct
    the appropriate ``ResultMetadata`` subclass at creation time.

    Parameterize with a metadata subclass for typed access to
    workflow-specific fields, e.g. ``WorkflowResult[DataCleaningMetadata]``.
    """

    name: str
    success: bool
    data: BaseModel
    errors: list[str] = field(default_factory=list)
    metadata: TMetadata = field(default_factory=_default_metadata)  # type: ignore[assignment]
    format: Literal["text", "json", "yaml"] = "text"

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
            lines.append(f"  Timestamp: {meta.timestamp.isoformat()}")
        if meta.execution_time_s is not None:
            lines.append(f"  Duration:  {meta.execution_time_s:.2f}s")
        if meta.dataset_id:
            lines.append(f"  Dataset:   {meta.dataset_id}")
        return lines

    def _report_text(self) -> str:
        """Return compact findings summary as a string."""
        lines: list[str] = []
        report = getattr(self.data, "report", None)
        if report is None:
            return f"{self.name}: no report available"

        findings = getattr(report, "findings", [])
        _prefix = {"ok": "  \u2713 ", "warning": "  ! ", "info": "    "}

        # Header
        lines.append("")
        lines.append("=" * 64)
        lines.append(f"  {report.summary}")
        lines.append("=" * 64)

        # Metadata summary
        meta_lines = self._metadata_text_lines()
        if meta_lines:
            lines.extend(meta_lines)
            lines.append("-" * 64)

        if not findings:
            lines.append("  No findings to report.")
            lines.append("=" * 64)
            return "\n".join(lines)

        # Findings, inserting blank lines between groups.
        # Group by the parenthetical context in the title, e.g. "(train)".
        last_ctx = None
        warnings = 0
        for finding in findings:
            severity = getattr(finding, "severity", "info")
            if severity == "warning":
                warnings += 1

            ctx = finding.title.rsplit("(", 1)[-1].rstrip(")") if "(" in finding.title else ""
            if last_ctx is not None and ctx != last_ctx:
                lines.append("")
            last_ctx = ctx

            prefix = _prefix.get(severity, "    ")
            desc = f" \u2014 {finding.description}" if finding.description else ""
            lines.append(f"{prefix}{finding.title}{desc}")

        # Footer
        lines.append("")
        if warnings:
            lines.append(f"  {warnings} issue(s) detected")
        else:
            lines.append("  No issues detected")
        lines.append("=" * 64)
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

    def execute(self, context: WorkflowContext, params: BaseModel | None = None) -> "WorkflowResult":
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
