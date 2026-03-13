"""Task configuration schema."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from dataeval_app.workflows.cleaning.params import DataCleaningParameters
    from dataeval_app.workflows.drift.params import DriftMonitoringParameters

AutoBinMethod = Literal["uniform_width", "uniform_count", "clusters"]


def _format_list(items: list[Any], indent: int = 0) -> list[str]:
    """Format a list with each element on its own line, indented by *indent* + 2."""
    lines: list[str] = []
    spaces = " " * (indent + 2)
    for item in items:
        if isinstance(item, dict):
            compact = ", ".join(f"{k}={v}" for k, v in item.items())
            lines.append(f"{spaces}{{{compact}}}")
        elif isinstance(item, list):
            lines.extend(_format_list(item, indent + 2))
        else:
            lines.append(f"{spaces}{item}")
    return lines


def _format_dict(d: dict[str, Any], indent: int = 0) -> list[str]:
    """Format a nested dict as aligned key: value lines."""
    lines: list[str] = []
    if not d:
        return lines
    pad = max(len(str(k)) for k in d) + 1
    spaces = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{spaces}{k:<{pad}}:")
            lines.extend(_format_dict(v, indent + 2))
        elif isinstance(v, list):
            lines.append(f"{spaces}{k:<{pad}}:")
            lines.extend(_format_list(v, indent))
        else:
            lines.append(f"{spaces}{k:<{pad}}: {v}")
    return lines


class TaskConfig(BaseModel):
    """Task/workflow configuration schema."""

    name: str
    workflow: str  # registered workflow name (e.g. "data-cleaning")
    datasets: str | list[str]  # reference to DatasetConfig.name
    models: str | Mapping[str, str] | None = None  # reference to ModelConfig.name
    preprocessors: str | Mapping[str, str] | None = None  # reference to PreprocessorConfig.name
    selections: str | Mapping[str, str] | None = None  # reference to SelectionConfig.name
    params: dict[str, Any] = Field(default_factory=dict)
    batch_size: int | None = Field(default=None)  # needed for tasks that involve embedding extraction
    output_format: Literal["json", "yaml", "text"] = "json"
    # Metadata configuration — applied uniformly across all datasets in this task
    metadata_auto_bin_method: AutoBinMethod | None = None
    metadata_exclude: list[str] = Field(default_factory=list)
    metadata_continuous_factor_bins: dict[str, int | list[float]] | None = None
    # Cache configuration - for caching expensive computations (embeddings, metadata stats) to disk across runs
    cache_dir: str | None = Field(
        default=None,
        description=(
            "Directory for disk-backed computation cache. "
            "When set, expensive computations (embeddings, metadata, hash stats) "
            "are cached to disk and reused across runs. "
            "In containers, point to a mounted volume (e.g. /cache)."
        ),
    )

    @model_validator(mode="after")
    def _require_batch_size_with_model(self) -> "TaskConfig":
        if self.models is not None and self.batch_size is None:
            raise ValueError(
                "batch_size is required when models is specified (embedding extraction needs a batch size)"
            )
        return self

    def summary(self) -> str:
        """Return a formatted summary of the task configuration."""
        lines = [f"Task: {self.name} ({self.workflow})"]

        # Collect key-value entries
        ds = self.datasets if isinstance(self.datasets, list) else [self.datasets]
        entries: list[tuple[str, str]] = [("datasets", ", ".join(ds))]
        if self.models is not None:
            entries.append(
                (
                    "models",
                    self.models if isinstance(self.models, str) else ", ".join(self.models.values()),
                )
            )
        if self.preprocessors is not None:
            entries.append(
                (
                    "preprocessors",
                    self.preprocessors
                    if isinstance(self.preprocessors, str)
                    else ", ".join(self.preprocessors.values()),
                )
            )
        if self.selections is not None:
            entries.append(
                (
                    "selections",
                    self.selections if isinstance(self.selections, str) else ", ".join(self.selections.values()),
                )
            )
        if self.batch_size is not None:
            entries.append(("batch_size", str(self.batch_size)))
        if self.cache_dir is not None:
            entries.append(("cache_dir", self.cache_dir))

        pad = max(len(k) for k, _ in entries)
        for key, val in entries:
            lines.append(f"  {key:<{pad}} : {val}")

        # Params
        params_data = self.params if isinstance(self.params, dict) else self.params.model_dump(mode="json")
        if params_data:
            lines.append("  params:")
            lines.extend(_format_dict(params_data, indent=4))

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the summary as the string representation of the config."""
        return self.summary()


class DataCleaningTaskConfig(TaskConfig):
    """Typed task config for the ``data-cleaning`` workflow.

    Provides static type narrowing: when passed to ``run_task()``, the
    return type is ``DataCleaningResult`` with fully typed ``metadata``
    and ``data`` fields.
    """

    workflow: str = "data-cleaning"
    params: "DataCleaningParameters"  # type: ignore[assignment]  # narrowed from dict[str, Any]

    @model_validator(mode="after")
    def _enforce_workflow(self) -> "DataCleaningTaskConfig":
        if self.workflow != "data-cleaning":
            raise ValueError(f"DataCleaningTaskConfig requires workflow='data-cleaning', got '{self.workflow}'")
        return self


class DriftMonitoringTaskConfig(TaskConfig):
    """Typed task config for the ``drift-monitoring`` workflow.

    Provides static type narrowing and validates that at least two
    datasets are specified (reference + test).
    """

    workflow: str = "drift-monitoring"
    params: "DriftMonitoringParameters"  # type: ignore[assignment]  # narrowed from dict[str, Any]

    @model_validator(mode="after")
    def _enforce_workflow(self) -> "DriftMonitoringTaskConfig":
        if self.workflow != "drift-monitoring":
            raise ValueError(f"DriftMonitoringTaskConfig requires workflow='drift-monitoring', got '{self.workflow}'")
        return self

    @model_validator(mode="after")
    def _require_multiple_datasets(self) -> "DriftMonitoringTaskConfig":
        ds = self.datasets if isinstance(self.datasets, list) else [self.datasets]
        if len(ds) < 2:
            raise ValueError(f"drift-monitoring requires at least 2 datasets (reference + test), got {len(ds)}: {ds}")
        return self


def _rebuild_deferred_models() -> None:
    """Rebuild models that use deferred forward references.

    Must be called once before typed task configs are used for validation.
    The workflow registry calls this during initialization.
    """
    from dataeval_app.workflows.cleaning.params import DataCleaningParameters
    from dataeval_app.workflows.drift.params import DriftMonitoringParameters

    DataCleaningTaskConfig.model_rebuild(_types_namespace={"DataCleaningParameters": DataCleaningParameters})
    DriftMonitoringTaskConfig.model_rebuild(_types_namespace={"DriftMonitoringParameters": DriftMonitoringParameters})
