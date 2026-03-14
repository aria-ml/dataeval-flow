"""Task configuration schema."""

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field, model_validator

AutoBinMethod = Literal["uniform_width", "uniform_count", "clusters"]


class TaskConfig(BaseModel):
    """Task/workflow configuration schema."""

    name: str
    workflow: str  # reference to WorkflowConfig.name (e.g. "clean_zscore_stats")
    datasets: str | Sequence[str]  # reference to DatasetConfig.name
    models: str | Mapping[str, str] | None = None  # reference to ModelConfig.name
    preprocessors: str | Mapping[str, str] | None = None  # reference to PreprocessorConfig.name
    selections: str | Mapping[str, str] | None = None  # reference to SelectionConfig.name
    batch_size: int | None = Field(default=None)  # needed for tasks that involve embedding extraction
    output_format: Literal["json", "yaml", "text"] = "json"
    # Metadata configuration — applied uniformly across all datasets in this task
    metadata_auto_bin_method: AutoBinMethod | None = None
    metadata_exclude: Sequence[str] = Field(default_factory=list)
    metadata_continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None
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


class DataCleaningTaskConfig(TaskConfig):
    """Task config for ``data-cleaning`` workflows.

    A typed subclass of :class:`TaskConfig` that enables typed overloads
    on :func:`~dataeval_flow.workflow.orchestrator.run_task`, returning a
    :class:`~dataeval_flow.workflows.cleaning.outputs.DataCleaningResult`
    with full access to cleaning-specific metadata (``mode``,
    ``clean_indices``, ``flagged_indices``, ``removed_count``).
    """


class DriftMonitoringTaskConfig(TaskConfig):
    """Task config that validates drift-monitoring constraints.

    Validates that at least two datasets are specified (reference + test).
    The ``workflow`` field references a workflow instance whose type must
    be ``drift-monitoring`` — enforced at runtime by the orchestrator.
    """

    @model_validator(mode="after")
    def _require_multiple_datasets(self) -> "DriftMonitoringTaskConfig":
        ds = self.datasets if isinstance(self.datasets, list) else [self.datasets]
        if len(ds) < 2:
            raise ValueError(f"drift-monitoring requires at least 2 datasets (reference + test), got {len(ds)}: {ds}")
        return self
