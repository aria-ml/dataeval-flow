"""Task configuration schema."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from dataeval_app.workflows.cleaning.params import DataCleaningParameters

AutoBinMethod = Literal["uniform_width", "uniform_count", "clusters"]


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


def _rebuild_deferred_models() -> None:
    """Rebuild models that use deferred forward references.

    Must be called once before ``DataCleaningTaskConfig`` is used for
    validation.  The workflow registry calls this during initialization.
    """
    from dataeval_app.workflows.cleaning.params import DataCleaningParameters

    DataCleaningTaskConfig.model_rebuild(_types_namespace={"DataCleaningParameters": DataCleaningParameters})
