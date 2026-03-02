"""Task configuration schema."""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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
