"""Task configuration schema."""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field

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
    output_format: Literal["json", "yaml", "terminal"] = "json"
    # Metadata configuration — applied uniformly across all datasets in this task
    metadata_auto_bin_method: AutoBinMethod | None = None
    metadata_exclude: list[str] = Field(default_factory=list)
    metadata_continuous_factor_bins: dict[str, list[float]] | None = None
