"""Task configuration schema."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    """Task/workflow configuration schema."""

    name: str
    dataset: str  # reference to DatasetConfig.name
    model: str | None = None  # reference to ModelConfig.name
    preprocessor: str | None = None  # reference to PreprocessorConfig.name
    selection: str | None = None  # reference to SelectionConfig.name
    params: dict[str, Any] = Field(default_factory=dict)
    output_format: Literal["json", "yaml", "terminal"] = "json"
