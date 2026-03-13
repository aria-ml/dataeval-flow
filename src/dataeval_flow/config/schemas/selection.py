"""Selection configuration schema."""

from typing import Any

from pydantic import BaseModel, Field


class SelectionStep(BaseModel):
    """Single selection step - pass-through to dataeval.selection.

    See: https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/selection/index.html
    """

    type: str = Field(description="Selection class from dataeval.selection")
    params: dict[str, Any] = Field(default_factory=dict)


class SelectionConfig(BaseModel):
    """Named selection pipeline configuration.

    Similar to PreprocessorConfig - defines reusable selection pipelines.
    """

    name: str
    steps: list[SelectionStep]
