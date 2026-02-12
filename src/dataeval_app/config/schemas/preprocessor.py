"""Preprocessor configuration schema."""

from pydantic import BaseModel

from dataeval_app.preprocessing import PreprocessingStep


class PreprocessorConfig(BaseModel):
    """Named preprocessor pipeline configuration.

    Uses existing PreprocessingStep which has 'step' field (not 'name').
    """

    name: str
    steps: list[PreprocessingStep]  # Reuses existing PreprocessingStep
