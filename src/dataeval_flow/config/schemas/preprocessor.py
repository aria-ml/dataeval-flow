"""Preprocessor configuration schema."""

from collections.abc import Sequence

from pydantic import BaseModel

from dataeval_flow.preprocessing import PreprocessingStep


class PreprocessorConfig(BaseModel):
    """Named preprocessor pipeline configuration.

    Uses existing PreprocessingStep which has 'step' field (not 'name').
    """

    name: str
    steps: Sequence[PreprocessingStep]  # Reuses existing PreprocessingStep
