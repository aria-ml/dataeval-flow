"""Preprocessor configuration schema."""

from collections.abc import Sequence

from pydantic import BaseModel

from dataeval_flow.preprocessing import PreprocessingStep


class PreprocessorConfig(BaseModel):
    """Named preprocessor pipeline configuration.

    Steps can be torchvision transforms (serializable, YAML-configurable)
    or arbitrary callables (programmatic only).
    """

    name: str
    steps: Sequence[PreprocessingStep]
