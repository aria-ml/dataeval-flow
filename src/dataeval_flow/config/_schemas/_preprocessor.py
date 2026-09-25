"""Preprocessor configuration schema."""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field


class PreprocessingStep(BaseModel):
    """One step of a preprocessor: a transform, by name, and the keyword arguments it is built with.

    A step names a registered transform (see :func:`~dataeval_flow.config.transforms.list_transforms`) or, failing
    that, a ``torchvision.transforms.v2`` transform (https://pytorch.org/vision/stable/transforms.html).

    Examples
    --------
    >>> from dataeval_flow.config import PreprocessingStep
    >>> step = PreprocessingStep(step="Resize", params={"size": 256, "antialias": True})
    >>> step = PreprocessingStep(step="Normalize", params={"mean": [0.485], "std": [0.229]})
    """

    step: str = Field(
        description="The transform: a registered one's name, such as `ToRGB`, or a torchvision.transforms.v2 one's."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments the transform is built with. For a torchvision transform, a `dtype` names a torch "
            "dtype and an `interpolation` an InterpolationMode."
        ),
    )


class PreprocessorConfig(BaseModel):
    """A named preprocessor: the steps an extractor applies to each image before it embeds it.

    An extractor names one by its ``preprocessor``. Its steps run in order, in one torchvision ``v2.Compose``.
    """

    name: str = Field(description="Identifier for the preprocessor, referenced by extractors.")
    steps: Sequence[PreprocessingStep] = Field(description="The transforms to apply, in order.")
