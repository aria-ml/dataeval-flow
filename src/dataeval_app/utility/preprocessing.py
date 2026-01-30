"""Preprocessing utilities for image transforms.

Provides configuration-driven preprocessing using torchvision.transforms.v2.
Any v2 transform can be specified by name in YAML config.

Example YAML:
    preprocessing:
      - step: Resize
        params: {size: [256, 256], antialias: true}
      - step: Normalize
        params: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}

Example Python:
    >>> from dataeval_app.utility.preprocessing import PreprocessingStep, build_preprocessing
    >>> steps = [PreprocessingStep(step="Resize", params={"size": 256})]
    >>> transform = build_preprocessing(steps)
    >>> output = transform(input_tensor)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from torchvision.transforms import v2


class PreprocessingStep(BaseModel):
    """Single preprocessing step.

    Pass-through to torchvision.transforms.v2 - any transform name is allowed.
    See: https://pytorch.org/vision/stable/transforms.html

    Example
    -------
    >>> step = PreprocessingStep(step="Resize", params={"size": 256, "antialias": True})
    >>> step = PreprocessingStep(step="Normalize", params={"mean": [0.485], "std": [0.229]})
    """

    step: str = Field(description="Transform name from torchvision.transforms.v2")
    params: dict[str, Any] = Field(default_factory=dict)


def build_preprocessing(steps: list[PreprocessingStep]) -> v2.Compose:
    """Build preprocessing pipeline from config.

    Pass-through to torchvision.transforms.v2 - no custom preprocessing logic.
    See: https://pytorch.org/vision/stable/transforms.html

    Parameters
    ----------
    steps : list[PreprocessingStep]
        List of preprocessing steps from config.

    Returns
    -------
    v2.Compose
        Composed transform pipeline.
    """
    import torch
    from torchvision.transforms import InterpolationMode, v2

    # Special parameter converters for non-primitive types
    param_converters = {
        "dtype": lambda x: getattr(torch, x),  # "float32" -> torch.float32
        "interpolation": lambda x: getattr(InterpolationMode, x),  # "BILINEAR" -> enum
    }

    ops = []
    for step in steps:
        params = dict(step.params)

        # Convert special parameter types
        for key, converter in param_converters.items():
            if key in params:
                params[key] = converter(params[key])

        # Get transform class and instantiate
        transform_cls = getattr(v2, step.step)
        ops.append(transform_cls(**params))

    return v2.Compose(ops)
