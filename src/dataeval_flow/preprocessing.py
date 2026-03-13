"""Preprocessing utilities for image transforms.

Provides configuration-driven preprocessing using torchvision.transforms.v2.
Any v2 transform can be specified by name in YAML config.

The returned callable accepts a numpy CHW array, converts to a torch tensor
for torchvision transforms, then converts back to a numpy CHW array so that
ONNX extractors receive the format they expect.

Example YAML:
    preprocessing:
      - step: Resize
        params: {size: [256, 256], antialias: true}
      - step: Normalize
        params: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}

Example Python:
    >>> from dataeval_flow.preprocessing import PreprocessingStep, build_preprocessing
    >>> steps = [PreprocessingStep(step="Resize", params={"size": 256})]
    >>> transform = build_preprocessing(steps)
    >>> output = transform(input_array)  # numpy CHW -> numpy CHW
"""

__all__ = ["PreprocessingStep", "build_preprocessing"]


import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

logger: logging.Logger = logging.getLogger(__name__)


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


def build_preprocessing(steps: list[PreprocessingStep]) -> Callable[[NDArray[Any]], NDArray[Any]]:
    """Build preprocessing pipeline from config.

    Builds a torchvision.transforms.v2 pipeline and wraps it so the returned
    callable accepts a **numpy CHW array** and returns a **numpy CHW array**.
    Internally the image is converted to a torch tensor for the v2 transforms,
    then converted back to numpy afterwards.

    See: https://pytorch.org/vision/stable/transforms.html

    Parameters
    ----------
    steps : list[PreprocessingStep]
        List of preprocessing steps from config.

    Returns
    -------
    Callable[[NDArray[Any]], NDArray[Any]]
        Wrapped transform: numpy CHW in, numpy CHW out.
    """
    import torch
    from torchvision.transforms import InterpolationMode, v2

    # Special parameter converters for non-primitive types
    def _resolve_dtype(name: object) -> torch.dtype:
        attr = str(name)
        result = getattr(torch, attr, None)
        if result is None or not isinstance(result, torch.dtype):
            raise ValueError(f"Unknown torch dtype: '{name}'. Example: 'float32', 'uint8'.")
        return result

    def _resolve_interpolation(name: object) -> InterpolationMode:
        attr = str(name)
        result = getattr(InterpolationMode, attr, None)
        if result is None:
            raise ValueError(f"Unknown InterpolationMode: '{name}'. Example: 'BILINEAR', 'NEAREST'.")
        return result

    param_converters = {
        "dtype": _resolve_dtype,
        "interpolation": _resolve_interpolation,
    }

    logger.debug("Building preprocessing pipeline: %s", [s.step for s in steps])

    ops: list[Any] = []
    for step in steps:
        params = dict(step.params)

        # Convert special parameter types
        for key, converter in param_converters.items():
            if key in params:
                params[key] = converter(params[key])

        # Get transform class and instantiate
        transform_cls = getattr(v2, step.step, None)
        if transform_cls is None:
            raise ValueError(f"Unknown transform: '{step.step}'. Check torchvision.transforms.v2 docs.")
        ops.append(transform_cls(**params))

    composed = v2.Compose(ops)

    def _apply(image: NDArray[Any]) -> NDArray[Any]:
        # numpy CHW -> torch tensor (zero-copy when possible)
        tensor = torch.as_tensor(np.ascontiguousarray(image))
        result = composed(tensor)
        # torch tensor -> numpy CHW (detach in case any transform tracked grads)
        return np.asarray(result.detach().cpu())

    return _apply
