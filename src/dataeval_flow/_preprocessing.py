"""Preprocessing utilities for image transforms.

Provides configuration-driven preprocessing using torchvision.transforms.v2.
A step names a registered transform (``dataeval_flow.config.transforms``, e.g. ``ToRGB``)
or any v2 transform; the registry refuses a transform that takes a v2 name, so the
two never collide.

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
    >>> from dataeval_flow._preprocessing import build_preprocessing
    >>> from dataeval_flow.config import PreprocessingStep
    >>> steps = [PreprocessingStep(step="Resize", params={"size": 256})]
    >>> transform = build_preprocessing(steps)
    >>> output = transform(input_array)  # numpy CHW -> numpy CHW
"""

__all__ = ["build_preprocessing"]


import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval_flow._logging import LogMessage
from dataeval_flow.config._schemas._preprocessor import PreprocessingStep
from dataeval_flow.config.transforms._base import Transform
from dataeval_flow.config.transforms._registry import resolve_step

_logger: logging.Logger = logging.getLogger(__name__)


class _PreprocessingTransform:
    """Callable wrapper around ``v2.Compose`` with a stable ``repr``.

    The default ``repr`` of a closure includes the memory address which
    changes every run, causing unnecessary cache misses when the cache
    key is derived from ``repr(transforms)``.  This wrapper delegates to
    ``v2.Compose.__repr__`` which is deterministic.
    """

    __slots__ = ("__wrapped__",)
    __wrapped__: Any

    def __init__(self, wrapped: Any) -> None:
        self.__wrapped__ = wrapped

    def __call__(self, image: NDArray[Any]) -> NDArray[Any]:
        import torch

        tensor = torch.as_tensor(np.ascontiguousarray(image))  # pyright: ignore[reportPrivateImportUsage]
        result = self.__wrapped__(tensor)
        return np.asarray(result.detach().cpu())

    def __repr__(self) -> str:
        return repr(self.__wrapped__)


def build_preprocessing(steps: Sequence[PreprocessingStep]) -> _PreprocessingTransform:
    """Build preprocessing pipeline from config.

    Builds a torchvision.transforms.v2 pipeline and wraps it so the returned
    callable accepts a **numpy CHW array** and returns a **numpy CHW array**.
    Internally the image is converted to a torch tensor for the v2 transforms,
    then converted back to numpy.

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
    def _resolve_dtype(name: object) -> torch.dtype:  # pyright: ignore[reportPrivateImportUsage]
        attr = str(name)
        result = getattr(torch, attr, None)
        if result is None or not isinstance(result, torch.dtype):  # pyright: ignore[reportPrivateImportUsage]
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

    _logger.debug(LogMessage(lambda: f"Building preprocessing pipeline: {[s.step for s in steps]}"))

    ops: list[Any] = []
    for step in steps:
        transform_cls = resolve_step(step.step)
        params = dict(step.params)

        # Convert torchvision's special parameter types. A registered transform takes its params as written.
        if not (isinstance(transform_cls, type) and issubclass(transform_cls, Transform)):
            for key, converter in param_converters.items():
                if key in params:
                    params[key] = converter(params[key])

        ops.append(transform_cls(**params))

    composed = v2.Compose(ops)
    return _PreprocessingTransform(composed)
