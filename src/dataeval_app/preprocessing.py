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
    >>> from dataeval_app.preprocessing import PreprocessingStep, build_preprocessing
    >>> steps = [PreprocessingStep(step="Resize", params={"size": 256})]
    >>> transform = build_preprocessing(steps)
    >>> output = transform(input_tensor)
"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

# TYPE_CHECKING-only: importing torchvision at module level is heavy (~2 s);
# it is imported lazily inside build_preprocessing() instead.  The quoted
# return annotation keeps full type-checker coverage with no runtime cost.
if TYPE_CHECKING:
    from torchvision.transforms import v2

__all__ = ["PreprocessingStep", "build_preprocessing"]


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


def build_preprocessing(steps: list[PreprocessingStep]) -> "v2.Compose":
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

    ops = []
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

    return v2.Compose(ops)
