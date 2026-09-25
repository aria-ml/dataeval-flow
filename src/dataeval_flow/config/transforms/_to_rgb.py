"""The built-in ``ToRGB`` transform."""

__all__ = ["ToRGB"]

from typing import Any, ClassVar

from dataeval_flow.config.transforms._base import Transform


class ToRGB(Transform):
    """Coerce a CHW tensor image to 3 channels (repeat grayscale, drop alpha).

    torchvision's ``v2.RGB`` expands 1->3 channels but leaves 4-channel RGBA
    untouched, so models expecting 3-channel input still fail on RGBA inputs.
    This handles both: 1/2-channel -> repeat luma to RGB, >=4-channel -> keep
    the first three (drop alpha); 3-channel passes through. The stable ``repr``
    keeps the preprocessing cache key deterministic.
    """

    name: ClassVar[str] = "ToRGB"
    description: ClassVar[str] = "Coerces an image to three channels: repeats grayscale, drops alpha."

    def __call__(self, image: Any, /) -> Any:
        """Coerce a CHW (or HW) tensor image to exactly 3 channels."""
        if image.ndim == 2:  # HW -> 1HW
            image = image.unsqueeze(0)
        channels = image.shape[0]
        if channels == 3:
            return image
        if channels > 3:  # RGBA (or more) -> drop alpha/extra channels
            return image[:3]
        return image[:1].repeat(3, 1, 1)  # gray (or gray+alpha) -> RGB

    def __repr__(self) -> str:
        """Return a deterministic repr for preprocessing cache-key stability."""
        return "ToRGB()"
