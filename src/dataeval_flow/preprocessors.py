"""Custom named preprocessors resolved alongside torchvision transforms.

These supplement ``torchvision.transforms.v2``: ``build_preprocessing`` checks
this registry first, then falls back to torchvision. Custom preprocessors use
**distinct names** and do **not** shadow or override any torchvision ``v2``
transform — by design the names never collide.

Each preprocessor is a tensor -> tensor callable composed into ``v2.Compose``
exactly like a torchvision transform.
"""

from typing import Any

__all__ = ["ToRGB", "CUSTOM_PREPROCESSORS", "resolve_custom"]


class ToRGB:
    """Coerce a CHW tensor image to 3 channels (repeat grayscale, drop alpha).

    torchvision's ``v2.RGB`` expands 1->3 channels but leaves 4-channel RGBA
    untouched, so models expecting 3-channel input still fail on RGBA inputs.
    This handles both: 1/2-channel -> repeat luma to RGB, >=4-channel -> keep
    the first three (drop alpha); 3-channel passes through. The stable ``repr``
    keeps the preprocessing cache key deterministic.
    """

    __slots__ = ()

    def __call__(self, image: Any) -> Any:
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


# Registry of custom preprocessors. Names here MUST be distinct from torchvision
# v2 transform names — custom preprocessors never shadow/override torchvision.
CUSTOM_PREPROCESSORS: dict[str, type] = {"ToRGB": ToRGB}


def resolve_custom(name: str) -> type | None:
    """Return the registered custom preprocessor class for ``name``, else ``None``.

    A ``None`` result signals the caller to fall back to torchvision. Custom
    names do not shadow torchvision transforms; the two name spaces are disjoint.
    """
    return CUSTOM_PREPROCESSORS.get(name)
