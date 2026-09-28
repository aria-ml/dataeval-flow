"""An item's preview: a small image of it, for a report to show beside the finding it evidences.

One previewer ships, for images: an array of channels × height × width, or height × width for one
channel, as every Flow dataset returns an image. Anything else, such as a signal, a volume or a
feature vector, has no previewer yet, and :func:`preview` raises :class:`NotAnImageError` for it, so its
cells name the item instead.
"""

__all__ = ["NotAnImageError", "preview"]

import base64
import io
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from dataeval_flow._blocks import Asset, ItemRef

# The long side of a thumbnail, in pixels: shrunk to fit, never enlarged.
SIZE = 192
_QUALITY = 80
# Around a detection box, as a share of its width and height, so that the crop shows what's next to it.
_MARGIN = 0.1


class NotAnImageError(TypeError):
    """The item isn't an image, which is the one kind of item that has a previewer."""


def _array(image: Any) -> NDArray[Any]:
    """The image as channels × height × width; a 2-D image is one channel."""
    array = np.asarray(image)
    if array.dtype.kind not in "biuf" or array.ndim not in (2, 3):
        raise NotAnImageError(f"a {array.ndim}-D array of {array.dtype} is not an image")
    return array[None] if array.ndim == 2 else array


def _crop(array: NDArray[Any], target: Any, box: int) -> NDArray[Any]:
    """The box's pixels, with a margin, from the item's annotation; ``LookupError`` or ``ValueError`` if it has none.

    MAITE's detection target holds its boxes as ``x0, y0, x1, y1`` in pixels, never normalized.
    """
    boxes = getattr(target, "boxes", None)
    if boxes is None:
        raise LookupError("its annotation has no boxes")
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not 0 <= box < len(boxes):
        raise LookupError(f"its annotation has no box {box}")
    x0, y0, x1, y1 = boxes[box]
    width, height = x1 - x0, y1 - y0
    if not (width > 0 and height > 0):
        raise ValueError(f"box {box} has no area")
    rows, columns = array.shape[1:]
    left, top = max(math.floor(x0 - _MARGIN * width), 0), max(math.floor(y0 - _MARGIN * height), 0)
    right, bottom = min(math.ceil(x1 + _MARGIN * width), columns), min(math.ceil(y1 + _MARGIN * height), rows)
    if right <= left or bottom <= top:
        raise ValueError(f"box {box} lies outside the image")
    return array[:, top:bottom, left:right]


def _to_uint8(array: NDArray[Any], value_range: tuple[float, float] | None) -> NDArray[np.uint8]:
    """The values scaled to 0–255: by the integer type's range, else the declared one, else the image's own.

    Nothing guesses a range from the largest value: a dark 0–255 image would read as 0–1.
    """
    if array.dtype == np.uint8:
        return array
    if array.dtype.kind in "iu":
        info = np.iinfo(array.dtype)
        low, high = float(info.min), float(info.max)
    elif value_range is not None:
        low, high = value_range
    else:
        finite = array[np.isfinite(array)]
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
    if high <= low:
        # A constant image: nothing to stretch, so mid-grey rather than a black or white that means something.
        return np.full(array.shape, 128, dtype=np.uint8)
    scaled = (array.astype(np.float32) - low) / (high - low) * 255
    return np.clip(np.nan_to_num(scaled), 0, 255).round().astype(np.uint8)


def preview(ref: ItemRef, image: Any, target: Any, value_range: tuple[float, float] | None) -> Asset:
    """The item's thumbnail: its image, or the box *ref* names cropped from it, as a WebP at most 192 px across.

    One channel is greyscale and three are RGB; four use the first three, since the fourth may be
    alpha or infrared, and any other count previews its first channel. *value_range* is what the
    dataset declares a float image's values span.

    Raises
    ------
    NotAnImageError
        When the item is no image: an array that isn't 2- or 3-D, or not an array of numbers.
    LookupError, ValueError
        When the box *ref* names isn't in the item's annotation, or has no area inside the image.
    """
    array = _array(image)
    if ref.target is not None:
        array = _crop(array, target, ref.target)
    channels = array[:3] if array.shape[0] in (3, 4) else array[:1]
    pixels = np.moveaxis(_to_uint8(channels, value_range), 0, -1)
    picture = Image.fromarray(pixels[..., 0] if pixels.shape[-1] == 1 else np.ascontiguousarray(pixels))
    picture.thumbnail((SIZE, SIZE))
    buffer = io.BytesIO()
    picture.save(buffer, format="WEBP", quality=_QUALITY)
    return Asset(
        item=ref,
        media_type="image/webp",
        width=picture.width,
        height=picture.height,
        data=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )
