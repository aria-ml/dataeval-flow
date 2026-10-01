"""Image classification data whose out-of-distribution images share a metadata factor, for the OOD port's tests."""

from collections.abc import Collection
from typing import Any

import numpy as np
from dataeval.protocols import DatasetMetadata


class FactorImages:
    """3x16x16 images over two classes, each with two numeric metadata factors: `altitude`, which a shifted image takes
    from far above the rest, and `hour`, which varies freely.

    Parameters
    ----------
    count : int
        How many images.
    seed : int
        Draws the images and the factors.
    shifted : Collection[int]
        The items brightened out of the distribution of the rest, and placed at altitude 800 to 900, not 100 to 300.
    factors : bool
        Whether each item's metadata holds the factors; ``False`` leaves only its `id`, as most image folders do.
    """

    def __init__(self, count: int = 40, seed: int = 0, *, shifted: Collection[int] = (), factors: bool = True) -> None:
        rng = np.random.default_rng(seed)
        self._images = [rng.integers(0, 120, (3, 16, 16), dtype=np.uint8) for _ in range(count)]
        self._altitude = rng.uniform(100.0, 300.0, count)
        self._hour = rng.integers(0, 24, count)
        for index in shifted:
            self._images[index] = (self._images[index].astype(np.int16) + 120).clip(0, 255).astype(np.uint8)
            self._altitude[index] = rng.uniform(800.0, 900.0)
        self._factors = factors
        self.metadata: DatasetMetadata = {"id": f"factors-{seed}-{count}", "index2label": {0: "a", 1: "b"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(2, dtype=np.float32)
        target[index % 2] = 1.0
        datum: dict[str, Any] = {"id": index}
        if self._factors:
            datum |= {"altitude": float(self._altitude[index]), "hour": int(self._hour[index])}
        return self._images[index], target, datum
