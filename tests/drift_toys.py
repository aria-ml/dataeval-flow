"""Small datasets for drift tests: labelled classification with chosen class sizes, and detection."""

from typing import Any

import numpy as np

CLASSES = {0: "cat", 1: "dog", 2: "bird"}


class ClassImages:
    """Image classification over `CLASSES`: `counts` items of each class, in class order, random 3x16x16 images.

    `bright` lifts every pixel by 100, out of the distribution of an unbrightened set, and `bright_classes` lifts only
    the items of those classes; `labeled=False` gives every item an empty target.
    """

    def __init__(
        self,
        counts: dict[int, int],
        seed: int = 0,
        *,
        bright: bool = False,
        bright_classes: frozenset[int] | set[int] = frozenset(),
        labeled: bool = True,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._labels = [cls for cls, n in sorted(counts.items()) for _ in range(n)]
        self._images = [
            np.clip(
                rng.integers(0, 155, (3, 16, 16)) + (100 if bright or label in bright_classes else 0) + 30 * label,
                0,
                255,
            ).astype(np.uint8)
            for label in self._labels
        ]
        self._labeled = labeled
        self.metadata: dict[str, Any] = {"id": f"classes-{seed}-{len(self._labels)}", "index2label": dict(CLASSES)}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        if not self._labeled:
            return self._images[index], np.zeros(0, dtype=np.float32), {"id": index}
        target = np.zeros(len(CLASSES), dtype=np.float32)
        target[self._labels[index]] = 1.0
        return self._images[index], target, {"id": index}


class _BoxTarget:
    """A duck-typed detection target; `dataeval.types` exports no constructible one."""

    def __init__(self, labels: Any, boxes: Any, scores: Any) -> None:
        self.labels, self.boxes, self.scores = labels, boxes, scores


class BoxImages:
    """Object detection over `CLASSES`: `count` images, each with two boxes of classes `i % 3` and `(i + 1) % 3`."""

    def __init__(self, count: int = 40, seed: int = 0, *, bright: bool = False) -> None:
        rng = np.random.default_rng(seed)
        shift = 100 if bright else 0
        self._images = [
            np.clip(rng.integers(0, 155, (3, 16, 16)) + shift, 0, 255).astype(np.uint8) for _ in range(count)
        ]
        self.metadata: dict[str, Any] = {"id": f"boxes-{seed}-{count}", "index2label": dict(CLASSES)}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = _BoxTarget(
            np.array([index % 3, (index + 1) % 3], dtype=np.intp),
            np.array([[1.0, 1.0, 8.0, 8.0], [8.0, 8.0, 15.0, 15.0]], dtype=np.float32),
            np.ones(2, dtype=np.float32),
        )
        return self._images[index], target, {"id": index}
