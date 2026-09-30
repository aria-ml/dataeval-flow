"""Datasets whose metadata triage has something to say about: a mixed column, an unpinned cut, and problem values
placed by item and by box (spec §10.10)."""

from typing import Any

import numpy as np
from dataeval.protocols import DatasetMetadata


class MixedWeightDataset:
    """Classification items whose ``weight`` reading mixes numerals with numerals wearing
    commas.

    Built by walking the dataset; ``Metadata.from_factors`` refuses a mixed-dtype column
    outright (``reject_mixed_values``). The held-back path this fixture needs exists only
    for metadata read off a dataset — see ``tests/test_binning.py::_MixedDataset``,
    which this mirrors.
    """

    def __init__(self, n: int = 60) -> None:
        self._n = n
        rng = np.random.default_rng(0)
        self._weight = rng.integers(1000, 9000, n)

    @property
    def metadata(self) -> DatasetMetadata:
        return {"id": "mixed-weight", "index2label": {0: "cat", 1: "dog"}}

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        one_hot = np.zeros(2, dtype=np.float32)
        one_hot[index % 2] = 1.0
        image = np.zeros((3, 8, 8), dtype=np.float32)
        raw = int(self._weight[index])
        # Every tenth reading is a numeral wearing commas rather than a plain number.
        weight: Any = f"{raw:,}" if index % 10 == 0 else raw
        datum: dict[str, Any] = {"id": index, "weight": weight}
        return image, one_hot, datum


class AltitudeDataset:
    """Classification items with a continuous ``altitude`` factor nobody pinned.

    All-numeric: the point is a column that reads cleanly and lands as ``unbinned``
    (a cut DataEval derived from this draw), so its suggestion is a bin count,
    not a correction.
    """

    def __init__(self, n: int = 60) -> None:
        self._n = n
        rng = np.random.default_rng(1)
        self._altitude = rng.uniform(0.0, 1000.0, n)

    @property
    def metadata(self) -> DatasetMetadata:
        return {"id": "altitude", "index2label": {0: "cat", 1: "dog"}}

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        one_hot = np.zeros(2, dtype=np.float32)
        one_hot[index % 2] = 1.0
        image = np.zeros((3, 8, 8), dtype=np.float32)
        datum: dict[str, Any] = {"id": index, "altitude": float(self._altitude[index])}
        return image, one_hot, datum


class LatitudeDataset:
    """Classification items whose ``latitude`` reads as a number, except where it reads ``'N'`` or ``'S'``."""

    metadata: DatasetMetadata = DatasetMetadata({"id": "latitude", "index2label": {0: "cat", 1: "dog"}})

    def __len__(self) -> int:
        return 60

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        one_hot = np.zeros(2, dtype=np.float32)
        one_hot[index % 2] = 1.0
        latitude: Any = "N" if index % 7 == 3 else "S" if index == 20 else float(index)
        return np.zeros((3, 8, 8), dtype=np.float32), one_hot, {"id": index, "latitude": latitude}


class OcclusionDataset:
    """Detections, two boxes an image, whose ``occlusion`` reads ``'high'`` on every fifth image's second box."""

    metadata: DatasetMetadata = DatasetMetadata({"id": "occlusion", "index2label": {0: "cat", 1: "dog"}})

    def __len__(self) -> int:
        return 30

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        from tests.test_coverage_workflow import _Target

        occlusion: list[Any] = [0.1 * index, "high" if index % 5 == 0 else 0.2]
        target = _Target([[2, 2, 20, 20], [8, 8, 30, 30]], [0, 1])
        return np.zeros((3, 32, 32), dtype=np.uint8), target, {"id": index, "occlusion": occlusion}
