"""The data-analysis runs the audit golden records: one pipeline per case, in data-analysis's settings and audit's.

The generator ran each case once on data-analysis and recorded what it produced; the agreement test runs the audit
preset's settings (audit spec §12). Sources are named `train`, `val` and `test`, so the first is train.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import PipelineConfig
from tests.chain_toys import ToyDetections, chain_pipeline
from tests.evaluator_toys import Items, ToyFactors, ToyImages


class ClassImages:
    """Images whose targets cycle through `classes`, with class names a, b and c declared whether used or not."""

    def __init__(self, count: int, classes: tuple[int, ...], seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._images = [rng.integers(0, 255, (3, 16, 16), dtype=np.uint8) for _ in range(count)]
        self._classes = classes
        self.metadata = {"id": f"classes-{seed}-{count}", "index2label": {0: "a", 1: "b", 2: "c"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(3, dtype=np.float32)
        target[self._classes[index % len(self._classes)]] = 1.0
        return self._images[index], target, {"id": index}


def _leaky_test() -> Items:
    """Train's first six items, then six of seed 3's: exact duplicates cross the split."""
    train = ToyImages(12, 0, near_duplicate=True)
    fresh = ToyImages(12, 3)
    return Items([train[i] for i in range(6)] + [fresh[i] for i in range(6)])


def _detections(dataset_id: str) -> ToyDetections:
    labels = [[0] if i % 2 == 0 else [1, 0] for i in range(12)]
    return ToyDetections(labels, {0: "a", 1: "b"}, duplicate_of={5: 0}, dataset_id=dataset_id)


@dataclass(frozen=True)
class Case:
    datasets: Callable[[], dict[str, Any]]
    """The sources, in order; the first is train."""
    legacy: dict[str, Any]
    """The data-analysis entry."""
    preset: dict[str, Any]
    """The audit entry."""
    extractor: bool


_LEGACY: dict[str, Any] = {"outlier_method": "zscore", "outlier_flags": ["pixel", "visual"]}
_PRESET: dict[str, Any] = {"outliers": {"flags": ["pixel", "visual"], "outlier_threshold": "zscore"}}


def _two() -> dict[str, Any]:
    return {"train": ToyImages(12, 0), "test": ToyImages(12, 1)}


CASES: dict[str, Case] = {
    "one": Case(lambda: {"train": ToyImages(12, 0)}, _LEGACY, _PRESET, False),
    "two": Case(_two, _LEGACY, _PRESET, False),
    "three": Case(
        lambda: {"train": ToyImages(12, 0), "val": ToyImages(12, 1), "test": ToyImages(12, 2)}, _LEGACY, _PRESET, False
    ),
    "duplicates": Case(
        lambda: {"train": ToyImages(12, 0, near_duplicate=True), "test": _leaky_test()}, _LEGACY, _PRESET, False
    ),
    "extractor": Case(
        _two,
        {**_LEGACY, "divergence_method": "mst"},
        {**_PRESET, "divergence": {"method": "mst"}},
        True,
    ),
    "detection": Case(
        lambda: {"train": _detections("det-train"), "test": _detections("det-test")}, _LEGACY, _PRESET, False
    ),
    "factors": Case(
        lambda: {"train": ToyFactors(60), "test": ToyFactors(30)},
        {**_LEGACY, "balance": True, "diversity_method": "simpson"},
        {**_PRESET, "diversity": {"method": "simpson"}},
        False,
    ),
    "one_class": Case(
        lambda: {"train": ClassImages(12, (0, 1), 0), "test": ClassImages(12, (0, 1, 2), 1)}, _LEGACY, _PRESET, False
    ),
}


def pipeline(name: str, *, legacy: bool) -> PipelineConfig:
    case = CASES[name]
    DatasetCache.clear_instances()
    entry = {"name": "w", "type": "data-analysis" if legacy else "audit", **(case.legacy if legacy else case.preset)}
    datasets = case.datasets()
    task: dict[str, Any] = {"name": "t", "workflow": "w", "sources": list(datasets)}
    if case.extractor:
        task["extractor"] = "flat"
    return chain_pipeline(workflows=[entry], tasks=[task], datasets=datasets, extractor=case.extractor)
