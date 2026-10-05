"""The data-coverage runs the agreement golden records: one pipeline per case, in legacy's settings and the preset's.

The generator ran each case once on legacy data-coverage and recorded what it produced; the agreement test runs the
preset's settings (coverage spec §8.2, §17). No existing toy combines factors, a declared class with no samples and
low-dimensional images, and `ToyDetections`' boxes are all one size, so the golden has its own.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dataeval_flow import PipelineConfig
from dataeval_flow._cache import DatasetCache
from tests.chain_toys import chain_pipeline

_ANIMALS = {0: "cat", 1: "dog", 2: "bird", 3: "owl"}
"""`owl` is declared and never labelled."""


class CoverageImages:
    """3x8x8 images over cat, dog and bird in proportion 5:3:2, `owl` declared and never labelled; `site` follows the
    class except for every tenth item, `angle` does not. 192 flat dimensions keep naive coverage under its overflow."""

    def __init__(self, count: int = 90, *, labeled: bool = True, factors: bool = True) -> None:
        rng = np.random.default_rng(0)
        self._images = [rng.integers(0, 255, (3, 8, 8), dtype=np.uint8) for _ in range(count)]
        self._labels = [0 if index % 10 < 5 else 1 if index % 10 < 8 else 2 for index in range(count)]
        self._labeled, self._factors = labeled, factors
        self.metadata = {"id": f"coverage-{count}-{int(labeled)}{int(factors)}", "index2label": dict(_ANIMALS)}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        if self._labeled:
            target = np.zeros(len(_ANIMALS), dtype=np.float32)
            target[self._labels[index]] = 1.0
        else:
            target = np.zeros(0, dtype=np.float32)
        label = self._labels[index]
        site = f"site-{(label + 1) % 3}" if index % 10 == 9 else f"site-{label}"
        factors = {"site": site, "angle": float(index % 5)} if self._factors else {}
        return self._images[index], target, {"id": index, **factors}


class _Boxes:
    """An object-detection target: boxes, their labels, and one score row per box."""

    def __init__(self, boxes: Any, labels: Any) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.scores = np.ones((len(self.labels), 3), dtype=np.float32)


class CoverageDetections:
    """3x16x16 images; each holds a 3x3 box and a 6x8 box (car and van, alternating), every fourth image none; `bus`
    is declared and never labelled. A `min_size` of 4 drops the small boxes only."""

    def __init__(self, count: int = 40) -> None:
        rng = np.random.default_rng(1)
        self._images = [rng.integers(0, 255, (3, 16, 16), dtype=np.uint8) for _ in range(count)]
        self.metadata = {"id": f"coverage-boxes-{count}", "index2label": {0: "car", 1: "van", 2: "bus"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        if index % 4 == 3:
            target = _Boxes(np.zeros((0, 4)), [])
        else:
            target = _Boxes([[1, 1, 4, 4], [6, 2, 12, 10]], [index % 2, (index + 1) % 2])
        return self._images[index], target, {"id": index, "site": ("north", "south")[index % 2]}


@dataclass(frozen=True)
class Case:
    """One golden case: its dataset, legacy's settings, the preset's, and whether the task names an extractor."""

    dataset: Any
    legacy: dict[str, Any] = field(default_factory=dict)
    preset: dict[str, Any] = field(default_factory=dict)
    extractor: bool = False


CASES: dict[str, Case] = {
    "no_extractor": Case(lambda: CoverageImages()),
    "adaptive": Case(lambda: CoverageImages(), extractor=True),
    "naive": Case(lambda: CoverageImages(), {"coverage_method": "naive"}, {"coverage": {"method": "naive"}}, True),
    "detection": Case(
        lambda: CoverageDetections(),
        {"crop_padding": 0.1, "crop_min_size": 4, "num_observations": 10, "min_class_samples": 5},
        {"crops": {"padding": 0.1, "min_size": 4}, "coverage": {"num_observations": 10, "min_class_samples": 5}},
        True,
    ),
    "few": Case(lambda: CoverageImages(count=40), extractor=True),
    "gaps_off": Case(lambda: CoverageImages(), {"run_gap_analysis": False}, {"gaps": None}),
    "shannon": Case(lambda: CoverageImages(), {"diversity_method": "shannon"}, {"diversity": "shannon"}),
    "expected": Case(
        lambda: CoverageImages(),
        {"ontology_expected": {"bird": 0.4, "nope": 0.1}},
        {"expected": {"bird": 0.4, "nope": 0.1}},
    ),
    "unlabelled": Case(lambda: CoverageImages(labeled=False, factors=False)),
    "no_factors": Case(lambda: CoverageImages(factors=False)),
    "strict": Case(
        lambda: CoverageImages(),
        {
            "health_thresholds": {
                "class_imbalance_ratio": 2.0,
                "gap_count": 1,
                "completeness_score": 0.9,
                "min_dispersion": 1.5,
            }
        },
        {
            "health_thresholds": {
                "class-imbalance": {"ratio": 2.0},
                "factor-coverage-gaps": {"count": 1},
                "dimensional-completeness": {"warning": 0.9, "info": 0.9},
                "class-coverage": {"dispersion": 1.5},
            }
        },
        True,
    ),
}


def pipeline(name: str, *, legacy: bool) -> PipelineConfig:
    """Case `name` as a one-task pipeline: legacy data-coverage's settings, or the preset's."""
    case = CASES[name]
    DatasetCache.clear_instances()
    entry = {"name": "w", "type": "data-coverage", **(case.legacy if legacy else case.preset)}
    task: dict[str, Any] = {"name": "t", "workflow": "w", "sources": ["src"]}
    if case.extractor:
        task["extractor"] = "flat"
    return chain_pipeline(workflows=[entry], tasks=[task], datasets={"src": case.dataset()}, extractor=case.extractor)
