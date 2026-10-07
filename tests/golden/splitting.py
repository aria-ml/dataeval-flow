"""The splits runs the agreement golden records: one pipeline per case, in legacy's settings and the
preset's.

The generator ran each case once on the legacy workflow and recorded what it produced; the agreement test runs the
preset's settings on whatever `splits` names today (data-splitting spec §9).
"""

from typing import Any

import numpy as np

from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import PipelineConfig
from tests.chain_toys import chain_pipeline

SEED = 0


class SplitImages:
    """`count` 3x8x8 images over three classes in proportion 5:3:2, each with a `site` (nine values, not following the
    class) and an `angle`, for `split_on`, `balance` and `diversity`."""

    def __init__(self, count: int = 90) -> None:
        rng = np.random.default_rng(0)
        self._images = [rng.integers(0, 255, (3, 8, 8), dtype=np.uint8) for _ in range(count)]
        self._labels = [0 if index % 10 < 5 else 1 if index % 10 < 8 else 2 for index in range(count)]
        self.metadata = {"id": f"split-{count}", "index2label": {0: "cat", 1: "dog", 2: "bird"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(3, dtype=np.float32)
        target[self._labels[index]] = 1.0
        return self._images[index], target, {"site": f"site-{index % 9}", "angle": float(index % 5)}


# Each case: (legacy settings, preset settings, with the flatten extractor).
CASES: dict[str, tuple[dict[str, Any], dict[str, Any], bool]] = {
    "defaults": ({}, {}, False),
    "kfold": ({"num_folds": 3}, {"folds": 3}, False),
    "rebalance_interclass": ({"rebalance_method": "interclass"}, {"rebalance": "interclass"}, False),
    "kfold_rebalance_global": (
        {"num_folds": 3, "rebalance_method": "global"},
        {"folds": 3, "rebalance": "global"},
        False,
    ),
    "split_on": ({"split_on": ["site"]}, {"split_on": ["site"]}, False),
    "unstratified": ({"stratify": False}, {"stratify": False}, False),
    "kfold_no_test": ({"num_folds": 3, "test_frac": 0.0}, {"folds": 3, "test_frac": 0.0}, False),
    "coverage": ({"coverage_percent": 0.1, "num_observations": 3}, {}, False),
}


# Legacy splits's class-imbalance limit, which judged no info band.
_IMBALANCE = {"class-imbalance": {"warning": 10.0, "info": None}}


def pipeline(name: str, *, legacy: bool) -> PipelineConfig:
    """Case `name`'s pipeline over `SplitImages`, seed 0: one `splits` task in legacy's settings, or task `t`
    in the preset's with task `b`, a `bias` entry on the same source judging the whole set's class balance at
    legacy's limit."""
    legacy_settings, settings, extractor = CASES[name]
    DatasetCache.clear_instances()
    task: dict[str, Any] = {"name": "t", "workflow": "split", "sources": ["src"]}
    if extractor:
        task["extractor"] = "flat"
    workflows: list[dict[str, Any]] = [{"name": "split", "type": "splits", **(legacy_settings if legacy else settings)}]
    tasks = [task]
    if not legacy:
        workflows.append({"name": "bias", "type": "bias", "checks": _IMBALANCE})
        tasks.append({"name": "b", "workflow": "bias", "sources": ["src"]})
    return chain_pipeline(
        workflows=workflows, tasks=tasks, datasets={"src": SplitImages()}, extractor=extractor, extra={"seed": SEED}
    )
