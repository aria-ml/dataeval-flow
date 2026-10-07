"""The data-cleaning runs the agreement golden records: one pipeline per case, and the findings it gives.

The same cases serve the generator, run once on the legacy workflow, and the agreement test, which runs whatever
`data-cleaning` names today: the legacy workflow before its port, the preset after it (spec §10.3).
"""

from collections.abc import Callable
from typing import Any

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import Finding
from tests.chain_toys import ToyDetections, chain_pipeline
from tests.evaluator_toys import ToyImages

_BASE: dict[str, Any] = {
    "name": "cleaning",
    "type": "data-cleaning",
    "outliers": {"flags": ["pixel", "visual"], "outlier_threshold": "zscore"},
}
_LENIENT = {
    "image-duplicates": {"exact": 50.0, "near": 50.0},
    "image-outliers": {"warning": 50.0},
    "target-outliers": {"warning": 50.0},
    "class-outliers": {"warning": 50.0},
}


def findings(dataset: Any, *, extractor: bool = False, **settings: Any) -> list[Finding]:
    """The findings a `data-cleaning` task gives on `dataset`, with `settings` over the base entry."""
    # Two toy datasets can share an id, and a dataset's cache is one per id within a process.
    DatasetCache.clear_instances()
    task: dict[str, Any] = {"name": "t", "workflow": "cleaning", "sources": ["src"]}
    if extractor:
        task["extractor"] = "flat"
    config = chain_pipeline(
        workflows=[{**_BASE, **settings}], tasks=[task], datasets={"src": dataset}, extractor=extractor
    )
    result = run_tasks(config)["t"]
    assert result.success, result.errors
    return list(result.findings)  # type: ignore[attr-defined]


def detections() -> ToyDetections:
    """20 detection images with 28 boxes: a copied image (9 of 4) and two bright boxes (item 3 box 0, item 8 box 1)."""
    return ToyDetections(
        [[0, 1], [1], [0], [1, 1], [0]] * 4,
        {0: "car", 1: "van", 2: "bus"},
        duplicate_of={9: 4},
        bright={(3, 0), (8, 1)},
    )


CASES: dict[str, Callable[[], list[Finding]]] = {
    "classification": lambda: findings(ToyImages(count=24)),
    "unlabelled": lambda: findings(ToyImages(count=24, labeled=False)),
    "detection": lambda: findings(detections()),
    "no_duplicates": lambda: findings(ToyImages(count=6)),
    "lenient_thresholds": lambda: findings(ToyImages(count=24), checks=_LENIENT),
    "cluster_mode": lambda: findings(
        ToyImages(count=40),
        extractor=True,
        outliers={**_BASE["outliers"], "cluster_threshold": 2.0},
        duplicates={"cluster_sensitivity": 1.0},
    ),
}
