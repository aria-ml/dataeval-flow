"""The data-prioritization runs the agreement golden records: one pipeline per case.

The generator ran each case once on the legacy workflow and recorded what it produced; the agreement test runs the same
pipelines on whatever `data-prioritization` names today (spec §10.9).
"""

from collections.abc import Callable
from typing import Any

from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import PipelineConfig
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages

_BASE: dict[str, Any] = {"name": "prio", "type": "data-prioritization", "method": "knn", "k": 3}
_CLEANING: dict[str, Any] = {"outliers": {"flags": ["pixel", "visual"], "outlier_threshold": "zscore"}}


def _pair(**pool: Any) -> Callable[[], dict[str, Any]]:
    """A reference of 16 toy images and one pool of 20, built with `pool`'s settings."""
    return lambda: {"ref": ToyImages(count=16), "pool": ToyImages(count=20, seed=1, **pool)}


CASES: dict[str, tuple[dict[str, Any], Callable[[], dict[str, Any]]]] = {
    "plain": ({}, _pair()),
    "easy_first": ({"order": "easy_first"}, _pair()),
    "cleaned": ({"cleaning": _CLEANING}, _pair()),
    "near_duplicates": ({"cleaning": _CLEANING}, _pair(near_duplicate=True)),
    "exact_only": ({"cleaning": {**_CLEANING, "dup_types": ["exact"]}}, _pair(near_duplicate=True)),
    "unlabelled_pool": ({"cleaning": _CLEANING}, _pair(labeled=False)),
    "two_pools": (
        {"cleaning": _CLEANING},
        lambda: {"ref": ToyImages(count=16), "p1": ToyImages(count=20, seed=1), "p2": ToyImages(count=12, seed=2)},
    ),
}


def pipeline(name: str) -> PipelineConfig:
    """Case `name`'s pipeline: one `data-prioritization` task over its sources, reference first, with an extractor."""
    settings, datasets = CASES[name]
    sources = datasets()
    # Two toy datasets can share an id, and a dataset's cache is one per id within a process.
    DatasetCache.clear_instances()
    return chain_pipeline(
        workflows=[{**_BASE, **settings}],
        tasks=[{"name": "t", "workflow": "prio", "sources": list(sources), "extractor": "flat"}],
        datasets=sources,
        extractor=True,
    )
