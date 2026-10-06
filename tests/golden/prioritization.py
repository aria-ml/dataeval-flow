"""The data-prioritization runs the agreement golden records: one pipeline per case.

The generator ran each case once on the legacy workflow and recorded what it produced; the agreement test runs the same
pipelines on whatever `data-prioritization` names today (spec §10.9). A case with cleaning runs the steps the preset's
removed `cleaning:` expanded to, in a custom workflow, then the preset as its step `rank`.
"""

from collections.abc import Callable
from typing import Any

from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import PipelineConfig
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages

_KNN: dict[str, Any] = {"method": "knn", "k": 3}
_BASE: dict[str, Any] = {"name": "prio", "type": "data-prioritization", "prioritization": _KNN}
_CLEANING: dict[str, Any] = {"dup_types": ["exact", "near"]}
_EVALUATORS = [
    {"name": "outliers", "type": "outliers", "flags": ["pixel", "visual"], "outlier_threshold": "zscore"},
    {"name": "duplicates", "type": "duplicates", "merge_near_duplicates": True},
]


def _pair(**pool: Any) -> Callable[[], dict[str, Any]]:
    """A reference of 16 toy images and one pool of 20, built with `pool`'s settings."""
    return lambda: {"ref": ToyImages(count=16), "pool": ToyImages(count=20, seed=1, **pool)}


CASES: dict[str, tuple[dict[str, Any], Callable[[], dict[str, Any]]]] = {
    "plain": ({}, _pair()),
    "easy_first": ({"prioritization": {**_KNN, "order": "easy_first"}}, _pair()),
    "cleaned": ({"cleaning": _CLEANING}, _pair()),
    "near_duplicates": ({"cleaning": _CLEANING}, _pair(near_duplicate=True)),
    "exact_only": ({"cleaning": {"dup_types": ["exact"]}}, _pair(near_duplicate=True)),
    "unlabelled_pool": ({"cleaning": _CLEANING}, _pair(labeled=False)),
    "two_pools": (
        {"cleaning": _CLEANING},
        lambda: {"ref": ToyImages(count=16), "p1": ToyImages(count=20, seed=1), "p2": ToyImages(count=12, seed=2)},
    ),
}


def _cleaned(prefix: str, source: str, dup_types: list[str]) -> list[dict[str, Any]]:
    """`source` without its outliers and duplicates, as `cleaning:` cleaned it: steps `outliers-<prefix>`,
    `duplicates-<prefix>` and `<prefix>-clean`."""
    return [
        {"name": f"outliers-{prefix}", "evaluator": "outliers", "input": source},
        {"name": f"duplicates-{prefix}", "evaluator": "duplicates", "input": source},
        {
            "name": f"{prefix}-clean",
            "transform": "remove",
            "input": source,
            "plans": {
                f"duplicates-{prefix}": {"dup_types": dup_types, "keep": "first"},
                f"outliers-{prefix}": {"min_flags": 1},
            },
        },
    ]


def pipeline(name: str) -> PipelineConfig:
    """Case `name`'s pipeline: one task over its sources, reference first, with an extractor. It runs
    `data-prioritization`, or with cleaning, a custom workflow that cleans each source and runs it as step `rank`."""
    settings, datasets = CASES[name]
    sources = datasets()
    # Two toy datasets can share an id, and a dataset's cache is one per id within a process.
    DatasetCache.clear_instances()
    settings = dict(settings)
    cleaning = settings.pop("cleaning", None)
    workflows: list[dict[str, Any]] = [{**_BASE, **settings}]
    if cleaning is not None:
        workflows.append(
            {
                "name": "cleaned",
                "inputs": ["reference", {"name": "pools", "list": True}],
                "steps": [
                    *_cleaned("reference", "reference", cleaning["dup_types"]),
                    *_cleaned("pool", "pools", cleaning["dup_types"]),
                    {"name": "rank", "workflow": "prio", "input": ["reference-clean", "pool-clean"]},
                ],
            }
        )
    return chain_pipeline(
        workflows=workflows,
        evaluators=_EVALUATORS if cleaning is not None else (),
        tasks=[
            {
                "name": "t",
                "workflow": "prio" if cleaning is None else "cleaned",
                "sources": list(sources),
                "extractor": "flat",
            }
        ],
        datasets=sources,
        extractor=True,
    )
