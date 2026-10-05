"""The metadata-triage runs the agreement golden records: one pipeline per case.

The generator ran each case once on the legacy workflow and recorded what it produced. The agreement test runs the same
pipelines on whatever `metadata-triage` names today (spec §10.10).
"""

from collections.abc import Callable
from typing import Any

from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import PipelineConfig
from tests.chain_toys import chain_pipeline
from tests.triage_toys import AltitudeDataset, LatitudeDataset, MixedWeightDataset, OcclusionDataset

# A policy that reads `weight`'s numerals wearing commas as numbers, as triage's own suggestion for it does.
WEIGHTS: dict[str, Any] = {
    "name": "weights",
    "corrections": [{"kind": "parse_value", "factor": "weight", "drop": [","]}],
}

CASES: dict[str, tuple[dict[str, Any], Callable[[], Any]]] = {
    "mixed_weight": ({}, MixedWeightDataset),
    "unverified": ({"verify": False}, MixedWeightDataset),
    "named_policy": ({"metadata": "weights"}, MixedWeightDataset),
    "altitude": ({}, AltitudeDataset),
    "latitude": ({}, LatitudeDataset),
    "occlusion": ({}, OcclusionDataset),
}


def pipeline(name: str) -> PipelineConfig:
    """Case `name`'s pipeline: one `metadata-triage` task over its dataset, with the `weights` policy defined."""
    settings, dataset = CASES[name]
    # Two toy datasets can share an id, and a dataset's cache is one per id within a process.
    DatasetCache.clear_instances()
    return chain_pipeline(
        workflows=[{"name": "triage", "type": "metadata-triage", **settings}],
        tasks=[{"name": "t", "workflow": "triage", "sources": ["src"]}],
        datasets={"src": dataset()},
        extra={"metadata": [WEIGHTS]},
    )
