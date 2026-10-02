"""The parameter-sweep combinations the agreement golden records, and the counts each gave.

The generator ran `legacy_counts` once on the legacy `parameter-sweep` workflow, before its removal. The agreement test
runs the same combinations as a data-cleaning task's matrix (task-matrix spec §11.3). They agree exactly on
classification data, with `duplicate_merge_near` at its default and no `value_range`: the conditions this module keeps.
"""

from typing import Any

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages

# Each key is a parameter-sweep field and the data-cleaning setting of the same name, in the sweep's product order.
GRID: dict[str, list[Any]] = {
    "outlier_method": ["zscore", "modzscore"],
    "outlier_threshold": [None, 2.0],
    "outlier_cluster_threshold": [None, 1.0],
    "duplicate_cluster_sensitivity": [None, 1.0],
}
FLAGS = ["dimension", "pixel", "visual"]
SEED = 0


def dataset() -> ToyImages:
    """A classification toy with a planted exact duplicate, outlier and near duplicate."""
    return ToyImages(count=24, near_duplicate=True)


def legacy_counts() -> list[dict[str, Any]]:
    """Each combination's swept values and counts, as the legacy workflow gave them, in its order."""
    from dataeval_flow.workflows.parameter_sweep import ParameterSweepConfig

    DatasetCache.clear_instances()
    sweep = ParameterSweepConfig(name="sweep", outlier_flags=FLAGS, **GRID)
    task = {"name": "t", "workflow": "sweep", "sources": ["src"], "extractor": "flat"}
    config = chain_pipeline(
        workflows=[sweep], tasks=[task], datasets={"src": dataset()}, extractor=True, extra={"seed": SEED}
    )
    result = run_tasks(config)["t"]
    assert result.success, result.errors
    return [
        {
            "params": {key: run.params[key] for key in GRID},
            "outlier_count": run.outlier_count,
            "near_duplicate_groups": run.near_duplicate_groups,
        }
        for run in result.output.raw.results
    ]
