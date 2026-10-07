"""The parameter-sweep combinations the agreement golden records, and the counts each gave.

Commit c93bf6a's generator ran the legacy `parameter-sweep` workflow on these combinations, before its removal. The
agreement test runs the same combinations as a quality task's matrix (task-matrix spec §11.3). They agree
exactly on classification data, with `merge_near_duplicates` at its default and no `value_range`: the conditions this
module keeps.
"""

from typing import Any

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages

# Each key is a quality setting, in the sweep's product order. The method and its bound are one setting here: the
# sweep's `outlier_method` and `outlier_threshold` fields, in its product order, as one value per pair.
GRID: dict[str, list[Any]] = {
    "outliers.outlier_threshold": ["zscore", ["zscore", 2.0], "modzscore", ["modzscore", 2.0]],
    "outliers.cluster_threshold": [None, 1.0],
    "duplicates.cluster_sensitivity": [None, 1.0],
}
FLAGS = ["dimension", "pixel", "visual"]
SEED = 0


def dataset() -> ToyImages:
    """A classification toy with a planted exact duplicate, outlier and near duplicate."""
    return ToyImages(count=24, near_duplicate=True)


def matrix_counts() -> list[dict[str, Any]]:
    """Each combination's values and counts as a quality task's matrix gives them, read from each run's steps."""
    import polars as pl

    from dataeval_flow import MatrixResult
    from dataeval_flow.steps import ChainResult

    DatasetCache.clear_instances()
    entry = {"name": "cleaning", "type": "quality", "outliers": {"flags": FLAGS, "outlier_threshold": "zscore"}}
    task = {"name": "t", "workflow": "cleaning", "sources": ["src"], "extractor": "flat", "matrix": GRID}
    config = chain_pipeline(
        workflows=[entry], tasks=[task], datasets={"src": dataset()}, extractor=True, extra={"seed": SEED}
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, MatrixResult)
    assert result.success, result.errors
    counts: list[dict[str, Any]] = []
    for run in result.runs:
        chain = run.result
        assert isinstance(chain, ChainResult)
        outliers = chain.steps["outliers"].output.data()
        dupes = chain.steps["duplicates"].output.data()
        near = dupes.filter((pl.col("dup_type") == "near") & (pl.col("level") == "item")) if len(dupes) else dupes
        counts.append(
            {
                "params": {key: run.values[key] for key in GRID},
                "outlier_count": outliers["item_index"].n_unique() if len(outliers) else 0,
                "near_duplicate_groups": len(near),
            }
        )
    return counts
