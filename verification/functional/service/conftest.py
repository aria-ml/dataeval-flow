"""A data root with a small COCO dataset, the pipelines a client would submit for it, and a wait for a run's status."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from verification.functional.integrity._data import write_coco

DATASET = {
    "datasets": [{"name": "ds", "format": "coco", "path": "fixture"}],
    "sources": [{"name": "data", "dataset": "ds"}],
}


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A data root holding a 12-image COCO detection dataset, ``fixture``."""
    root = tmp_path / "data"
    write_coco(root, count=12)
    return root


@pytest.fixture
def pipeline() -> dict[str, Any]:
    """Two quick tasks over the dataset: its content digest and its label health."""
    return {
        **DATASET,
        "evaluators": [{"name": "digest", "type": "content-digest"}, {"name": "labels", "type": "label-health"}],
        "tasks": [
            {"name": "digest", "evaluator": "digest", "sources": "data"},
            {"name": "labels", "evaluator": "labels", "sources": "data"},
        ],
    }


@pytest.fixture
def wait_for():
    """Wait for a run to reach one of `statuses`, failing the test once `timeout` seconds pass."""

    def wait(store: Any, run_id: str, statuses: set[str], timeout: float = 60) -> dict[str, Any]:
        end = time.monotonic() + timeout
        record = store.get(run_id)
        while record["status"] not in statuses:
            if time.monotonic() > end:
                pytest.fail(f"Run did not reach {statuses}: {record}")
            time.sleep(0.03)
            record = store.get(run_id)
        return record

    return wait
