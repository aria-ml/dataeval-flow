"""A small labelled COCO dataset on disk, the pipeline a client would submit for it, and a wait for a run's status."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def pipeline() -> dict:
    """A quality and a triage task over the whole fixture dataset."""
    return {
        "seed": 0,
        "datasets": [{"name": "fixture", "format": "coco", "path": "fixture"}],
        "sources": [{"name": "data", "dataset": "fixture"}],
        "workflows": [
            {
                "name": "quality",
                "type": "quality",
                "outliers": {"flags": ["dimension", "pixel", "visual"], "outlier_threshold": "adaptive"},
            },
            {"name": "metadata", "type": "triage"},
        ],
        "tasks": [
            {"name": "quality", "workflow": "quality", "sources": "data"},
            {"name": "metadata", "workflow": "metadata", "sources": "data"},
        ],
    }


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A data root holding a 12-image COCO detection dataset: two exact duplicates, a rare class, and telemetry."""
    root = tmp_path / "data"
    (root / "fixture" / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    duplicate = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    images, annotations = [], []
    for index in range(12):
        pixels = duplicate if index < 2 else rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
        name = f"images/{index:06d}.png"
        Image.fromarray(pixels).save(root / "fixture" / name)
        images.append(
            {
                "id": index,
                "file_name": name,
                "width": 48,
                "height": 32,
                "altitude": float(index + 1),
                "drone": "test",
                "latitude": 53.7 if index < 10 else "N",
            }
        )
        annotations.append(
            {"id": index, "image_id": index, "category_id": 0 if index < 11 else 1, "bbox": [1, 2, 5, 4], "iscrowd": 0}
        )
    categories = [{"id": 0, "name": "boat"}, {"id": 1, "name": "swimmer"}]
    (root / "fixture" / "instances.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )
    return root


@pytest.fixture
def wait_for():
    """Wait for a run to reach one of `statuses`, failing the test once `timeout` seconds pass."""

    def wait(store, run_id: str, statuses: set[str], timeout: float = 15) -> dict:
        end = time.monotonic() + timeout
        record = store.get(run_id)
        while record["status"] not in statuses:
            if time.monotonic() > end:
                pytest.fail(f"Run did not reach {statuses}: {record}")
            time.sleep(0.03)
            record = store.get(run_id)
        return record

    return wait
