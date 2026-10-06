"""Small labelled images exercise the real loader without network downloads."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def pipeline_for(dataset: str = "fixture") -> dict:
    """A quality and a triage task over the whole fixture, as a client would submit them."""
    return {
        "seed": 0,
        "datasets": [{"name": dataset, "format": "coco", "path": dataset}],
        "sources": [{"name": "data", "dataset": dataset}],
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
def staged(tmp_path: Path):
    """A 12-image COCO detection dataset under a data root: two exact duplicates, one rare class, telemetry."""
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
    return root, pipeline_for()
