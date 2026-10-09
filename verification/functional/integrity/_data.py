"""Synthetic datasets and pipelines shared by the integrity and service tests (tiny, deterministic, on disk)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

CLASSES = {0: "boat", 1: "swimmer"}


class Items:
    """A dataset of exactly the ``(image, target, metadata)`` items given, with its class names."""

    def __init__(self, items: Sequence[Any], index2label: Mapping[int, str] | None = None) -> None:
        self._items = list(items)
        self.metadata = {"id": "items", "index2label": dict(CLASSES if index2label is None else index2label)}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Any:
        return self._items[index]


def make_items(count: int = 6) -> list[tuple[Any, Any, dict[str, Any]]]:
    """Distinct classification items, each with a unique metadata ``id``."""
    rng = np.random.default_rng(0)
    items = []
    for index in range(count):
        target = np.zeros(2, dtype=np.float32)
        target[index % 2] = 1.0
        items.append((rng.integers(0, 255, (3, 8, 8), dtype=np.uint8), target, {"id": index, "site": "north"}))
    return items


def write_coco(root: Path, name: str = "fixture", count: int = 6) -> Path:
    """A COCO detection dataset ``root/name`` of `count` 32x48 images, one box each, with an ``altitude`` per image."""
    folder = root / name
    (folder / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    images, annotations = [], []
    for index in range(count):
        Image.fromarray(rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)).save(folder / "images" / f"{index:06d}.png")
        images.append(
            {
                "id": index,
                "file_name": f"images/{index:06d}.png",
                "width": 48,
                "height": 32,
                "altitude": float(index + 1),
            }
        )
        annotations.append(
            {"id": index, "image_id": index, "category_id": index % 2, "bbox": [1, 2, 5, 4], "iscrowd": 0}
        )
    categories = [{"id": key, "name": value} for key, value in CLASSES.items()]
    (folder / "instances.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )
    return folder


def invert_image(folder: Path, index: int) -> None:
    """Replace image `index` of a dataset written by :func:`write_coco` with its photographic negative."""
    path = folder / "images" / f"{index:06d}.png"
    Image.fromarray(255 - np.asarray(Image.open(path))).save(path)


def edit_annotations(folder: Path, edit: Any) -> None:
    """Rewrite a COCO dataset's ``instances.json`` after `edit` changed the loaded JSON in place."""
    path = folder / "instances.json"
    coco = json.loads(path.read_text())
    edit(coco)
    path.write_text(json.dumps(coco))


def digest_pipeline(**extra: Any) -> dict[str, Any]:
    """A pipeline with one ``content-digest`` task, ``digest``, over the COCO dataset ``fixture``."""
    return {
        "datasets": [{"name": "ds", "format": "coco", "path": "fixture"}],
        "sources": [{"name": "data", "dataset": "ds"}],
        "evaluators": [{"name": "digest", "type": "content-digest"}],
        "tasks": [{"name": "digest", "evaluator": "digest", "sources": "data"}],
        **extra,
    }


def write_pipeline(path: Path, pipeline: dict[str, Any]) -> Path:
    """Write `pipeline` as YAML to `path` and return the path."""
    path.write_text(yaml.safe_dump(pipeline))
    return path
