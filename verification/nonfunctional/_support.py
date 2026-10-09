"""Project builders shared by the non-functional and container tests (not a test module).

A *project* is a data root holding ``imgs/`` (a small ImageFolder) and ``config.yaml`` with ``quality`` tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from verification.fixtures import write_image_folder

IMAGE_SIZE = 32


def write_skewed_folder(root: Path) -> Path:
    """A two-class ImageFolder whose first class holds a single image, so a stratified split cannot be made."""
    rng = np.random.default_rng(3)
    for name, count in (("class_0", 1), ("class_1", 7)):
        (root / name).mkdir(parents=True)
        for i in range(count):
            pixels = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / name / f"img_{i}.png")
    return root


def pipeline_dict(
    *,
    n_tasks: int = 1,
    failing_task: bool = False,
    seed: int | None = None,
    max_processes: int | None = None,
    view: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The pipeline ``write_project`` writes, as a plain dict that ``PipelineConfig`` validates.

    ``n_tasks`` ordinary ``quality`` tasks are named ``clean_task``, ``clean_task_2``, ...; ``failing_task`` puts a
    ``bad_task`` first that fails while it runs; ``view`` is a list of view operations applied to the source.
    """
    sources: list[dict[str, Any]] = [{"name": "main", "dataset": "ds"}]
    views: list[dict[str, Any]] = []
    if view is not None:
        views = [{"name": "picked", "operations": view}]
        sources = [{"name": "main", "dataset": "ds", "view": "picked"}]
    datasets: list[dict[str, Any]] = [{"name": "ds", "format": "image_folder", "path": "imgs", "infer_labels": True}]
    workflows: list[dict[str, Any]] = [
        {"name": "clean", "type": "quality", "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}
    ]
    tasks: list[dict[str, Any]] = [
        {
            "name": "clean_task" if i == 1 else f"clean_task_{i}",
            "workflow": "clean",
            "sources": "main",
            "extractor": "flat",
        }
        for i in range(1, n_tasks + 1)
    ]
    if failing_task:
        datasets.append({"name": "skewed_ds", "format": "image_folder", "path": "skewed", "infer_labels": True})
        sources.append({"name": "skewed", "dataset": "skewed_ds"})
        workflows.append({"name": "split", "type": "splits"})
        tasks.insert(0, {"name": "bad_task", "workflow": "split", "sources": "skewed"})
    config: dict[str, Any] = {
        "datasets": datasets,
        "sources": sources,
        "extractors": [{"name": "flat", "model": "flatten", "batch_size": 8}],
        "workflows": workflows,
        "tasks": tasks,
    }
    if views:
        config["views"] = views
    if seed is not None:
        config["seed"] = seed
    if max_processes is not None:
        config["max_processes"] = max_processes
    return config


def write_project(
    root: Path,
    *,
    n_tasks: int = 1,
    failing_task: bool = False,
    seed: int | None = None,
    max_processes: int | None = None,
    view: list[dict[str, Any]] | None = None,
) -> Path:
    """Write ``imgs/`` and ``config.yaml`` (see :func:`pipeline_dict`) under *root* and return the config path."""
    write_image_folder(root / "imgs", n_per_class=8, n_classes=2, size=IMAGE_SIZE)
    if failing_task:
        write_skewed_folder(root / "skewed")
    path = root / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            pipeline_dict(n_tasks=n_tasks, failing_task=failing_task, seed=seed, max_processes=max_processes, view=view)
        )
    )
    return path


# Fields that differ between two runs of the same configuration: when they ran, how long they took, and the
# diagnostics DataEval warns about once per process.
VOLATILE = frozenset({"timestamp", "execution_time_s", "execution_time", "execution_duration", "elapsed", "elapsed_s"})
VOLATILE_DIAGNOSTICS = "diagnostics"


def stable(value: Any) -> Any:
    """A result's ``to_dict()`` (or any part of it) without the fields that differ between identical runs."""
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in VOLATILE and k != VOLATILE_DIAGNOSTICS}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value
