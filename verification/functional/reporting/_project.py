"""Small on-disk pipelines for the reporting, metadata, cache and command-line tests.

Every pipeline reads one synthetic ImageFolder dataset (``imgs/``) through the ``quality`` preset and a
``flatten`` extractor, so a run takes about a second and needs no model or network.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from verification.fixtures import write_image_folder

# Images are 32 px square: smaller ones are too small for perceptual hashing, which warns on every item.
IMAGE_SIZE = 32

QUALITY: dict[str, Any] = {
    "name": "q",
    "type": "quality",
    "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"},
}

# Asks for 500 clusters in a dataset of a few items, so its `outliers` step fails when it runs.
FAILING_QUALITY: dict[str, Any] = {
    "name": "q_fail",
    "type": "quality",
    "outliers": {
        "flags": ["pixel"],
        "outlier_threshold": "zscore",
        "cluster_threshold": 3.0,
        "cluster_algorithm": "kmeans",
        "n_clusters": 500,
    },
}

AUDIT: dict[str, Any] = {
    "name": "a",
    "type": "audit",
    "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"},
}


def task(name: str = "clean_task", workflow: str = "q", **fields: Any) -> dict[str, Any]:
    """A task reading the ``main`` source through the ``flat`` extractor."""
    return {"name": name, "workflow": workflow, "sources": "main", "extractor": "flat", **fields}


def write_images(
    root: Path, *, per_class: int = 4, duplicates: int = 0, seed: int = 0, name: str = "imgs", size: int = IMAGE_SIZE
) -> Path:
    """Write a two-class ImageFolder under *root*, with *duplicates* exact copies of class 0's first images."""
    folder = write_image_folder(root / name, n_per_class=per_class, n_classes=2, seed=seed, size=size)
    for i in range(duplicates):
        shutil.copy(folder / "class_0" / f"img_{i}.png", folder / "class_0" / f"copy_{i}.png")
    return folder


def pipeline(
    *,
    tasks: list[dict[str, Any]] | None = None,
    workflows: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
    **result: Any,
) -> dict[str, Any]:
    """A pipeline dict: ``q`` (quality) plus *workflows*, a default task unless *tasks*, ``result:`` from *result*."""
    config: dict[str, Any] = {
        "datasets": [{"name": "ds", "format": "image_folder", "path": "imgs", "infer_labels": True}],
        "sources": [{"name": "main", "dataset": "ds"}],
        "extractors": [{"name": "flat", "model": "flatten", "batch_size": 8}],
        "workflows": [QUALITY, *(workflows or [])],
        "tasks": tasks if tasks is not None else [task()],
        **(extra or {}),
    }
    if result:
        config["result"] = result
    return config


def write_project(
    root: Path,
    *,
    duplicates: int = 0,
    per_class: int = 4,
    size: int = IMAGE_SIZE,
    config: dict[str, Any] | None = None,
    **options: Any,
) -> Path:
    """Write ``imgs/`` and ``config.yaml`` under *root*; return the config path.

    *options* are :func:`pipeline`'s. ``duplicates`` plants exact copies, which the ``quality`` preset reports as a
    warning. *config* replaces the generated pipeline.
    """
    root.mkdir(parents=True, exist_ok=True)
    write_images(root, per_class=per_class, duplicates=duplicates, size=size)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config if config is not None else pipeline(**options)))
    return path


# ---------------------------------------------------------------------------
# Fixtures shared by the four test directories (each one's conftest.py imports them)
# ---------------------------------------------------------------------------


def reset_logging() -> None:
    """Remove the console and file handlers a command line run attached, so the next run attaches its own."""
    import dataeval_flow._logging as flow_logging

    flow_logging._initialized = False
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_dataeval_flow_console", False) or getattr(handler, "_dataeval_flow_file", False):
            handler.close()
            root.removeHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger("dataeval_flow").setLevel(logging.NOTSET)


@pytest.fixture(autouse=True)
def isolated_state() -> Iterator[None]:
    """Compute on the CPU, start with empty in-memory caches, and undo the logging a command line run sets up."""
    import dataeval.config

    from dataeval_flow import set_device
    from dataeval_flow._cache import DatasetCache

    set_device("cpu")
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()
    set_device(None)
    dataeval.config.set_device(None)
    reset_logging()


@dataclass
class Invocation:
    """What one command-line invocation left behind."""

    code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Callable[..., Invocation]:
    """Run ``dataeval-flow <args>`` in this process, with only the ``DATAEVAL_*`` variables given in ``env``.

    Returns the exit code and what the command printed. Running in-process spares each call the seconds it takes to
    import the stack; the real console script and ``python -m`` are exercised separately.
    """
    from dataeval_flow.__main__ import main

    def invoke(*args: str | Path, env: dict[str, str] | None = None) -> Invocation:
        reset_logging()
        for name in [name for name in os.environ if name.startswith("DATAEVAL_")]:
            monkeypatch.delenv(name)
        for name, value in (env or {}).items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(sys, "argv", ["dataeval_flow", *map(str, args)])
        with pytest.raises(SystemExit) as stopped:
            main()
        captured = capsys.readouterr()
        code = stopped.value.code
        return Invocation(code if isinstance(code, int) else 0, captured.out, captured.err)

    return invoke


def run_project(root: Path, **options: Any) -> dict[str, Any]:
    """Write a project under *root* and run its tasks on the CPU; the results are keyed by task name."""
    from dataeval_flow import load_config, run_tasks, set_device

    set_device("cpu")
    config = write_project(root, **options)
    return run_tasks(load_config(config), data_dir=root)
