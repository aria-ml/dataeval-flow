"""A pipeline run that recorded a manifest of its dataset, for the tests that check or compare against it."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from verification.functional.integrity._data import digest_pipeline, write_coco, write_pipeline
from verification.helpers import run_cli


@dataclass(frozen=True)
class RecordedRun:
    """A data root, the pipeline over it, and the output directory of one run of that pipeline."""

    data: Path
    config: Path
    output: Path

    @property
    def manifest(self) -> Path:
        return self.output / "results" / "manifests" / "digest" / "content-digest.json"

    def copy_data(self, destination: Path) -> Path:
        """A private copy of the data root, so a test can edit the dataset without touching the recorded run's."""
        shutil.copytree(self.data, destination)
        return destination


@pytest.fixture(scope="session")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> RecordedRun:
    """Run a ``content-digest`` task once over a six-image COCO dataset."""
    root = tmp_path_factory.mktemp("recorded")
    write_coco(root)
    config = write_pipeline(root / "pipeline.yaml", digest_pipeline())
    output = root / "out"
    proc = run_cli("-c", str(config), "-d", str(root), "-o", str(output))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return RecordedRun(root, config, output)
