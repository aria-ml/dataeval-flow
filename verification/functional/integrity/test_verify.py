"""TC-22-2 — `dataeval-flow verify`: whether a source still holds the items a run's manifest records."""

from __future__ import annotations

from pathlib import Path

import pytest

from verification.functional.integrity._data import edit_annotations, invert_image
from verification.functional.integrity.conftest import RecordedRun
from verification.helpers import run_cli

pytestmark = pytest.mark.required


def _verify(recorded: RecordedRun, data: Path, manifest: Path | None = None, source: str = "data"):
    return run_cli(
        "verify",
        str(manifest or recorded.manifest),
        "--config",
        str(recorded.config),
        "--source",
        source,
        "--data",
        str(data),
    )


class TestVerify:
    def test_unchanged_data_matches_its_manifest(self, recorded: RecordedRun) -> None:
        proc = _verify(recorded, recorded.data)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "OK: source 'data' holds the 6 items" in proc.stdout

    def test_an_edited_image_is_named_as_changed(self, recorded: RecordedRun, tmp_path: Path) -> None:
        data = recorded.copy_data(tmp_path / "data")
        invert_image(data / "fixture", 4)
        proc = _verify(recorded, data)
        assert proc.returncode == 1
        assert "MISMATCH" in proc.stdout
        assert "Changed: 1 (4)" in proc.stdout
        assert "Missing" not in proc.stdout
        assert "Added" not in proc.stdout

    def test_a_removed_item_is_named_as_missing(self, recorded: RecordedRun, tmp_path: Path) -> None:
        data = recorded.copy_data(tmp_path / "data")

        def drop_last(coco: dict) -> None:
            coco["images"] = [image for image in coco["images"] if image["id"] != 5]
            coco["annotations"] = [annotation for annotation in coco["annotations"] if annotation["image_id"] != 5]

        edit_annotations(data / "fixture", drop_last)
        proc = _verify(recorded, data)
        assert proc.returncode == 1
        assert "Missing: 1 (5)" in proc.stdout

    def test_an_added_item_is_named_as_added(self, recorded: RecordedRun, tmp_path: Path) -> None:
        data = recorded.copy_data(tmp_path / "data")
        (data / "fixture" / "images" / "000009.png").write_bytes(
            (data / "fixture" / "images" / "000000.png").read_bytes()
        )
        invert_image(data / "fixture", 9)

        def add(coco: dict) -> None:
            coco["images"].append({"id": 9, "file_name": "images/000009.png", "width": 48, "height": 32})
            coco["annotations"].append({"id": 9, "image_id": 9, "category_id": 0, "bbox": [1, 2, 5, 4], "iscrowd": 0})

        edit_annotations(data / "fixture", add)
        proc = _verify(recorded, data)
        assert proc.returncode == 1
        assert "Added: 1 (9)" in proc.stdout

    def test_renamed_classes_are_reported(self, recorded: RecordedRun, tmp_path: Path) -> None:
        data = recorded.copy_data(tmp_path / "data")

        def rename(coco: dict) -> None:
            coco["categories"][1]["name"] = "kayak"

        edit_annotations(data / "fixture", rename)
        proc = _verify(recorded, data)
        assert proc.returncode == 1
        assert "The class names differ." in proc.stdout

    def test_a_manifest_that_cannot_be_read_is_an_error(self, recorded: RecordedRun, tmp_path: Path) -> None:
        proc = _verify(recorded, recorded.data, manifest=tmp_path / "missing.json")
        assert proc.returncode == 1
        assert "ERROR" in proc.stderr

    def test_a_source_the_config_does_not_define_is_an_error(self, recorded: RecordedRun) -> None:
        proc = _verify(recorded, recorded.data, source="nothing")
        assert proc.returncode == 1
        assert "ERROR" in proc.stderr
        assert "Traceback" not in proc.stderr
