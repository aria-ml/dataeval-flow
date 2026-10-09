"""TC-22-3 — Dataset export: `exports:` writes COCO, YOLO, Hugging Face and VisDrone datasets with provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import dataset_digest, load_dataset
from dataeval_flow.config import ExportConfig
from verification.fixtures import write_image_folder
from verification.functional.integrity._data import digest_pipeline, write_coco, write_pipeline
from verification.helpers import run_cli

pytestmark = pytest.mark.required

_FORMATS = ("coco", "yolo", "huggingface_vision", "visdrone")
# What `load_dataset` takes to read an export back, for the formats Flow can read.
_RELOAD: dict[str, dict[str, Any]] = {
    "coco": {"dataset_format": "coco"},
    "yolo": {"dataset_format": "yolo"},
    "huggingface_vision": {"dataset_format": "huggingface", "task": "object_detection"},
}
# One file each format's layout must hold.
_LAYOUT = {
    "coco": "annotations/instances.json",
    "yolo": "data.yaml",
    "huggingface_vision": "data/metadata.jsonl",
    "visdrone": "VisDrone2019-DET-train/annotations/000000.txt",
}


@dataclass(frozen=True)
class Exported:
    """A data root, the pipeline over it, and the output of the run that exported one dataset per format."""

    data: Path
    config: Path
    output: Path

    def dataset(self, name: str) -> Path:
        return self.output / "datasets" / name

    def runs(self, name: str) -> list[dict[str, Any]]:
        return json.loads((self.dataset(name) / "provenance.json").read_text())["runs"]


def _pipeline(exports: list[dict[str, Any]]) -> dict[str, Any]:
    pipeline = digest_pipeline(exports=exports)
    pipeline["datasets"][0]["provenance"] = {"owner": "Survey team", "collected": "2025-06-01", "frames": 6}
    return pipeline


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Exported:
    """One run exporting the COCO fixture in every format, beside a content-digest task."""
    root = tmp_path_factory.mktemp("exported")
    write_coco(root)
    config = write_pipeline(
        root / "pipeline.yaml", _pipeline([{"name": f, "source": "data", "format": f} for f in _FORMATS])
    )
    proc = run_cli("-c", str(config), "-d", str(root), "-o", str(root / "out"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return Exported(root, config, root / "out")


class TestExportFormats:
    @pytest.mark.parametrize("fmt", _FORMATS)
    def test_each_format_is_written_under_datasets_with_its_layout(self, exported: Exported, fmt: str) -> None:
        assert (exported.dataset(fmt) / _LAYOUT[fmt]).is_file()
        images = [p for p in exported.dataset(fmt).rglob("*.png")]
        assert len(images) == 6

    @pytest.mark.parametrize("fmt", sorted(_RELOAD))
    def test_a_written_dataset_loads_back_with_every_item(self, exported: Exported, fmt: str) -> None:
        assert len(load_dataset(exported.dataset(fmt), **_RELOAD[fmt])) == 6

    def test_a_pipeline_with_exports_and_no_tasks_still_writes_them(self, tmp_path: Path) -> None:
        write_coco(tmp_path)
        pipeline = _pipeline([{"name": "only", "source": "data"}])
        del pipeline["tasks"], pipeline["evaluators"]
        config = write_pipeline(tmp_path / "pipeline.yaml", pipeline)
        proc = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(tmp_path / "out"))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (tmp_path / "out" / "datasets" / "only" / "annotations" / "instances.json").is_file()

    def test_a_run_with_no_output_directory_writes_no_export(self, tmp_path: Path) -> None:
        write_coco(tmp_path)
        config = write_pipeline(tmp_path / "pipeline.yaml", _pipeline([{"name": "ghost", "source": "data"}]))
        proc = run_cli("-c", str(config), "-d", str(tmp_path), cwd=tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert sorted(path.name for path in tmp_path.iterdir()) == ["fixture", "pipeline.yaml"]


class TestExportProvenance:
    @pytest.mark.parametrize("fmt", _FORMATS)
    def test_every_format_gets_a_provenance_record_of_one_write(self, exported: Exported, fmt: str) -> None:
        (run,) = exported.runs(fmt)
        assert run["tool"] == "dataeval-flow"
        assert run["tool_version"]
        assert run["created"]
        assert run["source"] == "data"
        assert [operand["dataset"] for operand in run["operands"]] == ["ds"]

    @pytest.mark.parametrize("fmt", sorted(_RELOAD))
    def test_the_recorded_digest_is_what_a_reader_loading_the_export_computes(
        self, exported: Exported, fmt: str
    ) -> None:
        (run,) = exported.runs(fmt)
        reloaded = dataset_digest(load_dataset(exported.dataset(fmt), **_RELOAD[fmt]))
        assert run["digest"]["content"] == reloaded.content
        assert run["digest"]["metadata"] == reloaded.metadata
        assert run["digest"]["items"] == 6

    def test_a_format_flow_cannot_read_back_records_no_digest_and_says_why(self, exported: Exported) -> None:
        (run,) = exported.runs("visdrone")
        assert run["digest"] is None
        assert run["digest_reason"] == "Flow can't read visdrone back"

    def test_each_operand_carries_its_dataset_s_provenance_facts(self, exported: Exported) -> None:
        (run,) = exported.runs("coco")
        (operand,) = run["operands"]
        assert operand["provenance"] == {"owner": "Survey team", "collected": "2025-06-01", "frames": 6}

    def test_coco_also_embeds_the_record_in_its_own_info_block(self, exported: Exported) -> None:
        coco = json.loads((exported.dataset("coco") / "annotations" / "instances.json").read_text())
        assert coco["info"]["source"] == "data"


class TestExportModes:
    @staticmethod
    def _again(exported: Exported, tmp_path: Path, mode: str | None) -> tuple[Any, Path]:
        """Run the export of `exported` a second time, into a copy of its output, with `mode`."""
        import shutil

        out = shutil.copytree(exported.output, tmp_path / "out")
        export = {"name": "coco", "source": "data", **({"mode": mode} if mode else {})}
        config = write_pipeline(tmp_path / "again.yaml", _pipeline([export]))
        return run_cli("-c", str(config), "-d", str(exported.data), "-o", str(out)), out

    def test_an_occupied_destination_is_refused_and_left_as_it_was(self, exported: Exported, tmp_path: Path) -> None:
        before = (exported.dataset("coco") / "provenance.json").read_text()
        proc, out = self._again(exported, tmp_path, None)
        assert proc.returncode != 0
        assert "already exists and is not empty" in proc.stdout + proc.stderr
        assert (out / "datasets" / "coco" / "provenance.json").read_text() == before

    def test_replace_clears_the_destination_and_records_one_write(self, exported: Exported, tmp_path: Path) -> None:
        stale = exported.dataset("coco") / "stale.txt"
        stale.write_text("left over")
        try:
            proc, out = self._again(exported, tmp_path, "replace")
        finally:
            stale.unlink()
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert not (out / "datasets" / "coco" / "stale.txt").exists()
        assert len(json.loads((out / "datasets" / "coco" / "provenance.json").read_text())["runs"]) == 1

    def test_append_writes_into_the_destination_and_records_both_writes(
        self, exported: Exported, tmp_path: Path
    ) -> None:
        proc, out = self._again(exported, tmp_path, "append")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        runs = json.loads((out / "datasets" / "coco" / "provenance.json").read_text())["runs"]
        assert len(runs) == 2
        reloaded = dataset_digest(load_dataset(out / "datasets" / "coco", dataset_format="coco"))
        assert runs[-1]["digest"]["content"] == reloaded.content

    def test_a_failed_export_does_not_cost_the_run_its_other_exports(self, tmp_path: Path) -> None:
        write_coco(tmp_path)
        write_image_folder(tmp_path / "classes", n_per_class=2, n_classes=2)
        pipeline = _pipeline([{"name": "bad", "source": "labelled"}, {"name": "good", "source": "data"}])
        pipeline["datasets"].append(
            {"name": "folder", "format": "image_folder", "path": "classes", "infer_labels": True}
        )
        pipeline["sources"].append({"name": "labelled", "dataset": "folder"})
        config = write_pipeline(tmp_path / "pipeline.yaml", pipeline)
        proc = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(tmp_path / "out"))
        assert proc.returncode != 0
        assert "Export 'bad' failed" in proc.stdout + proc.stderr
        assert (tmp_path / "out" / "datasets" / "good" / "annotations" / "instances.json").is_file()
        assert not (tmp_path / "out" / "datasets" / "bad" / "provenance.json").exists()


class TestExportConfig:
    def test_defaults_are_coco_and_refusing_an_occupied_destination(self) -> None:
        export = ExportConfig(name="out", source="data")
        assert (export.format, export.mode) == ("coco", "error")

    @pytest.mark.parametrize("name", ["a/b", "a\\b", ".", "..", ""])
    def test_a_name_that_is_not_one_directory_segment_is_refused(self, name: str) -> None:
        with pytest.raises(ValidationError, match="Export name"):
            ExportConfig(name=name, source="data")

    def test_an_unknown_format_or_mode_is_refused_at_load(self) -> None:
        with pytest.raises(ValidationError, match="format"):
            ExportConfig(name="out", source="data", format="tfrecord")  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="mode"):
            ExportConfig(name="out", source="data", mode="overwrite")  # type: ignore[arg-type]
