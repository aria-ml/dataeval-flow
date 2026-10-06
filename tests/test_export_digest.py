"""An export records the digest of what it wrote, read back, and each dataset's provenance (follow-ons spec §6)."""

import json
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import dataset_digest, load_dataset
from dataeval_flow._export import write_export
from dataeval_flow.config import CocoDatasetConfig, PipelineConfig, SourceConfig
from dataeval_flow.config._schemas._export import ExportConfig
from tests.test_export import _plain_config

_RELOAD: dict[str, dict[str, Any]] = {
    "coco": {"dataset_format": "coco"},
    "yolo": {"dataset_format": "yolo"},
    "huggingface_vision": {"dataset_format": "huggingface", "task": "object_detection"},
}


def _last_run(dest: Path) -> dict[str, Any]:
    return json.loads((dest / "provenance.json").read_text())["runs"][-1]


@pytest.mark.parametrize("fmt", sorted(_RELOAD))
def test_the_recorded_digest_is_what_a_reader_loading_the_export_computes(tmp_path: Path, fmt: str) -> None:
    dest = write_export(ExportConfig(name=fmt, source="plain", format=fmt), _plain_config(), tmp_path)  # type: ignore[arg-type]
    reloaded = load_dataset(dest, **_RELOAD[fmt])
    assert _last_run(dest)["digest"]["content"] == dataset_digest(reloaded).content


def test_a_format_flow_cannot_read_back_records_no_digest_and_says_why(tmp_path: Path) -> None:
    dest = write_export(ExportConfig(name="v", source="plain", format="visdrone"), _plain_config(), tmp_path)
    run = _last_run(dest)
    assert run["digest"] is None
    assert run["digest_reason"] == "Flow can't read visdrone back"


def test_an_appended_export_records_the_digest_of_the_whole_destination(tmp_path: Path) -> None:
    write_export(ExportConfig(name="d", source="plain"), _plain_config(), tmp_path)
    dest = write_export(ExportConfig(name="d", source="plain", mode="append"), _plain_config(), tmp_path)
    reloaded = load_dataset(dest, dataset_format="coco")
    assert _last_run(dest)["digest"]["items"] == len(reloaded)
    assert _last_run(dest)["digest"]["content"] == dataset_digest(reloaded).content


def test_each_operand_entry_carries_its_dataset_s_provenance(tmp_path: Path) -> None:
    write_export(ExportConfig(name="first", source="plain"), _plain_config(), tmp_path)
    config = PipelineConfig(
        datasets=[CocoDatasetConfig(name="ds", path="first", provenance={"owner": "survey team"})],
        sources=[SourceConfig(name="again", dataset="ds")],
    )
    dest = write_export(ExportConfig(name="second", source="again"), config, tmp_path, data_dir=tmp_path)
    assert [operand["provenance"] for operand in _last_run(dest)["operands"]] == [{"owner": "survey team"}]


def test_an_export_survives_a_failed_reload_and_records_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("boom")

    monkeypatch.setattr("dataeval_flow._dataset.load_dataset", boom)
    dest = write_export(ExportConfig(name="c", source="plain"), _plain_config(), tmp_path)
    run = _last_run(dest)
    assert run["digest"] is None
    assert "couldn't read coco back" in run["digest_reason"]
    assert "boom" in run["digest_reason"]
    assert "boom" in caplog.text
