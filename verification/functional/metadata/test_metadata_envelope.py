"""TC-13-1 — the metadata envelope every result carries."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import yaml

import dataeval_flow
from dataeval_flow import ResultMetadata, load_config, run_tasks, set_device
from dataeval_flow.steps import ChainResult
from verification.functional.reporting._project import pipeline, run_project, write_images, write_project

pytestmark = pytest.mark.required

SEED = 17


class Seeded:
    """A `quality` result on a pipeline that sets a seed, with when it started and finished."""

    def __init__(self, root: Path) -> None:
        self.started = datetime.now(UTC)
        self.result: ChainResult = run_project(root, extra={"seed": SEED})["clean_task"]
        self.finished = datetime.now(UTC)


@pytest.fixture(scope="module")
def seeded(tmp_path_factory: pytest.TempPathFactory) -> Seeded:
    return Seeded(tmp_path_factory.mktemp("seeded"))


@pytest.fixture(scope="module")
def conformed(tmp_path_factory: pytest.TempPathFactory) -> ChainResult:
    """A `quality` result whose source reads through a view that renames both classes to one."""
    root = tmp_path_factory.mktemp("conformed")
    config = pipeline(
        extra={
            "views": [
                {
                    "name": "coarse",
                    "operations": [
                        {
                            "type": "Relabel",
                            "params": {
                                "class_remap": {"class_0": "animal", "class_1": "animal"},
                                "target": ["animal", "thing"],
                            },
                        }
                    ],
                }
            ]
        }
    )
    config["sources"] = [{"name": "main", "dataset": "ds", "view": "coarse"}]
    write_project(root, config=config)
    set_device("cpu")
    return run_tasks(load_config(root / "config.yaml"), data_dir=root)["clean_task"]


@pytest.fixture(scope="module")
def two_source_evaluator(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The `duplicates` evaluator over two sources, from two datasets."""
    root = tmp_path_factory.mktemp("two_sources")
    write_images(root, name="imgs")
    write_images(root, name="imgs_b", seed=1)
    config = pipeline(tasks=[{"name": "dup", "evaluator": "dupes", "sources": ["a", "b"]}])
    config["datasets"] = [
        {"name": "a_ds", "format": "image_folder", "path": "imgs", "infer_labels": True},
        {"name": "b_ds", "format": "image_folder", "path": "imgs_b", "infer_labels": True},
    ]
    config["sources"] = [{"name": "a", "dataset": "a_ds"}, {"name": "b", "dataset": "b_ds"}]
    config["evaluators"] = [{"name": "dupes", "type": "duplicates"}]
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    set_device("cpu")
    return run_tasks(load_config(root / "config.yaml"), data_dir=root)["dup"]


@pytest.fixture(scope="module")
def small_images(tmp_path_factory: pytest.TempPathFactory) -> ChainResult:
    """A `quality` result over 8-pixel images, too small for perceptual hashing."""
    return run_project(tmp_path_factory.mktemp("small"), size=8)["clean_task"]


class TestEnvelope:
    def test_tool_and_versions_identify_what_made_the_result(self, seeded: Seeded) -> None:
        meta = seeded.result.metadata
        assert meta.version == "1.0"
        assert meta.tool == "dataeval-flow"
        assert meta.tool_version == dataeval_flow.__version__
        assert meta.tool_version

    def test_timestamp_is_utc_and_falls_inside_the_run(self, seeded: Seeded) -> None:
        stamp = seeded.result.metadata.timestamp
        assert stamp.tzinfo is not None
        assert stamp.utcoffset() == timedelta(0)
        assert seeded.started <= stamp <= seeded.finished

    def test_execution_time_is_positive_and_no_longer_than_the_run(self, seeded: Seeded) -> None:
        seconds = seeded.result.metadata.execution_time_s
        assert seconds is not None
        assert 0 < seconds <= (seeded.finished - seeded.started).total_seconds()

    def test_device_names_where_the_run_computed(self, seeded: Seeded) -> None:
        assert seeded.result.metadata.device == "cpu"

    def test_library_versions_record_the_installed_libraries(self, seeded: Seeded) -> None:
        versions = seeded.result.metadata.library_versions
        assert {"dataeval", "numpy", "pillow"} <= set(versions)
        for name, recorded in versions.items():
            assert recorded == version(name), name

    def test_sources_datasets_and_extractor_are_named(self, seeded: Seeded) -> None:
        meta = seeded.result.metadata
        assert meta.dataset_id == "ds"
        assert list(meta.source_descriptions) == ["main (ds)"]
        assert meta.model_id == "flat (flatten)"
        assert meta.label_source == "filepath"
        assert meta.selection_id is None  # the source reads through no view
        assert meta.preprocessor_id is None

    def test_resolved_config_holds_the_sources_workflow_extractor_and_seed(self, seeded: Seeded) -> None:
        resolved = seeded.result.metadata.resolved_config
        assert [source["name"] for source in resolved["sources"]] == ["main"]
        assert resolved["sources"][0]["dataset_config"]["path"] == "imgs"
        assert resolved["workflow"]["type"] == "quality"
        assert resolved["workflow"]["outliers"]["flags"] == ["pixel"]
        assert resolved["extractor"]["model"] == "flatten"
        assert resolved["seed"] == SEED

    def test_to_dict_and_export_carry_the_whole_envelope(self, seeded: Seeded) -> None:
        payload = seeded.result.to_dict()["metadata"]
        assert isinstance(payload, dict)
        assert set(ResultMetadata.model_fields) <= set(payload)
        assert payload["tool_version"] == dataeval_flow.__version__
        exported = json.loads(seeded.result.export())["metadata"]
        assert exported == payload
        assert datetime.fromisoformat(exported["timestamp"]).tzinfo is not None

    def test_envelope_records_how_metadata_factors_were_encoded(self, seeded: Seeded) -> None:
        meta = seeded.result.metadata
        assert meta.encoding_digest
        assert meta.metadata_binning is not None
        assert meta.metadata_binning["encoding_digest"] == meta.encoding_digest
        assert "filename" in meta.metadata_binning["factors"]

    def test_library_diagnostics_are_recorded(self, small_images: ChainResult, seeded: Seeded) -> None:
        assert any("too small for perceptual hashing" in message for message in small_images.metadata.diagnostics)
        assert not any("perceptual hashing" in message for message in seeded.result.metadata.diagnostics)


class TestLabelSpace:
    def test_a_result_that_conformed_no_labels_records_no_label_space(self, seeded: Seeded) -> None:
        assert list(seeded.result.metadata.label_space) == []
        assert seeded.result.metadata.label_space_digest is None

    def test_a_conformed_source_is_recorded_with_its_remap_target_and_digest(self, conformed: ChainResult) -> None:
        meta = conformed.metadata
        (record,) = meta.label_space
        assert record.source == "main"
        assert dict(record.class_remap) == {"class_0": "animal", "class_1": "animal"}
        assert list(record.target) == ["animal", "thing"]
        assert len(record.digest) == 12
        assert meta.label_space_digest == record.digest
        assert meta.selection_id == "coarse"
        assert list(meta.source_descriptions) == ["main (ds[coarse])"]

    def test_the_label_space_reaches_the_exported_envelope(self, conformed: ChainResult) -> None:
        exported = json.loads(conformed.export())["metadata"]
        assert exported["label_space"][0]["target"] == ["animal", "thing"]
        assert exported["label_space_digest"] == conformed.metadata.label_space_digest


class TestEvaluatorEnvelope:
    def test_dataset_id_lists_every_source_in_task_order(self, two_source_evaluator: Any) -> None:
        meta = two_source_evaluator.metadata
        assert two_source_evaluator.kind == "evaluator"
        assert meta.dataset_id == "a_ds,b_ds"
        assert list(meta.source_descriptions) == ["a (a_ds)", "b (b_ds)"]
        assert [source["name"] for source in meta.resolved_config["sources"]] == ["a", "b"]
        assert meta.resolved_config["evaluator"]["type"] == "duplicates"

    def test_the_envelope_names_the_evaluator_and_the_dataeval_call(self, two_source_evaluator: Any) -> None:
        meta = two_source_evaluator.metadata
        assert meta.evaluator == "duplicates"
        assert meta.dataeval.name.endswith("Duplicates.from_stats")
        assert meta.dataeval.version == version("dataeval")
        assert meta.tool_version == dataeval_flow.__version__
        assert meta.library_versions["dataeval"] == version("dataeval")
        assert meta.device == "cpu"
        assert meta.model_id is None  # no extractor
