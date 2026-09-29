"""The export step: a chain's Dataset written to disk, recorded in the result (spec §6.3)."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._graph import GraphError
from dataeval_flow.config import ExportConfig
from dataeval_flow.steps import ChainResult
from tests.chain_toys import ToyDetections, chain_pipeline
from tests.evaluator_toys import ToyImages

_DETECTIONS = ToyDetections([[0, 1], [1], [0], [1, 1], [0]], {0: "car", 1: "person"})


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _config(steps: list[dict[str, Any]], datasets: dict[str, Any] | None = None, **extra: Any):
    workflow = {"name": "w", "inputs": ["a"], "steps": steps}
    datasets = datasets or {"src": _DETECTIONS}
    return chain_pipeline(
        workflows=[workflow],
        tasks=[{"name": "t", "workflow": "w", "sources": list(datasets)}],
        datasets=datasets,
        **extra,
    )


def _instances(root: Path) -> dict[str, Any]:
    return json.loads((root / "annotations" / "instances.json").read_text())


def test_exporting_a_chain_input_writes_it_under_task_dot_step(tmp_path: Path) -> None:
    config = _config([{"name": "corpus", "transform": "export", "input": "a"}])
    result = run_tasks(config, output_dir=tmp_path)["t"]
    assert isinstance(result, ChainResult)
    record = result.steps["corpus"].output
    assert record.path == str(tmp_path / "datasets" / "t.corpus")
    written = _instances(tmp_path / "datasets" / "t.corpus")
    assert len(written["images"]) == 5
    assert len(written["annotations"]) == 7
    assert (tmp_path / "datasets" / "t.corpus" / "provenance.json").is_file()


def test_exporting_a_derived_dataset_encodes_its_pixels(tmp_path: Path) -> None:
    steps = [
        {"name": "few", "transform": "view", "input": "a", "operations": [{"type": "Limit", "params": {"size": 3}}]},
        {"name": "corpus", "transform": "export", "input": "few", "to": "first3"},
    ]
    result = run_tasks(_config(steps), output_dir=tmp_path)["t"]
    assert isinstance(result, ChainResult)
    assert result.success
    written = _instances(tmp_path / "datasets" / "first3")
    assert len(written["images"]) == 3
    assert result.steps["corpus"].output.items == 3
    provenance = json.loads((tmp_path / "datasets" / "first3" / "provenance.json").read_text())["runs"][-1]
    assert [record["name"] for record in provenance["lineage"]] == ["few", "a"]


def test_without_an_output_directory_the_export_is_skipped_and_the_task_passes() -> None:
    config = _config([{"name": "corpus", "transform": "export", "input": "a"}])
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success
    status_and_reason = (result.steps["corpus"].status, result.steps["corpus"].reason)
    assert status_and_reason == ("skipped", "the run has no output directory")


def test_the_record_is_in_the_json(tmp_path: Path) -> None:
    config = _config([{"name": "corpus", "transform": "export", "input": "a"}])
    payload = cast("dict[str, Any]", run_tasks(config, output_dir=tmp_path)["t"].to_dict())
    output = payload["steps"]["corpus"]["output"]
    assert (output["format"], output["mode"], output["items"]) == ("coco", "error", 5)
    assert output["provenance"]["tool"] == "dataeval-flow"


def test_two_exports_to_one_destination_fail_the_load() -> None:
    steps = [
        {"name": "one", "transform": "export", "input": "a", "to": "same"},
        {"name": "two", "transform": "export", "input": "a", "to": "same"},
    ]
    with pytest.raises(ValidationError, match="both export to `datasets/same`"):
        _config(steps)


def test_an_export_step_colliding_with_a_top_level_export_fails_the_load() -> None:
    step = {"name": "x", "transform": "export", "input": "a", "to": "corpus"}
    extra = {"exports": [ExportConfig(name="corpus", source="src")]}
    with pytest.raises(ValidationError, match="export 'corpus' and task 't' step 'x' both export"):
        _config([step], extra=extra)


def test_exporting_a_classification_dataset_fails_before_running(tmp_path: Path) -> None:
    config = _config([{"name": "corpus", "transform": "export", "input": "a"}], datasets={"src": ToyImages()})
    with pytest.raises(GraphError, match="`input` takes object_detection"):
        run_tasks(config, output_dir=tmp_path)
