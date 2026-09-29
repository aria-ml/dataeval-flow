"""The export step: a chain's Dataset written to disk, recorded in the result (spec §6.3)."""

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._graph import GraphError
from dataeval_flow.config import ExportConfig
from dataeval_flow.evaluators.scope import LabelAlignmentConfig
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


def test_the_provenance_records_each_conform_on_the_way_and_only_those(tmp_path: Path) -> None:
    steps = [
        {"name": "aligned", "evaluator": "align", "input": "a"},
        {"name": "c", "transform": "conform", "input": "a", "alignment": "aligned"},
        {"name": "few", "transform": "view", "input": "a", "operations": [{"type": "Limit", "params": {"size": 3}}]},
        {"name": "corpus", "transform": "export", "input": "c"},
        {"name": "plain", "transform": "export", "input": "few", "to": "plain"},
    ]
    concepts = [{"id": "Vehicle", "label": "Vehicle", "synonyms": ["car"]}, {"id": "Person", "label": "Person"}]
    ontology = {"name": "vehicles", "concepts": concepts}
    config = _config(
        steps,
        evaluators=[LabelAlignmentConfig(name="align", ontology="vehicles")],
        extra={"ontologies": [ontology]},
    )
    result = run_tasks(config, output_dir=tmp_path)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    conformed = json.loads((tmp_path / "datasets" / "t.corpus" / "provenance.json").read_text())["runs"][-1]
    (conform,) = conformed["conforms"]
    assert (conform["source"], conform["ontology"], conform["ontology_digest"]) == ("c", "vehicles", "7322513e6772")
    assert (conform["class_remap"], conform["target"]) == (
        {"car": "Vehicle", "person": "Person"},
        ["Vehicle", "Person"],
    )
    plain = json.loads((tmp_path / "datasets" / "plain" / "provenance.json").read_text())["runs"][-1]
    assert plain["conforms"] == []


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


_FIRST_TWO = {"type": "Limit", "params": {"size": 2}}


@pytest.mark.parametrize(
    ("before", "read", "element"),
    [
        ([], "all", "all[<key>]"),
        ([{"name": "folds", "transform": "kfold", "input": "one", "folds": 2}], "folds.train", "folds.train[0]"),
        ([{"name": "few", "transform": "view", "input": "all", "operations": [_FIRST_TWO]}], "few", "few[<key>]"),
    ],
    ids=["list input", "list output", "broadcast"],
)
def test_an_export_step_reading_a_list_fails_the_load(before: list[dict[str, Any]], read: str, element: str) -> None:
    steps = [*before, {"name": "corpus", "transform": "export", "input": read}]
    workflow = {"name": "w", "inputs": ["one", {"name": "all", "list": True}], "steps": steps}
    message = (
        f"Step 'corpus' reads `{read}`, a list, but transform 'export' writes one Dataset to one place, so it does "
        f"not run once per element: name one element, such as `{element}`."
    )
    with pytest.raises(ValidationError, match=re.escape(message)):
        chain_pipeline(workflows=[workflow], datasets={"src": _DETECTIONS})


def test_an_export_step_reading_one_element_of_a_list_writes_it(tmp_path: Path) -> None:
    steps = [
        {"name": "folds", "transform": "kfold", "input": "a", "folds": 2},
        {"name": "corpus", "transform": "export", "input": "folds.train[0]"},
    ]
    result = run_tasks(_config(steps), output_dir=tmp_path)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    items = {record.name: record.items for record in result.metadata.lineage}
    assert (items["folds.train[0]"], result.steps["corpus"].output.items) == (2, 2)
    assert len(_instances(tmp_path / "datasets" / "t.corpus")["images"]) == 2


@pytest.mark.parametrize(
    ("to", "message"),
    [
        (".", "Export destination '.' names a relative path rather than a directory"),
        ("..", r"Export destination '\.\.' names a relative path rather than a directory"),
        ("a/b", "Export destination 'a/b' must be one directory segment"),
    ],
)
def test_an_export_destination_that_is_not_one_plain_directory_fails_the_load(to: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _config([{"name": "corpus", "transform": "export", "input": "a", "to": to}])


def test_an_export_destination_that_is_one_plain_directory_loads() -> None:
    config = _config([{"name": "corpus", "transform": "export", "input": "a", "to": "corpus_v2"}])
    (step,) = config.workflows[0].steps  # type: ignore[index,union-attr]
    assert step.config.to == "corpus_v2"  # type: ignore[union-attr]


def test_an_export_step_colliding_with_a_top_level_export_fails_the_load() -> None:
    step = {"name": "x", "transform": "export", "input": "a", "to": "corpus"}
    extra = {"exports": [ExportConfig(name="corpus", source="src")]}
    with pytest.raises(ValidationError, match="export 'corpus' and task 't' step 'x' both export"):
        _config([step], extra=extra)


def test_exporting_a_classification_dataset_fails_before_running(tmp_path: Path) -> None:
    config = _config([{"name": "corpus", "transform": "export", "input": "a"}], datasets={"src": ToyImages()})
    with pytest.raises(GraphError, match="`input` takes object_detection"):
        run_tasks(config, output_dir=tmp_path)
