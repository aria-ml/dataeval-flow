"""Selections: a bin, a range, categories or flagged rows of a run, resolved exactly and served a page at a time."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

import polars as pl
from fastapi.testclient import TestClient

from dataeval_flow._service._app import create_app
from dataeval_flow._service._selections import Selections, flagged, matching
from dataeval_flow._service._store import RunStore
from dataeval_flow._sources import load_source
from dataeval_flow.config import PipelineConfig
from tests.service.test_evidence import _finished, _shuffled

_NUMERIC = {"type": "numeric", "histogram": {"edges": [0.0, 1.0, 2.0], "counts": [2, 2]}}
_CATEGORICAL = {"type": "categorical", "values": [{"value": "a", "count": 2}, {"value": 1, "count": 1}]}


def _rows(column: list) -> pl.DataFrame:
    return pl.DataFrame({"item": list(range(len(column))), "target": [None] * len(column), "x": column})


def _items(frame: pl.DataFrame) -> list[int]:
    return frame["item"].to_list()


def test_a_bin_holds_its_left_edge_and_the_last_its_right_edge_too() -> None:
    rows = _rows([0.0, 0.5, 1.0, 1.5, 2.0, None, float("nan")])
    assert _items(matching(rows, "x", {"kind": "bin", "index": 0}, _NUMERIC)) == [0, 1]
    assert _items(matching(rows, "x", {"kind": "bin", "index": 1}, _NUMERIC)) == [2, 3, 4]


def test_a_range_takes_its_bounds_as_declared_and_never_a_non_finite_value() -> None:
    rows = _rows([0.0, 1.0, 2.0, float("inf"), None])
    closed = {"kind": "range", "min": 0.0, "max": 2.0, "include_min": True, "include_max": True}
    assert _items(matching(rows, "x", closed, _NUMERIC)) == [0, 1, 2]
    half_open = {"kind": "range", "min": 0.0, "max": 2.0, "include_min": False, "include_max": False}
    assert _items(matching(rows, "x", half_open, _NUMERIC)) == [1]
    assert _items(matching(rows, "x", {"kind": "range"}, _NUMERIC)) == [0, 1, 2]


def test_missing_and_non_finite_values_are_selected_apart() -> None:
    rows = _rows([0.0, None, float("nan"), float("-inf")])
    assert _items(matching(rows, "x", {"kind": "missing"}, _NUMERIC)) == [1]
    assert _items(matching(rows, "x", {"kind": "non_finite"}, _NUMERIC)) == [2, 3]


def test_categories_keep_their_types_and_other_is_everything_unnamed() -> None:
    rows = _rows(['"a"', "1", '"1"', "true", '"a"', None])
    assert _items(matching(rows, "x", {"kind": "category", "values": [1]}, _CATEGORICAL)) == [1]
    assert _items(matching(rows, "x", {"kind": "category", "values": ["1", True]}, _CATEGORICAL)) == [2, 3]
    assert _items(matching(rows, "x", {"kind": "other"}, _CATEGORICAL)) == [2, 3]


def test_flagged_rows_count_targets_and_images_apart() -> None:
    rows = [
        {"item_index": 0, "target_index": 0, "metric_name": "brightness"},
        {"item_index": 0, "target_index": 1, "metric_name": "brightness"},
        {"item_index": 0, "target_index": 1, "metric_name": "contrast"},
        {"item_index": 1, "target_index": None, "metric_name": "brightness"},
    ]
    assert [row["item_index"] for row in flagged(rows, None, None)] == [0, 0, 0, 1]
    assert len(flagged(rows, "brightness", "target")) == 2
    assert flagged(rows, None, "image") == [rows[3]]


def _pipeline(pipeline: dict, profile: dict | None = None) -> dict:
    """The fixture's source with a content digest, a profile, and outliers at a threshold most images cross."""
    return {
        "datasets": pipeline["datasets"],
        "sources": pipeline["sources"],
        "evaluators": [
            {"name": "digest", "type": "content-digest"},
            {"name": "profile", "type": "profile", "flags": ["visual", "dimension"], **(profile or {})},
            {"name": "outliers", "type": "outliers", "flags": ["visual"], "outlier_threshold": ["zscore", 0.5]},
        ],
        "tasks": [
            {"name": "digest", "evaluator": "digest", "sources": "data"},
            {"name": "profile", "evaluator": "profile", "sources": "data"},
            {"name": "outliers", "evaluator": "outliers", "sources": "data"},
        ],
    }


@pytest.fixture
def served(data_root: Path, tmp_path: Path, pipeline: dict):
    run_id = _finished(data_root, tmp_path / "out", _pipeline(pipeline, {"bins": 4, "categories": 1}))
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        yield client, run_id


def _select(client: TestClient, run_id: str, **request) -> dict:
    response = client.post(f"/v1/runs/{run_id}/selections", json=request)
    assert response.status_code == 200, response.text
    return response.json()


def _members(client: TestClient, run_id: str, selection: str, limit: int = 5) -> list[dict]:
    pages, offset = [], 0
    while True:
        page = client.get(f"/v1/runs/{run_id}/selections/{selection}", params={"offset": offset, "limit": limit})
        members = page.json()["members"]
        pages += members
        offset += limit
        if len(members) < limit:
            return pages


def test_each_bin_selects_what_its_histogram_counts(served) -> None:
    client, run_id = served
    profile = client.get(f"/v1/runs/{run_id}/results").json()["profile"]["output"]["data"]
    (altitude,) = [field for field in profile["fields"] if field["name"] == "altitude"]
    seen = []
    for number, count in enumerate(altitude["histogram"]["counts"]):
        selection = _select(
            client, run_id, task="profile", field="altitude", predicate={"kind": "bin", "index": number}
        )
        members = _members(client, run_id, selection["id"])
        assert selection["total"] == len(members) == count
        seen += [member["index"] for member in members]
    assert sorted(seen) == list(range(12))


def test_pages_hold_every_member_once_and_name_the_source(served) -> None:
    client, run_id = served
    selection = _select(client, run_id, task="profile", field="altitude", predicate={"kind": "range"})
    members = _members(client, run_id, selection["id"], limit=5)
    assert [member["index"] for member in members] == list(range(12))
    assert {member["source"] for member in members} == {"data"}
    assert (selection["total"], selection["images"], selection["scope"]) == (12, 12, "image")


def test_a_selection_keeps_its_id_and_members_after_a_restart(tmp_path, served) -> None:
    client, run_id = served
    request = {"task": "profile", "field": "file_name", "predicate": {"kind": "other"}}
    first = _select(client, run_id, **request)
    assert _select(client, run_id, **request)["id"] == first["id"]
    restarted = Selections(RunStore(tmp_path / "out" / "runs")).page(run_id, first["id"], 0, 100)["members"]
    assert restarted == _members(client, run_id, first["id"])
    assert first["total"] == 11


def test_a_field_dataeval_could_not_read_says_why_and_selects_nothing(served) -> None:
    client, run_id = served
    profile = client.get(f"/v1/runs/{run_id}/results").json()["profile"]["output"]["data"]
    (latitude,) = [field for field in profile["fields"] if field["name"] == "latitude"]
    assert (latitude["type"], latitude["reason"]) == ("unsupported", "mixed_types")
    request = {"task": "profile", "field": "latitude", "predicate": {"kind": "missing"}}
    assert client.post(f"/v1/runs/{run_id}/selections", json=request).status_code == 422


def test_flagged_outliers_are_selected_with_their_metric_and_bound(served) -> None:
    client, run_id = served
    rows = client.get(f"/v1/runs/{run_id}/results").json()["outliers"]["output"]["rows"]
    selection = _select(client, run_id, task="outliers", predicate={"kind": "rows"})
    assert selection["total"] == len(rows)
    assert selection["images"] == len({row["item_index"] for row in rows})
    member = _members(client, run_id, selection["id"], limit=100)[0]
    assert {"metric_name", "metric_value", "bound", "direction"} <= set(member)


def test_an_image_selection_becomes_a_source_a_pipeline_can_load(data_root, served) -> None:
    client, run_id = served
    request = {"task": "profile", "field": "altitude", "predicate": {"kind": "bin", "index": 0}}
    selection = _select(client, run_id, **request)
    exported = client.get(f"/v1/runs/{run_id}/selections/{selection['id']}/view").json()
    config = PipelineConfig.model_validate(
        {"datasets": client.get(f"/v1/runs/{run_id}").json()["pipeline"]["datasets"], **exported}
    )
    assert len(load_source(config, exported["sources"][0]["name"], data_dir=data_root)) == selection["total"]


def test_a_target_selection_names_its_images_only_when_asked(served) -> None:
    client, run_id = served
    selection = _select(client, run_id, task="profile", field="brightness", scope="target", predicate={"kind": "range"})
    assert client.get(f"/v1/runs/{run_id}/selections/{selection['id']}/view").status_code == 422
    view = client.get(f"/v1/runs/{run_id}/selections/{selection['id']}/view", params={"parents": True})
    assert view.status_code == 200


@pytest.mark.parametrize(
    ("request_body", "where"),
    [
        ({"task": "profile", "field": "nothing", "predicate": {"kind": "missing"}}, ["body", "field"]),
        (
            {"task": "profile", "field": "drone", "predicate": {"kind": "bin", "index": 0}},
            ["body", "predicate", "kind"],
        ),
        (
            {"task": "profile", "field": "altitude", "predicate": {"kind": "bin", "index": 9}},
            ["body", "predicate", "index"],
        ),
        ({"task": "nothing", "predicate": {"kind": "rows"}}, ["body", "task"]),
        ({"task": "profile", "field": "brightness", "predicate": {"kind": "range"}}, ["body", "scope"]),
        ({"task": "profile", "predicate": {"kind": "rows"}}, ["body", "predicate", "kind"]),
    ],
    ids=["unknown-field", "bin-of-a-category", "no-such-bin", "unknown-task", "scope-needed", "rows-of-a-profile"],
)
def test_a_selection_the_run_cannot_answer_says_where_in_the_request(served, request_body, where) -> None:
    client, run_id = served
    response = client.post(f"/v1/runs/{run_id}/selections", json=request_body)
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert (error["loc"], error["type"]) == (where, "value_error")
    assert error["msg"]


def test_a_task_name_that_is_not_one_folder_is_refused_before_any_path_is_built(data_root, tmp_path, pipeline) -> None:
    profiled = _pipeline(pipeline)
    profiled["tasks"] = [
        {"name": "digest", "evaluator": "digest", "sources": "data"},
        {"name": "a/b", "evaluator": "profile", "sources": "data"},
    ]
    run_id = _finished(data_root, tmp_path / "out", profiled)
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        request = {"task": "a/b", "field": "altitude", "predicate": {"kind": "range"}}
        response = client.post(f"/v1/runs/{run_id}/selections", json=request)
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "task"]


def test_a_preset_s_flagged_rows_are_selected_through_its_step(data_root, tmp_path, pipeline) -> None:
    preset = {**pipeline, "tasks": [pipeline["tasks"][0]]}
    preset["workflows"] = [
        {**pipeline["workflows"][0], "outliers": {"flags": ["visual"], "outlier_threshold": ["zscore", 0.5]}}
    ]
    run_id = _finished(data_root, tmp_path / "out", preset)
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        rows = client.get(f"/v1/runs/{run_id}/results").json()["quality"]["steps"]["outliers"]["output"]["rows"]
        selection = _select(client, run_id, task="quality", step="outliers", predicate={"kind": "rows"})
        assert rows
        assert (selection["total"], selection["source"]) == (len(rows), "data")


def test_a_selection_over_a_source_each_task_drew_alone_is_refused(data_root, tmp_path, pipeline) -> None:
    run_id = _finished(data_root, tmp_path / "out", _shuffled(pipeline, seed=None))
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        request = {"task": "profile", "field": "altitude", "predicate": {"kind": "range"}}
        response = client.post(f"/v1/runs/{run_id}/selections", json=request)
        assert response.status_code == 422
        assert "seed" in response.text


def test_a_pipeline_seed_shuffles_every_task_alike_so_evidence_resolves(data_root, tmp_path, pipeline) -> None:
    run_id = _finished(data_root, tmp_path / "out", _shuffled(pipeline, seed=7))
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        request = {"task": "profile", "field": "altitude", "predicate": {"kind": "range"}}
        selection = _select(client, run_id, **request)
        members = _members(client, run_id, selection["id"], limit=12)
        for member in members:
            item = client.get(f"/v1/runs/{run_id}/items/data/{member['index']}").json()
            assert (item["status"], item["metadata"]["altitude"]) == ("verified", member["value"])
    assert sorted(member["index"] for member in members) == list(range(12))


def test_rows_a_step_flagged_on_a_derived_dataset_are_refused(data_root, tmp_path, pipeline) -> None:
    outliers = {"name": "outliers", "type": "outliers", "flags": ["visual"], "outlier_threshold": ["zscore", 0.5]}
    tail = {"type": "Indices", "params": {"indices": [11, 10, 9, 8]}}
    steps = [
        {"name": "tail", "transform": "view", "input": "data_in", "operations": [tail]},
        {"name": "flagged", "evaluator": "outliers", "input": "tail"},
        {"name": "direct", "evaluator": "outliers", "input": "data_in"},
    ]
    chained = {
        "datasets": pipeline["datasets"],
        "sources": pipeline["sources"],
        "evaluators": [outliers],
        "workflows": [{"name": "w", "inputs": ["data_in"], "steps": steps}],
        "tasks": [{"name": "t", "workflow": "w", "sources": "data"}],
    }
    run_id = _finished(data_root, tmp_path / "out", chained)
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        derived = client.post(
            f"/v1/runs/{run_id}/selections", json={"task": "t", "step": "flagged", "predicate": {"kind": "rows"}}
        )
        assert derived.status_code == 422
        assert "tail" in derived.text
        assert _select(client, run_id, task="t", step="direct", predicate={"kind": "rows"})["source"] == "data"


def test_a_field_both_supplied_and_computed_is_selected_by_its_origin(served) -> None:
    client, run_id = served
    ambiguous = {"task": "profile", "field": "width", "scope": "image", "predicate": {"kind": "range"}}
    refused = client.post(f"/v1/runs/{run_id}/selections", json=ambiguous)
    assert refused.status_code == 422
    assert "origin" in refused.text
    supplied = _select(client, run_id, **ambiguous, origin="supplied")
    computed = _select(client, run_id, **ambiguous, origin="computed")
    assert supplied["id"] != computed["id"]
    assert (supplied["definition"]["origin"], supplied["total"], computed["total"]) == ("supplied", 12, 12)
