"""TC-21-3 — Analysis service evidence: items, images and selections, served only while they match the manifest."""

from __future__ import annotations

import io
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dataeval_flow import load_source
from dataeval_flow._service._app import create_app
from dataeval_flow.config import PipelineConfig
from verification.functional.integrity._data import edit_annotations, invert_image, write_coco
from verification.functional.service.conftest import DATASET

pytestmark = pytest.mark.required


def _inspection(*, digest: bool = True) -> dict[str, Any]:
    """A pipeline that digests a source (so the run records its manifest), profiles it, and flags outliers."""
    evaluators: list[dict[str, Any]] = [
        {"name": "profile", "type": "profile", "flags": ["visual", "dimension"], "bins": 4, "categories": 1},
        {"name": "outliers", "type": "outliers", "flags": ["visual"], "outlier_threshold": ["zscore", 0.5]},
    ]
    tasks: list[dict[str, Any]] = [
        {"name": "profile", "evaluator": "profile", "sources": "data"},
        {"name": "outliers", "evaluator": "outliers", "sources": "data"},
    ]
    if digest:
        evaluators.insert(0, {"name": "digest", "type": "content-digest"})
        tasks.insert(0, {"name": "digest", "evaluator": "digest", "sources": "data"})
    return {**DATASET, "evaluators": evaluators, "tasks": tasks}


@dataclass(frozen=True)
class Copies:
    """Private copies of a data root and of the output of two finished runs of it."""

    data: Path
    output: Path
    run: str
    """The run that digested its source, so it recorded a manifest."""
    unrecorded: str
    """The run that did not."""


@dataclass(frozen=True)
class Served(Copies):
    """A running service over :class:`Copies`."""

    client: TestClient = None  # type: ignore[assignment]


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str, str]:
    """Two runs through a real service: one with a content-digest task, one without."""
    root = tmp_path_factory.mktemp("evidence")
    data, output = root / "data", root / "out"
    write_coco(data, count=12)
    app = create_app(data, output)
    with TestClient(app) as client:
        ids = [client.post("/v1/runs", json={"pipeline": _inspection(digest=d)}).json()["id"] for d in (True, False)]
        end = time.monotonic() + 180
        for run_id in ids:
            while (status := app.state.store.get(run_id)["status"]) in {"queued", "running"}:
                assert time.monotonic() < end, "runs did not finish"
                time.sleep(0.05)
            assert status == "succeeded", client.get(f"/v1/runs/{run_id}/logs").text
    return data, output, ids[0], ids[1]


@pytest.fixture
def copies(finished: tuple[Path, Path, str, str], tmp_path: Path) -> Copies:
    data, output, run, unrecorded = finished
    return Copies(shutil.copytree(data, tmp_path / "data"), shutil.copytree(output, tmp_path / "out"), run, unrecorded)


@pytest.fixture
def served(copies: Copies) -> Iterator[Served]:
    with TestClient(create_app(copies.data, copies.output)) as client:
        yield Served(copies.data, copies.output, copies.run, copies.unrecorded, client)


class TestItems:
    def test_an_item_is_served_with_its_boxes_labels_and_metadata(self, served: Served) -> None:
        item = served.client.get(f"/v1/runs/{served.run}/items/data/11").json()
        assert item["status"] == "verified"
        assert (item["source"], item["index"], item["root_index"]) == ("data", 11, 11)
        assert (item["width"], item["height"], item["box_format"]) == (48, 32, "xyxy")
        assert item["targets"] == [{"target": 0, "box": [1.0, 2.0, 6.0, 6.0], "label": 1, "class": "swimmer"}]
        assert item["metadata"]["altitude"] == 12.0
        assert item["image_url"] == f"/v1/runs/{served.run}/items/data/11/image"

    def test_pages_visit_every_item_once_in_the_order_the_run_read_them(self, served: Served) -> None:
        pages = [
            served.client.get(f"/v1/runs/{served.run}/items/data", params={"offset": offset, "limit": 5}).json()
            for offset in (0, 5, 10)
        ]
        assert {(page["status"], page["total"]) for page in pages} == {("recorded", 12)}
        assert [item["index"] for page in pages for item in page["items"]] == list(range(12))

    def test_a_page_larger_than_the_limit_is_refused(self, served: Served) -> None:
        response = served.client.get(f"/v1/runs/{served.run}/items/data", params={"limit": 101})
        assert response.status_code == 422

    def test_an_unknown_source_or_index_is_not_found(self, served: Served) -> None:
        assert served.client.get(f"/v1/runs/{served.run}/items/other/0").status_code == 404
        assert served.client.get(f"/v1/runs/{served.run}/items/data/12").status_code == 404


class TestItemsAgainstTheManifest:
    def test_an_edited_image_is_reported_changed_and_its_neighbours_stay_verified(self, served: Served) -> None:
        invert_image(served.data / "fixture", 4)
        changed = served.client.get(f"/v1/runs/{served.run}/items/data/4").json()
        assert changed["status"] == "input_changed"
        assert "targets" not in changed
        assert served.client.get(f"/v1/runs/{served.run}/items/data/5").json()["status"] == "verified"

    def test_edited_metadata_is_reported_changed(self, served: Served) -> None:
        def edit(coco: dict) -> None:
            coco["images"][6]["altitude"] = 99.0

        edit_annotations(served.data / "fixture", edit)
        assert served.client.get(f"/v1/runs/{served.run}/items/data/6").json()["status"] == "input_changed"

    def test_a_removed_image_file_is_reported_unavailable_with_a_reason(self, served: Served) -> None:
        (served.data / "fixture" / "images" / "000007.png").unlink()
        item = served.client.get(f"/v1/runs/{served.run}/items/data/7").json()
        assert item["status"] == "input_unavailable"
        assert item["reason"]

    def test_a_run_without_a_content_digest_task_has_no_evidence_to_serve(self, served: Served) -> None:
        page = served.client.get(f"/v1/runs/{served.unrecorded}/items/data").json()
        assert (page["status"], page["total"], page["items"]) == ("evidence_unavailable", 0, [])
        assert page["reason"]
        item = served.client.get(f"/v1/runs/{served.unrecorded}/items/data/0").json()
        assert item["status"] == "evidence_unavailable"

    def test_items_are_found_again_after_the_service_restarts(self, copies: Copies) -> None:
        for _ in range(2):  # a fresh service over the same runs, twice
            with TestClient(create_app(copies.data, copies.output)) as restarted:
                item = restarted.get(f"/v1/runs/{copies.run}/items/data/3").json()
            assert (item["status"], item["metadata"]["altitude"]) == ("verified", 4.0)


class TestImages:
    def test_an_image_is_served_as_a_png_of_the_item(self, served: Served) -> None:
        response = served.client.get(f"/v1/runs/{served.run}/items/data/3/image")
        assert (response.status_code, response.headers["content-type"]) == (200, "image/png")
        original = np.asarray(Image.open(served.data / "fixture" / "images" / "000003.png"))
        assert np.array_equal(np.asarray(Image.open(io.BytesIO(response.content))), original)

    def test_a_box_can_be_cropped_with_a_margin(self, served: Served) -> None:
        response = served.client.get(f"/v1/runs/{served.run}/items/data/3/image", params={"target": 0})
        assert Image.open(io.BytesIO(response.content)).size == (7, 6)

    def test_an_image_can_be_shrunk_to_fit_a_side(self, served: Served) -> None:
        response = served.client.get(f"/v1/runs/{served.run}/items/data/3/image", params={"max_side": 24})
        assert Image.open(io.BytesIO(response.content)).size == (24, 16)

    def test_a_box_the_item_does_not_hold_is_not_found(self, served: Served) -> None:
        assert served.client.get(f"/v1/runs/{served.run}/items/data/3/image", params={"target": 5}).status_code == 404

    def test_the_image_of_a_changed_item_is_refused_with_its_status(self, served: Served) -> None:
        invert_image(served.data / "fixture", 4)
        response = served.client.get(f"/v1/runs/{served.run}/items/data/4/image")
        assert response.status_code == 409
        assert response.json()["status"] == "input_changed"


def _select(served: Served, **request: Any) -> dict[str, Any]:
    response = served.client.post(f"/v1/runs/{served.run}/selections", json=request)
    assert response.status_code == 200, response.text
    return response.json()


def _members(served: Served, selection: str, limit: int = 5) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    while True:
        page = served.client.get(
            f"/v1/runs/{served.run}/selections/{selection}", params={"offset": len(members), "limit": limit}
        ).json()["members"]
        members += page
        if len(page) < limit:
            return members


class TestSelections:
    def test_each_bin_selects_what_its_histogram_counts_and_together_they_hold_every_item(self, served: Served) -> None:
        results = served.client.get(f"/v1/runs/{served.run}/results").json()
        (altitude,) = [f for f in results["profile"]["output"]["data"]["fields"] if f["name"] == "altitude"]
        seen: list[int] = []
        for number, count in enumerate(altitude["histogram"]["counts"]):
            selection = _select(served, task="profile", field="altitude", predicate={"kind": "bin", "index": number})
            members = _members(served, selection["id"])
            assert selection["total"] == len(members) == count
            seen += [member["index"] for member in members]
        assert sorted(seen) == list(range(12))

    def test_pages_hold_every_member_once_and_name_the_source_item_and_value(self, served: Served) -> None:
        selection = _select(served, task="profile", field="altitude", predicate={"kind": "range"})
        members = _members(served, selection["id"], limit=5)
        assert [member["index"] for member in members] == list(range(12))
        assert {member["source"] for member in members} == {"data"}
        assert [member["value"] for member in members] == [float(i + 1) for i in range(12)]
        assert (selection["total"], selection["images"], selection["scope"]) == (12, 12, "image")

    def test_the_same_request_gives_the_same_id_before_and_after_a_restart(self, copies: Copies) -> None:
        request = {"task": "profile", "field": "altitude", "predicate": {"kind": "range", "min": 3, "max": 6}}
        with TestClient(create_app(copies.data, copies.output)) as client:
            first = client.post(f"/v1/runs/{copies.run}/selections", json=request).json()
            assert client.post(f"/v1/runs/{copies.run}/selections", json=request).json()["id"] == first["id"]
        with TestClient(create_app(copies.data, copies.output)) as restarted:
            again = restarted.post(f"/v1/runs/{copies.run}/selections", json=request).json()
            page = restarted.get(f"/v1/runs/{copies.run}/selections/{first['id']}").json()
        assert again["id"] == first["id"]
        assert [member["index"] for member in page["members"]] == [2, 3, 4]

    def test_a_range_includes_its_minimum_and_excludes_its_maximum_by_default(self, served: Served) -> None:
        selection = _select(served, task="profile", field="altitude", predicate={"kind": "range", "min": 3, "max": 6})
        assert [m["value"] for m in _members(served, selection["id"])] == [3.0, 4.0, 5.0]

    def test_flagged_outliers_are_selected_with_their_metric_and_bound(self, served: Served) -> None:
        rows = served.client.get(f"/v1/runs/{served.run}/results").json()["outliers"]["output"]["rows"]
        selection = _select(served, task="outliers", predicate={"kind": "rows"})
        assert selection["total"] == len(rows) > 0
        assert selection["images"] == len({row["item_index"] for row in rows})
        assert {"metric_name", "metric_value", "bound", "direction"} <= set(_members(served, selection["id"], 100)[0])

    def test_an_image_selection_becomes_a_source_a_later_pipeline_can_load(self, served: Served) -> None:
        selection = _select(served, task="profile", field="altitude", predicate={"kind": "range", "min": 3, "max": 6})
        exported = served.client.get(f"/v1/runs/{served.run}/selections/{selection['id']}/view").json()
        datasets = served.client.get(f"/v1/runs/{served.run}").json()["pipeline"]["datasets"]
        config = PipelineConfig.model_validate({"datasets": datasets, **exported})
        loaded = load_source(config, exported["sources"][0]["name"], data_dir=served.data)
        assert len(loaded) == selection["total"] == 3

    @pytest.mark.parametrize(
        ("request_body", "where"),
        [
            ({"task": "profile", "field": "nothing", "predicate": {"kind": "missing"}}, ["body", "field"]),
            (
                {"task": "profile", "field": "altitude", "predicate": {"kind": "bin", "index": 9}},
                ["body", "predicate", "index"],
            ),
            ({"task": "nothing", "predicate": {"kind": "rows"}}, ["body", "task"]),
            ({"task": "profile", "predicate": {"kind": "rows"}}, ["body", "predicate", "kind"]),
        ],
        ids=["unknown-field", "no-such-bin", "unknown-task", "rows-of-a-profile"],
    )
    def test_a_selection_the_run_cannot_answer_is_refused_with_where_in_the_request(
        self, served: Served, request_body: dict[str, Any], where: list[str]
    ) -> None:
        response = served.client.post(f"/v1/runs/{served.run}/selections", json=request_body)
        assert response.status_code == 422
        (error,) = response.json()["detail"]
        assert (error["loc"], error["type"]) == (where, "value_error")
        assert error["msg"]

    def test_an_unknown_selection_or_run_is_not_found(self, served: Served) -> None:
        assert served.client.get(f"/v1/runs/{served.run}/selections/nothing").status_code == 404
        response = served.client.post(
            "/v1/runs/missing/selections", json={"task": "profile", "predicate": {"kind": "rows"}}
        )
        assert response.status_code == 404
