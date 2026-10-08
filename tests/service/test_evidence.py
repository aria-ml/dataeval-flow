"""A run's items over HTTP: each read back from its source and served only while it matches the run's manifest."""

import io
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from dataeval_flow._runner import run
from dataeval_flow._service._app import create_app
from dataeval_flow._service._store import RunStore


def _inspection(pipeline: dict, view: dict | None = None, seed: int | None = None) -> dict:
    """The fixture pipeline with only a content-digest task over its source, viewed through `view`."""
    source = {"name": "data", "dataset": "fixture", **({"view": view["name"]} if view else {})}
    return {
        **({"seed": seed} if seed is not None else {}),
        "datasets": pipeline["datasets"],
        **({"views": [view]} if view else {}),
        "sources": [source],
        "evaluators": [{"name": "digest", "type": "content-digest"}],
        "tasks": [{"name": "digest", "evaluator": "digest", "sources": "data"}],
    }


def _finished(data_root: Path, out: Path, pipeline: dict) -> str:
    """A run the store records as succeeded, its files written by the batch command as the worker writes them."""
    store = RunStore(out / "runs")
    record = store.create({"tasks": [task["name"] for task in pipeline["tasks"]]}, pipeline)
    directory = store.directory(record["id"])
    assert run(directory / "pipeline.json", directory, data_dir=data_root, report_images=False) == 0
    store.update(record["id"], status="succeeded", exit_code=0)
    return record["id"]


@pytest.fixture
def client(data_root: Path, tmp_path: Path):
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        yield client


def test_an_item_is_served_with_its_boxes_labels_and_metadata(data_root, tmp_path, pipeline, client) -> None:
    run_id = _finished(data_root, tmp_path / "out", _inspection(pipeline))
    item = client.get(f"/v1/runs/{run_id}/items/data/11").json()
    assert item["status"] == "verified"
    assert (item["width"], item["height"], item["box_format"]) == (48, 32, "xyxy")
    assert item["targets"] == [{"target": 0, "box": [1.0, 2.0, 6.0, 6.0], "label": 1, "class": "swimmer"}]
    assert item["metadata"]["altitude"] == 12.0
    assert item["image_url"] == f"/v1/runs/{run_id}/items/data/11/image"


def test_a_page_visits_every_item_once(data_root, tmp_path, pipeline, client) -> None:
    run_id = _finished(data_root, tmp_path / "out", _inspection(pipeline))
    pages = [client.get(f"/v1/runs/{run_id}/items/data", params={"offset": o, "limit": 5}).json() for o in (0, 5, 10)]
    assert {page["total"] for page in pages} == {12}
    assert [item["index"] for page in pages for item in page["items"]] == list(range(12))


def test_an_image_is_served_as_a_png_of_the_item(data_root, tmp_path, pipeline, client) -> None:
    run_id = _finished(data_root, tmp_path / "out", _inspection(pipeline))
    response = client.get(f"/v1/runs/{run_id}/items/data/3/image")
    assert response.headers["content-type"] == "image/png"
    served = np.asarray(Image.open(io.BytesIO(response.content)))
    assert np.array_equal(served, np.asarray(Image.open(data_root / "fixture" / "images" / "000003.png")))
    crop = Image.open(io.BytesIO(client.get(f"/v1/runs/{run_id}/items/data/3/image", params={"target": 0}).content))
    assert crop.size == (7, 6)


def test_a_shrunk_image_fits_the_side_asked_for(data_root, tmp_path, pipeline, client) -> None:
    run_id = _finished(data_root, tmp_path / "out", _inspection(pipeline))
    response = client.get(f"/v1/runs/{run_id}/items/data/3/image", params={"max_side": 24})
    assert Image.open(io.BytesIO(response.content)).size == (24, 16)


def test_an_edited_image_is_reported_changed_and_not_served(data_root, tmp_path, pipeline, client) -> None:
    run_id = _finished(data_root, tmp_path / "out", _inspection(pipeline))
    path = data_root / "fixture" / "images" / "000004.png"
    Image.fromarray(255 - np.asarray(Image.open(path))).save(path)
    assert client.get(f"/v1/runs/{run_id}/items/data/4").json()["status"] == "input_changed"
    assert client.get(f"/v1/runs/{run_id}/items/data/4/image").status_code == 409
    assert client.get(f"/v1/runs/{run_id}/items/data/5").json()["status"] == "verified"


def test_edited_metadata_is_reported_changed(data_root, tmp_path, pipeline, client) -> None:
    run_id = _finished(data_root, tmp_path / "out", _inspection(pipeline))
    annotations = data_root / "fixture" / "instances.json"
    coco = json.loads(annotations.read_text())
    coco["images"][6]["altitude"] = 99.0
    annotations.write_text(json.dumps(coco))
    assert client.get(f"/v1/runs/{run_id}/items/data/6").json()["status"] == "input_changed"


def test_a_removed_image_is_reported_unavailable(data_root, tmp_path, pipeline, client) -> None:
    run_id = _finished(data_root, tmp_path / "out", _inspection(pipeline))
    (data_root / "fixture" / "images" / "000007.png").unlink()
    item = client.get(f"/v1/runs/{run_id}/items/data/7").json()
    assert item["status"] == "input_unavailable"
    assert item["reason"]


def test_a_run_without_a_manifest_says_its_evidence_is_unavailable(data_root, tmp_path, pipeline, client) -> None:
    unrecorded = {**_inspection(pipeline), "evaluators": [{"name": "labels", "type": "label-health"}]}
    unrecorded["tasks"] = [{"name": "labels", "evaluator": "labels", "sources": "data"}]
    run_id = _finished(data_root, tmp_path / "out", unrecorded)
    page = client.get(f"/v1/runs/{run_id}/items/data").json()
    assert (page["status"], page["total"], page["items"]) == ("evidence_unavailable", 0, [])
    assert client.get(f"/v1/runs/{run_id}/items/data/0").json()["status"] == "evidence_unavailable"


def test_items_of_a_viewed_source_are_found_after_a_restart(data_root, tmp_path, pipeline) -> None:
    view = {"name": "tail", "operations": [{"type": "Indices", "params": {"indices": [9, 2, 4]}}]}
    run_id = _finished(data_root, tmp_path / "out", _inspection(pipeline, view))
    for _ in range(2):  # a fresh service over the same runs
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            item = client.get(f"/v1/runs/{run_id}/items/data/1").json()
            assert (item["status"], item["root_index"], item["metadata"]["altitude"]) == ("verified", 2, 3.0)


def _shuffled(pipeline: dict, seed: int | None) -> dict:
    """A digest and a profile over the fixture through a `Shuffle` with no seed of its own."""
    view = {"name": "shuffled", "operations": [{"type": "Shuffle", "params": {}}]}
    shuffled = _inspection(pipeline, view, seed)
    shuffled["evaluators"].append({"name": "profile", "type": "profile", "flags": ["visual"]})
    shuffled["tasks"].append({"name": "profile", "evaluator": "profile", "sources": "data"})
    return shuffled


def test_a_shuffle_with_no_seed_anywhere_leaves_no_evidence_to_resolve(data_root, tmp_path, pipeline, client) -> None:
    run_id = _finished(data_root, tmp_path / "out", _shuffled(pipeline, seed=None))
    page = client.get(f"/v1/runs/{run_id}/items/data").json()
    assert (page["status"], page["total"]) == ("evidence_unavailable", 0)
    assert "seed" in page["reason"]
    assert client.get(f"/v1/runs/{run_id}/items/data/0").json()["status"] == "evidence_unavailable"


def test_unknown_sources_and_indices_are_not_found(data_root, tmp_path, pipeline, client) -> None:
    run_id = _finished(data_root, tmp_path / "out", _inspection(pipeline))
    assert client.get(f"/v1/runs/{run_id}/items/other/0").status_code == 404
    assert client.get(f"/v1/runs/{run_id}/items/data/12").status_code == 404
    assert client.get(f"/v1/runs/{run_id}/items/data", params={"limit": 101}).status_code == 422


def test_a_box_the_item_does_not_hold_is_not_found(data_root, tmp_path, pipeline, client) -> None:
    run_id = _finished(data_root, tmp_path / "out", _inspection(pipeline))
    assert client.get(f"/v1/runs/{run_id}/items/data/3/image", params={"target": 5}).status_code == 404


def test_a_manifest_still_being_written_reads_as_not_yet_recorded(data_root, tmp_path, pipeline, client) -> None:
    run_id = _finished(data_root, tmp_path / "out", _inspection(pipeline))
    manifest = tmp_path / "out" / "runs" / run_id / "results" / "manifests" / "digest" / "content-digest.json"
    manifest.write_text(manifest.read_text()[:100])
    assert client.get(f"/v1/runs/{run_id}/items/data/0").json()["status"] == "evidence_unavailable"


def test_index_0_of_two_sources_names_each_source_s_own_item(data_root, tmp_path, pipeline, client) -> None:
    tail = {"name": "tail", "operations": [{"type": "Indices", "params": {"indices": [11, 10]}}]}
    two = _inspection(pipeline)
    two["views"] = [tail]
    two["sources"].append({"name": "tail", "dataset": "fixture", "view": "tail"})
    two["tasks"].append({"name": "digest-tail", "evaluator": "digest", "sources": "tail"})
    run_id = _finished(data_root, tmp_path / "out", two)
    data, tail_item = (client.get(f"/v1/runs/{run_id}/items/{source}/0").json() for source in ("data", "tail"))
    assert (data["status"], data["root_index"], data["metadata"]["altitude"]) == ("verified", 0, 1.0)
    assert (tail_item["status"], tail_item["root_index"], tail_item["metadata"]["altitude"]) == ("verified", 11, 12.0)
