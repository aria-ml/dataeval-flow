"""Run activity through the service: closing on cancel, death and restart; snapshots, pages and old runs (builder
spec §6)."""

import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

from fastapi.testclient import TestClient

from dataeval_flow._activity import ACTIVITY_FILE, ActivityLog, read_events
from dataeval_flow._service._app import create_app
from dataeval_flow._service._store import TERMINAL, RunStore
from tests.service.builder_fixtures import fixture_pipeline

pytestmark = pytest.mark.optional

# A stand-in worker: records task `review` with `dupes` running and `judge` pending, then sleeps or dies.
_WORKER = """
import os, signal, sys, time
from pathlib import Path
from dataeval_flow._activity import ActivityLog
log = ActivityLog(Path(sys.argv[1]))
log.task("review", "pending")
log.task("review", "running")
for name in ("dupes", "judge"):
    log.step("review", None, name, "pending")
log.step("review", None, "dupes", "running")
if sys.argv[2] == "die":
    os.kill(os.getpid(), signal.SIGKILL)
time.sleep(60)
"""


def _worker(mode: str):
    return lambda directory: [sys.executable, "-c", _WORKER, str(directory / ACTIVITY_FILE), mode]


def _wait_until(check, timeout: float = 15) -> None:
    end = time.monotonic() + timeout
    while not check():
        if time.monotonic() > end:
            pytest.fail("timed out")
        time.sleep(0.03)


def _running(store: RunStore, run_id: str) -> bool:
    events = read_events(store.root / run_id / ACTIVITY_FILE)
    return any(event["kind"] == "step_state" and event["step"]["status"] == "running" for event in events)


def _steps(snapshot: dict) -> dict[str, dict]:
    return {row["node_id"]: row for row in snapshot["steps"]}


def test_cancelling_a_running_run_cancels_its_running_step_and_explains_the_rest(data_root, tmp_path, wait_for):
    app = create_app(data_root, tmp_path / "out")
    app.state.manager.command = _worker("sleep")
    store = app.state.store
    with TestClient(app) as client:
        assert client.get("/v1/capabilities").json()["features"]["step_activity"] == 1
        run = client.post("/v1/runs", json={"pipeline": fixture_pipeline()}).json()
        _wait_until(lambda: _running(store, run["id"]))
        client.post(f"/v1/runs/{run['id']}/cancel")
        record = wait_for(store, run["id"], {"cancelled"})
        snapshot = client.get(f"/v1/runs/{run['id']}/activity").json()
        assert (snapshot["run_status"], snapshot["schema_fingerprint"]) == ("cancelled", record["schema_fingerprint"])
        steps = _steps(snapshot)
        assert steps["review:dupes"]["status"] == "cancelled"
        assert (steps["review:judge"]["status"], steps["review:judge"]["reason"]) == (
            "pending",
            "the run was cancelled before it started",
        )
        assert snapshot["tasks"][0]["status"] == "cancelled"
        assert snapshot["sequence"] == len(read_events(store.root / run["id"] / ACTIVITY_FILE))


def test_cancelling_a_queued_run_closes_its_activity(data_root, tmp_path, wait_for):
    app = create_app(data_root, tmp_path / "out")
    app.state.manager.command = _worker("sleep")
    store = app.state.store
    with TestClient(app) as client:
        first = client.post("/v1/runs", json={"pipeline": fixture_pipeline()}).json()
        second = client.post("/v1/runs", json={"pipeline": fixture_pipeline()}).json()
        client.post(f"/v1/runs/{second['id']}/cancel")
        events = client.get(f"/v1/runs/{second['id']}/events").json()["events"]
        assert [event["run"]["status"] for event in events] == ["queued", "cancelled"]
        client.post(f"/v1/runs/{first['id']}/cancel")
        wait_for(store, first["id"], {"cancelled"})


def test_a_worker_killed_mid_step_interrupts_it(data_root, tmp_path, wait_for):
    app = create_app(data_root, tmp_path / "out")
    app.state.manager.command = _worker("die")
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"pipeline": fixture_pipeline()}).json()
        wait_for(app.state.store, run["id"], {"failed"})
        snapshot = client.get(f"/v1/runs/{run['id']}/activity").json()
        dupes = _steps(snapshot)["review:dupes"]
        assert snapshot["run_status"] == "failed"
        assert dupes["status"] == "interrupted"
        assert "signal 9" in dupes["reason"]


def test_activity_closes_before_the_record_turns_terminal(data_root, tmp_path, wait_for, monkeypatch):
    app = create_app(data_root, tmp_path / "out")
    app.state.manager.command = _worker("sleep")
    store = app.state.store
    update = store.update
    seen: list[tuple[str, str]] = []

    def checked(run_id: str, **changes):
        if changes.get("status") in TERMINAL:
            last = read_events(store.root / run_id / ACTIVITY_FILE)[-1]
            seen.append((last["run"]["status"], changes["status"]))
        return update(run_id, **changes)

    monkeypatch.setattr(store, "update", checked)
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"pipeline": fixture_pipeline()}).json()
        _wait_until(lambda: _running(store, run["id"]))
        client.post(f"/v1/runs/{run['id']}/cancel")
        wait_for(store, run["id"], {"cancelled"})
    assert seen == [("cancelled", "cancelled")]


def test_a_restart_interrupts_what_a_lost_run_was_doing(data_root, tmp_path):
    store = RunStore(tmp_path / "out" / "runs")
    record = store.create({"tasks": ["review"]}, fixture_pipeline(), "sha256:" + "1" * 64)
    log = ActivityLog(store.root / record["id"] / ACTIVITY_FILE)
    log.append("run_state", {"status": "queued", "reason": None})
    log.append("run_state", {"status": "running", "reason": None})
    log.task("review", "running")
    log.step("review", None, "dupes", "running")
    store.update(record["id"], status="running", started_at=time.time())
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        snapshot = client.get(f"/v1/runs/{record['id']}/activity").json()
        assert snapshot["run_status"] == "interrupted"
        assert _steps(snapshot)["review:dupes"]["status"] == "interrupted"
        assert client.get(f"/v1/runs/{record['id']}").json()["status"] == "interrupted"


def test_a_run_queued_across_a_restart_continues_its_sequence(data_root, tmp_path, wait_for):
    first = create_app(data_root, tmp_path / "out")
    first.state.manager.command = lambda _: [sys.executable, "-c", "import time; time.sleep(60)"]
    with TestClient(first) as client:
        running = client.post("/v1/runs", json={"pipeline": fixture_pipeline()}).json()
        wait_for(first.state.store, running["id"], {"running"})  # so the second stays queued through the restart
        waiting = client.post("/v1/runs", json={"pipeline": fixture_pipeline()}).json()
    second = create_app(data_root, tmp_path / "out")
    second.state.manager.command = _worker("sleep")
    with TestClient(second) as client:
        _wait_until(lambda: _running(second.state.store, waiting["id"]))
        events = read_events(Path(second.state.store.root) / waiting["id"] / ACTIVITY_FILE)
        assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
        assert [event["run"]["status"] for event in events if event["kind"] == "run_state"] == ["queued", "running"]
        client.post(f"/v1/runs/{waiting['id']}/cancel")
        wait_for(second.state.store, waiting["id"], {"cancelled"})


def test_events_page_through_the_activity_with_a_cursor(data_root, tmp_path, wait_for):
    app = create_app(data_root, tmp_path / "out")
    app.state.manager.command = _worker("die")
    with TestClient(app) as client:
        run_id = client.post("/v1/runs", json={"pipeline": fixture_pipeline()}).json()["id"]
        wait_for(app.state.store, run_id, {"failed"})
        total = client.get(f"/v1/runs/{run_id}/activity").json()["sequence"]
        first = client.get(f"/v1/runs/{run_id}/events", params={"after": 0, "limit": 2}).json()
        assert ([event["sequence"] for event in first["events"]], first["has_more"]) == ([1, 2], True)
        assert first["events"][0]["run_id"] == run_id
        rest = client.get(f"/v1/runs/{run_id}/events", params={"after": first["next_after"]}).json()
        assert rest["events"][-1]["sequence"] == total
        assert rest["has_more"] is False
        beyond = client.get(f"/v1/runs/{run_id}/events", params={"after": total + 1})
        assert (beyond.status_code, beyond.json()["detail"][0]["loc"]) == (422, ["query", "after"])
        for limit in (0, 101):
            assert client.get(f"/v1/runs/{run_id}/events", params={"limit": limit}).status_code == 422


def test_a_run_from_before_activity_was_recorded_is_409(data_root, tmp_path):
    app = create_app(data_root, tmp_path / "out")
    record = app.state.store.create({"tasks": ["review"]}, fixture_pipeline())
    with TestClient(app) as client:
        for route in ("activity", "events"):
            response = client.get(f"/v1/runs/{record['id']}/{route}")
            assert (response.status_code, response.json()["code"]) == (409, "activity_unavailable")


def test_a_real_run_converges_with_its_results(data_root, tmp_path, wait_for):
    app = create_app(data_root, tmp_path / "out")
    with TestClient(app) as client:
        run_id = client.post("/v1/runs", json={"pipeline": fixture_pipeline()}).json()["id"]
        wait_for(app.state.store, run_id, {"succeeded"}, timeout=300)
        snapshot = client.get(f"/v1/runs/{run_id}/activity").json()
        assert snapshot["run_status"] == "succeeded"
        assert [task["status"] for task in snapshot["tasks"]] == ["ok"]
        assert all(row["status"] in {"ok", "skipped"} for row in snapshot["steps"])
        results = client.get(f"/v1/runs/{run_id}/results").json()
        assert {"/".join(row["path"][1:]) for row in snapshot["steps"]} == set(results["review"]["steps"])
