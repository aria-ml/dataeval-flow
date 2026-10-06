"""Real child processes verify queueing, cancellation, interruption, and restart receipts."""

import json
import signal
import sys
import time

import pytest

from dataeval_flow_service._manager import RunManager
from dataeval_flow_service._store import RunStore


def wait_for(store, run_id, statuses, timeout=12):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        record = store.get(run_id)
        if record["status"] in statuses:
            return record
        time.sleep(0.03)
    pytest.fail(f"Run did not reach {statuses}: {store.get(run_id)}")


def test_store_recovery_and_immutability(tmp_path):
    store = RunStore(tmp_path)
    queued = store.create({"dataset_id": "one"}, {"tasks": []})
    lost = store.create({}, {})
    store.update(lost["id"], status="running")
    assert json.loads((store.directory(queued["id"]) / "request.json").read_text()) == queued["request"]
    with pytest.raises(KeyError):
        store.update("missing", status="failed")
    with pytest.raises(KeyError):
        store.directory("../escape")
    restarted = RunStore(tmp_path)
    restarted.recover()
    assert restarted.get(queued["id"])["status"] == "queued"
    assert restarted.get(lost["id"])["status"] == "interrupted"
    assert restarted.update(lost["id"], status="succeeded")["status"] == "interrupted"


def test_queue_and_process_failures(tmp_path):
    store = RunStore(tmp_path / "runs")

    def command(directory):
        mode = json.loads((directory / "request.json").read_text())["mode"]
        if mode == "launch-error":
            return ["/nonexistent/worker"]
        code = "import sys,json,time,pathlib; time.sleep(.05); "
        if mode != "no-receipt":
            code += f"pathlib.Path(sys.argv[1]).write_text(json.dumps({{'success': {mode == 'ok'}, 'error': None}})); "
        return [sys.executable, "-c", code, str(directory / "outcome.json")]

    manager = RunManager(store, tmp_path, tmp_path / "cache", command)
    ids = [manager.submit({"mode": mode}, {})["id"] for mode in ("ok", "failed", "no-receipt", "launch-error")]
    assert manager._command(tmp_path)[2:4] == ["-m", "dataeval_flow_service._worker"]
    manager.start()
    try:
        other = RunManager(RunStore(store.root), tmp_path, tmp_path / "cache")
        with pytest.raises(RuntimeError, match="Another"):
            other.start()
        for run_id, status in zip(ids, ("succeeded", "failed", "failed", "failed"), strict=True):
            assert wait_for(store, run_id, {status})["status"] == status
        with pytest.raises(ValueError, match="already finished"):
            manager.cancel(ids[0])
    finally:
        manager.stop()
    assert not manager.thread.is_alive()
    assert "without a complete receipt" in store.get(ids[2])["error"]


def test_queued_cancellation_and_running_shutdown(tmp_path):
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store, tmp_path, tmp_path, lambda _: [sys.executable, "-c", "import time; time.sleep(60)"])
    queued = manager.submit({}, {})
    assert manager.cancel(queued["id"])["status"] == "cancelled"
    active = manager.submit({}, {})
    manager.start()
    wait_for(store, active["id"], {"running"})
    manager.stop()
    assert store.get(active["id"])["status"] == "interrupted"
    manager._finish()  # Shutdown leaves no active process to finish or signal.
    manager._signal(signal.SIGTERM)


def test_cancellation_escalates_for_resistant_workers(tmp_path):
    store = RunStore(tmp_path / "runs")
    code = (
        "import signal,time,pathlib,sys; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"
    )
    manager = RunManager(
        store, tmp_path, tmp_path, lambda directory: [sys.executable, "-c", code, str(directory / "ready")]
    )
    active = manager.submit({}, {})
    manager.start()
    try:
        wait_for(store, active["id"], {"running"})
        ready = store.directory(active["id"]) / "ready"
        end = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < end:
            time.sleep(0.03)
        assert ready.exists()
        assert manager.cancel(active["id"])["status"] == "cancelling"
        assert manager.cancel(active["id"])["status"] == "cancelling"
        assert wait_for(store, active["id"], {"cancelled"})["status"] == "cancelled"
    finally:
        manager.stop()


def test_shutdown_escalates_for_resistant_workers(tmp_path):
    store = RunStore(tmp_path / "runs")
    code = (
        "import signal,time,pathlib,sys; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"
    )
    manager = RunManager(
        store, tmp_path, tmp_path, lambda directory: [sys.executable, "-c", code, str(directory / "ready")]
    )
    active = manager.submit({}, {})
    manager.start()
    try:
        wait_for(store, active["id"], {"running"})
        end = time.monotonic() + 5
        while not (store.directory(active["id"]) / "ready").exists() and time.monotonic() < end:
            time.sleep(0.03)
    finally:
        manager.stop()
    assert store.get(active["id"])["status"] == "interrupted"
