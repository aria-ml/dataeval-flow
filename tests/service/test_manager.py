"""Real child processes: queueing, exit codes, cancellation, interruption, console relay and restart."""

import io
import json
import logging
import signal
import sys
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

from dataeval_flow._service import _manager, _worker
from dataeval_flow._service._manager import RunManager
from dataeval_flow._service._store import RunStore, UnknownRunError

pytestmark = pytest.mark.optional

_STUBBORN = (
    "import signal,time,pathlib,sys; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
    "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"
)


def _ready(path, timeout=5):
    end = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < end:
        time.sleep(0.03)
    assert path.exists()


def test_store_recovery_order_and_immutability(tmp_path):
    store = RunStore(tmp_path)
    queued = store.create({"tasks": ["one"]}, {"tasks": []})
    lost = store.create({}, {})
    later = store.create({}, {})
    store.update(lost["id"], status="running")
    assert json.loads((store.directory(queued["id"]) / "request.json").read_text()) == queued["request"]
    first = store.next_queued()
    assert first is not None
    assert first["id"] == queued["id"]
    assert [run["id"] for run in store.history()] == [later["id"], lost["id"], queued["id"]]
    with pytest.raises(UnknownRunError):
        store.update("missing", status="failed")
    with pytest.raises(UnknownRunError):
        store.directory("../escape")
    restarted = RunStore(tmp_path)
    restarted.recover()
    assert restarted.get(queued["id"])["status"] == "queued"
    assert restarted.get(lost["id"])["status"] == "interrupted"
    assert restarted.update(lost["id"], status="succeeded")["status"] == "interrupted"
    assert restarted.usable()
    restarted.database.write_bytes(b"not a database")
    assert not restarted.usable()


def test_command_runs_the_batch_command_on_the_snapshot(tmp_path):
    store = RunStore(tmp_path / "runs")
    run = store.create({"tasks": ["quality", "metadata"]}, {})
    directory = store.directory(run["id"])
    command = RunManager(store, tmp_path / "data", tmp_path / "cache")._command(directory)
    assert command[1:3] == ["-m", "dataeval_flow._service._worker"]
    assert command[command.index("--config") + 1] == str(directory / "pipeline.json")
    assert command[command.index("--output") + 1] == str(directory)
    assert command[-4:] == ["--task", "quality", "--task", "metadata"]


def test_exit_codes_decide_status_and_output_is_kept(tmp_path, monkeypatch, wait_for):
    monkeypatch.setenv("DATAEVAL_REQUIRE", "ready")  # The service's own settings never reach a run.
    store = RunStore(tmp_path / "runs")

    def command(directory):
        mode = json.loads((directory / "request.json").read_text())["mode"]
        if mode == "missing":
            return ["/nonexistent/command"]
        if mode == "killed":
            return [sys.executable, "-c", "import os,signal; os.kill(os.getpid(), signal.SIGKILL)"]
        code = (
            "import os,sys; print('INFO line'); print('WARNING: careful'); print('to stderr', file=sys.stderr); "
            f"assert 'DATAEVAL_REQUIRE' not in os.environ; sys.exit({mode})"
        )
        return [sys.executable, "-c", code]

    manager = RunManager(store, tmp_path, tmp_path / "cache", command)
    modes = {0: "succeeded", 3: "succeeded", 4: "succeeded", 1: "failed", "killed": "failed", "missing": "failed"}
    ids = {mode: manager.submit({"mode": mode}, {})["id"] for mode in modes}
    manager.start()
    try:
        with pytest.raises(RuntimeError, match="Another service"):
            RunManager(RunStore(store.root), tmp_path, tmp_path / "cache").start()
        for mode, status in modes.items():
            assert wait_for(store, ids[mode], {status})["status"] == status
        with pytest.raises(ValueError, match="already finished"):
            manager.cancel(ids[0])
    finally:
        manager.stop()
    assert manager.thread is not None
    assert not manager.thread.is_alive()
    assert store.get(ids[3])["exit_code"] == 3
    assert "killed by signal 9" in store.get(ids["killed"])["error"]
    assert "exited with code 1" in store.get(ids[1])["error"]
    assert "could not start" in store.get(ids["missing"])["error"]
    log = (store.directory(ids[0]) / "console.log").read_text()
    assert "INFO line" in log
    assert "WARNING: careful" in log
    assert "to stderr" in log


def test_relay_logs_each_line_at_its_level(tmp_path, caplog):
    stream = io.BytesIO(b"plain info\n\nWARNING: careful\nERROR: broken\n")
    with caplog.at_level(logging.INFO, logger=_manager.__name__):
        _manager._relay("0123456789", stream, tmp_path / "console.log", logging.INFO)
        _manager._relay("0123456789", io.BytesIO(b"FutureWarning: soon\n"), tmp_path / "console.log", logging.WARNING)
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.INFO, "run 01234567: plain info"),
        (logging.WARNING, "run 01234567: careful"),
        (logging.ERROR, "run 01234567: broken"),
        (logging.WARNING, "run 01234567: FutureWarning: soon"),
    ]
    assert (tmp_path / "console.log").read_bytes().endswith(b"ERROR: broken\nFutureWarning: soon\n")


def test_a_failed_step_does_not_stop_the_queue(tmp_path, monkeypatch, wait_for, caplog):
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store, tmp_path, tmp_path, lambda _: [sys.executable, "-c", "pass"])
    real = store.next_queued
    failures = iter([True])
    monkeypatch.setattr(store, "next_queued", lambda: real() if not next(failures, False) else 1 / 0)
    run = manager.submit({}, {})
    manager.start()
    try:
        assert wait_for(store, run["id"], {"succeeded"})["exit_code"] == 0
        assert manager.alive
    finally:
        manager.stop()
    assert "failed a step" in caplog.text


def test_queued_cancellation_and_running_shutdown(tmp_path, wait_for):
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


def test_cancellation_escalates_for_stubborn_runs(tmp_path, wait_for):
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store, tmp_path, tmp_path, lambda d: [sys.executable, "-c", _STUBBORN, str(d / "ready")])
    active = manager.submit({}, {})
    manager.start()
    try:
        wait_for(store, active["id"], {"running"})
        _ready(store.directory(active["id"]) / "ready")
        assert manager.cancel(active["id"])["status"] == "cancelling"
        assert manager.cancel(active["id"])["status"] == "cancelling"
        assert wait_for(store, active["id"], {"cancelled"})["status"] == "cancelled"
    finally:
        manager.stop()


def test_shutdown_escalates_for_stubborn_runs(tmp_path, wait_for):
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store, tmp_path, tmp_path, lambda d: [sys.executable, "-c", _STUBBORN, str(d / "ready")])
    active = manager.submit({}, {})
    manager.start()
    try:
        wait_for(store, active["id"], {"running"})
        _ready(store.directory(active["id"]) / "ready")
    finally:
        manager.stop()
    assert store.get(active["id"])["status"] == "interrupted"


def test_worker_watches_the_service_and_runs_the_batch_command(monkeypatch):
    parents = iter([10, 11])
    signals = []
    monkeypatch.setattr(_worker.os, "getppid", lambda: next(parents))
    monkeypatch.setattr(_worker.os, "getpgrp", lambda: 99)
    monkeypatch.setattr(_worker.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(_worker.time, "sleep", lambda _: None)
    _worker._watch_parent(10)
    assert signals == [(99, signal.SIGTERM)]
    ran = []
    monkeypatch.setattr(_worker.os, "getppid", lambda: 10)
    monkeypatch.setattr(_worker.threading, "Thread", lambda **_kwargs: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(_worker, "run_command", lambda: ran.append(True))
    _worker.main()
    assert ran == [True]
