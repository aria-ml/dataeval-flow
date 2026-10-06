"""Single-worker persistent queue with process cancellation and restart accounting."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from dataeval_flow_service._store import TERMINAL, RunStore


class RunManager:
    """Own worker processes independently of HTTP requests and browser sessions."""

    def __init__(
        self, store: RunStore, data_root: Path, cache_root: Path, command: Callable[[Path], list[str]] | None = None
    ) -> None:
        self.store = store
        self.data_root = data_root.resolve()
        self.cache_root = cache_root.resolve()
        self.command = command or self._command
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.process: subprocess.Popen[bytes] | None = None
        self.log: BinaryIO | None = None
        self.active: str | None = None
        self.cancel_deadline: float | None = None
        self.thread: threading.Thread | None = None
        self.lease: BinaryIO | None = None

    def _command(self, directory: Path) -> list[str]:
        return [
            sys.executable,
            "-u",
            "-m",
            "dataeval_flow_service._worker",
            str(directory),
            str(self.data_root),
            str(self.cache_root),
        ]

    def start(self) -> None:
        """Recover history and start one queue consumer."""
        lease = (self.store.root / "service.lock").open("ab")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            raise RuntimeError("Another Flow service owns this run directory") from None
        self.lease = lease
        self.stopping.clear()
        self.store.recover()
        self.thread = threading.Thread(target=self._loop, name="flow-queue", daemon=True)
        self.thread.start()

    def submit(self, request: dict[str, Any], pipeline: dict[str, Any]) -> dict[str, Any]:
        """Durably submit a validated execution snapshot."""
        with self.lock:
            record = self.store.create(request, pipeline)
            self.wake.set()
            return record

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Cancel queued work or request termination of the entire active process group."""
        with self.lock:
            run = self.store.get(run_id)
            if run["status"] in TERMINAL:
                raise ValueError("Run has already finished")
            if run["status"] == "queued":
                return self.store.update(run_id, status="cancelled", finished_at=time.time())
            if run["status"] != "cancelling":
                self.store.update(run_id, status="cancelling")
                self.cancel_deadline = time.monotonic() + 3
                self._signal(signal.SIGTERM)
            self.wake.set()
            return self.store.get(run_id)

    def _signal(self, sig: int) -> None:
        if self.process is not None:
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass

    def stop(self) -> None:
        """Stop execution on service shutdown, preserving queued work and interrupted receipts."""
        self.stopping.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        if self.lease is not None:
            self.lease.close()
            self.lease = None

    def _launch(self, run: dict[str, Any]) -> None:
        directory = self.store.directory(run["id"])
        self.active = run["id"]
        try:
            self.log = (directory / "worker.log").open("ab")
            self.process = subprocess.Popen(
                self.command(directory),
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "HF_HUB_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                    "OMP_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
                },
            )
            self.store.update(run["id"], status="running", started_at=time.time())
        except OSError as error:
            self.store.update(run["id"], status="failed", finished_at=time.time(), error=str(error))
            self._clear()

    def _clear(self) -> None:
        if self.log is not None:
            self.log.close()
        self.log = None
        self.process = None
        self.active = None
        self.cancel_deadline = None

    def _finish(self) -> None:
        if self.active is None or self.process is None:
            return
        run = self.store.get(self.active)
        if run["status"] == "cancelling":
            self._signal(signal.SIGKILL)  # Also reap descendants if the parent exited first.
            self.store.update(self.active, status="cancelled", finished_at=time.time())
        else:
            outcome_path = self.store.directory(self.active) / "outcome.json"
            try:
                outcome = json.loads(outcome_path.read_text())
                success = self.process.returncode == 0 and outcome["success"] is True
                error = outcome.get("error")
            except (OSError, ValueError, KeyError):
                success, error = False, f"Worker exited ({self.process.returncode}) without a complete receipt"
            self.store.update(
                self.active, status="succeeded" if success else "failed", finished_at=time.time(), error=error
            )
        self._clear()

    def _loop(self) -> None:
        while not self.stopping.is_set():
            with self.lock:
                if self.process is not None:
                    if self.cancel_deadline is not None and time.monotonic() >= self.cancel_deadline:
                        self._signal(signal.SIGKILL)
                    if self.process.poll() is not None:
                        self._finish()
                if self.process is None:
                    queued = [run for run in reversed(self.store.list()) if run["status"] == "queued"]
                    if queued:
                        self._launch(queued[0])
            self.wake.wait(0.1)
            self.wake.clear()
        with self.lock:
            if self.process is not None and self.active is not None:
                self._signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._signal(signal.SIGKILL)
                    self.process.wait(timeout=3)
                self._signal(signal.SIGKILL)
                self.store.update(self.active, status="interrupted", finished_at=time.time(), error="Service stopped")
                self._clear()
