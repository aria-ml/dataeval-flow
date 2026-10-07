"""One queue consumer: runs each queued snapshot as a batch ``dataeval-flow`` process, and cancels or recovers it."""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from dataeval_flow._service._store import TERMINAL, RunStore

_logger = logging.getLogger(__name__)

# What a batch run that ran to its end exits with: 0, or a gate on what its tasks found (3 warnings, 4 a verdict).
_RAN = frozenset({0, 3, 4})
# A plain-format console line from WARNING up opens with its level; any other line takes its stream's level.
_LEVELS = {"WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
# Seconds a run has to stop after SIGTERM before its process group is killed.
_GRACE = 3.0


def _relay(run_id: str, stream: IO[bytes], log: Path, level: int) -> None:
    """Keep a run's console output in its directory, and log each line on the service's console, naming the run."""
    with stream, log.open("ab", buffering=0) as saved:
        for raw in stream:
            saved.write(raw)
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            prefix, _, message = line.partition(": ")
            if prefix in _LEVELS:
                _logger.log(_LEVELS[prefix], "run %s: %s", run_id[:8], message)
            else:
                _logger.log(level, "run %s: %s", run_id[:8], line)


class RunManager:
    """Own each run's process, independently of the HTTP requests that queued it."""

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
        self.relays: list[threading.Thread] = []
        self.active: str | None = None
        self.cancel_deadline: float | None = None
        self.thread: threading.Thread | None = None
        self.lease: IO[bytes] | None = None

    @property
    def alive(self) -> bool:
        """Whether the queue consumer is running and not shutting down."""
        return self.thread is not None and self.thread.is_alive() and not self.stopping.is_set()

    def _command(self, directory: Path) -> list[str]:
        """The batch command that runs a queued snapshot, writing into the run's own directory."""
        tasks = json.loads((directory / "request.json").read_text())["tasks"]
        return [
            sys.executable,
            "-m",
            "dataeval_flow._service._worker",
            "--log-format",
            "plain",
            "-vv",
            "--config",
            str(directory / "pipeline.json"),
            "--data",
            str(self.data_root),
            "--output",
            str(directory),
            "--cache",
            str(self.cache_root),
            *(argument for task in tasks for argument in ("--task", task)),
        ]

    def start(self) -> None:
        """Take the queue's lease, recover runs a stopped service lost, and start the queue consumer."""
        lease = (self.store.root / "service.lock").open("ab")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            raise RuntimeError(f"Another service is running the queue in {self.store.root}") from None
        self.lease = lease
        self.stopping.clear()
        self.store.recover()
        self.thread = threading.Thread(target=self._loop, name="dataeval-flow-queue", daemon=True)
        self.thread.start()

    def submit(self, request: dict[str, Any], pipeline: dict[str, Any]) -> dict[str, Any]:
        """Durably queue a validated snapshot."""
        with self.lock:
            record = self.store.create(request, pipeline)
            self.wake.set()
        _logger.info("Run %s queued", record["id"])
        return record

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Cancel a queued run, or terminate a running one's whole process group."""
        with self.lock:
            run = self.store.get(run_id)
            if run["status"] in TERMINAL:
                raise ValueError("The run has already finished")
            if run["status"] == "queued":
                _logger.info("Run %s cancelled before it started", run_id)
                return self.store.update(run_id, status="cancelled", finished_at=time.time())
            if run["status"] != "cancelling":
                self.store.update(run_id, status="cancelling")
                self.cancel_deadline = time.monotonic() + _GRACE
                self._signal(signal.SIGTERM)
            self.wake.set()
            return self.store.get(run_id)

    def stop(self) -> None:
        """Interrupt the running run, keep the queued ones, and release the queue's lease."""
        self.stopping.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        if self.lease is not None:
            self.lease.close()
            self.lease = None

    def _signal(self, sig: int) -> None:
        if self.process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, sig)

    def _launch(self, run: dict[str, Any]) -> None:
        directory = self.store.directory(run["id"])
        # The snapshot alone defines a run: no DATAEVAL_* setting of the service's own reaches it.
        env = {name: value for name, value in os.environ.items() if not name.startswith("DATAEVAL_")}
        try:
            process = subprocess.Popen(  # noqa: S603 - our own interpreter and module, with paths we made
                self.command(directory),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env={**env, "PYTHONUNBUFFERED": "1"},
            )
        except (OSError, ValueError, KeyError) as error:
            _logger.error("Run %s could not start: %s", run["id"], error)
            self.store.update(
                run["id"], status="failed", finished_at=time.time(), error=f"The run could not start: {error}"
            )
            return
        self.process, self.active = process, run["id"]
        log = directory / "console.log"
        streams = ((process.stdout, logging.INFO), (process.stderr, logging.WARNING))
        self.relays = [
            threading.Thread(target=_relay, args=(run["id"], stream, log, level), daemon=True)
            for stream, level in streams
            if stream is not None
        ]
        for relay in self.relays:
            relay.start()
        self.store.update(run["id"], status="running", started_at=time.time())
        _logger.info("Run %s started", run["id"])

    def _drain(self) -> None:
        """Wait, briefly, for a stopped run's last output; a descendant holding its pipe open cannot stall the queue."""
        deadline = time.monotonic() + 5
        for relay in self.relays:
            relay.join(timeout=max(0.0, deadline - time.monotonic()))

    def _finish(self) -> None:
        if self.active is None or self.process is None:
            return
        cancelling = self.store.get(self.active)["status"] == "cancelling"
        if cancelling:
            self._signal(signal.SIGKILL)  # Reap descendants that outlive the run's own process.
        self._drain()
        code = self.process.returncode
        changes: dict[str, Any] = {"finished_at": time.time(), "exit_code": code}
        if cancelling:
            record = self.store.update(self.active, status="cancelled", **changes)
        elif code in _RAN:
            record = self.store.update(self.active, status="succeeded", **changes)
        else:
            how = f"was killed by signal {-code}" if code < 0 else f"exited with code {code}"
            record = self.store.update(
                self.active, status="failed", error=f"The run {how}; its log says why", **changes
            )
        _logger.info("Run %s %s", self.active, record["status"])
        self._clear()

    def _clear(self) -> None:
        self.process = None
        self.relays = []
        self.active = None
        self.cancel_deadline = None

    def _step(self) -> None:
        with self.lock:
            if self.process is not None:
                if self.cancel_deadline is not None and time.monotonic() >= self.cancel_deadline:
                    self._signal(signal.SIGKILL)
                if self.process.poll() is not None:
                    self._finish()
            if self.process is None and (run := self.store.next_queued()) is not None:
                self._launch(run)

    def _loop(self) -> None:
        while not self.stopping.is_set():
            try:
                self._step()
            except Exception:  # A failed step must not stop the queue: the next one retries it.
                _logger.exception("The run queue failed a step; retrying")
                self.stopping.wait(1)
            self.wake.wait(0.1)
            self.wake.clear()
        self._interrupt()

    def _interrupt(self) -> None:
        with self.lock:
            if self.process is None or self.active is None:
                return
            self._signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=_GRACE)
            except subprocess.TimeoutExpired:
                self._signal(signal.SIGKILL)
                self.process.wait(timeout=_GRACE)
            self._signal(signal.SIGKILL)
            self._drain()
            self.store.update(
                self.active,
                status="interrupted",
                finished_at=time.time(),
                exit_code=self.process.returncode,
                error="The service stopped mid-run",
            )
            _logger.info("Run %s interrupted", self.active)
            self._clear()
