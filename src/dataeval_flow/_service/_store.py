"""Durable run records in SQLite; only the service writes a run's lifecycle state."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})


class UnknownRunError(KeyError):
    """No run has this ID."""


class RunStore:
    """Run records under ``root``: one SQLite table, and a directory per run holding its inputs and what it wrote."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "runs.sqlite3"
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runs "
                "(id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at REAL NOT NULL, record TEXT NOT NULL)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS runs_by_status ON runs (status, created_at)")

    @contextmanager
    def connect(self, timeout: float = 30) -> Iterator[sqlite3.Connection]:
        """Open a connection owned by the current operation, safe across request threads."""
        connection = sqlite3.connect(self.database, timeout=timeout)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def usable(self) -> bool:
        """Whether the store answers a query within a second and its directory takes new runs."""
        try:
            with self.connect(timeout=1) as connection:
                connection.execute("SELECT 1 FROM runs LIMIT 1").fetchall()
        except sqlite3.Error:
            return False
        return os.access(self.root, os.W_OK)

    def create(self, request: dict[str, Any], pipeline: dict[str, Any]) -> dict[str, Any]:
        """Persist an immutable input snapshot before putting the run on the queue."""
        run_id = str(uuid4())
        directory = self.root / run_id
        directory.mkdir()
        (directory / "pipeline.json").write_text(json.dumps(pipeline, indent=2))
        (directory / "request.json").write_text(json.dumps(request, indent=2))
        record: dict[str, Any] = {
            "api_version": 1,
            "id": run_id,
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "request": request,
            "pipeline": pipeline,
            "error": None,
        }
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?)", (run_id, "queued", record["created_at"], json.dumps(record))
            )
        return record

    def get(self, run_id: str) -> dict[str, Any]:
        """Return a stored run, rejecting unknown identifiers before any path is built from one."""
        with self.connect() as connection:
            row = connection.execute("SELECT record FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise UnknownRunError(run_id)
        return json.loads(row[0])

    def history(self) -> list[dict[str, Any]]:
        """Return durable history, newest submissions first."""
        # ponytail: whole history in one response; page it when histories grow to thousands of runs.
        with self.connect() as connection:
            rows = connection.execute("SELECT record FROM runs ORDER BY created_at DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def next_queued(self) -> dict[str, Any] | None:
        """The oldest run still waiting, if any."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT record FROM runs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        """Update a record transactionally; a finished run's record never changes."""
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT record FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise UnknownRunError(run_id)
            record = json.loads(row[0])
            if record["status"] in TERMINAL:
                return record
            record.update(changes)
            connection.execute(
                "UPDATE runs SET status = ?, record = ? WHERE id = ?", (record["status"], json.dumps(record), run_id)
            )
        return record

    def recover(self) -> None:
        """Mark runs a stopped service lost as interrupted; keep queued ones queued."""
        with self.connect() as connection:
            lost = connection.execute("SELECT id FROM runs WHERE status IN ('running', 'cancelling')").fetchall()
        for (run_id,) in lost:
            self.update(run_id, status="interrupted", finished_at=time.time(), error="The service stopped mid-run")

    def directory(self, run_id: str) -> Path:
        """The directory of a known run."""
        self.get(run_id)
        return self.root / run_id
