"""Durable run records; a worker never writes lifecycle state."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})


class RunStore:
    """Store run receipts in SQLite outside the service process and browser session."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "runs.sqlite3"
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record TEXT NOT NULL)")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection owned by the current operation, safe across request threads."""
        connection = sqlite3.connect(self.database, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create(self, request: dict[str, Any], pipeline: dict[str, Any]) -> dict[str, Any]:
        """Persist an immutable input snapshot before putting the run on the queue."""
        run_id = str(uuid4())
        directory = self.root / run_id
        directory.mkdir()
        (directory / "pipeline.json").write_text(json.dumps(pipeline, indent=2))
        (directory / "request.json").write_text(json.dumps(request, indent=2))
        record = {
            "api_version": 1,
            "id": run_id,
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "request": request,
            "pipeline": pipeline,
            "error": None,
        }
        with self.connect() as connection:
            connection.execute("INSERT INTO runs VALUES (?, ?)", (run_id, json.dumps(record)))
        return record

    def get(self, run_id: str) -> dict[str, Any]:
        """Return a stored run, rejecting unknown identifiers before resolving any path."""
        with self.connect() as connection:
            row = connection.execute("SELECT record FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row[0])

    def list(self) -> list[dict[str, Any]]:
        """Return durable history, newest submissions first."""
        with self.connect() as connection:
            rows = connection.execute("SELECT record FROM runs").fetchall()
        return sorted((json.loads(row[0]) for row in rows), key=lambda row: row["created_at"], reverse=True)

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        """Update a receipt transactionally; terminal states cannot be overwritten."""
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT record FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            record = json.loads(row[0])
            if record["status"] in TERMINAL:
                return record
            record.update(changes)
            connection.execute("UPDATE runs SET record = ? WHERE id = ?", (json.dumps(record), run_id))
        return record

    def recover(self) -> None:
        """Mark executions lost across service restart as interrupted; retain queued work."""
        for run in self.list():
            if run["status"] in {"running", "cancelling"}:
                self.update(
                    run["id"], status="interrupted", finished_at=time.time(), error="Service execution was lost"
                )

    def directory(self, run_id: str) -> Path:
        """Resolve the directory of a known run."""
        self.get(run_id)
        return self.root / run_id
