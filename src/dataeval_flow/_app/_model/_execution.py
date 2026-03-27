"""Per-task execution state tracking for the dashboard.

Ephemeral — not saved to config files.  Thread-safe since Textual
workers run in background threads.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dataeval_flow.workflow import WorkflowResult

__all__ = ["ExecutionState", "TaskExecution"]


@dataclass
class TaskExecution:
    """Runtime state for a single task execution."""

    task_name: str
    status: Literal["idle", "running", "completed", "failed"] = "idle"
    result: WorkflowResult[Any, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def elapsed_s(self) -> float | None:
        """Return elapsed seconds, or ``None`` if not completed."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


class ExecutionState:
    """Tracks per-task execution status.  Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskExecution] = {}

    def mark_running(self, name: str) -> TaskExecution:
        """Set task to running.  Clears any previous result."""
        with self._lock:
            entry = TaskExecution(
                task_name=name,
                status="running",
                started_at=datetime.now(timezone.utc),
            )
            self._tasks[name] = entry
            return entry

    def mark_completed(self, name: str, result: WorkflowResult[Any, Any]) -> TaskExecution:
        """Set task to completed with its result."""
        with self._lock:
            entry = self._tasks.get(name)
            if entry is None:
                entry = TaskExecution(task_name=name)
                self._tasks[name] = entry
            entry.status = "completed"
            entry.result = result
            entry.error = None
            entry.finished_at = datetime.now(timezone.utc)
            return entry

    def mark_failed(self, name: str, error: str) -> TaskExecution:
        """Set task to failed with an error message."""
        with self._lock:
            entry = self._tasks.get(name)
            if entry is None:
                entry = TaskExecution(task_name=name)
                self._tasks[name] = entry
            entry.status = "failed"
            entry.error = error
            entry.result = None
            entry.finished_at = datetime.now(timezone.utc)
            return entry

    def get(self, name: str) -> TaskExecution | None:
        """Return execution entry for *name*, or ``None``."""
        with self._lock:
            return self._tasks.get(name)

    def clear(self, name: str | None = None) -> None:
        """Clear execution state.  If *name* is ``None``, clear all."""
        with self._lock:
            if name is None:
                self._tasks.clear()
            else:
                self._tasks.pop(name, None)

    def entries(self) -> list[TaskExecution]:
        """Return a snapshot of all execution entries."""
        with self._lock:
            return list(self._tasks.values())

    def completed_results(self) -> list[tuple[str, WorkflowResult[Any, Any]]]:
        """Return ``(name, result)`` pairs for all completed tasks."""
        with self._lock:
            return [
                (e.task_name, e.result)
                for e in self._tasks.values()
                if e.status == "completed" and e.result is not None
            ]
