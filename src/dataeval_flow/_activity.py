"""A run's live state, as events appended to one file the service serves (builder spec §6).

The service passes a worker ``--activity <path>``, and the worker records each task's and step's state as it changes.
With no file open, the module's functions do nothing, so a batch run and the Python API record nothing. One process
writes a run's file at a time: the service before the worker starts and after it ends, the worker in between.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataeval_flow.steps._result import StepResult

__all__ = [
    "ACTIVITY_FILE",
    "ActivityLog",
    "close",
    "fold",
    "page",
    "read_events",
    "recording",
    "snapshot",
    "step_state",
    "task_state",
]

ACTIVITY_FILE = "activity.jsonl"
_ROW = {"run_state": "run", "task_state": "task", "step_state": "step"}
_FINISHED = frozenset({"ok", "failed", "skipped", "cancelled", "interrupted"})


def read_events(path: Path) -> list[dict[str, Any]]:
    """Every complete event in `path`, in order; a last line still being written is left out."""
    if not path.exists():
        return []
    *lines, _unfinished = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines]


class ActivityLog:
    """Appends one run's events, numbered on from those already written."""

    def __init__(self, path: Path) -> None:
        self.path = path
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if not text.endswith("\n") and text:  # a writer died mid-line: drop the partial event before appending
            path.write_text(text[: text.rfind("\n") + 1], encoding="utf-8")
        self.sequence = len(read_events(path))
        self.started: dict[str, float] = {}

    def append(self, kind: str, row: dict[str, Any]) -> None:
        """Write one event holding `row`, the full state of the run, task or step it is about."""
        self.sequence += 1
        event = {"sequence": self.sequence, "timestamp": time.time(), "kind": kind, _ROW[kind]: row}
        # ponytail: flushed as the file closes, never fsynced; a machine crash may lose the last events.
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")

    def task(self, name: str, status: str, reason: str | None = None) -> None:
        """A task's state, and why it ended as it did."""
        now = time.time()
        if status == "running":
            self.started[f"task {name}"] = now
        self.append(
            "task_state",
            {
                "task": name,
                "status": status,
                "started_at": self.started.get(f"task {name}"),
                "finished_at": now if status in _FINISHED else None,
                "reason": reason,
            },
        )

    def step(self, task: str, run: int | None, name: str, status: str, record: StepResult | None = None) -> None:
        """A step's state; `run` is its task-matrix run, ``None`` outside a matrix, and `record` its outcome."""
        node = f"{task}:{name}" if run is None else f"{task}:{name}#{run}"
        now = time.time()
        if status == "running":
            self.started[node] = now
        self.append(
            "step_state",
            {
                "node_id": node,
                "definition_node_id": f"{task}:{name}",
                "task": task,
                "path": [task, *name.split("/")],
                "instances": [] if run is None else [{"kind": "matrix", "key": str(run)}],
                "status": status,
                "started_at": self.started.get(node),
                "finished_at": now if status in _FINISHED else None,
                "reason": _reason(record),
                "not_assessed": None if record is None else record.not_assessed,
                "elapsed_s": None if record is None else round(record.elapsed, 3),
                "elements": None
                if record is None or record.elements is None
                else {
                    key: {"status": element.status, "reason": _reason(element), "not_assessed": element.not_assessed}
                    for key, element in record.elements.items()
                },
            },
        )


def _reason(record: StepResult | None) -> str | None:
    """Why a step ended as it did: its reason, or else its errors."""
    if record is None:
        return None
    return record.reason or "; ".join(record.errors) or None


def fold(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The state `events` leave: the last run row, and each task's and step's last row, in the order first seen."""
    run: dict[str, Any] | None = None
    tasks: dict[str, dict[str, Any]] = {}
    steps: dict[str, dict[str, Any]] = {}
    for event in events:
        kind = event["kind"]
        row = event[_ROW[kind]]
        if kind == "run_state":
            run = row
        elif kind == "task_state":
            tasks[row["task"]] = row
        else:
            steps[row["node_id"]] = row
    sequence = events[-1]["sequence"] if events else 0
    return {"sequence": sequence, "run": run, "tasks": list(tasks.values()), "steps": list(steps.values())}


def snapshot(events: Sequence[dict[str, Any]], run_id: str, schema_fingerprint: str | None) -> dict[str, Any]:
    """`GET /v1/runs/{id}/activity`: the state `events` leave, consistent at their last sequence."""
    state = fold(events)
    return {
        "schema": 1,
        "run_id": run_id,
        "schema_fingerprint": schema_fingerprint,
        "sequence": state["sequence"],
        "first_available_sequence": 1,
        "run_status": None if state["run"] is None else state["run"]["status"],
        "tasks": state["tasks"],
        "steps": state["steps"],
    }


def page(events: Sequence[dict[str, Any]], run_id: str, after: int, limit: int) -> dict[str, Any]:
    """`GET /v1/runs/{id}/events`: up to `limit` events after sequence `after`. Every event is kept for the run's life,
    so sequence ``n`` is ``events[n - 1]``."""
    chosen = [{**event, "run_id": run_id} for event in events[after : after + limit]]
    last = chosen[-1]["sequence"] if chosen else after
    return {
        "schema": 1,
        "run_id": run_id,
        "after": after,
        "next_after": last,
        "snapshot_sequence": len(events),
        "first_available_sequence": 1,
        "has_more": last < len(events),
        "events": chosen,
    }


def close(path: Path, status: str, reason: str | None) -> None:
    """Append what a run that ended leaves (builder spec §6.4).

    A step or task still running was cancelled with the run, or else interrupted; one never started stays pending, now
    saying why. Then the run's own final state. A run with no activity file, queued before activity was recorded, gets
    none.
    """
    if not path.exists():
        return
    log = ActivityLog(path)
    state = fold(read_events(path))
    cancelled = status == "cancelled"
    unstarted = "the run was cancelled before it started" if cancelled else "the run stopped before it started"
    now = time.time()
    for kind, rows in (("step_state", state["steps"]), ("task_state", state["tasks"])):
        for row in rows:
            if row["status"] == "running":
                ended = "cancelled" if cancelled else "interrupted"
                log.append(kind, {**row, "status": ended, "finished_at": now, "reason": reason})
            elif row["status"] == "pending":
                log.append(kind, {**row, "reason": unstarted})
    log.append("run_state", {"status": status, "reason": reason})


_log: ActivityLog | None = None


@contextmanager
def recording(path: Path) -> Iterator[ActivityLog]:
    """Record this process's tasks and steps to `path` while the block runs."""
    global _log
    previous, _log = _log, ActivityLog(path)
    try:
        yield _log
    finally:
        _log = previous


def task_state(name: str, status: str, reason: str | None = None) -> None:
    """Record task `name`'s state, when recording."""
    if _log is not None:
        _log.task(name, status, reason)


def step_state(task: str, run: int | None, name: str, status: str, record: StepResult | None = None) -> None:
    """Record step `name`'s state in `task` (and matrix `run`), when recording."""
    if _log is not None:
        _log.step(task, run, name, status, record)
