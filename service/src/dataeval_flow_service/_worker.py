"""Execute Flow in a child process, saving structured partial outcomes and offline reports."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from dataeval.config import set_max_processes
from dataeval_flow import run_task, set_device
from dataeval_flow.config import PipelineConfig


def write_json(path: Path, value: Any) -> None:
    """Publish JSON atomically so API readers never see a partial document."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def event(directory: Path, kind: str, **details: Any) -> None:
    """Append a structured execution event with a UTC epoch timestamp."""
    with (directory / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), "type": kind, **details}) + "\n")


def execute(directory: Path, data_root: Path, cache_root: Path) -> bool:
    """Run each snapshotted task and persist successes and failures without interpreting process exit codes."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    payload: dict[str, Any] = {"format": 1, "tasks": {}}
    try:
        config = PipelineConfig.model_validate_json((directory / "pipeline.json").read_text())
        request = json.loads((directory / "request.json").read_text())
        set_device("cpu")
        set_max_processes(2)
        for task in request["tasks"]:
            event(directory, "task_started", task=task)
            result = run_task(config=config, task=task, data_dir=data_root, cache_dir=cache_root)
            payload["tasks"][task] = {"success": result.success, "result": result.to_dict()}
            write_json(directory / "results.json", payload)
            (directory / f"{task}.html").write_text(result.to_html())
            event(directory, "task_finished", task=task, success=result.success, errors=list(result.errors))
        success = bool(payload["tasks"]) and all(task["success"] for task in payload["tasks"].values())
        write_json(
            directory / "outcome.json", {"success": success, "error": None if success else "One or more tasks failed"}
        )
        event(directory, "run_finished", success=success)
        return success
    except Exception as error:
        logging.exception("Assessment execution failed")
        write_json(directory / "outcome.json", {"success": False, "error": f"{type(error).__name__}: {error}"})
        event(directory, "run_failed", error=str(error))
        return False


def _watch_parent(parent: int) -> None:
    while os.getppid() == parent:
        time.sleep(0.5)
    os.killpg(os.getpgrp(), signal.SIGTERM)


def main() -> None:
    """Execute the isolated worker and terminate its process group if the service disappears."""
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()
    directory, data_root, cache_root = map(Path, sys.argv[1:])
    sys.exit(0 if execute(directory, data_root, cache_root) else 1)


if __name__ == "__main__":
    main()
