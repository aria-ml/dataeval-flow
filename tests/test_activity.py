"""A run's activity file: appending, reading, folding, paging and closing (builder spec §6)."""

from pathlib import Path

from dataeval_flow import _activity
from dataeval_flow._activity import ActivityLog, close, fold, page, read_events, recording, snapshot
from dataeval_flow.steps._result import StepResult


def _started(path: Path, steps: list[tuple[str, str]]) -> Path:
    """A run queued, then running task `t` with `steps` at the statuses given, as a worker would write it."""
    log = ActivityLog(path)
    log.append("run_state", {"status": "queued", "reason": None})
    log.append("run_state", {"status": "running", "reason": None})
    log.task("t", "running")
    for name, status in steps:
        log.step("t", None, name, status)
    return path


def test_events_are_numbered_in_order_across_writers(tmp_path):
    path = tmp_path / "activity.jsonl"
    ActivityLog(path).append("run_state", {"status": "queued", "reason": None})
    ActivityLog(path).append("run_state", {"status": "running", "reason": None})
    assert [(event["sequence"], event["run"]["status"]) for event in read_events(path)] == [
        (1, "queued"),
        (2, "running"),
    ]


def test_a_partial_last_line_is_dropped_before_appending(tmp_path):
    path = tmp_path / "activity.jsonl"
    ActivityLog(path).append("run_state", {"status": "running", "reason": None})
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"sequence": 2, "kind": "step_st')
    assert len(read_events(path)) == 1
    ActivityLog(path).append("run_state", {"status": "failed", "reason": "boom"})
    assert [event["sequence"] for event in read_events(path)] == [1, 2]
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_nothing_is_recorded_outside_a_recording(tmp_path):
    _activity.task_state("t", "running")
    with recording(tmp_path / "a.jsonl"):
        _activity.task_state("t", "running")
    _activity.task_state("t", "ok")
    assert [event["task"]["status"] for event in read_events(tmp_path / "a.jsonl")] == ["running"]


def test_a_step_row_carries_its_outcome_and_each_elements(tmp_path):
    element = StepResult(
        name="k", kind="transform", type="toy-keep", inputs=["a[x]"], status="failed", errors=["RuntimeError: boom"]
    )
    record = StepResult(
        name="k", kind="transform", type="toy-keep", inputs=["a"], status="ok", elapsed=1.23456, elements={"x": element}
    )
    with recording(tmp_path / "a.jsonl"):
        _activity.step_state("t", None, "q/k", "running")
        _activity.step_state("t", None, "q/k", "ok", record)
    running, done = (event["step"] for event in read_events(tmp_path / "a.jsonl"))
    assert (running["status"], running["finished_at"]) == ("running", None)
    assert done == {
        "node_id": "t:q/k",
        "definition_node_id": "t:q/k",
        "task": "t",
        "path": ["t", "q", "k"],
        "instances": [],
        "status": "ok",
        "started_at": running["started_at"],
        "finished_at": done["finished_at"],
        "reason": None,
        "not_assessed": None,
        "elapsed_s": 1.235,
        "elements": {"x": {"status": "failed", "reason": "RuntimeError: boom", "not_assessed": None}},
    }
    assert done["finished_at"] >= done["started_at"]


def test_a_failed_steps_reason_is_its_errors(tmp_path):
    record = StepResult(name="b", kind="transform", type="toy-explode", inputs=["a"], status="failed", errors=["E: x"])
    with recording(tmp_path / "a.jsonl"):
        _activity.step_state("t", None, "b", "failed", record)
    assert read_events(tmp_path / "a.jsonl")[0]["step"]["reason"] == "E: x"


def test_a_matrix_runs_steps_are_instances_of_one_node(tmp_path):
    with recording(tmp_path / "a.jsonl"):
        _activity.step_state("t", 2, "few", "pending")
    row = read_events(tmp_path / "a.jsonl")[0]["step"]
    assert (row["node_id"], row["definition_node_id"], row["instances"]) == (
        "t:few#2",
        "t:few",
        [{"kind": "matrix", "key": "2"}],
    )


def test_closing_a_cancelled_run_cancels_what_ran_and_explains_what_never_started(tmp_path):
    path = _started(tmp_path / "a.jsonl", [("a", "pending"), ("b", "pending"), ("a", "running")])
    close(path, "cancelled", "Cancelled while it ran")
    state = fold(read_events(path))
    steps = {row["node_id"]: row for row in state["steps"]}
    assert (steps["t:a"]["status"], steps["t:a"]["reason"]) == ("cancelled", "Cancelled while it ran")
    assert steps["t:a"]["finished_at"] is not None
    assert (steps["t:b"]["status"], steps["t:b"]["reason"]) == ("pending", "the run was cancelled before it started")
    assert state["tasks"][0]["status"] == "cancelled"
    assert state["run"] == {"status": "cancelled", "reason": "Cancelled while it ran"}
    assert read_events(path)[-1]["kind"] == "run_state"


def test_closing_a_run_that_died_interrupts_what_ran_and_invents_no_skip(tmp_path):
    path = _started(tmp_path / "a.jsonl", [("a", "ok"), ("b", "running"), ("c", "pending")])
    close(path, "failed", "The run was killed by signal 9; its log says why")
    statuses = {row["node_id"]: row["status"] for row in fold(read_events(path))["steps"]}
    assert statuses == {"t:a": "ok", "t:b": "interrupted", "t:c": "pending"}


def test_closing_a_run_without_an_activity_file_writes_none(tmp_path):
    close(tmp_path / "a.jsonl", "interrupted", "The service stopped mid-run")
    assert not (tmp_path / "a.jsonl").exists()


def test_a_page_follows_its_cursor(tmp_path):
    path = tmp_path / "a.jsonl"
    log = ActivityLog(path)
    for status in ("queued", "running", "succeeded"):
        log.append("run_state", {"status": status, "reason": None})
    events = read_events(path)
    first = page(events, "r", 0, 2)
    assert [event["sequence"] for event in first["events"]] == [1, 2]
    assert (first["next_after"], first["snapshot_sequence"], first["has_more"]) == (2, 3, True)
    assert first["events"][0]["run_id"] == "r"
    last = page(events, "r", 2, 2)
    assert ([event["sequence"] for event in last["events"]], last["next_after"], last["has_more"]) == ([3], 3, False)
    empty = page(events, "r", 3, 2)
    assert (empty["events"], empty["next_after"], empty["has_more"]) == ([], 3, False)


def test_a_snapshot_then_the_events_after_it_give_the_final_snapshot(tmp_path):
    path = _started(tmp_path / "a.jsonl", [("a", "pending"), ("a", "running")])
    early = read_events(path)
    seen = snapshot(early, "r", "sha256:x")
    assert (seen["sequence"], seen["run_status"], seen["first_available_sequence"]) == (len(early), "running", 1)
    log = ActivityLog(path)
    log.step("t", None, "a", "ok")
    log.task("t", "ok")
    log.append("run_state", {"status": "succeeded", "reason": None})
    later = page(read_events(path), "r", seen["sequence"], 100)["events"]
    replayed = early + [{key: value for key, value in event.items() if key != "run_id"} for event in later]
    assert snapshot(replayed, "r", "sha256:x") == snapshot(read_events(path), "r", "sha256:x")
