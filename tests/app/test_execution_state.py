"""Tests for ExecutionState and TaskExecution."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from dataeval_flow._app._model._execution import ExecutionState, TaskExecution

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_result(**kwargs: Any) -> MagicMock:
    """Return a mock standing in for WorkflowResult."""
    return MagicMock(**kwargs)


# ---------------------------------------------------------------------------
# TaskExecution dataclass
# ---------------------------------------------------------------------------


class TestTaskExecution:
    """Tests for the TaskExecution dataclass."""

    def test_defaults(self) -> None:
        entry = TaskExecution(task_name="t1")
        assert entry.task_name == "t1"
        assert entry.status == "idle"
        assert entry.result is None
        assert entry.error is None
        assert entry.started_at is None
        assert entry.finished_at is None

    def test_elapsed_s_none_when_no_times(self) -> None:
        entry = TaskExecution(task_name="t")
        assert entry.elapsed_s is None

    def test_elapsed_s_none_when_only_started(self) -> None:
        entry = TaskExecution(task_name="t", started_at=datetime.now(timezone.utc))
        assert entry.elapsed_s is None

    def test_elapsed_s_none_when_only_finished(self) -> None:
        entry = TaskExecution(task_name="t", finished_at=datetime.now(timezone.utc))
        assert entry.elapsed_s is None

    def test_elapsed_s_calculated(self) -> None:
        start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
        entry = TaskExecution(task_name="t", started_at=start, finished_at=end)
        assert entry.elapsed_s == pytest.approx(5.0)

    def test_elapsed_s_sub_second(self) -> None:
        start = datetime(2025, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 0, 0, 500_000, tzinfo=timezone.utc)
        entry = TaskExecution(task_name="t", started_at=start, finished_at=end)
        assert entry.elapsed_s == pytest.approx(0.5)

    def test_status_values(self) -> None:
        for status in ("idle", "running", "completed", "failed"):
            entry = TaskExecution(task_name="t", status=status)  # type: ignore[arg-type]
            assert entry.status == status


# ---------------------------------------------------------------------------
# ExecutionState — basic operations
# ---------------------------------------------------------------------------


class TestExecutionState:
    """Tests for ExecutionState."""

    def test_initial_state_empty(self) -> None:
        state = ExecutionState()
        assert state.entries() == []
        assert state.completed_results() == []
        assert state.get("nonexistent") is None

    # -- mark_running -------------------------------------------------------

    def test_mark_running_creates_entry(self) -> None:
        state = ExecutionState()
        entry = state.mark_running("task_a")
        assert entry.task_name == "task_a"
        assert entry.status == "running"
        assert entry.started_at is not None
        assert entry.result is None
        assert entry.error is None
        assert entry.finished_at is None

    def test_mark_running_replaces_previous(self) -> None:
        state = ExecutionState()
        state.mark_running("task_a")
        result = _fake_result()
        state.mark_completed("task_a", result)
        # Now re-run: previous result should be gone
        entry = state.mark_running("task_a")
        assert entry.status == "running"
        assert entry.result is None
        assert entry.error is None

    def test_mark_running_sets_utc_timestamp(self) -> None:
        state = ExecutionState()
        before = datetime.now(timezone.utc)
        entry = state.mark_running("t")
        after = datetime.now(timezone.utc)
        assert entry.started_at is not None
        assert before <= entry.started_at <= after

    # -- mark_completed -----------------------------------------------------

    def test_mark_completed_after_running(self) -> None:
        state = ExecutionState()
        state.mark_running("task_a")
        result = _fake_result()
        entry = state.mark_completed("task_a", result)
        assert entry.status == "completed"
        assert entry.result is result
        assert entry.error is None
        assert entry.finished_at is not None

    def test_mark_completed_without_prior_running(self) -> None:
        """mark_completed should work even if mark_running was never called."""
        state = ExecutionState()
        result = _fake_result()
        entry = state.mark_completed("orphan", result)
        assert entry.task_name == "orphan"
        assert entry.status == "completed"
        assert entry.result is result
        assert entry.started_at is None  # never started via mark_running
        assert entry.finished_at is not None

    def test_mark_completed_clears_error(self) -> None:
        state = ExecutionState()
        state.mark_failed("t", "boom")
        result = _fake_result()
        entry = state.mark_completed("t", result)
        assert entry.error is None
        assert entry.status == "completed"

    # -- mark_failed --------------------------------------------------------

    def test_mark_failed_after_running(self) -> None:
        state = ExecutionState()
        state.mark_running("task_a")
        entry = state.mark_failed("task_a", "something broke")
        assert entry.status == "failed"
        assert entry.error == "something broke"
        assert entry.result is None
        assert entry.finished_at is not None

    def test_mark_failed_without_prior_running(self) -> None:
        state = ExecutionState()
        entry = state.mark_failed("orphan", "error msg")
        assert entry.task_name == "orphan"
        assert entry.status == "failed"
        assert entry.error == "error msg"
        assert entry.started_at is None
        assert entry.finished_at is not None

    def test_mark_failed_clears_result(self) -> None:
        state = ExecutionState()
        result = _fake_result()
        state.mark_completed("t", result)
        entry = state.mark_failed("t", "oops")
        assert entry.result is None
        assert entry.error == "oops"

    # -- get ----------------------------------------------------------------

    def test_get_returns_entry(self) -> None:
        state = ExecutionState()
        state.mark_running("task_a")
        entry = state.get("task_a")
        assert entry is not None
        assert entry.task_name == "task_a"

    def test_get_returns_none_for_unknown(self) -> None:
        state = ExecutionState()
        assert state.get("no_such_task") is None

    # -- entries ------------------------------------------------------------

    def test_entries_returns_snapshot(self) -> None:
        state = ExecutionState()
        state.mark_running("a")
        state.mark_running("b")
        entries = state.entries()
        assert len(entries) == 2
        names = {e.task_name for e in entries}
        assert names == {"a", "b"}

    def test_entries_is_copy(self) -> None:
        """Mutating the returned list should not affect internal state."""
        state = ExecutionState()
        state.mark_running("x")
        snapshot = state.entries()
        snapshot.clear()
        assert len(state.entries()) == 1

    # -- clear --------------------------------------------------------------

    def test_clear_all(self) -> None:
        state = ExecutionState()
        state.mark_running("a")
        state.mark_running("b")
        state.clear()
        assert state.entries() == []
        assert state.get("a") is None
        assert state.get("b") is None

    def test_clear_single(self) -> None:
        state = ExecutionState()
        state.mark_running("a")
        state.mark_running("b")
        state.clear("a")
        assert state.get("a") is None
        assert state.get("b") is not None

    def test_clear_nonexistent_does_not_raise(self) -> None:
        state = ExecutionState()
        state.clear("nonexistent")  # should be a no-op

    def test_clear_none_explicit(self) -> None:
        """Passing None explicitly clears all."""
        state = ExecutionState()
        state.mark_running("a")
        state.clear(None)
        assert state.entries() == []

    # -- completed_results --------------------------------------------------

    def test_completed_results_only_completed(self) -> None:
        state = ExecutionState()
        r1 = _fake_result()
        r2 = _fake_result()
        state.mark_running("running_task")
        state.mark_completed("done_task", r1)
        state.mark_failed("failed_task", "err")
        state.mark_completed("done_task2", r2)

        results = state.completed_results()
        assert len(results) == 2
        result_names = {name for name, _ in results}
        assert result_names == {"done_task", "done_task2"}

    def test_completed_results_empty(self) -> None:
        state = ExecutionState()
        assert state.completed_results() == []

    def test_completed_results_excludes_none_result(self) -> None:
        """A completed task whose result is somehow None should be excluded."""
        state = ExecutionState()
        result = _fake_result()
        state.mark_completed("t", result)
        # Manually force result to None to test the filter
        entry = state.get("t")
        assert entry is not None
        entry.result = None
        assert state.completed_results() == []

    def test_completed_results_returns_correct_pairs(self) -> None:
        state = ExecutionState()
        r1 = _fake_result(name="result1")
        r2 = _fake_result(name="result2")
        state.mark_completed("alpha", r1)
        state.mark_completed("beta", r2)
        results = dict(state.completed_results())
        assert results["alpha"] is r1
        assert results["beta"] is r2


# ---------------------------------------------------------------------------
# Lifecycle / state transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    """Test realistic lifecycle sequences."""

    def test_full_lifecycle_success(self) -> None:
        state = ExecutionState()
        entry = state.mark_running("task")
        assert entry.status == "running"
        assert entry.started_at is not None
        started = entry.started_at

        result = _fake_result()
        entry = state.mark_completed("task", result)
        assert entry.status == "completed"
        assert entry.result is result
        assert entry.finished_at is not None
        # started_at should be preserved from mark_running
        assert entry.started_at == started
        assert entry.elapsed_s is not None
        assert entry.elapsed_s >= 0.0

    def test_full_lifecycle_failure(self) -> None:
        state = ExecutionState()
        state.mark_running("task")
        entry = state.mark_failed("task", "kaboom")
        assert entry.status == "failed"
        assert entry.error == "kaboom"
        assert entry.elapsed_s is not None

    def test_retry_after_failure(self) -> None:
        state = ExecutionState()
        state.mark_running("task")
        state.mark_failed("task", "first fail")
        # Retry
        entry = state.mark_running("task")
        assert entry.status == "running"
        assert entry.error is None
        assert entry.result is None

        result = _fake_result()
        entry = state.mark_completed("task", result)
        assert entry.status == "completed"
        assert entry.result is result

    def test_multiple_tasks_independent(self) -> None:
        state = ExecutionState()
        state.mark_running("a")
        state.mark_running("b")
        state.mark_completed("a", _fake_result())
        state.mark_failed("b", "err")

        a = state.get("a")
        b = state.get("b")
        assert a is not None
        assert a.status == "completed"
        assert b is not None
        assert b.status == "failed"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Verify ExecutionState behaves correctly under concurrent access."""

    def test_concurrent_mark_running(self) -> None:
        state = ExecutionState()
        num_threads = 50
        barrier = threading.Barrier(num_threads)

        def worker(i: int) -> None:
            barrier.wait()
            state.mark_running(f"task_{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(state.entries()) == num_threads

    def test_concurrent_mixed_operations(self) -> None:
        state = ExecutionState()
        num_tasks = 20

        def run_lifecycle(i: int) -> None:
            name = f"task_{i}"
            state.mark_running(name)
            if i % 2 == 0:
                state.mark_completed(name, _fake_result())
            else:
                state.mark_failed(name, f"error_{i}")

        threads = [threading.Thread(target=run_lifecycle, args=(i,)) for i in range(num_tasks)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(state.entries()) == num_tasks
        completed = state.completed_results()
        assert len(completed) == num_tasks // 2
