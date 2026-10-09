"""TC-12-4 — `result: fail_on` and `result: require` set the exit code, never whether results are written."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from verification.functional.reporting._project import AUDIT, FAILING_QUALITY, Invocation, task, write_project

pytestmark = pytest.mark.required

Cli = Callable[..., Invocation]


def _audit_task() -> dict:
    return task("audit_task", "a", sources=["main"])


class TestFailOn:
    def test_default_fails_on_a_failed_task_and_not_on_warnings(self, tmp_path: Path, cli: Cli) -> None:
        warned = write_project(tmp_path / "warned", duplicates=2)
        done = cli("-c", warned, "-d", tmp_path / "warned")
        assert done.code == 0
        assert "Health warnings raised by: clean_task" in done.output

        failing = write_project(tmp_path / "failing", workflows=[FAILING_QUALITY], tasks=[task("bad", "q_fail")])
        done = cli("-c", failing, "-d", tmp_path / "failing")
        assert done.code == 1
        assert "FAILED: bad" in done.output

    def test_warning_exits_3_when_a_task_succeeds_with_a_warning(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(tmp_path, duplicates=2, fail_on="warning")
        done = cli("-c", config, "-d", tmp_path, "-o", tmp_path / "out")
        assert done.code == 3
        assert "Failing on health warnings" in done.output
        assert (tmp_path / "out" / "results" / "result.json").is_file()  # the gate decides the code, not the files

    def test_warning_exits_0_when_nothing_warns(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(tmp_path, fail_on="warning")
        assert cli("-c", config, "-d", tmp_path).code == 0

    def test_a_failed_task_outranks_a_warning(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(
            tmp_path,
            duplicates=2,
            workflows=[FAILING_QUALITY],
            tasks=[task("bad", "q_fail"), task("clean_task")],
            fail_on="warning",
        )
        assert cli("-c", config, "-d", tmp_path).code == 1

    def test_never_exits_0_for_a_failed_task_and_still_writes_its_results(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(
            tmp_path,
            duplicates=2,
            workflows=[FAILING_QUALITY],
            tasks=[task("bad", "q_fail"), task("clean_task")],
            fail_on="never",
        )
        done = cli("-c", config, "-d", tmp_path, "-o", tmp_path / "out")
        assert done.code == 0
        assert "FAILED: bad" in done.output
        assert (tmp_path / "out" / "results" / "result.json").is_file()


class TestRequire:
    def test_a_verdict_short_of_the_required_level_exits_4(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(
            tmp_path, duplicates=2, workflows=[AUDIT], tasks=[_audit_task()], require="ready", fail_on="never"
        )
        done = cli("-c", config, "-d", tmp_path, "-o", tmp_path / "out")
        assert done.code == 4  # `fail_on: never` does not silence a verdict gate
        assert "Verdicts short of `require: ready`: audit_task (Ready with caveats)" in done.output
        assert (tmp_path / "out" / "results" / "result.json").is_file()

    def test_a_verdict_that_meets_the_level_exits_0_and_the_flag_overrides_the_config(
        self, tmp_path: Path, cli: Cli
    ) -> None:
        config = write_project(
            tmp_path, duplicates=2, workflows=[AUDIT], tasks=[_audit_task()], require="ready-with-caveats"
        )
        assert cli("-c", config, "-d", tmp_path).code == 0
        assert cli("-c", config, "-d", tmp_path, "--require", "ready").code == 4

    def test_a_verdict_outranks_a_warning_gate(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(
            tmp_path, duplicates=2, workflows=[AUDIT], tasks=[_audit_task()], require="ready", fail_on="warning"
        )
        done = cli("-c", config, "-d", tmp_path)
        assert done.code == 4
        assert "Health warnings raised by: audit_task" in done.output  # the warning is still said

    def test_require_with_no_task_that_gives_a_verdict_is_refused_before_any_task_runs(
        self, tmp_path: Path, cli: Cli
    ) -> None:
        config = write_project(tmp_path, require="ready")
        done = cli("-c", config, "-d", tmp_path, "-o", tmp_path / "out")
        assert done.code == 1
        assert "gates on a verdict, and no task this run runs gives one" in done.output
        assert not (tmp_path / "out" / "results").exists()
