"""TC-15-4 — exit codes 0, 1, 2, 3 and 4, and the order they take when more than one applies."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from verification.functional.reporting._project import AUDIT, FAILING_QUALITY, Invocation, task, write_project
from verification.helpers import run_cli

pytestmark = pytest.mark.required

Cli = Callable[..., Invocation]


class TestExit0:
    def test_a_run_whose_tasks_all_succeed_exits_0_even_with_warnings(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(tmp_path, duplicates=2)
        done = cli("-c", config, "-d", tmp_path)
        assert done.code == 0
        assert "Health warnings raised by: clean_task" in done.output  # said, but not fatal

    def test_a_config_that_defines_no_tasks_exits_0(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(tmp_path, tasks=[])
        done = cli("-c", config, "-d", tmp_path, "-vv")
        assert done.code == 0
        assert "No tasks defined in config." in done.stdout


class TestExit1:
    def test_a_missing_config_exits_1_with_the_path(self, tmp_path: Path) -> None:
        proc = run_cli("-c", "nope.yaml", "-d", str(tmp_path))
        assert proc.returncode == 1
        assert f"Config path not found: {tmp_path / 'nope.yaml'}" in proc.stdout + proc.stderr
        assert "Traceback" not in proc.stdout + proc.stderr

    def test_a_missing_data_root_exits_1_naming_it(self, tmp_path: Path, cli: Cli) -> None:
        done = cli("-d", tmp_path / "no_such_data_root")
        assert done.code == 1
        assert "no_such_data_root" in done.output
        assert "Traceback" not in done.output

    def test_an_invalid_config_exits_1_naming_the_problem_and_runs_nothing(self, tmp_path: Path, cli: Cli) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("sources: this is not a list\n")
        done = cli("-c", bad, "-d", tmp_path, "-o", tmp_path / "out")
        assert done.code == 1
        assert "validation error" in done.output
        assert "Traceback" not in done.output
        assert not (tmp_path / "out" / "results").exists()

    def test_a_failed_task_exits_1_after_the_remaining_tasks_have_run(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(
            tmp_path, workflows=[FAILING_QUALITY], tasks=[task("bad_task", "q_fail"), task("clean_task")]
        )
        done = cli("-c", config, "-d", tmp_path, "-o", tmp_path / "out")
        assert done.code == 1
        assert "FAILED: bad_task" in done.output
        assert "Done. 1/2 succeeded." in (tmp_path / "out" / "result.log").read_text()
        assert "OK: clean_task" in (tmp_path / "out" / "result.log").read_text()

    def test_an_unknown_task_and_an_unreadable_variable_exit_1_not_2(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(tmp_path)
        assert cli("-c", config, "-d", tmp_path, "-t", "nosuch").code == 1
        assert cli("-c", config, "-d", tmp_path, env={"DATAEVAL_MAX_PROCESSES": "x"}).code == 1

    def test_a_failed_export_exits_1_and_still_writes_the_results(self, tmp_path: Path, cli: Cli) -> None:
        # Exports write object-detection datasets, so the image-classification source below is refused.
        config = write_project(tmp_path, extra={"exports": [{"name": "conformed", "source": "main", "format": "coco"}]})
        done = cli("-c", config, "-d", tmp_path, "-o", tmp_path / "out")
        assert done.code == 1
        assert "Export 'conformed' failed" in done.output
        assert (tmp_path / "out" / "results" / "result.json").is_file()


class TestExit2:
    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--bogus"], "unrecognized arguments: --bogus"),
            (["--log-format", "json"], "invalid choice: 'json'"),
            (["--require", "perfect"], "invalid choice: 'perfect'"),
            (["--report-width", "10"], "must be at least 40"),
            (["--max-processes", "many"], "invalid int value: 'many'"),
            (["--task"], "expected one argument"),
            (["encoding"], "the following arguments are required: result"),
            (["verify"], "the following arguments are required"),
        ],
    )
    def test_a_mistyped_command_line_exits_2_with_the_usage_and_runs_nothing(
        self, tmp_path: Path, cli: Cli, args: list[str], message: str
    ) -> None:
        done = cli(*args)
        assert done.code == 2
        assert done.stderr.startswith("usage: dataeval_flow")
        assert message in done.stderr

    def test_the_real_process_exits_2_on_an_unknown_option(self) -> None:
        proc = run_cli("--bogus")
        assert proc.returncode == 2
        assert "unrecognized arguments: --bogus" in proc.stderr


class TestExit3:
    def test_fail_on_warning_exits_3_when_a_task_succeeds_with_a_warning(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(tmp_path, duplicates=2)
        done = cli("-c", config, "-d", tmp_path, "--fail-on-warning", "-o", tmp_path / "out")
        assert done.code == 3
        assert "--fail-on-warning" in done.output
        assert (tmp_path / "out" / "results" / "result.json").is_file()

    def test_without_a_warning_the_flag_leaves_the_exit_code_at_0(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(tmp_path)
        assert cli("-c", config, "-d", tmp_path, "--fail-on-warning").code == 0

    def test_the_flag_the_variable_and_the_config_each_set_the_gate_and_no_fail_on_warning_lifts_it(
        self, tmp_path: Path, cli: Cli
    ) -> None:
        config = write_project(tmp_path, duplicates=2, fail_on="warning")
        base = ("-c", config, "-d", tmp_path)
        assert cli(*base).code == 3  # the config's `fail_on: warning`
        assert cli(*base, "--no-fail-on-warning").code == 0
        assert cli(*base, env={"DATAEVAL_FAIL_ON_WARNING": "false"}).code == 0
        assert cli(*base, "--fail-on-warning", env={"DATAEVAL_FAIL_ON_WARNING": "false"}).code == 3

    def test_the_real_process_exits_3(self, tmp_path: Path) -> None:
        config = write_project(tmp_path, duplicates=2)
        assert run_cli("-c", str(config), "-d", str(tmp_path), "--fail-on-warning").returncode == 3


class TestExit4:
    @pytest.fixture
    def audit(self, tmp_path: Path) -> Path:
        """An audit of one split with a planted duplicate: Ready with caveats, and warning."""
        return write_project(
            tmp_path, duplicates=2, workflows=[AUDIT], tasks=[task("audit_task", "a", sources=["main"])]
        )

    def test_a_verdict_worse_than_require_exits_4_and_names_the_task_and_verdict(
        self, audit: Path, tmp_path: Path, cli: Cli
    ) -> None:
        done = cli("-c", audit, "-d", tmp_path, "--require", "ready")
        assert done.code == 4
        assert "Verdicts short of `require: ready`: audit_task (Ready with caveats)" in done.output

    def test_a_verdict_that_meets_require_exits_0(self, audit: Path, tmp_path: Path, cli: Cli) -> None:
        assert cli("-c", audit, "-d", tmp_path, "--require", "ready-with-caveats").code == 0

    def test_the_variable_sets_require_and_the_flag_replaces_it(self, audit: Path, tmp_path: Path, cli: Cli) -> None:
        assert cli("-c", audit, "-d", tmp_path, env={"DATAEVAL_REQUIRE": "ready"}).code == 4
        relaxed = cli("-c", audit, "-d", tmp_path, "--require", "ready-with-caveats", env={"DATAEVAL_REQUIRE": "ready"})
        assert relaxed.code == 0

    def test_require_for_a_run_with_no_verdict_is_refused_with_exit_1(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(tmp_path)
        done = cli("-c", config, "-d", tmp_path, "--require", "ready")
        assert done.code == 1
        assert "gates on a verdict, and no task this run runs gives one" in done.output


class TestOrder:
    def test_a_failed_task_outranks_the_verdict_and_the_verdict_outranks_the_warning_gate(
        self, tmp_path: Path, cli: Cli
    ) -> None:
        config = write_project(
            tmp_path,
            duplicates=2,
            workflows=[AUDIT, FAILING_QUALITY],
            tasks=[task("audit_task", "a", sources=["main"]), task("bad_task", "q_fail")],
        )
        base = ("-c", config, "-d", tmp_path, "--require", "ready", "--fail-on-warning")
        assert cli(*base).code == 1  # the failed task
        assert cli(*base, "-t", "audit_task").code == 4  # the verdict, though the warning gate also trips
        assert cli("-c", config, "-d", tmp_path, "--fail-on-warning", "-t", "audit_task").code == 3
