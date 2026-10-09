"""TC-15-3 — choosing tasks with `--task`, the console log format, and how much the console prints."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from verification.functional.reporting._project import Invocation, task, write_project

pytestmark = pytest.mark.required

Cli = Callable[..., Invocation]

TIMESTAMPED = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z \[(?P<level>[A-Z ]+)\] ", re.MULTILINE)


def _ran(out: Path) -> list[str]:
    return list(json.loads((out / "results" / "result.json").read_text()))


@pytest.fixture
def three_tasks(tmp_path: Path) -> Path:
    """Tasks `first`, `second` and `dormant`, the last of which the config switches off."""
    return write_project(tmp_path, tasks=[task("first"), task("second"), task("dormant", enabled=False)])


class TestTaskSelection:
    def test_without_task_every_enabled_task_runs_and_a_disabled_one_does_not(
        self, three_tasks: Path, tmp_path: Path, cli: Cli
    ) -> None:
        done = cli("-c", three_tasks, "-d", tmp_path, "-o", tmp_path / "out")
        assert done.code == 0, done.output
        assert _ran(tmp_path / "out") == ["first", "second"]

    def test_task_runs_only_the_named_one(self, three_tasks: Path, tmp_path: Path, cli: Cli) -> None:
        done = cli("-c", three_tasks, "-d", tmp_path, "-o", tmp_path / "out", "--task", "second")
        assert done.code == 0, done.output
        assert _ran(tmp_path / "out") == ["second"]

    def test_repeating_task_runs_them_in_the_order_given_and_a_name_twice_runs_once(
        self, three_tasks: Path, tmp_path: Path, cli: Cli
    ) -> None:
        done = cli(
            "-c", three_tasks, "-d", tmp_path, "-o", tmp_path / "out", "-t", "second", "-t", "first", "-t", "second"
        )
        assert done.code == 0, done.output
        assert _ran(tmp_path / "out") == ["second", "first"]

    def test_a_named_task_runs_even_when_the_config_disables_it(
        self, three_tasks: Path, tmp_path: Path, cli: Cli
    ) -> None:
        done = cli("-c", three_tasks, "-d", tmp_path, "-o", tmp_path / "out", "--task", "dormant")
        assert done.code == 0, done.output
        assert _ran(tmp_path / "out") == ["dormant"]

    def test_the_variable_lists_tasks_separated_by_commas_and_the_flag_replaces_it(
        self, three_tasks: Path, tmp_path: Path, cli: Cli
    ) -> None:
        env = {"DATAEVAL_TASKS": "second, dormant"}
        done = cli("-c", three_tasks, "-d", tmp_path, "-o", tmp_path / "from_env", env=env)
        assert done.code == 0, done.output
        assert _ran(tmp_path / "from_env") == ["second", "dormant"]

        done = cli("-c", three_tasks, "-d", tmp_path, "-o", tmp_path / "from_flag", "--task", "first", env=env)
        assert done.code == 0, done.output
        assert _ran(tmp_path / "from_flag") == ["first"]  # replaced, not appended to

    def test_an_unknown_task_name_exits_1_listing_the_tasks_there_are_before_anything_runs(
        self, three_tasks: Path, tmp_path: Path, cli: Cli
    ) -> None:
        done = cli("-c", three_tasks, "-d", tmp_path, "-o", tmp_path / "out", "--task", "first", "--task", "nosuch")
        assert done.code == 1
        assert "Unknown task: 'nosuch'" in done.output
        assert "first" in done.output
        assert not (tmp_path / "out" / "results").exists()  # `first` did not run either


class TestLogFormat:
    @pytest.fixture
    def warned(self, tmp_path: Path) -> Path:
        """A run that logs a warning to the console: its duplicates check warns."""
        return write_project(tmp_path, duplicates=2)

    def test_structured_is_the_default_and_prefixes_a_utc_timestamp_and_the_level(
        self, warned: Path, tmp_path: Path, cli: Cli
    ) -> None:
        done = cli("-c", warned, "-d", tmp_path)
        line = next(line for line in done.stdout.splitlines() if "Health warnings raised by" in line)
        assert TIMESTAMPED.match(line)
        assert "[WARNING]" in line

    def test_plain_prints_bare_messages_with_a_level_only_from_warning_up(
        self, warned: Path, tmp_path: Path, cli: Cli
    ) -> None:
        done = cli("-c", warned, "-d", tmp_path, "-vv", "--log-format", "plain")
        assert "WARNING:   Health warnings raised by: clean_task" in done.stdout
        assert "Running 1 task(s)" in done.stdout  # INFO, with no prefix at all
        assert not TIMESTAMPED.search(done.stdout)

    def test_the_variable_selects_the_format_and_the_flag_replaces_it(
        self, warned: Path, tmp_path: Path, cli: Cli
    ) -> None:
        plain = cli("-c", warned, "-d", tmp_path, env={"DATAEVAL_LOG_FORMAT": "plain"})
        assert "WARNING:   Health warnings raised by: clean_task" in plain.stdout
        assert not TIMESTAMPED.search(plain.stdout)

        structured = cli(
            "-c", warned, "-d", tmp_path, "--log-format", "structured", env={"DATAEVAL_LOG_FORMAT": "plain"}
        )
        assert TIMESTAMPED.search(structured.stdout)

    def test_the_log_file_keeps_its_own_timestamped_format_whatever_the_console_uses(
        self, warned: Path, tmp_path: Path, cli: Cli
    ) -> None:
        done = cli("-c", warned, "-d", tmp_path, "-o", tmp_path / "out", "--log-format", "plain")
        assert done.code == 0, done.output
        log = (tmp_path / "out" / "result.log").read_text()
        assert re.search(r"^\d{4}-\d{2}-\d{2}T[\d:]+Z \[DEBUG\] dataeval_flow\.", log, flags=re.MULTILINE)
        assert re.search(r"\[WARNING\] dataeval_flow\._runner:", log)

    def test_a_format_other_than_the_two_is_a_usage_error(self, warned: Path, tmp_path: Path, cli: Cli) -> None:
        done = cli("-c", warned, "-d", tmp_path, "--log-format", "json")
        assert done.code == 2
        assert "invalid choice: 'json'" in done.stderr
        assert "structured" in done.stderr
        assert "plain" in done.stderr


class TestVerbosity:
    def test_the_console_prints_the_summary_by_default_the_full_report_with_v_and_logs_with_vv(
        self, tmp_path: Path, cli: Cli
    ) -> None:
        config = write_project(tmp_path, duplicates=2)

        quiet = cli("-c", config, "-d", tmp_path)
        assert "SUMMARY" in quiet.stdout
        assert "From Duplicates" not in quiet.stdout
        assert "Running 1 task(s)" not in quiet.stdout

        report = cli("-c", config, "-d", tmp_path, "-v")
        assert "From Duplicates" in report.stdout
        assert "Running 1 task(s)" not in report.stdout

        info = cli("-c", config, "-d", tmp_path, "-vv")
        assert "From Duplicates" in info.stdout
        assert "Running 1 task(s)" in info.stdout
        assert "Memory hit" not in info.stdout

        debug = cli("-c", config, "-d", tmp_path, "-vvv")
        assert "Memory hit" in debug.stdout
        assert "[DEBUG]" in debug.stdout

    def test_a_failed_task_is_always_reported_on_the_console(self, tmp_path: Path, cli: Cli) -> None:
        from verification.functional.reporting._project import FAILING_QUALITY

        config = write_project(tmp_path, workflows=[FAILING_QUALITY], tasks=[task("bad", "q_fail")])
        done = cli("-c", config, "-d", tmp_path)
        assert done.code == 1
        assert "FAILED: bad" in done.output
        assert "n_expected_clusters=500 should be less than dataset size" in done.output
