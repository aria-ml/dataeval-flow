"""TC-15-2 — a headless run: configured by flags or by `DATAEVAL_*` variables, with the flag taking precedence."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from verification.functional.reporting._project import Invocation, write_project
from verification.helpers import run_cli

pytestmark = pytest.mark.required

Cli = Callable[..., Invocation]


def _results(out: Path) -> dict:
    return json.loads((out / "results" / "result.json").read_text())


class TestHeadlessRun:
    def test_flags_run_the_pipeline_and_write_results_and_a_log(self, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        out = tmp_path / "out"

        proc = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(out))

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert list(_results(out)) == ["clean_task"]
        assert (out / "results" / "result.txt").read_text().strip()
        assert (out / "results" / "result.html").is_file()
        log = (out / "result.log").read_text()
        assert "Task 'clean_task': finished" in log
        assert "Done. 1/1 succeeded." in log

    def test_environment_variables_alone_configure_data_config_and_output(self, tmp_path: Path) -> None:
        write_project(tmp_path)
        out = tmp_path / "out"
        env = {"DATAEVAL_DATA": str(tmp_path), "DATAEVAL_CONFIG": "config.yaml", "DATAEVAL_OUTPUT": str(out)}

        proc = run_cli(env=env)

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert list(_results(out)) == ["clean_task"]

    def test_the_summary_is_printed_to_the_console_by_default(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(tmp_path)
        done = cli("-c", config, "-d", tmp_path)
        assert done.code == 0
        assert "Health: All checks passed" in done.stdout
        assert "Wrote" not in done.output  # no output directory, so no files


class TestFlagsOverrideVariables:
    def test_a_flag_beats_the_variable_for_config_data_output_and_cache(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(tmp_path / "real")
        decoy = tmp_path / "decoy"
        decoy.mkdir()
        env = {
            "DATAEVAL_CONFIG": str(decoy / "nope.yaml"),
            "DATAEVAL_DATA": str(decoy),
            "DATAEVAL_OUTPUT": str(decoy / "out"),
            "DATAEVAL_CACHE": str(decoy / "cache"),
        }
        out, cache = tmp_path / "out", tmp_path / "cache"

        done = cli("-c", config, "-d", tmp_path / "real", "-o", out, "-k", cache, env=env)

        assert done.code == 0, done.output
        assert list(_results(out)) == ["clean_task"]
        assert list(cache.rglob("stats_*.parquet"))
        assert sorted(path.name for path in decoy.iterdir()) == []  # nothing reached the variables' paths

    def test_variables_apply_when_no_flag_is_given(self, tmp_path: Path, cli: Cli) -> None:
        write_project(tmp_path)
        out, cache = tmp_path / "out", tmp_path / "cache"

        done = cli(
            env={
                "DATAEVAL_DATA": str(tmp_path),
                "DATAEVAL_CONFIG": "config.yaml",  # relative: resolved against the data root
                "DATAEVAL_OUTPUT": str(out),
                "DATAEVAL_CACHE": str(cache),
            }
        )

        assert done.code == 0, done.output
        assert (out / "results" / "result.json").is_file()
        assert list(cache.rglob("stats_*.parquet"))

    def test_a_relative_config_path_resolves_against_the_data_root(self, tmp_path: Path, cli: Cli) -> None:
        write_project(tmp_path / "root")
        done = cli("-c", "config.yaml", "-d", tmp_path / "root")
        assert done.code == 0, done.output

    def test_without_a_config_the_data_root_is_searched_for_one(self, tmp_path: Path, cli: Cli) -> None:
        write_project(tmp_path)
        out = tmp_path / "out"
        done = cli("-d", tmp_path, "-o", out)
        assert done.code == 0, done.output
        assert list(_results(out)) == ["clean_task"]

    def test_the_current_directory_is_the_default_data_root(
        self, tmp_path: Path, cli: Cli, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert cli().code == 0

    def test_boolean_and_choice_variables_are_read_and_overridden_by_their_flags(
        self, tmp_path: Path, cli: Cli
    ) -> None:
        config = write_project(tmp_path, duplicates=2)
        base = ("-c", config, "-d", tmp_path)

        assert cli(*base, env={"DATAEVAL_FAIL_ON_WARNING": "yes"}).code == 3
        assert cli(*base, "--no-fail-on-warning", env={"DATAEVAL_FAIL_ON_WARNING": "yes"}).code == 0
        assert cli(*base, "--fail-on-warning", env={"DATAEVAL_FAIL_ON_WARNING": "off"}).code == 3

    def test_max_processes_comes_from_the_flag_else_the_variable(self, tmp_path: Path, cli: Cli) -> None:
        from dataeval.config import get_max_processes, set_max_processes

        config = write_project(tmp_path)
        out = tmp_path / "out"
        try:
            assert cli("-c", config, "-d", tmp_path, "-o", out, env={"DATAEVAL_MAX_PROCESSES": "2"}).code == 0
            assert get_max_processes() == 2
            assert _results(out)["clean_task"]["metadata"]["resolved_config"]["max_processes"] == 2

            assert (
                cli(
                    "-c", config, "-d", tmp_path, "-o", out, "--max-processes", "1", env={"DATAEVAL_MAX_PROCESSES": "2"}
                ).code
                == 0
            )
            assert get_max_processes() == 1
            assert _results(out)["clean_task"]["metadata"]["resolved_config"]["max_processes"] == 1
        finally:
            set_max_processes(None)

    def test_the_verbosity_variable_applies_until_a_flag_replaces_it(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(tmp_path)
        assert "Running 1 task(s)" in cli("-c", config, "-d", tmp_path, env={"DATAEVAL_VERBOSITY": "2"}).stdout
        replaced = cli("-c", config, "-d", tmp_path, "-v", env={"DATAEVAL_VERBOSITY": "2"})
        assert "Running 1 task(s)" not in replaced.stdout  # `-v` means 1, whatever the variable says
        assert "STEPS" in replaced.stdout


class TestInvalidVariables:
    @pytest.mark.parametrize(
        ("variable", "value", "message"),
        [
            ("DATAEVAL_MAX_PROCESSES", "many", "DATAEVAL_MAX_PROCESSES must be an integer, got 'many'"),
            ("DATAEVAL_FAIL_ON_WARNING", "maybe", "DATAEVAL_FAIL_ON_WARNING must be one of"),
            ("DATAEVAL_LOG_FORMAT", "json", "DATAEVAL_LOG_FORMAT must be one of"),
            ("DATAEVAL_REQUIRE", "perfect", "DATAEVAL_REQUIRE must be one of"),
            ("DATAEVAL_REPORT_IMAGES", "sometimes", "DATAEVAL_REPORT_IMAGES must be one of"),
            ("DATAEVAL_REPORT_WIDTH", "wide", "DATAEVAL_REPORT_WIDTH must be an integer"),
            ("DATAEVAL_REPORT_WIDTH", "39", "DATAEVAL_REPORT_WIDTH must be at least 40, got 39"),
            ("DATAEVAL_VERBOSITY", "loud", "DATAEVAL_VERBOSITY must be an integer"),
            ("DATAEVAL_TASKS", ",", "DATAEVAL_TASKS was set but lists no values"),
        ],
    )
    def test_a_malformed_value_exits_1_naming_the_variable_and_prints_no_traceback(
        self, tmp_path: Path, cli: Cli, variable: str, value: str, message: str
    ) -> None:
        config = write_project(tmp_path)
        out = tmp_path / "out"

        done = cli("-c", config, "-d", tmp_path, "-o", out, env={variable: value})

        assert done.code == 1
        assert message in done.stderr
        assert "Traceback" not in done.output
        assert not out.exists()  # refused before anything ran

    def test_a_blank_variable_counts_as_unset(self, tmp_path: Path, cli: Cli) -> None:
        config = write_project(tmp_path)
        done = cli("-c", config, "-d", tmp_path, env={"DATAEVAL_FAIL_ON_WARNING": "  ", "DATAEVAL_TASKS": ""})
        assert done.code == 0, done.output
