"""TC-15-1 and TC-26-1 — headless command-line runs: flags, environment variables, exit codes, errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verification.fixtures import write_cli_project
from verification.helpers import run_cli

pytestmark = pytest.mark.required


def _output(proc) -> str:
    return proc.stdout + proc.stderr


class TestHeadlessCLI:
    def test_headless_run_with_flags_writes_results(self, tmp_path: Path) -> None:
        config = write_cli_project(tmp_path)
        out = tmp_path / "out"

        proc = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(out))

        assert proc.returncode == 0, _output(proc)
        results = json.loads((out / "results" / "result.json").read_text())
        assert set(results) == {"clean_task"}
        assert (out / "results" / "result.txt").read_text().strip()
        assert (out / "result.log").is_file()

    def test_environment_variables_configure_the_run_and_flags_win(self, tmp_path: Path) -> None:
        write_cli_project(tmp_path)
        env_out = tmp_path / "out_env"
        flag_out = tmp_path / "out_flag"
        env = {
            "DATAEVAL_DATA": str(tmp_path),
            "DATAEVAL_CONFIG": "config.yaml",
            "DATAEVAL_OUTPUT": str(env_out),
        }

        # Variables alone configure data root, config, and output.
        proc = run_cli(env=env)
        assert proc.returncode == 0, _output(proc)
        assert (env_out / "results" / "result.json").is_file()

        # A conflicting flag takes precedence over the variable.
        proc = run_cli("-o", str(flag_out), env={**env, "DATAEVAL_OUTPUT": str(tmp_path / "ignored")})
        assert proc.returncode == 0, _output(proc)
        assert (flag_out / "results" / "result.json").is_file()
        assert not (tmp_path / "ignored").exists()

    def test_fail_on_warning_exits_nonzero_on_warnings(self, tmp_path: Path) -> None:
        config = write_cli_project(tmp_path, plant_duplicate=True)
        base = ["-c", str(config), "-d", str(tmp_path)]

        # Warnings alone do not fail the run ...
        proc = run_cli(*base)
        assert proc.returncode == 0, _output(proc)
        assert "Health warnings raised by: clean_task" in _output(proc)

        # ... unless asked to, by flag or by variable; the flag overrides the variable.
        proc = run_cli(*base, "--fail-on-warning")
        assert proc.returncode == 1
        assert "--fail-on-warning" in _output(proc)
        assert run_cli(*base, env={"DATAEVAL_FAIL_ON_WARNING": "true"}).returncode == 1
        assert run_cli(*base, "--no-fail-on-warning", env={"DATAEVAL_FAIL_ON_WARNING": "true"}).returncode == 0

    def test_missing_config_reports_an_error_and_exits_nonzero(self, tmp_path: Path) -> None:
        proc = run_cli("-c", "nope.yaml", "-d", str(tmp_path))
        assert proc.returncode != 0
        assert "Config path not found" in _output(proc)

    def test_missing_data_directory_reports_an_error_and_exits_nonzero(self, tmp_path: Path) -> None:
        config = write_cli_project(tmp_path / "project")
        proc = run_cli("-c", str(config), "-d", str(tmp_path / "no_such_data_root"))
        assert proc.returncode != 0
        assert "no_such_data_root" in _output(proc)
        assert "Traceback" not in _output(proc)

    def test_invalid_config_reports_an_error_and_exits_nonzero(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("sources: this is not a list\n")
        proc = run_cli("-c", str(bad), "-d", str(tmp_path))
        assert proc.returncode != 0
        assert "validation error" in _output(proc)
        assert "Traceback" not in _output(proc)

    def test_failing_task_sets_nonzero_exit_after_remaining_tasks_run(self, tmp_path: Path) -> None:
        config = write_cli_project(tmp_path, failing_task=True)
        out = tmp_path / "out"

        proc = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(out))

        assert proc.returncode == 1
        assert "FAILED: bad_task" in _output(proc)
        # The task after the failing one still ran and its result was written.
        assert set(json.loads((out / "results" / "result.json").read_text())) == {"clean_task"}
