"""TC-36-1 (NFR-6) — bad input and failing tasks produce clear errors and an exit code a script can read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from verification.helpers import run_cli
from verification.nonfunctional._support import pipeline_dict, write_project

pytestmark = pytest.mark.required


def _output(run) -> str:
    """What a user sees: errors are logged to the console (stdout), argument errors go to stderr."""
    return run.stdout + run.stderr


class TestBadInput:
    def test_missing_config_reports_an_error_and_exits_nonzero(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.yaml"

        run = run_cli("-c", str(missing), "-d", str(tmp_path), "-o", str(tmp_path / "out"))

        assert run.returncode == 1
        assert "[ERROR] Config path not found" in run.stdout
        assert str(missing) in run.stdout

    def test_a_data_root_with_no_config_reports_an_error_and_exits_nonzero(self, tmp_path: Path) -> None:
        run = run_cli("-d", str(tmp_path), "-o", str(tmp_path / "out"))

        assert run.returncode == 1
        assert "No valid pipeline config files found" in run.stdout

    def test_missing_data_directory_reports_an_error_and_exits_nonzero(self, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        nowhere = tmp_path / "nowhere"

        run = run_cli("-c", str(config), "-d", str(nowhere), "-o", str(tmp_path / "out"))

        assert run.returncode == 1
        assert "[ERROR] Image folder not found" in run.stdout
        assert str(nowhere / "imgs") in run.stdout

    def test_invalid_config_reports_an_error_and_exits_nonzero(self, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        broken = tmp_path / "broken.yaml"
        broken.write_text(yaml.safe_dump({**yaml.safe_load(config.read_text()), "seeed": 3}))

        run = run_cli("-c", str(broken), "-d", str(tmp_path), "-o", str(tmp_path / "out"))

        assert run.returncode == 1
        assert "Unknown top-level key 'seeed' (did you mean 'seed'?)" in run.stdout
        assert not (tmp_path / "out" / "results").exists()

    def test_a_task_naming_an_undefined_workflow_is_refused_before_anything_runs(self, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        pipeline = pipeline_dict()
        pipeline["tasks"][0]["workflow"] = "missing"
        config.write_text(yaml.safe_dump(pipeline))

        run = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(tmp_path / "out"))

        assert run.returncode == 1
        assert "Task 'clean_task' names workflow 'missing'" in run.stdout

    def test_a_malformed_environment_variable_is_reported_by_name(self) -> None:
        run = run_cli(env={"DATAEVAL_MAX_PROCESSES": "many"})

        assert run.returncode == 1
        assert "DATAEVAL_MAX_PROCESSES must be an integer, got 'many'" in _output(run)

    def test_naming_a_task_the_config_does_not_define_is_reported_with_the_known_ones(self, tmp_path: Path) -> None:
        config = write_project(tmp_path)

        run = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(tmp_path / "out"), "-t", "nope")

        assert run.returncode == 1
        assert "Unknown task: 'nope'. Available: ['clean_task']" in run.stdout


class TestFailingTasks:
    def test_failing_task_sets_nonzero_exit_after_remaining_tasks_run(self, tmp_path: Path) -> None:
        config = write_project(tmp_path, failing_task=True, n_tasks=2)
        out = tmp_path / "out"

        run = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(out), "-vv")

        assert run.returncode == 1
        results = json.loads((out / "results" / "result.json").read_text())
        assert list(results) == ["bad_task", "clean_task", "clean_task_2"]
        assert results["bad_task"]["health"]["status"] == "failed"
        assert results["bad_task"]["health"]["failed_steps"] == ["split"]
        assert any("Unable to stratify" in error for error in results["bad_task"]["errors"])
        for name in ("clean_task", "clean_task_2"):
            assert results[name]["health"]["status"] == "ok"
        assert "FAILED: bad_task" in run.stdout
        assert "Done. 2/3 succeeded." in run.stdout

    def test_the_failure_is_in_the_returned_result_and_the_other_tasks_still_ran(self, tmp_path: Path) -> None:
        from dataeval_flow import load_config, run_tasks

        config = write_project(tmp_path, failing_task=True)

        results = run_tasks(load_config(config), data_dir=tmp_path)

        assert list(results) == ["bad_task", "clean_task"]
        assert not results["bad_task"].success
        assert results["bad_task"].errors
        assert results["clean_task"].success

    def test_a_run_whose_tasks_all_succeed_exits_zero(self, tmp_path: Path) -> None:
        config = write_project(tmp_path, n_tasks=2)

        run = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(tmp_path / "out"))

        assert run.returncode == 0, run.stdout + run.stderr


class TestUnknownNames:
    @pytest.mark.parametrize(
        ("command", "kind"),
        [("workflows", "workflow"), ("evaluators", "evaluator")],
    )
    def test_an_unknown_workflow_or_evaluator_name_is_reported_with_the_installed_ones(
        self, command: str, kind: str
    ) -> None:
        run = run_cli(command, "no-such-type")

        assert run.returncode == 1
        assert f"ERROR: Unknown {kind}: 'no-such-type'. Installed: [" in run.stderr

    def test_an_unknown_step_name_is_reported(self) -> None:
        run = run_cli("steps", "no-such-step")

        assert run.returncode == 1
        assert "ERROR: 'no-such-step' names no step" in run.stderr

    def test_get_workflow_unknown_raises(self) -> None:
        from dataeval_flow.workflows import get_workflow

        with pytest.raises(ValueError, match="Unknown workflow: 'no-such-type'"):
            get_workflow("no-such-type")

    def test_a_config_naming_a_workflow_type_that_does_not_exist_is_refused_naming_it(self, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        pipeline = pipeline_dict()
        pipeline["workflows"][0] = {"name": "clean", "type": "data-cleaning"}
        config.write_text(yaml.safe_dump(pipeline))

        run = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(tmp_path / "out"))

        assert run.returncode == 1
        assert "Unknown workflow: 'data-cleaning'" in run.stdout

    @pytest.mark.parametrize(
        ("args", "named"),
        [(["--log-format", "xml"], "--log-format"), (["--no-such-option"], "--no-such-option")],
        ids=["bad-value", "unknown-option"],
    )
    def test_an_unsupported_option_exits_two_and_names_it(self, args: list[str], named: str) -> None:
        run = run_cli(*args)

        assert run.returncode == 2
        assert named in run.stderr
