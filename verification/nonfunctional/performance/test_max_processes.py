"""TC-35-1 — the worker-process limit reaches DataEval for every task and is recorded in the result."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from dataeval_flow import run_tasks
from verification.nonfunctional._support import write_project

pytestmark = [pytest.mark.required, pytest.mark.performance]


@pytest.fixture
def set_max_processes_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Record every ``dataeval.config.set_max_processes`` call (still forwarding to DataEval)."""
    import dataeval.config

    calls: list[int] = []
    real = dataeval.config.set_max_processes

    def spy(max_processes: int | None) -> None:
        calls.append(max_processes)  # type: ignore[arg-type]
        real(max_processes)

    monkeypatch.setattr(dataeval.config, "set_max_processes", spy)
    yield calls
    real(None)


@pytest.fixture
def restore_root_logging() -> Iterator[None]:
    """Undo the console handler that running the CLI in-process attaches to the root logger."""
    from dataeval_flow import _logging

    root = logging.getLogger()
    handlers, level, initialized = list(root.handlers), root.level, _logging._initialized
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    _logging._initialized = initialized


def _run_cli_in_process(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    from dataeval_flow.__main__ import main

    monkeypatch.setattr(sys, "argv", ["dataeval_flow", *argv])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0


class TestMaxProcesses:
    def test_config_field_reaches_dataeval_for_every_task_and_is_recorded(
        self, tmp_path: Path, set_max_processes_calls: list[int]
    ) -> None:
        from dataeval_flow import load_config

        config_path = write_project(tmp_path, n_tasks=2)
        cfg = load_config(config_path).model_copy(update={"max_processes": 3})

        results = run_tasks(cfg, data_dir=tmp_path)

        assert list(results) == ["clean_task", "clean_task_2"]
        assert set_max_processes_calls == [3, 3]
        assert [r.metadata.resolved_config["max_processes"] for r in results.values()] == [3, 3]

    def test_no_setting_leaves_dataeval_default(self, tmp_path: Path, set_max_processes_calls: list[int]) -> None:
        from dataeval_flow import load_config

        results = run_tasks(load_config(write_project(tmp_path)), data_dir=tmp_path)

        assert set_max_processes_calls == []
        assert "max_processes" not in results["clean_task"].metadata.resolved_config

    @pytest.mark.parametrize(
        ("argv", "env", "expected"),
        [
            pytest.param(["--max-processes", "2"], {}, 2, id="flag"),
            pytest.param([], {"DATAEVAL_MAX_PROCESSES": "3"}, 3, id="environment-variable"),
            pytest.param(["--max-processes", "2"], {"DATAEVAL_MAX_PROCESSES": "3"}, 2, id="flag-beats-variable"),
        ],
    )
    def test_flag_and_variable_reach_dataeval_for_every_task_and_are_recorded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        set_max_processes_calls: list[int],
        restore_root_logging: None,
        argv: list[str],
        env: dict[str, str],
        expected: int,
    ) -> None:
        config_path = write_project(tmp_path, n_tasks=2)
        out = tmp_path / "out"
        for name, value in env.items():
            monkeypatch.setenv(name, value)

        _run_cli_in_process(monkeypatch, "-c", str(config_path), "-d", str(tmp_path), "-o", str(out), *argv)

        assert set_max_processes_calls == [expected, expected]
        results = json.loads((out / "results" / "result.json").read_text())
        assert set(results) == {"clean_task", "clean_task_2"}
        for result in results.values():
            assert result["metadata"]["resolved_config"]["max_processes"] == expected

    def test_flag_overrides_the_config_field(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        set_max_processes_calls: list[int],
        restore_root_logging: None,
    ) -> None:
        import yaml

        config_path = write_project(tmp_path)
        config = yaml.safe_load(config_path.read_text())
        config["max_processes"] = 4
        config_path.write_text(yaml.safe_dump(config))

        _run_cli_in_process(monkeypatch, "-c", str(config_path), "-d", str(tmp_path), "--max-processes", "2")

        assert set_max_processes_calls == [2]
