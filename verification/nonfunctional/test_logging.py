"""TC-34-1 (NFR-4) — stdlib logging integration."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from verification.helpers import run_cli
from verification.nonfunctional._support import write_project

pytestmark = pytest.mark.required

# A console record in the structured format: ISO-8601 UTC time, then the level in brackets, then the message.
STAMPED = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z \[(INFO|WARNING|ERROR)\]\s*\S")


class TestLogging:
    def test_logging_module_exports_setup(self) -> None:
        from dataeval_flow import _logging

        assert hasattr(_logging, "setup_logging"), "expected setup_logging entrypoint"

    def test_setup_logging_attaches_root_handler(self) -> None:
        from dataeval_flow import _logging
        from dataeval_flow._logging import setup_logging

        root = logging.getLogger()
        original_handlers = list(root.handlers)
        original_level = root.level
        original_initialized = _logging._initialized
        try:
            for h in original_handlers:
                root.removeHandler(h)
            _logging._initialized = False
            setup_logging()
            assert len(root.handlers) >= 1
            setup_logging()  # a second call adds nothing
            assert len(root.handlers) == 1
        finally:
            for h in list(root.handlers):
                root.removeHandler(h)
            for h in original_handlers:
                root.addHandler(h)
            root.setLevel(original_level)
            _logging._initialized = original_initialized

    @pytest.mark.parametrize("module", ["dataeval_flow._runner", "dataeval_flow._orchestrator", "dataeval_flow._cache"])
    def test_module_logger_is_logging_logger(self, module: str) -> None:
        import importlib

        logger = importlib.import_module(module)._logger

        assert type(logger) is logging.Logger
        assert logger.name.startswith("dataeval_flow.")

    def test_importing_the_package_attaches_no_root_handler(self) -> None:
        """Library use must not configure logging: no root handler, and module loggers are plain loggers."""
        code = (
            "import logging\n"
            "import dataeval_flow\n"
            "import dataeval_flow._runner, dataeval_flow._orchestrator, dataeval_flow._cache\n"
            "assert not logging.getLogger().handlers, logging.getLogger().handlers\n"
            "for mod in (dataeval_flow._runner, dataeval_flow._orchestrator, dataeval_flow._cache):\n"
            "    assert type(mod._logger) is logging.Logger, mod\n"
            "assert all(isinstance(h, logging.NullHandler) for h in logging.getLogger('dataeval_flow').handlers)\n"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_console_records_carry_utc_timestamp_and_level_for_each_log_format(self, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        args = ("-c", str(config), "-d", str(tmp_path), "-vv")

        before = datetime.now(UTC).replace(microsecond=0)
        structured = run_cli(*args, "--log-format", "structured")
        after = datetime.now(UTC)
        assert structured.returncode == 0, structured.stdout + structured.stderr
        records = [m for line in structured.stdout.splitlines() if (m := STAMPED.match(line))]
        assert any(m.group(2) == "INFO" for m in records)
        for m in records:
            moment = datetime.fromisoformat(m.group(1)).replace(tzinfo=UTC)
            assert before - timedelta(seconds=1) <= moment <= after + timedelta(seconds=1)

        plain = run_cli(*args, "--log-format", "plain")
        assert plain.returncode == 0, plain.stdout + plain.stderr
        assert "OK: clean_task" in plain.stdout
        assert not any(STAMPED.match(line) for line in plain.stdout.splitlines())
        assert not re.search(r"^\d{4}-\d{2}-\d{2}T", plain.stdout, re.MULTILINE)

    def test_the_log_format_comes_from_the_environment_and_the_option_beats_it(self, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        args = ("-c", str(config), "-d", str(tmp_path), "-vv")

        from_env = run_cli(*args, env={"DATAEVAL_LOG_FORMAT": "plain"})
        flag_wins = run_cli(*args, "--log-format", "structured", env={"DATAEVAL_LOG_FORMAT": "plain"})

        assert from_env.returncode == 0, from_env.stdout + from_env.stderr
        assert not any(STAMPED.match(line) for line in from_env.stdout.splitlines())
        assert any(STAMPED.match(line) for line in flag_wins.stdout.splitlines())

    def test_a_run_writes_a_debug_log_with_utc_timestamps_beside_its_results(self, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        out = tmp_path / "out"

        run = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(out))

        assert run.returncode == 0, run.stdout + run.stderr
        lines = (out / "result.log").read_text(encoding="utf-8").splitlines()
        record = re.compile(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z \[(DEBUG|INFO |WARNING|ERROR)\] dataeval_flow[\w.]*: "
        )
        records = [m.group(1) for line in lines if (m := record.match(line))]
        assert "DEBUG" in records  # the file keeps more than the console shows at the default verbosity
        assert any("OK: clean_task" in line for line in lines)
