"""TC-16-1 (NFR-5) — stdlib logging integration."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from verification.fixtures import write_cli_project
from verification.helpers import run_cli

pytestmark = pytest.mark.required


class TestLogging:
    def test_logging_module_exports_setup(self) -> None:
        from dataeval_flow import _logging

        assert hasattr(_logging, "setup_logging"), "expected setup_logging entrypoint"

    def test_setup_logging_attaches_root_handler(self) -> None:
        import logging

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
        finally:
            for h in list(root.handlers):
                root.removeHandler(h)
            for h in original_handlers:
                root.addHandler(h)
            root.setLevel(original_level)
            _logging._initialized = original_initialized

    def test_module_logger_is_logging_logger(self) -> None:
        import dataeval_flow.runner as runner

        assert isinstance(runner._logger, logging.Logger)

    def test_importing_the_package_attaches_no_root_handler(self) -> None:
        """Library use must not configure logging: no root handler, and module loggers are plain loggers."""
        code = (
            "import logging\n"
            "import dataeval_flow\n"
            "import dataeval_flow.runner, dataeval_flow.workflow.orchestrator, dataeval_flow.cache\n"
            "assert not logging.getLogger().handlers, logging.getLogger().handlers\n"
            "for mod in (dataeval_flow.runner, dataeval_flow.workflow.orchestrator, dataeval_flow.cache):\n"
            "    assert type(mod._logger) is logging.Logger, mod\n"
            "assert all(isinstance(h, logging.NullHandler) for h in logging.getLogger('dataeval_flow').handlers)\n"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_console_records_carry_utc_timestamp_and_level_for_each_log_format(self, tmp_path: Path) -> None:
        config = write_cli_project(tmp_path)
        args = ("-c", str(config), "-d", str(tmp_path), "-vv")
        stamped = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z \[(INFO|WARNING|ERROR)\] \S")

        before = datetime.now(timezone.utc).replace(microsecond=0)
        structured = run_cli(*args, "--log-format", "structured")
        after = datetime.now(timezone.utc)
        assert structured.returncode == 0, structured.stdout + structured.stderr
        records = [m for line in structured.stdout.splitlines() if (m := stamped.match(line))]
        assert any(m.group(2) == "INFO" for m in records)
        for m in records:
            moment = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
            assert before - timedelta(seconds=1) <= moment <= after + timedelta(seconds=1)

        plain = run_cli(*args, "--log-format", "plain")
        assert plain.returncode == 0, plain.stdout + plain.stderr
        assert "OK: clean_task" in plain.stdout
        assert not any(stamped.match(line) for line in plain.stdout.splitlines())
        assert not re.search(r"^\d{4}-\d{2}-\d{2}T", plain.stdout, re.MULTILINE)
