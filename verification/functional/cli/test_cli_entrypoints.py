"""TC-16-1 — CLI entrypoints (headless, app, config)."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest


@pytest.mark.test_case("16-1")
class TestCLIEntrypoints:
    def test_console_script_on_path(self) -> None:
        assert shutil.which("dataeval-flow") is not None

    def test_module_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dataeval_flow", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "dataeval" in result.stdout.lower()

    def test_app_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dataeval_flow", "app", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_config_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dataeval_flow", "config", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
