"""TC-16-1 — CLI entrypoints (headless, app, config)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.required


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

    def test_workflows_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dataeval_flow", "workflows", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_version_reports_the_installed_build(self) -> None:
        """A signed multi-variant image has to be able to say which build it is."""
        from dataeval_flow import __version__

        result = subprocess.run(
            [sys.executable, "-m", "dataeval_flow", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert __version__ in result.stdout

    def test_workflows_lists_the_registered_types(self) -> None:
        """Workflow discovery from the command line, for images without the TUI."""
        result = subprocess.run(
            [sys.executable, "-m", "dataeval_flow", "workflows", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        entries = json.loads(result.stdout)
        assert {"data-cleaning", "data-analysis"} <= {e["name"] for e in entries}
