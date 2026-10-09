"""TC-16-1 — the container's help text and reference page agree with the command line and the code.

The entrypoint prints this help when the container starts with no data or with ``--help``, and
``docs/source/reference/containers.md`` repeats it. These checks run without Docker: they compare both texts
with the argument parser and with the environment variables the code reads.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess

import pytest

from verification.helpers import REPO_ROOT

pytestmark = pytest.mark.required

ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"
REFERENCE = REPO_ROOT / "docs" / "source" / "reference" / "containers.md"

# The operational options and the variable that sets each one in a container.
OPTION_VARIABLES = {
    "--config": "DATAEVAL_CONFIG",
    "--data": "DATAEVAL_DATA",
    "--output": "DATAEVAL_OUTPUT",
    "--cache": "DATAEVAL_CACHE",
    "--task": "DATAEVAL_TASKS",
    "--log-format": "DATAEVAL_LOG_FORMAT",
    "--fail-on-warning": "DATAEVAL_FAIL_ON_WARNING",
    "--max-processes": "DATAEVAL_MAX_PROCESSES",
    "--verbose": "DATAEVAL_VERBOSITY",
}


def _parser() -> argparse.ArgumentParser:
    from dataeval_flow.__main__ import _build_parser

    return _build_parser()


def _commands() -> list[str]:
    parser = _parser()
    return sorted(next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices)


def _flags() -> set[str]:
    return {s for a in _parser()._actions for s in a.option_strings if s.startswith("--")}


def _section(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def _variables_read_by_the_code() -> set[str]:
    names: set[str] = set()
    for path in (REPO_ROOT / "src" / "dataeval_flow").rglob("*.py"):
        names |= set(re.findall(r'"(DATAEVAL_[A-Z_]+)"', path.read_text()))
    return names


def _entrypoint_options() -> str:
    return _section(ENTRYPOINT.read_text(), "COMMAND-LINE OPTIONS", "COMMANDS (optional")


def _entrypoint_variables() -> str:
    return _section(ENTRYPOINT.read_text(), "ENVIRONMENT VARIABLES", "COMMAND-LINE OPTIONS")


class TestEntrypointHelp:
    def test_the_entrypoint_script_is_valid_bash(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash is not installed")

        run = subprocess.run([bash, "-n", str(ENTRYPOINT)], capture_output=True, text=True, check=False)  # noqa: S603

        assert run.returncode == 0, run.stderr

    @pytest.mark.parametrize("command", _commands())
    def test_help_lists_every_command_the_cli_offers(self, command: str) -> None:
        assert re.search(rf"^\s+{command}\s{{2,}}\S", ENTRYPOINT.read_text(), re.MULTILINE), command

    def test_every_option_the_help_lists_exists(self) -> None:
        listed = set(re.findall(r"(--[a-z][\w-]*)", _entrypoint_options()))

        assert {"--config", "--data", "--output", "--cache", "--task", "--log-format"} <= listed
        assert listed - {"--no-fail-on-warning"} <= _flags(), sorted(listed - _flags())

    def test_every_variable_the_help_lists_is_read_by_the_code(self) -> None:
        listed = set(re.findall(r"^\s+(DATAEVAL_[A-Z_]+)", _entrypoint_variables(), re.MULTILINE))

        assert listed
        assert listed <= _variables_read_by_the_code() | {"DATAEVAL_SERVICE_HOST", "DATAEVAL_SERVICE_PORT"}

    @pytest.mark.parametrize(("option", "variable"), sorted(OPTION_VARIABLES.items()))
    def test_help_names_the_variable_for_each_operational_option(self, option: str, variable: str) -> None:
        assert variable in _entrypoint_variables()
        assert option in _flags()
        assert variable in _variables_read_by_the_code()

    def test_help_says_that_no_secrets_are_needed(self) -> None:
        assert "No secret mounts are required" in ENTRYPOINT.read_text()
        assert "None are read from the environment or baked into the image" in ENTRYPOINT.read_text()

    def test_help_shows_how_to_mount_data_output_and_cache_and_to_run_the_service(self) -> None:
        text = ENTRYPOINT.read_text()

        for target in ("$DATA_DIR", "$OUTPUT_DIR", "$CACHE_DIR"):
            assert f"target={target}" in text
        assert "python -m dataeval_flow serve" in text
        assert "-p 8001:8001" in text


class TestContainerReference:
    @pytest.mark.parametrize("command", _commands())
    def test_reference_lists_every_command_the_cli_offers(self, command: str) -> None:
        assert re.search(rf"^\| `{command}`\s+\|", REFERENCE.read_text(), re.MULTILINE), command

    def test_every_option_the_reference_lists_exists(self) -> None:
        table = _section(REFERENCE.read_text(), "## Command-line options", "Optional sub-commands")
        listed = {
            flag for row in table.splitlines() if row.startswith("|") for flag in re.findall(r"`(--[a-z][\w-]*)", row)
        }

        assert {"--config", "--data", "--output", "--cache"} <= listed
        assert listed <= _flags(), sorted(listed - _flags())

    def test_every_variable_the_reference_lists_is_read_by_the_code(self) -> None:
        table = _section(REFERENCE.read_text(), "## Environment variables", "`DATAEVAL_DATA` and")
        listed = set(re.findall(r"^\| `(DATAEVAL_[A-Z_]+)`", table, re.MULTILINE))

        assert {"DATAEVAL_DATA", "DATAEVAL_OUTPUT", "DATAEVAL_CACHE"} <= listed
        assert listed <= _variables_read_by_the_code()

    def test_the_reference_documents_the_service_port_and_the_three_health_endpoints(self) -> None:
        text = REFERENCE.read_text()

        assert "`DATAEVAL_SERVICE_PORT`" in text
        assert "| `8001`" in text
        for endpoint in ("/healthz", "/readyz", "/livez"):
            assert re.search(rf"^\| `{endpoint}`", text, re.MULTILINE), endpoint

    def test_the_documented_defaults_match_the_dockerfile(self) -> None:
        text = (REPO_ROOT / "docker" / "Dockerfile.cpu").read_text()
        reference = REFERENCE.read_text()

        assert "ENV DATAEVAL_DATA=/dataeval" in text
        assert "ENV DATAEVAL_OUTPUT=/output" in text
        assert re.search(r"\| `DATAEVAL_DATA`\s+\|[^|]*\|\s*`/dataeval` in the container", reference)
        assert re.search(r"\| `DATAEVAL_OUTPUT`\s+\|[^|]*\|\s*`/output` in the container", reference)
        assert "`0.0.0.0` in the container" in reference
        assert "ENV DATAEVAL_SERVICE_HOST=0.0.0.0" in text
