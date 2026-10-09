"""The container's help text and reference page list every command the CLI offers (DR-2.6-H-2)."""

import argparse
import re
from pathlib import Path

import pytest

from dataeval_flow.__main__ import _build_parser

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = sorted(next(a for a in _build_parser()._actions if isinstance(a, argparse._SubParsersAction)).choices)


@pytest.mark.parametrize("command", COMMANDS)
def test_entrypoint_help_lists_the_command(command: str):
    help_text = (ROOT / "docker" / "entrypoint.sh").read_text()
    assert re.search(rf"^\s+{command}\s{{2,}}\S", help_text, re.M), f"docker/entrypoint.sh help omits `{command}`"


@pytest.mark.parametrize("command", COMMANDS)
def test_container_reference_lists_the_command(command: str):
    reference = (ROOT / "docs" / "source" / "reference" / "containers.md").read_text()
    assert re.search(rf"^\| `{command}`\s+\|", reference, re.M), f"containers.md omits `{command}`"
