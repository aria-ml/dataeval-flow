"""TC-1-1 — Python version compatibility."""

from __future__ import annotations

import importlib.metadata as md
import sys

import pytest
from packaging.specifiers import SpecifierSet


@pytest.mark.test_case("1-1")
class TestPythonVersions:
    def test_running_on_supported_version(self) -> None:
        requires_python = md.metadata("dataeval-flow")["Requires-Python"]
        spec = SpecifierSet(requires_python)
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert current in spec, f"Python {current} is outside {requires_python}"

    def test_import_succeeds_on_current_version(self) -> None:
        import dataeval_flow

        assert dataeval_flow is not None
