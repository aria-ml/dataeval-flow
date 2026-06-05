"""TC-1-7 — packaging metadata."""

from __future__ import annotations

import importlib.metadata as md
from pathlib import Path

import pytest

DIST = "dataeval-flow"


@pytest.mark.test_case("1-7")
class TestVersionMetadata:
    def test_version_is_valid_pep440(self) -> None:
        from packaging.version import Version

        v = md.version(DIST)
        assert v
        Version(v)  # raises InvalidVersion if not PEP 440 compliant

    def test_package_name(self) -> None:
        meta = md.metadata(DIST)
        assert meta["Name"].lower() == DIST

    def test_requires_python(self) -> None:
        meta = md.metadata(DIST)
        assert meta["Requires-Python"].strip().startswith(">=3.10")

    def test_license_set(self) -> None:
        meta = md.metadata(DIST)
        license_val = meta.get("License") or meta.get("License-Expression") or ""
        assert "MIT" in license_val.upper()

    def test_py_typed_marker_present(self) -> None:
        import dataeval_flow

        pkg_root = Path(dataeval_flow.__file__).parent
        assert (pkg_root / "py.typed").exists()

    def test_no_test_files_in_installed_package(self) -> None:
        import dataeval_flow

        pkg_root = Path(dataeval_flow.__file__).parent
        assert not any(p.name == "tests" for p in pkg_root.iterdir())
        assert not any(p.name == "verification" for p in pkg_root.iterdir())
