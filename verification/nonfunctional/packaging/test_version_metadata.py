"""TC-32-1 (NFR-2) — packaging metadata and what the distribution contains."""

from __future__ import annotations

import importlib.metadata as md
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from verification.helpers import REPO_ROOT

pytestmark = pytest.mark.required

DIST = "dataeval-flow"


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


class TestVersionMetadata:
    def test_version_is_valid_pep440(self) -> None:
        v = md.version(DIST)
        assert v
        Version(v)  # raises InvalidVersion if not PEP 440 compliant

    def test_the_package_reports_the_version_the_metadata_records(self) -> None:
        import dataeval_flow

        assert dataeval_flow.__version__ == md.version(DIST)

    def test_package_name(self) -> None:
        meta = md.metadata(DIST)
        assert meta["Name"].lower() == DIST

    def test_requires_python(self) -> None:
        meta = md.metadata(DIST)
        assert SpecifierSet(meta["Requires-Python"]) == SpecifierSet(">=3.11,<3.15")

    def test_license_set(self) -> None:
        meta = md.metadata(DIST)
        license_val = meta.get("License") or meta.get("License-Expression") or ""
        assert "MIT" in license_val.upper()

    def test_the_metadata_names_the_readme_and_project_urls(self) -> None:
        meta = md.metadata(DIST)

        assert meta["Summary"]
        assert meta["Description-Content-Type"].startswith("text/markdown")
        assert any(url.startswith("Documentation, https://") for url in meta.get_all("Project-URL") or [])

    def test_py_typed_marker_present(self) -> None:
        import dataeval_flow

        pkg_root = Path(dataeval_flow.__file__).parent
        assert (pkg_root / "py.typed").exists()

    def test_no_test_files_in_installed_package(self) -> None:
        import dataeval_flow

        pkg_root = Path(dataeval_flow.__file__).parent
        assert not any(p.name == "tests" for p in pkg_root.iterdir())
        assert not any(p.name == "verification" for p in pkg_root.iterdir())

    def test_no_test_or_verification_file_is_recorded_for_the_distribution(self) -> None:
        files = md.distribution(DIST).files or []

        assert files
        assert not [f for f in files if f.parts[0] in ("tests", "verification", "docs")]


class TestBuildConfiguration:
    def test_the_build_follows_pep_517_and_518(self) -> None:
        build = _pyproject()["build-system"]

        assert build["build-backend"] == "hatchling.build"
        assert "hatchling" in build["requires"]

    def test_the_wheel_and_sdist_include_only_the_package_source(self) -> None:
        targets = _pyproject()["tool"]["hatch"]["build"]["targets"]

        assert targets["wheel"]["include"] == ["src/dataeval_flow"]
        assert targets["sdist"]["include"] == ["src/dataeval_flow"]
        assert targets["wheel"]["sources"] == {"src/dataeval_flow": "dataeval_flow"}

    def test_the_version_comes_from_the_repository_tags_and_is_not_declared_by_hand(self) -> None:
        project = _pyproject()

        assert "version" in project["project"]["dynamic"]
        assert "version" not in project["project"]
        assert project["tool"]["hatch"]["version"]["source"] == "vcs"
