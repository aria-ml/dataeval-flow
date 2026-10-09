"""TC-31-1 (NFR-1) — Python version compatibility, and that the places naming the supported versions agree."""

from __future__ import annotations

import importlib.metadata as md
import re
import sys
import tomllib

import pytest
import yaml
from packaging.specifiers import SpecifierSet

from verification.helpers import REPO_ROOT

pytestmark = pytest.mark.required

SUPPORTED = ["3.11", "3.12", "3.13", "3.14"]


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


class TestPythonVersions:
    def test_running_on_supported_version(self) -> None:
        requires_python = md.metadata("dataeval-flow")["Requires-Python"]
        spec = SpecifierSet(requires_python)
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert current in spec, f"Python {current} is outside {requires_python}"

    def test_import_succeeds_on_current_version(self) -> None:
        import dataeval_flow

        assert dataeval_flow is not None

    def test_requires_python_admits_exactly_the_supported_versions(self) -> None:
        spec = SpecifierSet(_pyproject()["project"]["requires-python"])
        assert SpecifierSet(md.metadata("dataeval-flow")["Requires-Python"]) == spec

        minors = [f"3.{minor}" for minor in range(8, 18)]
        assert [m for m in minors if f"{m}.0" in spec and f"{m}.99" in spec] == SUPPORTED

    def test_the_trove_classifiers_name_the_supported_versions(self) -> None:
        classifiers = _pyproject()["project"]["classifiers"]
        versions = [
            m.group(1) for c in classifiers if (m := re.fullmatch(r"Programming Language :: Python :: (3\.\d+)", c))
        ]

        assert versions == SUPPORTED

    def test_the_ci_test_and_verify_matrix_covers_every_supported_version(self) -> None:
        text = (REPO_ROOT / ".gitlab-ci.yml").read_text()

        for job in ("test", "verify"):
            block = re.search(rf"job_name: {job}\n(?:.*\n)*?\s+python_versions: (\[.*\])", text)
            assert block is not None, f"no python_versions for the {job} job"
            assert yaml.safe_load(block.group(1)) == SUPPORTED

    def test_the_conda_environment_and_container_use_a_supported_python(self) -> None:
        spec = SpecifierSet(_pyproject()["project"]["requires-python"])
        env = yaml.safe_load((REPO_ROOT / "environment.yml").read_text())
        pins = [d for d in env["dependencies"] if isinstance(d, str) and d.startswith("python")]
        variants = yaml.safe_load((REPO_ROOT / "docker" / "variants.yaml").read_text())

        assert len(pins) == 1
        assert SpecifierSet(pins[0].removeprefix("python")) == spec
        assert f"{variants['python_version']}.0" in spec
