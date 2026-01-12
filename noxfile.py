"""Nox automation for DataEval App."""

import os

import nox
import nox_uv

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "type", "test", "check"]  # Default sessions to run

IS_CI = bool(os.environ.get("CI"))


@nox_uv.session(uv_only_groups=["lint"], uv_no_install_project=True)
def lint(session: nox.Session) -> None:
    """Run linters and formatters (Ruff + Codespell)."""
    session.run("ruff", "check", "--show-fixes", "--exit-non-zero-on-fix", "--fix")
    session.run("ruff", "format", "--check" if IS_CI else ".")
    session.run("codespell")


@nox_uv.session(uv_groups=["type"])
def type(session: nox.Session) -> None:  # noqa: A001
    """Run static type checking (Pyright)."""
    session.run("pyright", "--stats", "src/", "tests/")
    session.run("pyright", "--ignoreexternal", "--verifytypes", "dataeval_app")


@nox_uv.session(uv_groups=["test"])
def test(session: nox.Session) -> None:
    """Run tests with coverage (90% threshold enforced)."""
    session.run(
        "pytest",
        "-n4",
        "--dist=loadfile",
        "--cov=src/dataeval_app",
        "--cov-report=term",
        "--cov-report=xml:output/coverage.xml",
        "--cov-report=html:output/htmlcov",
        "--cov-fail-under=90",
        "--junitxml=output/junit.xml",
    )


@nox_uv.session(uv_only_groups=["base"])
def check(session: nox.Session) -> None:
    """Validate lock file is up to date."""
    session.run("uv", "lock", "--check")
