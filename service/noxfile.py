"""Validate the companion package in an already prepared Flow CPU environment."""

import sys

import nox

nox.options.default_venv_backend = "none"


@nox.session
def lint(session: nox.Session) -> None:
    """Check formatting and Python lint rules without changing source files."""
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def schema(session: nox.Session) -> None:
    """Verify runtime discovery and OpenAPI agree with accepted assessment configuration."""
    session.run("pytest", "tests/test_schema.py", "-q")


@nox.session(name="type")
def typecheck(session: nox.Session) -> None:
    """Check the companion package against the prepared runtime's types."""
    session.run("pyright", "--pythonpath", sys.executable, "src")


@nox.session
def test(session: nox.Session) -> None:
    """Exercise API, real workflows, persistence, and process lifecycle with coverage."""
    session.run("pytest", "-q", "--cov=dataeval_flow_service", "--cov-report=term-missing", "--cov-fail-under=90")
