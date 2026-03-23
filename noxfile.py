"""Nox automation for DataEval Workflows."""

import os

import nox
import nox_uv

nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = "always"
nox.options.sessions = ["lint", "type", "test", "schema", "check"]  # Default sessions to run

IS_CI = bool(os.environ.get("CI"))
UV_EXTRAS_OVERRIDE = os.environ.get("DATAEVAL_NOX_UV_EXTRAS_OVERRIDE", "")
if not UV_EXTRAS_OVERRIDE:
    if os.path.exists(".cuda-version"):
        with open(".cuda-version") as f:
            UV_EXTRAS_OVERRIDE = f.read().strip()
    if UV_EXTRAS_OVERRIDE not in ["cpu", "cu118", "cu124", "cu128"]:
        UV_EXTRAS_OVERRIDE = "cu118"

UV_EXTRAS = [UV_EXTRAS_OVERRIDE]
UV_EXTRAS_WITH_ONNX = UV_EXTRAS + ["onnx" if UV_EXTRAS_OVERRIDE == "cpu" else "onnx-gpu"]
UV_EXTRAS_WITH_ONNX_AND_OPENCV = UV_EXTRAS_WITH_ONNX + ["opencv"]


@nox_uv.session(uv_only_groups=["lint"], uv_no_install_project=True)
def lint(session: nox.Session) -> None:
    """Run linters and formatters (Ruff + Codespell)."""
    session.run("ruff", "check", "--show-fixes", "--exit-non-zero-on-fix", "--fix")
    session.run("ruff", "format", "--check" if IS_CI else ".")
    session.run("codespell")


@nox_uv.session(uv_groups=["type"], uv_extras=["cpu", "onnx"])
def type(session: nox.Session) -> None:  # noqa: A001
    """Run static type checking (Pyright).

    Two checks are run:
    1. Basic type check [TR-7-H-2 Hard requirement]
    2. Type completeness [TR-8-S-1 Soft requirement]
    """
    # TR-7-H-2 [Hard]: Static type checking with pyright
    session.run("pyright", "--stats", "src/", "tests/")

    # TR-8-S-1 [Soft]: Type completeness for public API
    session.run(
        "pyright",
        "--ignoreexternal",
        "--verifytypes",
        "dataeval_flow",
    )


@nox_uv.session(uv_groups=["test"], uv_extras=UV_EXTRAS)
def test(session: nox.Session) -> None:
    """Run tests with coverage (90% threshold enforced)."""
    session.run(
        "pytest",
        "-n4",
        "--dist=loadfile",
        "--cov=src/dataeval_flow",
        "--cov-report=term",
        "--cov-report=xml:output/coverage.xml",
        "--cov-report=html:output/htmlcov",
        "--cov-fail-under=90",
        "--junitxml=output/junit.xml",
    )


@nox_uv.session(uv_only_groups=["docs"], uv_no_install_project=True)
def docsync(session: nox.Session) -> None:
    """Sync notebook .py (percent) files with .ipynb via jupytext.

    For each .py file in docs/source/notebooks/:
      - If no .ipynb exists, generate one from the .py file.
      - If .ipynb exists and is newer than .py, sync ipynb -> py.
      - Otherwise, sync py -> ipynb.
    """
    from pathlib import Path

    nb_dir = Path("docs/source/notebooks")
    for py_file in sorted(nb_dir.glob("*.py")):
        ipynb_file = py_file.with_suffix(".ipynb")
        if not ipynb_file.exists():
            session.log(f"{ipynb_file.name} missing — generating from {py_file.name}")
            session.run("jupytext", "--to", "ipynb", str(py_file))
        elif ipynb_file.stat().st_mtime > py_file.stat().st_mtime:
            session.log(f"{ipynb_file.name} is newer — syncing ipynb -> py")
            session.run("jupytext", "--sync", str(ipynb_file))
        else:
            session.log(f"{py_file.name} is newer — syncing py -> ipynb")
            session.run("jupytext", "--sync", str(py_file))


@nox_uv.session(uv_groups=["docs"], uv_extras=UV_EXTRAS_WITH_ONNX_AND_OPENCV)
def docs(session: nox.Session) -> None:
    """Build Sphinx documentation."""
    session.run(
        "sphinx-build",
        "--fail-on-warning",
        "--keep-going",
        "--fresh-env",
        "--builder",
        "html",
        "docs/source",
        "output/docs",
    )


@nox_uv.session(uv_extras=["cpu"])
def schema(session: nox.Session) -> None:
    """Regenerate config/params.schema.json from PipelineConfig and verify it is up to date.

    Usage:
      nox -s schema           # Check only (CI-friendly) — fails if schema is stale
      nox -s schema -- fix    # Regenerate and overwrite the file
    """
    args = ["python", "config/sync_schema.py"]
    if "fix" in session.posargs or "--fix" in session.posargs:
        args.append("--fix")
    session.run(*args)


@nox_uv.session(uv_only_groups=["base"])
def check(session: nox.Session) -> None:
    """Validate lock file is up to date."""
    session.run("uv", "lock", "--check")


@nox_uv.session(uv_only_groups=["docker"], uv_no_install_project=True)
def docker_gen(session: nox.Session) -> None:
    """Generate Dockerfile.<variant> files from docker/Dockerfile.j2 template."""
    session.run("python", "docker/generate.py")
