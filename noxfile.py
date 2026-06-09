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
    if UV_EXTRAS_OVERRIDE not in ["cpu", "cu118", "cu128"]:
        UV_EXTRAS_OVERRIDE = "cpu"

UV_EXTRAS = [UV_EXTRAS_OVERRIDE] + ["app"]
UV_EXTRAS_WITH_ONNX = UV_EXTRAS + ["onnx" if UV_EXTRAS_OVERRIDE == "cpu" else "onnx-gpu"]
UV_EXTRAS_WITH_ONNX_AND_OPENCV = UV_EXTRAS_WITH_ONNX + ["opencv"]

DOCS_ENVS = {"LANG": "C", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYDEVD_DISABLE_FILE_VALIDATION": "1"}


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
    """Run unit tests with coverage (90% threshold enforced).

    Only ``tests/`` runs here. The requirements verification suite lives in its
    own ``verify`` session so it stays out of the unit-coverage gate.
    """
    session.run(
        "pytest",
        "-n4",
        "--dist=loadscope",
        "--cov=src/dataeval_flow",
        "--cov-report=term",
        "--cov-report=xml:output/coverage.xml",
        "--cov-report=html:output/htmlcov",
        "--cov-fail-under=90",
        "--junitxml=output/junit.xml",
    )


@nox_uv.session(uv_groups=["verify"], uv_extras=UV_EXTRAS)
def verify(session: nox.Session) -> None:
    """Run the requirements verification suite (FR/NFR compliance).

    Runs ``verification/`` separately from unit tests: its own JUnit report and
    no coverage gate. The verification conftest writes
    ``output/verification_report.json`` (test-case traceability) on finish.
    """
    session.run(
        "pytest",
        "verification/",
        "--tb=short",
        "--junitxml=output/verify.xml",
        *session.posargs,
    )


@nox_uv.session(python="3.10", uv_only_groups=["base"], reuse_venv=False)
def deps(session: nox.Session) -> None:
    """Run unit tests against minimum supported Python with lowest declared dependencies and no optional extras.

    Combines two SDP checks into one fast session:
      * TR-1-H-4 / TR-2-H-5 — verify the project still works at the *minimum*
        declared versions of every direct dependency (``--resolution=lowest-direct``).
      * TR-2-H-3 / TR-8-H-1 — verify the project works *without* its optional
        extras (no ``onnx``, ``opencv``, or ``app``). The ``cpu`` extra is the
        only one installed so torch/torchvision are available for tests that
        touch tensors; everything else is excluded via ``-m "not optional"`` and
        by ignoring ``tests/app`` (which imports ``textual`` at module level).
    """
    session.run_install("uv", "pip", "install", ".[cpu]", "--resolution=lowest-direct")
    session.run_install("uv", "pip", "install", "pytest", "pytest-asyncio", "pytest-xdist")
    session.run("pytest", "-m", "not optional", "--ignore=tests/app", "-n4", "--dist=loadscope")


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
    """Build Sphinx documentation.

    Notebooks are executed from a jupyter-cache that is fetched from the
    ``docs-artifacts/<branch>`` orphan branch, so cached notebooks are not
    re-executed (see docs/CACHE_MANAGEMENT.md).

    Pass 'clean' to clear the jupyter cache and force re-execution:
        nox -s docs -- clean
    Pass 'skip' to skip notebook execution entirely:
        nox -s docs -- skip
    """
    skip_notebooks = "skip" in session.posargs
    clean_notebooks = "clean" in session.posargs

    notebook_dir = "docs/source/notebooks"
    cache_dir = "docs/source/.jupyter_cache"

    # Convert py:percent notebooks to ipynb (py is source of truth, ignores timestamps)
    session.run("jupytext", "--to", "notebook", "--update", notebook_dir + "/*.py")

    if clean_notebooks:
        # Clear local jupyter cache to force re-execution of all notebooks
        session.log(f"Clearing jupyter cache at {cache_dir} to force re-execution...")
        session.run("rm", "-rf", cache_dir, external=True)
    elif not skip_notebooks:
        # Fetch cached notebook results from orphan artifact branch
        session.run("bash", "docs/fetch-docs-cache.sh", external=True)

    session.run("rm", "-rf", "output/docs", external=True)

    if not skip_notebooks:
        # Fix inconsistent cache state before building (db records without folders or vice versa)
        session.run("python", "docs/check_notebook_cache.py", "--fix")

    session.run(
        "sphinx-build",
        "--fail-on-warning",
        "--keep-going",
        "--fresh-env",
        "--show-traceback",
        "--builder",
        "html",
        "docs/source",
        "output/docs",
        env={**DOCS_ENVS, **({"NB_EXECUTION_MODE_OVERRIDE": "off"} if skip_notebooks else {})},
    )

    if not skip_notebooks:
        # Clean up stale cache entries after sphinx-build updates the cache
        session.run("python", "docs/check_notebook_cache.py", "--clean")


@nox_uv.session(uv_extras=["cpu"])
def schema(session: nox.Session) -> None:
    """Regenerate config/params.schema.json from PipelineConfig and verify it is up to date.

    Usage:
      nox -s schema           # Auto-fix locally; check-only in CI ($CI set)
      nox -s schema -- fix    # Regenerate and overwrite the file (explicit)
    """
    args = ["python", "config/sync_schema.py"]
    if "fix" in session.posargs or "--fix" in session.posargs or not os.environ.get("CI"):
        args.append("--fix")
    session.run(*args)


@nox_uv.session(uv_only_groups=["lock"], uv_sync_locked=False)
def lock(session: nox.Session) -> None:
    """Lock dependencies for uv, Poetry, and conda.

    Regenerates `uv.lock`, `requirements.txt`, `poetry.lock`, and
    `environment.yml`. Pass `upgrade` to bump dependencies to the latest
    versions satisfying constraints.

      nox -s lock                # refresh lockfiles preserving pins
      nox -s lock -- upgrade     # bump to latest compatible versions
    """
    upgrade_args = ["--upgrade"] if "upgrade" in session.posargs else []
    session.run("uv", "lock", *upgrade_args)
    session.run("uv", "export", "--no-emit-project", "-o", "requirements.txt")
    session.run("poetry", "lock")
    session.run(
        "p2c",
        "yaml",
        "--pyproject",
        "pyproject.toml",
        "--python-include",
        "infer",
        "-n",
        "dataeval-flow",
        "-o",
        "environment.yml",
    )


@nox_uv.session(uv_only_groups=["lock"])
def check(session: nox.Session) -> None:
    """Validate lock files are up to date (uv.lock + poetry.lock)."""
    session.run("uv", "lock", "--check")
    session.run("poetry", "check", "--lock")


@nox_uv.session(uv_only_groups=["docker"], uv_no_install_project=True)
def docker_gen(session: nox.Session) -> None:
    """Generate Dockerfile.<variant> files from docker/Dockerfile.j2 template."""
    session.run("python", "docker/generate.py")


@nox_uv.session(uv_groups=["test"], uv_extras=UV_EXTRAS)
def docker_smoke(session: nox.Session) -> None:
    """Container-focused smoke test invoked from the Dockerfile `test` stage.

    Skips repo-state checks (lint, schema, lockfile validation) and the 90%
    coverage gate — those are already enforced by the MR pipeline before any
    Docker build runs. What's left is the slice that only the *built image*
    can validate: that the frozen, variant-specific venv (cpu / cu118 / cu128)
    actually produces a runnable package end-to-end.

    Coverage:
      1. Import smoke — package + key submodules import cleanly with the
         variant's torch/onnx wheels (catches missing extras, wrong wheel
         platform, ABI mismatch).
      2. CLI smoke — ``python -m dataeval_flow --help`` exits 0 (validates the
         entrypoint and argparse wiring without requiring a config).
      3. Wiring tests — fast integration tests against the installed venv:
         config loading, runner, main entrypoint, and e2e orchestration.
    """
    session.run("python", "-c", "import dataeval_flow; from dataeval_flow import runner, workflow")
    session.run("python", "-m", "dataeval_flow", "--help")
    session.run(
        "pytest",
        "tests/test_config_loader.py",
        "tests/test_main_run.py",
        "tests/test_runner.py",
        "tests/test_e2e.py",
        "-n4",
        "--dist=loadscope",
        "-x",
    )
