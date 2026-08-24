"""Nox automation for DataEval Workflows."""

import argparse
import os
import shutil
import sys
from pathlib import Path

import nox
import nox_uv

nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = "always"
nox.options.sessions = ["lint", "type", "test", "schema", "check"]  # Default sessions to run

IS_CI = bool(os.environ.get("CI"))
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
PYTHON_DEFAULT = "3.11"
DEVICE_VARIANTS = ["cpu", "cu126", "cu130"]
DEVICE_DEFAULT = "cpu"
VENV_DEFAULT = ".venv"
CUDA_VERSION_FILE = ".cuda-version"
UV_EXTRAS_OVERRIDE = os.environ.get("DATAEVAL_NOX_UV_EXTRAS_OVERRIDE", "")
if not UV_EXTRAS_OVERRIDE:
    if os.path.exists(CUDA_VERSION_FILE):
        with open(CUDA_VERSION_FILE) as f:
            UV_EXTRAS_OVERRIDE = f.read().strip()
    if UV_EXTRAS_OVERRIDE not in DEVICE_VARIANTS:
        UV_EXTRAS_OVERRIDE = DEVICE_DEFAULT


def onnx_extra(device: str) -> str:
    """Name the onnx extra matching a device variant.

    CPU wheels for cpu, and for CUDA the GPU extra built against that same CUDA major --
    an onnxruntime-gpu wheel links against one specific CUDA runtime, so `cu130` has to
    pull `onnx-cu130` rather than a shared `onnx-gpu`.
    """
    return "onnx" if device == "cpu" else f"onnx-{device}"


UV_EXTRAS = [UV_EXTRAS_OVERRIDE] + ["app"]
UV_EXTRAS_WITH_ONNX = UV_EXTRAS + [onnx_extra(UV_EXTRAS_OVERRIDE)]
UV_EXTRAS_WITH_ONNX_AND_OPENCV = UV_EXTRAS_WITH_ONNX + ["opencv"]

DOCS_ENVS = {
    "LANG": "C",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYDEVD_DISABLE_FILE_VALIDATION": "1",
    "IPYTHONDIR": os.path.abspath("docs/source/.ipython"),
}


def python_version(session: nox.Session) -> str:
    """Return the ``major.minor`` version of the session's interpreter.

    Test artifacts are suffixed with this. The CI matrix (3.10/3.11/3.12) merges
    every job's ``output/`` into one artifact set, so unsuffixed reports would
    overwrite each other and leave nothing for the `coverage` job to combine.
    """
    out = session.run(
        "python",
        "-c",
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        silent=True,
        log=False,
    )
    # session.run returns True under --no-install/dry-run; fall back to the nox interpreter.
    return out.strip() if isinstance(out, str) else f"{sys.version_info.major}.{sys.version_info.minor}"


def resolve_option(session: nox.Session, label: str, provided: "str | None", allowed: list[str], default: str) -> str:
    """Validate an option value, prompting for it when it was not supplied on the command line.

    A value typed at the prompt falls back to `default` when unrecognized, but a value passed
    as a flag is an error, so that a typo such as `-d cu124` cannot silently install cpu.
    """
    value = provided
    interactive = value is None
    if interactive:
        prompt = f"Enter desired {label} [supported: {' '.join(allowed)}] [default: {default}]: "
        value = input(prompt).strip() if sys.stdin.isatty() else ""
    if not value:
        return default
    # Accept a full patch version such as "3.11.4" for a "3.11" option (a no-op for device names).
    value = ".".join(value.split(".")[:2])
    if value not in allowed:
        if not interactive:
            session.error(f"Unrecognized {label} '{value}' (supported: {', '.join(allowed)})")
        session.warn(f"Unrecognized {label} '{value}', defaulting to {default}")
        return default
    return value


# Declared with nox.session rather than nox_uv.session: this session builds the project
# environment itself, so it must not ask nox to create one first.
@nox.session(venv_backend="none")
def dev(session: nox.Session) -> None:
    """Create a local development environment. Prompts for any option not passed on a terminal.

    Usage: `uvx --with nox-uv nox -s dev -- [-p VERSION] [-d DEVICE] [-n NAME]`

    Bootstrap this one with `uvx`, not `uv run nox -s dev` -- nox itself lives in the
    environment the session rebuilds. `--with nox-uv` is required for every other session
    in this file, since it imports nox_uv at module scope.
    """
    parser = argparse.ArgumentParser(prog="nox -s dev --", add_help=False)
    parser.add_argument("-p", "--python", dest="python")
    parser.add_argument("-d", "--device", dest="device")
    parser.add_argument("-n", "--name", dest="name", default=VENV_DEFAULT)
    try:
        args = parser.parse_args(session.posargs)
    except SystemExit:
        session.error("Usage: nox -s dev -- [-p|--python VERSION] [-d|--device DEVICE] [-n|--name NAME]")

    if shutil.which("uv") is None:
        session.error("Install uv to continue: https://docs.astral.sh/uv/")

    python = resolve_option(session, "version of python", args.python, PYTHON_VERSIONS, PYTHON_DEFAULT)
    device = resolve_option(session, "device variant", args.device, DEVICE_VARIANTS, DEVICE_DEFAULT)
    venv_path = Path(args.name).resolve()

    # `uv sync` recreates the environment, which would pull the interpreter out from under a
    # nox running inside it.
    if Path(sys.prefix).resolve() == venv_path:
        session.error(
            f"Refusing to rebuild '{args.name}' while running from it. "
            "Use `uvx --with nox-uv nox -s dev` instead of `uv run nox -s dev`."
        )
    if venv_path.exists():
        if not (venv_path / "pyvenv.cfg").exists():
            session.error(f"'{args.name}' exists but is not a virtual environment; refusing to remove it.")
        session.log(f"Removing existing virtual environment at {args.name}...")
        shutil.rmtree(venv_path)

    session.log(f"Installing Python {python}+{device} to {args.name}...")
    # Mirrors UV_EXTRAS_WITH_ONNX, the set the test and type sessions build against.
    extras: list[str] = []
    for extra in [device, onnx_extra(device), "app"]:
        extras += ["--extra", extra]
    session.run(
        "uv",
        "sync",
        "-p",
        python,
        *extras,
        external=True,
        env={"UV_PROJECT_ENVIRONMENT": str(venv_path)},
    )
    # Recorded for DATAEVAL_NOX_UV_EXTRAS_OVERRIDE so later sessions pick up the same device.
    Path(CUDA_VERSION_FILE).write_text(f"{device}\n")

    session.log(f"Finished installing dataeval-flow for python {python} to {args.name}")
    session.log(f"Activate it with `source {args.name}/bin/activate`.")


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

    Reports are suffixed with the interpreter version so the CI matrix jobs can
    coexist in one artifact set, and the combined data file is moved under
    ``output/`` for the pipeline's `coverage` job to ``coverage combine``.
    """
    py = python_version(session)
    session.run(
        "pytest",
        "-n4",
        "--dist=loadscope",
        "--cov=src/dataeval_flow",
        "--cov-report=term",
        f"--cov-report=xml:output/coverage.{py}.xml",
        f"--cov-report=html:output/htmlcov.{py}",
        "--cov-fail-under=90",
        f"--junitxml=output/junit.{py}.xml",
    )
    session.run("mv", ".coverage", f"output/.coverage.{py}", external=True)


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
    # Render the meta repo artifacts (test-case stubs + VCRM) from that report.
    # Runs on every verify, not just at publish time, so a registry.yaml that no
    # longer matches the test suite fails in the MR rather than at release.
    session.run("python", "verification/generate_metarepo.py")


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
    """Lock dependencies for uv, pip, and conda.

    Regenerates `uv.lock`, `requirements.txt`, and `environment.yml`. Pass `upgrade`
    to bump dependencies to the latest versions satisfying constraints.

      nox -s lock                # refresh lockfiles preserving pins
      nox -s lock -- upgrade     # bump to latest compatible versions
    """
    upgrade_args = ["--upgrade"] if "upgrade" in session.posargs else []
    session.run("uv", "lock", *upgrade_args)
    session.run("uv", "export", "--no-emit-project", "-o", "requirements.txt")
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
    """Validate lock file is up to date."""
    session.run("uv", "lock", "--check")


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
    can validate: that the frozen, variant-specific venv (cpu / cu126 / cu130)
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
