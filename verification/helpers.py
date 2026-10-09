"""Shared helpers for verification tests (paths, importlib utilities)."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent


def safe_import(module_name: str) -> ModuleType | None:
    """Import a module by name and return it, or None if unavailable."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, AttributeError):
        return None


def run_cli(*args: str, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``python -m dataeval_flow <args>`` with no inherited ``DATAEVAL_*`` variables, plus *env*."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith("DATAEVAL_")}
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "dataeval_flow", *args],
        capture_output=True,
        text=True,
        check=False,
        env={**clean, **(env or {})},
        cwd=cwd,
    )
