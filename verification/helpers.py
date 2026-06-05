"""Shared helpers for verification tests (paths, importlib utilities)."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent


def safe_import(module_name: str) -> ModuleType | None:
    """Import a module by name and return it, or None if unavailable."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, AttributeError):
        return None
