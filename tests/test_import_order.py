"""Every public package imports on its own, first, in a fresh interpreter."""

import importlib
import subprocess
import sys

import pytest

WORKFLOW_PACKAGES = [
    "data_analysis",
    "data_cleaning",
    "data_coverage",
    "data_prioritization",
    "data_splitting",
    "drift_monitoring",
    "metadata_triage",
    "ood_detection",
    "parameter_sweep",
]
PUBLIC = [
    "dataeval_flow",
    "dataeval_flow.config",
    "dataeval_flow.workflows",
    "dataeval_flow.evaluators",
    "dataeval_flow.evaluators.quality",
    "dataeval_flow.config.extractors",
    "dataeval_flow.config.transforms",
    *(f"dataeval_flow.workflows.{name}" for name in WORKFLOW_PACKAGES),
]


@pytest.mark.parametrize("module", PUBLIC)
def test_a_public_package_imports_first(module: str) -> None:
    done = subprocess.run(  # noqa: S603 -- fixed argv, no shell, no untrusted input
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize(
    "module", ["dataeval_flow.workflow", "dataeval_flow.evaluator", "dataeval_flow.workflows.cleaning"]
)
def test_the_old_packages_are_gone(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)
