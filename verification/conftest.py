"""Verification test configuration and report generation plugin.

Provides:
- JSON report of every test's outcome, keyed by node id (read by generate_metarepo.py)
- Terminal summary of verification results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from dataeval_flow import PipelineConfig

VERIFICATION_DIR = Path(__file__).parent
OUTPUT_DIR = VERIFICATION_DIR.parent / "output"

_PROJECT_ROOT = str(VERIFICATION_DIR.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    reports = getattr(item, "_verification_reports", {})
    reports[call.when] = report
    item._verification_reports = reports


def _get_test_status(item):
    reports = getattr(item, "_verification_reports", {})
    for phase in ("setup", "teardown"):
        r = reports.get(phase)
        if r is not None and r.failed:
            return "error"
    call = reports.get("call")
    if call is not None:
        if call.passed:
            return "passed"
        if call.skipped:
            return "skipped"
        return "failed"
    setup = reports.get("setup")
    if setup is not None and setup.skipped:
        return "skipped"
    return "error"


def pytest_sessionfinish(session, exitstatus):
    """Write ``output/verification_report.json``: the outcome of every collected test by node id.

    The registry (``verification/registry.yaml``) maps test cases and their steps to node ids, so
    the generator reads results from this map rather than from markers on the tests.
    """
    nodes = {item.nodeid: _get_test_status(item) for item in session.items}
    if not nodes:
        return

    counts = {s: sum(1 for v in nodes.values() if v == s) for s in ("passed", "failed", "error", "skipped")}
    report = {"summary": {"total_tests": len(nodes), **counts}, "nodes": dict(sorted(nodes.items()))}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "verification_report.json").write_text(json.dumps(report, indent=2))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a compact verification summary after the normal pytest output."""
    report_path = OUTPUT_DIR / "verification_report.json"
    if not report_path.exists():
        return

    report = json.loads(report_path.read_text())
    summary = report["summary"]

    terminalreporter.section("Verification Report")
    terminalreporter.write_line(
        f"Tests: {summary['total_tests']} total, {summary['passed']} passed, "
        f"{summary['failed']} failed, {summary['error']} errored, {summary['skipped']} skipped",
    )
    terminalreporter.write_line(f"Report: {report_path}")
    for node, status in report["nodes"].items():
        if status in ("failed", "error"):
            terminalreporter.write_line(f"  {status.upper()}: {node}")


# ---------------------------------------------------------------------------
# Shared workflow fixtures (TC-6-1 and downstream tasks)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_pipeline_config(tmp_path: Path) -> tuple[object, Path]:
    """Return a ``PipelineConfig`` + ``data_dir`` that runs a single trivial workflow.

    The fixture writes a tiny synthetic ImageFolder to ``tmp_path/imgs`` and
    composes a single ``data-cleaning`` task referencing it.  Field names follow
    the actual pydantic schemas (``datasets``/``sources``/``workflows``/``tasks``
    as lists of named items) rather than the dict-of-config shape sketched in
    the plan.
    """
    from dataeval_flow import (
        DataCleaningWorkflowConfig,
        FlattenExtractorConfig,
        ImageFolderDatasetConfig,
        PipelineConfig,
        SourceConfig,
        TaskConfig,
    )
    from verification.fixtures import write_image_folder

    write_image_folder(tmp_path / "imgs", n_per_class=4, n_classes=2)
    cfg = PipelineConfig(
        datasets=[
            ImageFolderDatasetConfig(
                name="main_ds",
                format="image_folder",
                path="imgs",
                infer_labels=True,
            ),
        ],
        sources=[
            SourceConfig(name="main", dataset="main_ds"),
        ],
        extractors=[
            FlattenExtractorConfig(name="flat", model="flatten"),
        ],
        workflows=[
            DataCleaningWorkflowConfig(
                name="clean_main",
                type="data-cleaning",
                outlier_method="zscore",
                outlier_flags=["dimension", "pixel"],
            ),
        ],
        tasks=[
            TaskConfig(
                name="clean_task",
                workflow="clean_main",
                sources="main",
                extractor="flat",
            ),
        ],
    )
    return cfg, tmp_path


@pytest.fixture
def image_folder_pipeline_builder(
    tmp_path: Path,
) -> Callable[..., tuple[PipelineConfig, Path]]:
    """Factory for building an ImageFolder-backed ``PipelineConfig``.

    Returns a callable that workflow tests invoke with their workflow + task
    config classes. Centralizes the boilerplate so per-workflow tests are
    just the configs unique to that workflow.

    Each invocation writes a fresh synthetic image folder to ``tmp_path`` so
    multiple builds in one test do not collide.
    """
    from collections.abc import Iterable, Sequence

    from dataeval_flow import (
        FlattenExtractorConfig,
        ImageFolderDatasetConfig,
        PipelineConfig,
        SourceConfig,
        TaskConfig,
    )
    from verification.fixtures import write_image_folder

    def _build(
        *,
        sources: Sequence[tuple[str, int]] = (("main", 0),),
        workflows: Iterable[object] = (),
        tasks: Iterable[TaskConfig] = (),
        n_per_class: int = 4,
        n_classes: int = 2,
        include_extractor: bool = True,
        extractor_batch_size: int | None = 8,
    ) -> tuple[PipelineConfig, Path]:
        ds_configs: list[ImageFolderDatasetConfig] = []
        src_configs: list[SourceConfig] = []
        for src_name, seed in sources:
            write_image_folder(
                tmp_path / src_name,
                n_per_class=n_per_class,
                n_classes=n_classes,
                seed=seed,
            )
            ds_configs.append(
                ImageFolderDatasetConfig(
                    name=f"{src_name}_ds",
                    format="image_folder",
                    path=src_name,
                    infer_labels=True,
                ),
            )
            src_configs.append(SourceConfig(name=src_name, dataset=f"{src_name}_ds"))

        extractors: list[FlattenExtractorConfig] = (
            [
                FlattenExtractorConfig(
                    name="flat",
                    model="flatten",
                    batch_size=extractor_batch_size,
                ),
            ]
            if include_extractor
            else []
        )

        cfg = PipelineConfig(
            datasets=ds_configs,
            sources=src_configs,
            extractors=extractors,
            workflows=list(workflows),
            tasks=list(tasks),
        )
        return cfg, tmp_path

    return _build


@pytest.fixture
def mixed_outcome_pipeline(
    image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
) -> tuple[PipelineConfig, Path]:
    """A two-task pipeline whose first task fails and whose second succeeds.

    ``bad_task`` asks for cluster-based outlier detection with no extractor, which the
    cleaning workflow rejects at run time; ``ok_task`` is an ordinary cleaning run.
    """
    from dataeval_flow import DataCleaningWorkflowConfig, TaskConfig

    return image_folder_pipeline_builder(
        workflows=[
            DataCleaningWorkflowConfig(
                name="bad",
                type="data-cleaning",
                outlier_method="zscore",
                outlier_flags=["pixel"],
                outlier_cluster_threshold=3.0,
            ),
            DataCleaningWorkflowConfig(
                name="ok", type="data-cleaning", outlier_method="zscore", outlier_flags=["pixel"]
            ),
        ],
        tasks=[
            TaskConfig(name="bad_task", workflow="bad", sources="main"),
            TaskConfig(name="ok_task", workflow="ok", sources="main", extractor="flat"),
        ],
    )
