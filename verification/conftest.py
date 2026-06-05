"""Verification test configuration and report generation plugin.

Provides:
- ``test_case(*ids)`` marker linking tests to ``test-case-<id>.md`` in the meta repo
- JSON report generation mapping test case numbers to pass/fail results
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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "test_case(*ids): link test to one or more test-case-<id>.md files in the meta repo",
    )


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


def _tc_status(tests: list[dict]) -> str:
    statuses = {t["status"] for t in tests}
    if statuses & {"failed", "error"}:
        return "failed"
    if statuses == {"skipped"}:
        return "skipped"
    return "passed"


def pytest_sessionfinish(session, exitstatus):
    results: dict[str, list[dict]] = {}
    for item in session.items:
        status = _get_test_status(item)
        for marker in item.iter_markers("test_case"):
            for tc_num in marker.args:
                tc_id = f"test-case-{tc_num}"
                results.setdefault(tc_id, []).append(
                    {
                        "test": item.nodeid,
                        "file": str(Path(item.path).relative_to(VERIFICATION_DIR)),
                        "status": status,
                    },
                )

    if not results:
        return

    tc_statuses = {tc_id: _tc_status(tests) for tc_id, tests in results.items()}
    passed = sum(1 for s in tc_statuses.values() if s == "passed")
    failed = sum(1 for s in tc_statuses.values() if s == "failed")
    skipped = sum(1 for s in tc_statuses.values() if s == "skipped")

    report = {
        "summary": {
            "total_test_cases": len(results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "test_cases": {
            tc_id: {
                "meta_repo_file": f"test-cases/{tc_id}.md",
                "status": _tc_status(tests),
                "tests": tests,
            }
            for tc_id, tests in sorted(results.items())
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    report_path = OUTPUT_DIR / "verification_report.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text())
    summary = report["summary"]
    terminalreporter.section("Verification Report")
    terminalreporter.write_line(
        f"Test Cases: {summary['total_test_cases']} total, "
        f"{summary['passed']} passed, "
        f"{summary['failed']} failed, "
        f"{summary['skipped']} skipped",
    )
    terminalreporter.write_line(f"Report: {report_path}")
    for tc_id, tc_data in report["test_cases"].items():
        if tc_data["status"] == "failed":
            terminalreporter.write_line(f"  FAILED: {tc_id} ({tc_data['meta_repo_file']})")
            for test in tc_data["tests"]:
                if test["status"] in ("failed", "error"):
                    terminalreporter.write_line(f"    - {test['test']}")


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
