"""Verification test configuration and report generation plugin.

Provides:
- ``test_case(*ids)`` marker linking tests to ``test-case-<id>.md`` in the meta repo
- JSON report generation mapping test case numbers to pass/fail results, plus
  run metadata and every test's status for the metarepo test-results log
- Terminal summary of verification results
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
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


def pytest_sessionstart(session):
    session.config._verification_started = datetime.now(UTC)


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
        # An xfail lands as `skipped` with `wasxfail` set. Keep it distinct: a known,
        # documented gap is not the same as an unrun test, and a strict xpass stays
        # `failed` so the mark gets removed when the gap closes.
        if call.skipped and hasattr(call, "wasxfail"):
            return "xfailed"
        if call.passed:
            return "passed"
        if call.skipped:
            return "skipped"
        return "failed"
    setup = reports.get("setup")
    if setup is not None and setup.skipped:
        return "skipped"
    return "error"


def _get_xfail_reason(item) -> str | None:
    """Return the xfail mark's reason for an xfailed test, so the report can quote it."""
    call = getattr(item, "_verification_reports", {}).get("call")
    return getattr(call, "wasxfail", None) or None


def _get_test_message(item) -> str | None:
    """Return the first line of an item's skip, xfail, or failure reason, if any."""
    reports = getattr(item, "_verification_reports", {})
    for phase in ("setup", "call", "teardown"):
        r = reports.get(phase)
        if r is None:
            continue
        if r.skipped:
            reason = getattr(r, "wasxfail", None)
            if reason is None and isinstance(r.longrepr, tuple):
                reason = r.longrepr[2].removeprefix("Skipped: ")
            return reason.splitlines()[0][:200] if reason else None
        if r.failed:
            crash = getattr(r.longrepr, "reprcrash", None)
            message = crash.message if crash is not None else str(r.longrepr)
            return message.splitlines()[0][:200] if message else None
    return None


def _utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _tc_status(tests: list[dict]) -> str:
    statuses = {t["status"] for t in tests}
    if statuses & {"failed", "error"}:
        return "failed"
    if statuses == {"skipped"}:
        return "skipped"
    # One xfail keeps the whole case off a clean pass, so a known gap is not
    # hidden in the VCRM by the passing tests beside it.
    if "xfailed" in statuses:
        return "xfailed"
    return "passed"


def pytest_sessionfinish(session, exitstatus):
    results: dict[str, list[dict]] = {}
    all_tests: dict[str, dict] = {}
    for item in session.items:
        status = _get_test_status(item)
        message = _get_test_message(item)
        all_tests[item.nodeid] = {"status": status, **({"message": message} if message else {})}
        for marker in item.iter_markers("test_case"):
            for tc_num in marker.args:
                tc_id = f"test-case-{tc_num}"
                entry = {
                    "test": item.nodeid,
                    "file": str(Path(item.path).relative_to(VERIFICATION_DIR)),
                    "status": status,
                }
                reason = _get_xfail_reason(item)
                if reason:
                    entry["reason"] = reason
                results.setdefault(tc_id, []).append(entry)

    if not results:
        return

    tc_statuses = {tc_id: _tc_status(tests) for tc_id, tests in results.items()}
    passed = sum(1 for s in tc_statuses.values() if s == "passed")
    failed = sum(1 for s in tc_statuses.values() if s == "failed")
    skipped = sum(1 for s in tc_statuses.values() if s == "skipped")
    xfailed = sum(1 for s in tc_statuses.values() if s == "xfailed")

    report = {
        "summary": {
            "total_test_cases": len(results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "xfailed": xfailed,
        },
        "test_cases": {
            tc_id: {
                "meta_repo_file": f"test-cases/{tc_id}.md",
                "status": _tc_status(tests),
                "tests": tests,
            }
            for tc_id, tests in sorted(results.items())
        },
        "run": {
            "started": _utc(session.config._verification_started),
            "finished": _utc(datetime.now(UTC)),
            "exit_status": int(exitstatus),
            "command": " ".join(["pytest", *session.config.invocation_params.args]),
            "python": platform.python_version(),
            "platform": f"{platform.system().lower()} {platform.machine()}",
        },
        "tests": all_tests,
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
        f"{summary['skipped']} skipped, "
        f"{summary.get('xfailed', 0)} xfailed",
    )
    terminalreporter.write_line(f"Report: {report_path}")
    for tc_id, tc_data in report["test_cases"].items():
        if tc_data["status"] == "failed":
            terminalreporter.write_line(f"  FAILED: {tc_id} ({tc_data['meta_repo_file']})")
            for test in tc_data["tests"]:
                if test["status"] in ("failed", "error"):
                    terminalreporter.write_line(f"    - {test['test']}")
        elif tc_data["status"] == "xfailed":
            terminalreporter.write_line(f"  XFAILED: {tc_id} ({tc_data['meta_repo_file']})")
            for test in tc_data["tests"]:
                if test["status"] == "xfailed":
                    terminalreporter.write_line(f"    - {test['test']}: {test.get('reason', '')}")


# ---------------------------------------------------------------------------
# Shared workflow fixtures (TC-6-1 and downstream tasks)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_pipeline_config(tmp_path: Path) -> tuple[object, Path]:
    """Return a ``PipelineConfig`` + ``data_dir`` that runs a single trivial workflow.

    The fixture writes a tiny synthetic ImageFolder to ``tmp_path/imgs`` and
    composes a single ``data-cleaning`` task referencing it.  Field names follow
    the actual pydantic schemas (``datasets``/``sources``/``workflows``/``tasks``
    as lists of named items).
    """
    from dataeval_flow import PipelineConfig
    from dataeval_flow.config import ImageFolderDatasetConfig, SourceConfig, TaskConfig
    from dataeval_flow.config.extractors import FlattenExtractorConfig
    from dataeval_flow.workflows.data_cleaning import DataCleaningConfig
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
            DataCleaningConfig(
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

    from dataeval_flow import PipelineConfig
    from dataeval_flow.config import ImageFolderDatasetConfig, SourceConfig, TaskConfig
    from dataeval_flow.config.extractors import FlattenExtractorConfig
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
