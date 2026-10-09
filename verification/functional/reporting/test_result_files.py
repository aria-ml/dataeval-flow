"""TC-12-2 — the files a command-line run writes for its results, and the `result:` block that shapes them."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import load_config
from dataeval_flow.config import ResultConfig
from verification.functional.reporting._project import FAILING_QUALITY, Invocation, task, write_project
from verification.helpers import run_cli

pytestmark = pytest.mark.required

ALL_FORMATS = ["json", "text", "html", "junit", "markdown"]


@dataclass
class Run:
    """A finished command-line run: its exit code and the directory holding its results."""

    code: int
    results: Path

    def names(self) -> list[str]:
        return sorted(path.name for path in self.results.iterdir())


def _run(root: Path, **options: Any) -> Run:
    config = write_project(root, **options)
    proc = run_cli("-c", str(config), "-d", str(root), "-o", str(root / "out"))
    return Run(proc.returncode, root / "out" / "results")


@pytest.fixture(scope="module")
def default_run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """Defaults throughout, over a dataset whose duplicates check warns."""
    return _run(tmp_path_factory.mktemp("default"), duplicates=2)


@pytest.fixture(scope="module")
def all_formats_run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """Every format, over a failing task followed by a warning one."""
    return _run(
        tmp_path_factory.mktemp("all_formats"),
        duplicates=2,
        workflows=[FAILING_QUALITY],
        tasks=[task("bad_task", "q_fail"), task("clean_task")],
        formats=ALL_FORMATS,
    )


@pytest.fixture(scope="module")
def per_task_run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """Named files, one set per task, at summary detail."""
    return _run(
        tmp_path_factory.mktemp("per_task"),
        duplicates=2,
        tasks=[task("first"), task("second")],
        name="release",
        formats=["json", "text", "html"],
        per_task=True,
        detail="summary",
    )


class TestResultFiles:
    def test_default_run_writes_json_text_and_html_keyed_by_task(self, default_run: Run) -> None:
        assert default_run.code == 0
        assert default_run.names() == ["encoding.json", "result.html", "result.json", "result.txt"]
        results = json.loads((default_run.results / "result.json").read_text())
        assert list(results) == ["clean_task"]
        assert results["clean_task"]["kind"] == "workflow"
        assert results["clean_task"]["health"]["status"] == "warning"
        text = (default_run.results / "result.txt").read_text()
        assert "  QUALITY\n" in text
        assert "From Duplicates" in text  # the default detail is full
        assert (default_run.results / "result.html").read_text().lstrip().lower().startswith("<!doctype html")
        assert (default_run.results.parent / "result.log").is_file()

    def test_every_format_writes_a_file_with_its_own_extension(self, all_formats_run: Run) -> None:
        assert {"result.json", "result.txt", "result.html", "result.xml", "result.md"} <= set(all_formats_run.names())

    def test_a_failed_workflow_keeps_its_partial_steps_in_json_text_and_html(self, all_formats_run: Run) -> None:
        results = json.loads((all_formats_run.results / "result.json").read_text())
        assert list(results) == ["bad_task", "clean_task"]  # a failed task does not stop the next one
        assert results["bad_task"]["health"]["status"] == "failed"
        assert results["bad_task"]["health"]["failed_steps"] == ["outliers"]
        assert results["bad_task"]["steps"]["outliers"]["status"] == "failed"
        assert results["bad_task"]["errors"] == [
            "outliers: ValueError: n_expected_clusters=500 should be less than dataset size (10)"
        ]
        assert results["clean_task"]["health"]["status"] == "warning"
        assert "Health: failed" in (all_formats_run.results / "result.txt").read_text()
        assert "n_expected_clusters=500" in (all_formats_run.results / "result.html").read_text()

    def test_junit_report_has_a_suite_per_task_and_a_case_per_finding(self, all_formats_run: Run) -> None:
        root = ET.fromstring((all_formats_run.results / "result.xml").read_text())  # noqa: S314 - our own output
        assert root.tag == "testsuites"
        suites = {suite.get("name"): suite for suite in root.iter("testsuite")}
        assert list(suites) == ["bad_task", "clean_task"]
        # The failed task: an error case for its failed step.
        errors = suites["bad_task"].findall("./testcase[error]")
        assert [case.get("name") for case in errors] == ["step: outliers"]
        # The warning task: one failing case, the duplicates finding.
        clean_cases = {case.get("name"): case for case in suites["clean_task"].iter("testcase")}
        assert set(clean_cases) == {"Image Outliers", "Class Outliers", "Image Duplicates"}
        failing = [name for name, case in clean_cases.items() if case.find("failure") is not None]
        assert failing == ["Image Duplicates"]

    def test_markdown_summary_names_every_task_including_the_failed_one(self, all_formats_run: Run) -> None:
        text = (all_formats_run.results / "result.md").read_text()
        assert text.startswith("# dataeval-flow results")
        assert "## bad\\_task" in text
        assert "**Failed steps:** outliers" in text
        assert "## clean\\_task" in text
        assert "| warning | Image Duplicates | 4 exact (40.0%), 0 near (0.0%) |" in text

    def test_named_per_task_files_replace_the_single_run_files(self, per_task_run: Run) -> None:
        assert per_task_run.code == 0
        assert per_task_run.names() == [
            "encoding.json",
            "release-first.html",
            "release-first.json",
            "release-first.txt",
            "release-second.html",
            "release-second.json",
            "release-second.txt",
        ]
        assert list(json.loads((per_task_run.results / "release-first.json").read_text())) == ["first"]
        assert list(json.loads((per_task_run.results / "release-second.json").read_text())) == ["second"]

    def test_summary_detail_shortens_the_text_and_html_files_but_not_the_json(
        self, per_task_run: Run, default_run: Run
    ) -> None:
        summary_text = (per_task_run.results / "release-first.txt").read_text()
        full_text = (default_run.results / "result.txt").read_text()
        assert "SUMMARY" in summary_text
        assert "From Duplicates" not in summary_text
        assert len(summary_text.splitlines()) < len(full_text.splitlines()) * 0.7
        assert (
            len((per_task_run.results / "release-first.html").read_text())
            < len((default_run.results / "result.html").read_text()) * 0.8
        )
        summary_json = json.loads((per_task_run.results / "release-first.json").read_text())["first"]
        full_json = json.loads((default_run.results / "result.json").read_text())["clean_task"]
        assert summary_json["steps"]["duplicates"]["output"] == full_json["steps"]["duplicates"]["output"]

    def test_only_the_requested_formats_are_written_and_no_temporary_file_is_left(
        self, tmp_path: Path, cli: Callable[..., Invocation]
    ) -> None:
        config = write_project(tmp_path, formats=["markdown"])
        done = cli("-c", config, "-d", tmp_path, "-o", tmp_path / "out")
        assert done.code == 0, done.output
        written = {path.name for path in (tmp_path / "out" / "results").iterdir()}
        assert written - {"encoding.json"} == {"result.md"}  # no leftover .tmp, and no json, text or html

    def test_the_console_prints_the_summary_and_v_prints_the_full_report(
        self, tmp_path: Path, cli: Callable[..., Invocation]
    ) -> None:
        config = write_project(tmp_path, duplicates=2, detail="summary")
        quiet = cli("-c", config, "-d", tmp_path)
        assert "Health: 1 warning(s)" in quiet.stdout
        assert "From Duplicates" not in quiet.stdout
        verbose = cli("-c", config, "-d", tmp_path, "-v")
        assert "From Duplicates" in verbose.stdout  # whatever `detail` says, -v prints it all

    def test_without_an_output_directory_nothing_is_written(
        self, tmp_path: Path, cli: Callable[..., Invocation]
    ) -> None:
        config = write_project(tmp_path)
        before = {path.name for path in tmp_path.iterdir()}
        done = cli("-c", config, "-d", tmp_path)
        assert done.code == 0
        assert "Health: All checks passed" in done.stdout
        assert {path.name for path in tmp_path.iterdir()} == before


class TestReportWidthOnTheCommandLine:
    def _widest(self, tmp_path: Path) -> int:
        return max(len(line) for line in (tmp_path / "out" / "results" / "result.txt").read_text().splitlines())

    def test_result_width_sets_the_text_file_and_the_console(
        self, tmp_path: Path, cli: Callable[..., Invocation]
    ) -> None:
        config = write_project(tmp_path, width=60)
        done = cli("-c", config, "-d", tmp_path, "-o", tmp_path / "out")
        assert done.code == 0, done.output
        assert self._widest(tmp_path) == 60
        assert "=" * 60 in done.stdout
        assert "=" * 61 not in done.stdout

    def test_the_flag_overrides_the_config_and_the_environment_overrides_the_config(
        self, tmp_path: Path, cli: Callable[..., Invocation]
    ) -> None:
        config = write_project(tmp_path, width=60)
        out = tmp_path / "out"
        assert cli("-c", config, "-d", tmp_path, "-o", out, "--report-width", "50").code == 0
        assert self._widest(tmp_path) == 50
        assert cli("-c", config, "-d", tmp_path, "-o", out, env={"DATAEVAL_REPORT_WIDTH": "100"}).code == 0
        assert self._widest(tmp_path) == 100
        flagged = cli(
            "-c", config, "-d", tmp_path, "-o", out, "--report-width", "70", env={"DATAEVAL_REPORT_WIDTH": "100"}
        )
        assert flagged.code == 0
        assert self._widest(tmp_path) == 70

    def test_a_width_below_40_is_refused_before_any_task_runs(
        self, tmp_path: Path, cli: Callable[..., Invocation]
    ) -> None:
        config = write_project(tmp_path)
        from_flag = cli("-c", config, "-d", tmp_path, "-o", tmp_path / "out", "--report-width", "39")
        assert from_flag.code == 2
        assert "must be at least 40" in from_flag.stderr
        from_env = cli("-c", config, "-d", tmp_path, "-o", tmp_path / "out", env={"DATAEVAL_REPORT_WIDTH": "39"})
        assert from_env.code == 1
        assert "DATAEVAL_REPORT_WIDTH must be at least 40" in from_env.output
        assert not (tmp_path / "out" / "results").exists()


class TestResultBlockValidation:
    def test_defaults_write_json_text_and_html_at_full_detail(self) -> None:
        settings = ResultConfig()
        assert settings.name == "result"
        assert settings.formats == ["json", "text", "html"]
        assert (settings.detail, settings.per_task, settings.fail_on, settings.require) == (
            "full",
            False,
            "failure",
            None,
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("formats", ["pdf"]),
            ("formats", []),
            ("detail", "brief"),
            ("fail_on", "error"),
            ("require", "perfect"),
            ("width", 39),
            ("max_rows", 0),
            ("max_images", -2),
            ("name", "../escape"),
            ("name", "encoding"),
            ("unknown_key", 1),
        ],
    )
    def test_a_value_the_command_cannot_honour_is_refused(self, field: str, value: Any) -> None:
        with pytest.raises(ValidationError, match=field):
            ResultConfig.model_validate({field: value})

    def test_a_bad_result_block_fails_the_config_load(self, tmp_path: Path) -> None:
        config = write_project(tmp_path, width=39)
        with pytest.raises(ValidationError, match="result.width"):
            load_config(config)

    def test_a_format_listed_twice_is_written_once(self) -> None:
        assert ResultConfig(formats=["json", "text", "json"]).formats == ["json", "text"]
