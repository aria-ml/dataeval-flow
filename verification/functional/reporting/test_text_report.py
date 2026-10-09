"""TC-12-1 — text report, export, and the health values a script can gate on."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from dataeval_flow import Result
from dataeval_flow.steps import ChainResult
from verification.functional.reporting._project import FAILING_QUALITY, run_project, task

pytestmark = pytest.mark.required

DUPLICATE_GROUPS = 2  # planted by the fixtures below: img_0 and img_1 each have a copy


@pytest.fixture(scope="module")
def warned(tmp_path_factory: pytest.TempPathFactory) -> ChainResult:
    """A `quality` result whose duplicates check warns."""
    return run_project(tmp_path_factory.mktemp("warned"), duplicates=DUPLICATE_GROUPS)["clean_task"]


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> ChainResult:
    """A `quality` result with nothing to warn about."""
    return run_project(tmp_path_factory.mktemp("clean"))["clean_task"]


@pytest.fixture(scope="module")
def failed(tmp_path_factory: pytest.TempPathFactory) -> ChainResult:
    """A `quality` result whose `outliers` step raised."""
    results = run_project(
        tmp_path_factory.mktemp("failed"), duplicates=1, workflows=[FAILING_QUALITY], tasks=[task("bad_task", "q_fail")]
    )
    return results["bad_task"]


@pytest.fixture(scope="module")
def evaluator_result() -> Result[Any, Any]:
    """The result of the `duplicates` evaluator over a dataset in memory."""
    import numpy as np

    from dataeval_flow import run
    from dataeval_flow.evaluators.quality import DuplicatesConfig

    class Images:
        metadata = {"id": "report-images", "index2label": {0: "a", 1: "b"}}

        def __init__(self) -> None:
            rng = np.random.default_rng(0)
            self.images = [rng.integers(0, 255, (3, 32, 32), dtype=np.uint8) for _ in range(6)]
            self.images[4] = self.images[1].copy()

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
            target = np.zeros(2, dtype=np.float32)
            target[index % 2] = 1.0
            return self.images[index], target, {"id": index}

    return run(DuplicatesConfig(), Images())


class TestTextReport:
    def test_report_is_a_multi_line_text_with_banner_summary_health_and_configuration(
        self, warned: ChainResult
    ) -> None:
        text = warned.report()
        assert isinstance(text, str)
        assert len(text.splitlines()) > 50
        for heading in ("QUALITY", "SUMMARY", "IMAGE DUPLICATES", "STEPS", "CONFIGURATION"):
            assert f"  {heading}\n" in text or f"  {heading} " in text, heading
        assert "Workflow:  q (quality)" in text
        assert "Source:    main (ds)" in text

    def test_summary_report_keeps_the_summary_and_drops_the_evidence(self, warned: ChainResult) -> None:
        detailed = warned.report()
        summary = warned.report(detailed=False)
        assert len(summary.splitlines()) < len(detailed.splitlines())
        assert "SUMMARY" in summary
        assert "Health:" in summary
        assert "CONFIGURATION" in summary
        assert "From Duplicates" in detailed
        assert "From Duplicates" not in summary

    def test_report_width_sets_the_line_length(self, warned: ChainResult) -> None:
        narrow = warned.report(width=60).splitlines()
        wide = warned.report(width=120).splitlines()
        assert max(map(len, narrow)) == 60  # the banner rules fill the width
        assert max(map(len, wide)) == 120
        assert max(map(len, warned.report().splitlines())) == 80  # the default

    def test_report_width_below_the_minimum_is_refused(self, warned: ChainResult) -> None:
        with pytest.raises(ValueError, match="at least 40"):
            warned.report(width=39)
        warned.report(width=40)  # the minimum itself draws

    def test_warnings_are_counted_in_the_health_line(self, warned: ChainResult) -> None:
        assert warned.warning_count == 1
        assert "Health: 1 warning(s)" in warned.report()
        assert "Image Duplicates" in warned.report(detailed=False)

    def test_clean_run_says_all_checks_passed(self, clean: ChainResult) -> None:
        assert clean.warning_count == 0
        assert "Health: All checks passed" in clean.report()

    def test_failed_workflow_report_names_the_step_and_the_error(self, failed: ChainResult) -> None:
        assert not failed.success
        for text in (failed.report(), failed.report(detailed=False)):
            assert "Health: failed" in text
            assert "step `outliers` failed" in text
        assert "ValueError: n_expected_clusters=500" in failed.report()
        assert "1 failed" in failed.report()

    def test_evaluator_report_has_its_own_banner_and_judges_no_health(self, evaluator_result: Result[Any, Any]) -> None:
        text = evaluator_result.report()
        assert "  DUPLICATES\n" in text
        assert "Evaluator: duplicates" in text
        assert "1, 4" in text  # the planted pair
        assert "Health:" not in text


class TestExport:
    def test_export_writes_json_and_returns_the_path(self, warned: ChainResult, tmp_path: Path) -> None:
        written = warned.export(tmp_path / "result.json")
        assert written == tmp_path / "result.json"
        assert json.loads(written.read_text()) == warned.to_dict()

    def test_export_writes_yaml(self, warned: ChainResult, tmp_path: Path) -> None:
        written = warned.export(tmp_path / "result.yaml", fmt="yaml")
        assert yaml.safe_load(written.read_text()) == warned.to_dict()

    def test_export_to_a_directory_writes_results_file_inside_it(self, warned: ChainResult, tmp_path: Path) -> None:
        assert warned.export(tmp_path / "out") == tmp_path / "out" / "results.json"
        assert warned.export(tmp_path / "out", fmt="yaml") == tmp_path / "out" / "results.yaml"

    def test_export_without_a_path_returns_the_serialized_string(self, warned: ChainResult) -> None:
        as_json = warned.export()
        as_yaml = warned.export(fmt="yaml")
        assert isinstance(as_json, str)
        assert json.loads(as_json) == warned.to_dict()
        assert yaml.safe_load(as_yaml) == warned.to_dict()

    def test_workflow_dict_holds_kind_metadata_health_steps_and_findings(self, warned: ChainResult) -> None:
        payload = warned.to_dict()
        assert payload["kind"] == "workflow"
        assert {"metadata", "health", "steps", "findings"} <= set(payload)
        steps = payload["steps"]
        assert isinstance(steps, dict)
        assert list(steps)[:2] == ["outliers", "label-health"]
        assert steps["duplicates"]["status"] == "ok"
        findings = payload["findings"]
        assert isinstance(findings, list)
        assert [f["title"] for f in findings] == ["Image Outliers", "Class Outliers", "Image Duplicates"]
        assert [f["severity"] for f in findings] == ["ok", "ok", "warning"]

    def test_evaluator_dict_holds_kind_metadata_and_output_but_no_health(
        self, evaluator_result: Result[Any, Any]
    ) -> None:
        payload = evaluator_result.to_dict()
        assert payload["kind"] == "evaluator"
        assert {"metadata", "output"} <= set(payload)
        assert "health" not in payload
        output = payload["output"]
        assert isinstance(output, dict)
        assert output["rows"][0]["item_indices"] == [1, 4]

    def test_exported_json_is_strict(self, warned: ChainResult) -> None:
        """No NaN or infinity: strict parsers refuse the literals Python would otherwise write."""
        json.loads(warned.export(), parse_constant=lambda name: pytest.fail(f"non-finite {name} in the export"))


class TestHealthValues:
    def test_health_warning_count_and_findings_agree_with_the_report(self, warned: ChainResult) -> None:
        warnings = [f for f in warned.findings if f.severity == "warning"]
        assert warned.warning_count == len(warnings) == 1
        assert warned.health == {
            "status": "warning",
            "warnings": 1,
            "findings": len(warned.findings),
            "failed_steps": [],
        }
        assert warned.to_dict()["health"] == warned.health

    def test_clean_health_is_ok(self, clean: ChainResult) -> None:
        assert clean.health["status"] == "ok"
        assert clean.health["warnings"] == 0
        assert clean.findings
        assert all(f.severity == "ok" for f in clean.findings)

    def test_failed_health_names_the_failed_steps(self, failed: ChainResult) -> None:
        assert failed.health["status"] == "failed"
        assert failed.health["failed_steps"] == ["outliers"]
        assert failed.warning_count == 1  # the duplicates check still ran and still warned
