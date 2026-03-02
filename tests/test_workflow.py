"""Tests for workflow/__init__.py — WorkflowResult.report() and discovery helpers."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from dataeval_app.config.schemas.metadata import ResultMetadata
from dataeval_app.workflow import WorkflowResult, get_workflow, list_workflows
from dataeval_app.workflow.base import Reportable, WorkflowReportBase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyReport(WorkflowReportBase):
    summary: str = "Test Summary"
    findings: list[Reportable] = []


class _DummyOutputWithReport(BaseModel):
    value: int = 42
    report: _DummyReport = _DummyReport()


class _DummyOutputNoReport(BaseModel):
    value: int = 99


def _make_result(
    *,
    data: BaseModel | None = None,
    metadata: ResultMetadata | None = None,
    fmt: Literal["text", "json", "yaml"] = "text",
) -> WorkflowResult:
    return WorkflowResult(
        name="test-workflow",
        success=True,
        data=data or _DummyOutputWithReport(),
        metadata=metadata or ResultMetadata(),
        format=fmt,
    )


# ---------------------------------------------------------------------------
# WorkflowResult.report() — format dispatch
# ---------------------------------------------------------------------------


class TestReportFormatDispatch:
    def test_text_format_explicit(self):
        result = _make_result()
        out = result.report(format="text")
        assert isinstance(out, str)

    def test_default_format_uses_instance_format(self):
        """format=None falls back to self.format (line 120)."""
        result = _make_result(fmt="text")
        out = result.report()
        assert isinstance(out, str)

    def test_json_format_returns_string(self):
        result = _make_result()
        out = result.report(format="json")
        assert isinstance(out, str)
        parsed = json.loads(out)
        assert "metadata" in parsed

    def test_yaml_format_returns_string(self):
        """YAML path (lines 213-216)."""
        result = _make_result()
        out = result.report(format="yaml")
        assert isinstance(out, str)
        assert "metadata:" in out

    def test_unknown_format_raises(self):
        """Unknown format raises ValueError (lines 125-126)."""
        result = _make_result()
        with pytest.raises(ValueError, match="Unknown format"):
            result.report(format="csv")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# WorkflowResult._report_text()
# ---------------------------------------------------------------------------


class TestReportText:
    def test_no_report_attribute(self):
        """data without .report returns 'no report available' (line 147)."""
        result = _make_result(data=_DummyOutputNoReport())
        out = result.report(format="text")
        assert "no report available" in out

    def test_empty_findings(self):
        """Empty findings list shows 'No findings to report.' (lines 164-167)."""
        result = _make_result()
        out = result.report(format="text")
        assert "No findings to report." in out
        assert "Test Summary" in out

    def test_findings_with_warnings(self):
        """Findings with warnings show count in footer (lines 189-190)."""
        findings = [
            Reportable(
                report_type="text", severity="warning", title="Bad Image (train)", data="detail", description="desc1"
            ),
            Reportable(
                report_type="text", severity="warning", title="Corrupt File (train)", data="detail", description=None
            ),
            Reportable(
                report_type="text", severity="ok", title="All Good (train)", data="detail", description="ok desc"
            ),
        ]
        report = _DummyReport(summary="Findings Test", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "2 issue(s) detected" in out
        assert "Findings Test" in out

    def test_findings_no_warnings(self):
        """Findings with no warnings show 'No issues detected' (lines 191-192)."""
        findings = [
            Reportable(report_type="text", severity="ok", title="All Good (train)", data="detail", description="fine"),
        ]
        report = _DummyReport(summary="Clean Report", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "No issues detected" in out

    def test_findings_group_separator(self):
        """Blank line inserted between different context groups (lines 179-180)."""
        findings = [
            Reportable(report_type="text", severity="info", title="Check (train)", data="d"),
            Reportable(report_type="text", severity="info", title="Check (val)", data="d"),
        ]
        report = _DummyReport(summary="Grouped", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        lines = out.split("\n")
        # There should be a blank line between the two context groups
        check_indices = [i for i, line in enumerate(lines) if "Check" in line]
        assert len(check_indices) == 2
        # A blank line should exist between them
        assert any(lines[i] == "" for i in range(check_indices[0] + 1, check_indices[1]))

    def test_finding_without_description(self):
        """Finding with no description omits the dash (line 184)."""
        findings = [
            Reportable(report_type="text", severity="info", title="NoDesc", data="d", description=None),
        ]
        report = _DummyReport(summary="S", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "NoDesc" in out
        assert "\u2014" not in out.split("NoDesc")[1].split("\n")[0]

    def test_finding_with_description(self):
        """Finding with description includes the dash (line 184)."""
        findings = [
            Reportable(report_type="text", severity="info", title="WithDesc", data="d", description="some detail"),
        ]
        report = _DummyReport(summary="S", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "\u2014 some detail" in out


# ---------------------------------------------------------------------------
# WorkflowResult._metadata_text_lines()
# ---------------------------------------------------------------------------


class TestMetadataTextLines:
    def test_all_metadata_fields(self):
        """All metadata fields present (lines 132-140)."""
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        meta = ResultMetadata(timestamp=ts, execution_time_s=1.23, dataset_id="ds-1")
        result = _make_result(metadata=meta)
        out = result.report(format="text")
        assert "2025-06-15" in out
        assert "1.23s" in out
        assert "ds-1" in out

    def test_metadata_no_optional_fields(self):
        """Empty metadata produces no metadata section."""
        from unittest.mock import MagicMock

        meta = MagicMock()
        meta.timestamp = None
        meta.execution_time_s = None
        meta.dataset_id = ""
        meta.model_dump = MagicMock(return_value={})
        result = _make_result(metadata=meta)
        out = result.report(format="text")
        assert "Timestamp" not in out
        assert "Duration" not in out
        assert "Dataset" not in out


# ---------------------------------------------------------------------------
# WorkflowResult._report_serialized() — file paths
# ---------------------------------------------------------------------------


class TestReportSerialized:
    def test_json_to_directory(self, tmp_path: Path):
        """JSON written to a directory creates results.json."""
        result = _make_result()
        out = result.report(format="json", path=tmp_path)
        assert isinstance(out, Path)
        assert out.name == "results.json"
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert "metadata" in parsed

    def test_json_to_file_path(self, tmp_path: Path):
        """JSON written to a specific file path (line 226)."""
        dest = tmp_path / "sub" / "output.json"
        result = _make_result()
        out = result.report(format="json", path=dest)
        assert isinstance(out, Path)
        assert out == dest
        assert out.exists()

    def test_yaml_to_directory(self, tmp_path: Path):
        """YAML written to a directory creates results.yaml."""
        result = _make_result()
        out = result.report(format="yaml", path=tmp_path)
        assert isinstance(out, Path)
        assert out.name == "results.yaml"
        assert out.exists()

    def test_yaml_to_file_path(self, tmp_path: Path):
        """YAML written to a specific file path."""
        dest = tmp_path / "deep" / "nested" / "out.yaml"
        result = _make_result()
        out = result.report(format="yaml", path=dest)
        assert isinstance(out, Path)
        assert out == dest
        assert out.exists()

    def test_json_no_path_returns_string(self):
        """path=None returns serialized string (line 219)."""
        result = _make_result()
        out = result.report(format="json", path=None)
        assert isinstance(out, str)
        parsed = json.loads(out)
        assert "metadata" in parsed

    def test_yaml_no_path_returns_string(self):
        """path=None returns YAML string (lines 213-216, 219)."""
        result = _make_result()
        out = result.report(format="yaml", path=None)
        assert isinstance(out, str)
        assert "metadata:" in out

    def test_directory_without_suffix(self, tmp_path: Path):
        """Path without suffix treated as directory (line 222)."""
        dest = tmp_path / "no_suffix_dir"
        result = _make_result()
        out = result.report(format="json", path=dest)
        assert isinstance(out, Path)
        assert out.name == "results.json"
        assert out.parent == dest


# ---------------------------------------------------------------------------
# Workflow discovery
# ---------------------------------------------------------------------------


class TestWorkflowDiscovery:
    def test_get_workflow_known(self):
        wf = get_workflow("data-cleaning")
        assert wf.name == "data-cleaning"

    def test_get_workflow_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown workflow"):
            get_workflow("nonexistent-workflow")

    def test_list_workflows(self):
        workflows = list_workflows()
        assert isinstance(workflows, list)
        assert len(workflows) >= 1
        names = [w["name"] for w in workflows]
        assert "data-cleaning" in names
        for w in workflows:
            assert "name" in w
            assert "description" in w
