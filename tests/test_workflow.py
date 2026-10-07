"""Tests for workflow/__init__.py — a workflow result's report(), export(), and discovery helpers."""

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from dataeval_flow import ResultMetadata
from dataeval_flow._blocks import Fields, Paragraph, Section
from dataeval_flow._blocks._text import Frame, render_text
from dataeval_flow._result import _envelope_items
from dataeval_flow.config import ViewOperation
from dataeval_flow.steps import ChainMetadata, ChainResult, Finding
from dataeval_flow.workflows import DatasetContext, get_workflow, list_workflows
from dataeval_flow.workflows._result import finding_section
from tests.workflow_toys import count_result

pytestmark = pytest.mark.required

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_failed() -> ChainResult:
    return ChainResult.failed(type="test.count", errors=["boom"])


# ---------------------------------------------------------------------------
# A workflow result's health — the roll-up an automated gate reads
# ---------------------------------------------------------------------------


class TestResultHealth:
    def _result(self, *severities: Literal["ok", "info", "warning"]) -> ChainResult:
        return count_result(*(Finding(severity=sev, title=f"f{i}") for i, sev in enumerate(severities)))

    def test_warning_count_counts_only_warnings(self):
        assert self._result("warning", "info", "warning", "ok").warning_count == 2

    def test_warning_count_is_zero_without_findings(self):
        assert count_result().warning_count == 0

    def test_warning_count_is_zero_for_a_failed_run(self):
        """A run that did not complete has nothing to gate on."""
        assert _make_failed().warning_count == 0

    def test_findings_property_is_empty_for_a_failed_run(self):
        assert _make_failed().findings == []

    def test_health_status_is_warning_when_any_finding_warns(self):
        health = {"status": "warning", "warnings": 1, "findings": 2, "failed_steps": []}
        assert self._result("info", "warning").health == health

    def test_health_status_is_ok_when_none_do(self):
        assert self._result("info", "ok").health == {"status": "ok", "warnings": 0, "findings": 2, "failed_steps": []}

    def test_health_matches_the_rendered_health_line(self):
        """The roll-up and the text report must not be able to disagree."""
        result = self._result("warning", "warning", "info")
        assert f"{result.warning_count} warning(s)" in result.report()

    def test_envelope_carries_health(self):
        payload = self._result("warning", "info").to_dict()
        assert payload["health"] == {"status": "warning", "warnings": 1, "findings": 2, "failed_steps": []}

    def test_envelope_still_carries_metadata_and_data(self):
        """health is additive — it must not displace what the envelope already held."""
        result = count_result()
        payload = result.to_dict()
        assert "metadata" in payload
        assert payload["steps"] == {"at-least": result.steps["at-least"].to_dict()}


# ---------------------------------------------------------------------------
# A workflow result's report() — text output
# ---------------------------------------------------------------------------


class TestReportFormatDispatch:
    def test_report_returns_text(self):
        result = count_result()
        out = result.report()
        assert isinstance(out, str)

    def test_report_detailed_false(self):
        result = count_result()
        out = result.report(detailed=False)
        assert isinstance(out, str)

    def test_export_json_returns_string(self):
        result = count_result()
        out = result.export()
        assert isinstance(out, str)
        parsed = json.loads(out)
        assert "metadata" in parsed

    def test_export_yaml_returns_string(self):
        result = count_result()
        out = result.export(fmt="yaml")
        assert isinstance(out, str)
        assert "metadata:" in out


# ---------------------------------------------------------------------------
# A workflow result's text report
# ---------------------------------------------------------------------------


class TestReportText:
    def test_a_failed_run_reports_its_errors(self):
        """A failed workflow is titled by its type and shows FAILED and each error, as a failed evaluator does."""
        out = _make_failed().report()
        assert "  TEST.COUNT" in out
        lines = out.splitlines()
        assert lines[lines.index("  FAILED") + 2] == "  boom"
        assert "No findings to report." not in out

    def test_empty_findings(self):
        """Empty findings list shows 'No findings to report.'."""
        result = count_result()
        out = result.report()
        assert "No findings to report." in out

    def test_findings_with_warnings(self):
        """Findings with warnings show count in health line."""
        findings = [
            Finding(severity="warning", title="Bad Image", description="desc1"),
            Finding(severity="warning", title="Corrupt File", description=None),
            Finding(severity="ok", title="All Good", description="ok desc"),
        ]
        result = count_result(*findings)
        out = result.report()
        assert "2 warning(s)" in out

    def test_findings_no_warnings(self):
        """Findings with no warnings show 'All checks passed' in health line."""
        findings = [
            Finding(severity="ok", title="All Good", description="fine"),
        ]
        result = count_result(*findings)
        out = result.report()
        assert "All checks passed [ok]" in out

    def test_summary_section_present(self):
        """Report includes a SUMMARY section with dotted lines."""
        findings = [
            Finding(severity="info", title="Check A", description="desc a"),
            Finding(severity="info", title="Check B", description="desc b"),
        ]
        result = count_result(*findings)
        out = result.report()
        assert "SUMMARY" in out
        assert "Check A" in out
        assert "Check B" in out

    def test_detail_sections_present(self):
        """Each finding gets a detail section with uppercased title."""
        findings = [
            Finding(severity="info", title="My Finding", description="some detail"),
        ]
        result = count_result(*findings)
        out = result.report()
        assert "MY FINDING" in out
        assert "some detail" in out

    def test_finding_without_description(self):
        """Finding with no description still renders its detail section."""
        findings = [
            Finding(severity="info", title="NoDesc", description=None),
        ]
        result = count_result(*findings)
        out = result.report()
        assert "NODESC" in out

    def test_health_line_with_warnings(self):
        """Health line shows warning count when warnings exist."""
        findings = [
            Finding(severity="warning", title="Issue", description="bad"),
        ]
        result = count_result(*findings)
        out = result.report()
        assert "1 warning(s)" in out

    def test_health_line_without_warnings(self):
        """Health line shows 'No issues detected' when no warnings."""
        findings = [
            Finding(severity="info", title="Info", description="ok"),
        ]
        result = count_result(*findings)
        out = result.report()
        assert "Health: All checks passed [ok]" in out

    def test_warning_marker_in_summary(self):
        """Warning findings get [!!] marker in summary line."""
        findings = [
            Finding(severity="warning", title="Bad Thing", description="bad"),
        ]
        result = count_result(*findings)
        out = result.report()
        assert "[!!]" in out


# ---------------------------------------------------------------------------
# finding_section — a finding's detail section
# ---------------------------------------------------------------------------


class TestFindingSection:
    """A finding's detail section: its title, brief and severity, the description as the lede, then its blocks."""

    def test_the_description_leads_the_blocks(self):
        evidence = Fields(items=[("Total", 9)])
        finding = Finding(
            title="Pruning", severity="warning", brief="3 items", description="Removed three.", blocks=[evidence]
        )
        assert finding_section(finding) == Section(
            title="Pruning",
            brief="3 items",
            severity="warning",
            blocks=[Paragraph(text="Removed three."), evidence],
        )

    def test_an_empty_brief_and_no_description_add_nothing(self):
        assert finding_section(Finding(title="Bare", brief="")) == Section(title="Bare", severity="info")


# ---------------------------------------------------------------------------
# A workflow result's metadata lines
# ---------------------------------------------------------------------------


class TestMetadataTextLines:
    def test_all_metadata_fields(self):
        """All metadata fields present (lines 132-140)."""
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
        meta = ChainMetadata(timestamp=ts, execution_time_s=1.23, dataset_id="ds-1")
        result = count_result(metadata=meta)
        out = result.report()
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
        meta.model_id = None
        meta.preprocessor_id = None
        meta.selection_id = None
        meta.resolved_config = {}
        meta.workflow = ""
        meta.model_dump = MagicMock(return_value={})
        result = count_result(metadata=meta)
        out = result.report()
        assert "Timestamp" not in out
        assert "Duration" not in out
        assert "Dataset" not in out
        assert "Model" not in out
        assert "Preprocessor" not in out
        assert "Selection" not in out

    def test_metadata_with_model(self):
        """Model ID appears in metadata block."""
        meta = ChainMetadata(model_id="resnet50")
        result = count_result(metadata=meta)
        out = result.report()
        assert "Model:" in out
        assert "resnet50" in out

    def test_metadata_with_preprocessor(self):
        """Preprocessor ID appears in metadata block."""
        meta = ChainMetadata(preprocessor_id="resnet50_preprocessor")
        result = count_result(metadata=meta)
        out = result.report()
        assert "Preprocessor:" in out
        assert "resnet50_preprocessor" in out

    def test_metadata_with_selection(self):
        """Selection ID appears in metadata block."""
        meta = ChainMetadata(selection_id="training_subset")
        result = count_result(metadata=meta)
        out = result.report()
        assert "Selection:" in out
        assert "training_subset" in out

    def test_metadata_dataset_line_includes_label_source(self):
        """Dataset line includes label_source in parentheses when present."""
        meta = ChainMetadata(dataset_id="my-dataset", label_source="annotations")
        result = count_result(metadata=meta)
        out = result.report()
        assert "my-dataset" in out
        assert "(annotations)" in out


# ---------------------------------------------------------------------------
# A workflow result's export() — file paths
# ---------------------------------------------------------------------------


class TestReportSerialized:
    def test_json_to_directory(self, tmp_path: Path):
        """JSON written to a directory creates results.json."""
        result = count_result()
        out = result.export(tmp_path)
        assert isinstance(out, Path)
        assert out.name == "results.json"
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert "metadata" in parsed

    def test_json_to_file_path(self, tmp_path: Path):
        """JSON written to a specific file path."""
        dest = tmp_path / "sub" / "output.json"
        result = count_result()
        out = result.export(dest)
        assert isinstance(out, Path)
        assert out == dest
        assert out.exists()

    def test_yaml_to_directory(self, tmp_path: Path):
        """YAML written to a directory creates results.yaml."""
        result = count_result()
        out = result.export(tmp_path, fmt="yaml")
        assert isinstance(out, Path)
        assert out.name == "results.yaml"
        assert out.exists()

    def test_yaml_to_file_path(self, tmp_path: Path):
        """YAML written to a specific file path."""
        dest = tmp_path / "deep" / "nested" / "out.yaml"
        result = count_result()
        out = result.export(dest, fmt="yaml")
        assert isinstance(out, Path)
        assert out == dest
        assert out.exists()

    def test_json_no_path_returns_string(self):
        """path=None returns serialized string."""
        result = count_result()
        out = result.export()
        assert isinstance(out, str)
        parsed = json.loads(out)
        assert "metadata" in parsed

    def test_yaml_no_path_returns_string(self):
        """path=None returns YAML string."""
        result = count_result()
        out = result.export(fmt="yaml")
        assert isinstance(out, str)
        assert "metadata:" in out

    def test_directory_without_suffix(self, tmp_path: Path):
        """Path without suffix treated as directory."""
        dest = tmp_path / "no_suffix_dir"
        result = count_result()
        out = result.export(dest)
        assert isinstance(out, Path)
        assert out.name == "results.json"
        assert out.parent == dest


# ---------------------------------------------------------------------------
# Workflow discovery
# ---------------------------------------------------------------------------


class TestWorkflowDiscovery:
    def test_get_workflow_known(self):
        wf = get_workflow("quality")
        assert wf.name == "quality"

    def test_get_workflow_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown workflow"):
            get_workflow("nonexistent-workflow")

    def test_list_workflows(self):
        workflows = list_workflows()
        assert isinstance(workflows, list)
        assert len(workflows) >= 1
        names = [w.name for w in workflows]
        assert "quality" in names
        assert names == sorted(names)
        for w in workflows:
            assert w.description


# ---------------------------------------------------------------------------
# DatasetContext — view_operations
# ---------------------------------------------------------------------------


class TestDatasetContextViewOperations:
    _OPS = [ViewOperation(type="Limit", params={"size": 5})]

    def test_view_operations_no_warning(self):
        """Passing view_operations directly stores them and fires no warning."""
        with warnings.catch_warnings(action="error"):  # any warning becomes a test failure
            ctx = DatasetContext(name="s", dataset=object(), view_operations=self._OPS)  # type: ignore
        assert ctx.view_operations == self._OPS

    def test_no_view_no_warning(self):
        """Omitting view_operations leaves it as None with no warning."""
        with warnings.catch_warnings(action="error"):
            ctx = DatasetContext(name="s", dataset=object())  # type: ignore
        assert ctx.view_operations is None


# ---------------------------------------------------------------------------
# the envelope's sources — multi-source continuation
# ---------------------------------------------------------------------------


def _source_lines(meta: ResultMetadata) -> list[str]:
    """The envelope's source lines as a report draws them."""
    items = [item for item in _envelope_items(meta) if item[0] in ("Source", "Dataset", "Selection")]
    return render_text([Fields(items=items)], Frame(indent="  "))


class TestSourceLines:
    def test_multiple_source_descriptions(self):
        """Multiple source_descriptions use continuation indent (line 73)."""
        meta = ResultMetadata(source_descriptions=["Source A: train (100 imgs)", "Source B: test (50 imgs)"])
        lines = _source_lines(meta)
        assert len(lines) == 2
        assert "Source:" in lines[0]
        assert "Source A" in lines[0]
        assert "Source B" in lines[1]
        # Continuation line should NOT have "Source:" label
        assert "Source:" not in lines[1]

    def test_single_source_description(self):
        """Single source_description uses Source label."""
        meta = ResultMetadata(source_descriptions=["My dataset (200 imgs)"])
        lines = _source_lines(meta)
        assert len(lines) == 1
        assert "Source:" in lines[0]

    def test_no_sources_falls_back_to_dataset(self):
        """No source_descriptions falls back to dataset_id."""
        meta = ResultMetadata(dataset_id="ds-1")
        lines = _source_lines(meta)
        assert len(lines) == 1
        assert "ds-1" in lines[0]
