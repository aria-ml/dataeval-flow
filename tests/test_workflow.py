"""Tests for workflow/__init__.py — WorkflowResult.report(), .export(), and discovery helpers."""

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from dataeval_flow import ResultMetadata
from dataeval_flow._result import _source_lines
from dataeval_flow.config import ViewOperation
from dataeval_flow.workflows import (
    DatasetContext,
    Finding,
    WorkflowOutput,
    WorkflowRawOutput,
    WorkflowReport,
    WorkflowResult,
    get_workflow,
    list_workflows,
)

pytestmark = pytest.mark.required

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyRawOutput(WorkflowRawOutput):
    dataset_size: int = 1
    value: int = 42


class _DummyReport(WorkflowReport):
    summary: str = "Test Summary"


def _output(report: _DummyReport | None = None) -> WorkflowOutput[_DummyRawOutput, _DummyReport]:
    return WorkflowOutput[_DummyRawOutput, _DummyReport](raw=_DummyRawOutput(), report=report or _DummyReport())


def _make_result(
    *,
    output: WorkflowOutput[_DummyRawOutput, _DummyReport] | None = None,
    metadata: ResultMetadata | None = None,
) -> WorkflowResult:
    return WorkflowResult(
        type="test-workflow",
        success=True,
        output=output or _output(),
        metadata=metadata or ResultMetadata(),
    )


def _make_failed() -> WorkflowResult:
    return WorkflowResult(type="test-workflow", success=False, metadata=ResultMetadata(), errors=["boom"])


# ---------------------------------------------------------------------------
# WorkflowResult health — the roll-up an automated gate reads
# ---------------------------------------------------------------------------


class TestResultHealth:
    def _result(self, *severities: Literal["ok", "info", "warning"]) -> WorkflowResult:
        findings = [
            Finding(report_type="text", severity=sev, title=f"f{i}", data="detail") for i, sev in enumerate(severities)
        ]
        return _make_result(output=_output(_DummyReport(findings=findings)))

    def test_warning_count_counts_only_warnings(self):
        assert self._result("warning", "info", "warning", "ok").warning_count == 2

    def test_warning_count_is_zero_without_findings(self):
        assert _make_result().warning_count == 0

    def test_warning_count_is_zero_for_a_failed_run(self):
        """A run that did not complete has nothing to gate on."""
        assert _make_failed().warning_count == 0

    def test_findings_property_is_empty_for_a_failed_run(self):
        assert _make_failed().findings == []

    def test_health_status_is_warning_when_any_finding_warns(self):
        assert self._result("info", "warning").health == {"status": "warning", "warnings": 1, "findings": 2}

    def test_health_status_is_ok_when_none_do(self):
        assert self._result("info", "ok").health == {"status": "ok", "warnings": 0, "findings": 2}

    def test_health_matches_the_rendered_health_line(self):
        """The roll-up and the text report must not be able to disagree."""
        result = self._result("warning", "warning", "info")
        assert f"{result.warning_count} warning(s)" in result.report()

    def test_envelope_carries_health(self):
        payload = self._result("warning", "info").to_dict()
        assert payload["health"] == {"status": "warning", "warnings": 1, "findings": 2}

    def test_envelope_still_carries_metadata_and_data(self):
        """health is additive — it must not displace what the envelope already held."""
        payload = _make_result().to_dict()
        assert "metadata" in payload
        assert payload["raw"] == {"dataset_size": 1, "value": 42}


# ---------------------------------------------------------------------------
# WorkflowResult.report() — text output
# ---------------------------------------------------------------------------


class TestReportFormatDispatch:
    def test_report_returns_text(self):
        result = _make_result()
        out = result.report()
        assert isinstance(out, str)

    def test_report_detailed_false(self):
        result = _make_result()
        out = result.report(detailed=False)
        assert isinstance(out, str)

    def test_export_json_returns_string(self):
        result = _make_result()
        out = result.export()
        assert isinstance(out, str)
        parsed = json.loads(out)
        assert "metadata" in parsed

    def test_export_yaml_returns_string(self):
        result = _make_result()
        out = result.export(fmt="yaml")
        assert isinstance(out, str)
        assert "metadata:" in out


# ---------------------------------------------------------------------------
# WorkflowResult._report_text()
# ---------------------------------------------------------------------------


class TestReportText:
    def test_a_failed_run_reports_its_errors(self):
        """A failed workflow is titled by its type and shows FAILED and each error, as a failed evaluator does."""
        out = _make_failed().report()
        assert "  TEST-WORKFLOW" in out
        assert "  FAILED\n    boom" in out
        assert "No findings to report." not in out

    def test_empty_findings(self):
        """Empty findings list shows 'No findings to report.'."""
        result = _make_result()
        out = result.report()
        assert "No findings to report." in out
        assert "TEST SUMMARY" in out  # summary is uppercased in the banner

    def test_findings_with_warnings(self):
        """Findings with warnings show count in health line."""
        findings = [
            Finding(report_type="text", severity="warning", title="Bad Image", data="detail", description="desc1"),
            Finding(report_type="text", severity="warning", title="Corrupt File", data="detail", description=None),
            Finding(report_type="text", severity="ok", title="All Good", data="detail", description="ok desc"),
        ]
        report = _DummyReport(summary="Findings Test", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "2 warning(s)" in out
        assert "FINDINGS TEST" in out

    def test_findings_no_warnings(self):
        """Findings with no warnings show 'All checks passed' in health line."""
        findings = [
            Finding(report_type="text", severity="ok", title="All Good", data="detail", description="fine"),
        ]
        report = _DummyReport(summary="Clean Report", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "All checks passed [ok]" in out

    def test_summary_section_present(self):
        """Report includes a SUMMARY section with dotted lines."""
        findings = [
            Finding(report_type="text", severity="info", title="Check A", data="d", description="desc a"),
            Finding(report_type="text", severity="info", title="Check B", data="d", description="desc b"),
        ]
        report = _DummyReport(summary="Summary Test", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "SUMMARY" in out
        assert "Check A" in out
        assert "Check B" in out

    def test_detail_sections_present(self):
        """Each finding gets a detail section with uppercased title."""
        findings = [
            Finding(report_type="text", severity="info", title="My Finding", data="d", description="some detail"),
        ]
        report = _DummyReport(summary="S", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "MY FINDING" in out
        assert "some detail" in out

    def test_finding_without_description(self):
        """Finding with no description still renders its detail section."""
        findings = [
            Finding(report_type="text", severity="info", title="NoDesc", data="d", description=None),
        ]
        report = _DummyReport(summary="S", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "NODESC" in out

    def test_outlier_per_metric_breakdown(self):
        """Image Outliers finding renders per-metric table in detail section."""
        findings = [
            Finding(
                report_type="key_value",
                severity="warning",
                title="Image Outliers",
                data={
                    "brief": "10 images (5.0%)",
                    "multi_metric_subject": "images",
                    "count": 10,
                    "percentage": 5.0,
                    "per_metric": {"brightness": 7, "contrast": 5},
                    "total_flags": 12,
                    "dataset_size": 200,
                },
                description="10 images (5.0%) flagged as outliers.",
            ),
        ]
        report = _DummyReport(summary="S", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "IMAGE OUTLIERS" in out
        assert "brightness" in out
        assert "contrast" in out
        assert "multiple metrics" in out

    def test_duplicate_detail_renders(self):
        """Duplicate finding renders group details, methods, orientations."""
        findings = [
            Finding(
                report_type="key_value",
                severity="info",
                title="Duplicates",
                data={
                    "brief": "2 exact, 3 near",
                    "detail_lines": [
                        "2 exact-duplicate groups (6 images)",
                        "3 near-duplicate groups (10 images)",
                        "  Methods: dhash, phash",
                        "  Orientations: 1 rotated, 2 same",
                    ],
                    "exact_groups": 2,
                    "near_groups": 3,
                    "exact_affected": 6,
                    "near_affected": 10,
                    "near_methods": ["dhash", "phash"],
                    "near_orientations": {"same": 2, "rotated": 1},
                },
                description="2 exact duplicate groups, 3 near-duplicate groups found.",
            ),
        ]
        report = _DummyReport(summary="S", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "DUPLICATES" in out
        assert "2 exact-duplicate groups (6 images)" in out
        assert "3 near-duplicate groups (10 images)" in out
        assert "dhash, phash" in out
        assert "rotated" in out

    def test_label_distribution_bar_chart(self):
        """Label Distribution finding renders bar chart with block characters."""
        findings = [
            Finding(
                report_type="table",
                severity="info",
                title="Label Distribution",
                data={
                    "brief": "2 classes, 150 items",
                    "table_data": {"cat": 100, "dog": 50},
                    "table_headers": ("Class", "Count"),
                    "footer_lines": ["Imbalance ratio: 2.0 (max/min)"],
                    "label_counts": {"cat": 100, "dog": 50},
                    "class_count": 2,
                    "item_count": 150,
                    "imbalance_ratio": 2.0,
                },
                description="2 classes, 150 items.",
            ),
        ]
        report = _DummyReport(summary="S", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "LABEL DISTRIBUTION" in out
        assert "cat" in out
        assert "dog" in out
        assert "\u2588" in out  # bar chart block character
        assert "Imbalance ratio: 2.0" in out

    def test_health_line_with_warnings(self):
        """Health line shows warning count when warnings exist."""
        findings = [
            Finding(report_type="text", severity="warning", title="Issue", data="d", description="bad"),
        ]
        report = _DummyReport(summary="S", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "1 warning(s)" in out

    def test_health_line_without_warnings(self):
        """Health line shows 'No issues detected' when no warnings."""
        findings = [
            Finding(report_type="text", severity="info", title="Info", data="d", description="ok"),
        ]
        report = _DummyReport(summary="S", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "Health: All checks passed [ok]" in out

    def test_warning_marker_in_summary(self):
        """Warning findings get [!!] marker in summary line."""
        findings = [
            Finding(report_type="text", severity="warning", title="Bad Thing", data="d", description="bad"),
        ]
        report = _DummyReport(summary="S", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "[!!]" in out

    def test_duplicate_exact_only_no_methods_line(self):
        """Exact-only duplicates don't render Methods/Orientations lines."""
        findings = [
            Finding(
                report_type="key_value",
                severity="info",
                title="Duplicates",
                data={
                    "brief": "1 exact, 0 near",
                    "detail_lines": [
                        "1 exact-duplicate groups (3 images)",
                    ],
                    "exact_groups": 1,
                    "near_groups": 0,
                    "exact_affected": 3,
                    "near_affected": 0,
                    "near_methods": [],
                    "near_orientations": {},
                },
                description="1 exact duplicate groups, 0 near-duplicate groups found.",
            ),
        ]
        report = _DummyReport(summary="S", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "1 exact-duplicate groups (3 images)" in out
        # No Methods/Orientations lines when near_groups=0
        assert "Methods:" not in out
        assert "Orientations:" not in out

    def test_label_distribution_balanced_no_imbalance_line(self):
        """Balanced labels (imbalance_ratio=1.0) suppress the imbalance line."""
        findings = [
            Finding(
                report_type="table",
                severity="info",
                title="Label Distribution",
                data={
                    "brief": "2 classes, 100 items",
                    "table_data": {"a": 50, "b": 50},
                    "table_headers": ("Class", "Count"),
                    "footer_lines": [],
                    "label_counts": {"a": 50, "b": 50},
                    "class_count": 2,
                    "item_count": 100,
                    "imbalance_ratio": 1.0,
                },
                description="2 classes, 100 items.",
            ),
        ]
        report = _DummyReport(summary="S", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "Imbalance" not in out

    def test_target_outlier_multiple_metrics_message(self):
        """Target outlier detail says 'targets' not 'images' in multiple-metrics message."""
        findings = [
            Finding(
                report_type="key_value",
                severity="warning",
                title="Target Outliers",
                data={
                    "brief": "2 targets",
                    "multi_metric_subject": "targets",
                    "count": 2,
                    "per_metric": {"brightness": 2, "contrast": 1},
                    "total_flags": 3,
                },
                description="2 bounding-box targets flagged as outliers.",
            ),
        ]
        report = _DummyReport(summary="S", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        assert "Some targets trigger multiple metrics" in out
        assert "Some images" not in out

    def test_bar_chart_uses_left_filling_blocks(self):
        """Bar chart uses left-filling fractional blocks, not bottom-filling."""
        findings = [
            Finding(
                report_type="table",
                severity="info",
                title="Label Distribution",
                data={
                    "brief": "2 classes, 175 items",
                    "table_data": {"a": 100, "b": 75},
                    "table_headers": ("Class", "Count"),
                    "footer_lines": ["Imbalance ratio: 1.3 (max/min)"],
                    "label_counts": {"a": 100, "b": 75},
                    "class_count": 2,
                    "item_count": 175,
                    "imbalance_ratio": 1.3,
                },
                description="2 classes, 175 items.",
            ),
        ]
        report = _DummyReport(summary="S", findings=findings)
        result = _make_result(output=_output(report))
        out = result.report()
        # Ensure no bottom-filling blocks are present (U+2581-U+2587)
        bottom_blocks = set("\u2581\u2582\u2583\u2584\u2585\u2586\u2587")
        assert not any(ch in bottom_blocks for ch in out), "Should use left-filling blocks, not bottom-filling"


# ---------------------------------------------------------------------------
# WorkflowResult._metadata_text_lines()
# ---------------------------------------------------------------------------


class TestMetadataTextLines:
    def test_all_metadata_fields(self):
        """All metadata fields present (lines 132-140)."""
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
        meta = ResultMetadata(timestamp=ts, execution_time_s=1.23, dataset_id="ds-1")
        result = _make_result(metadata=meta)
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
        meta.model_dump = MagicMock(return_value={})
        result = _make_result(metadata=meta)
        out = result.report()
        assert "Timestamp" not in out
        assert "Duration" not in out
        assert "Dataset" not in out
        assert "Model" not in out
        assert "Preprocessor" not in out
        assert "Selection" not in out

    def test_metadata_with_model(self):
        """Model ID appears in metadata block."""
        meta = ResultMetadata(model_id="resnet50")
        result = _make_result(metadata=meta)
        out = result.report()
        assert "Model:" in out
        assert "resnet50" in out

    def test_metadata_with_preprocessor(self):
        """Preprocessor ID appears in metadata block."""
        meta = ResultMetadata(preprocessor_id="resnet50_preprocessor")
        result = _make_result(metadata=meta)
        out = result.report()
        assert "Preprocessor:" in out
        assert "resnet50_preprocessor" in out

    def test_metadata_with_selection(self):
        """Selection ID appears in metadata block."""
        meta = ResultMetadata(selection_id="training_subset")
        result = _make_result(metadata=meta)
        out = result.report()
        assert "Selection:" in out
        assert "training_subset" in out

    def test_metadata_dataset_line_includes_label_source(self):
        """Dataset line includes label_source in parentheses when present."""
        meta = ResultMetadata(dataset_id="my-dataset", label_source="annotations")
        result = _make_result(metadata=meta)
        out = result.report()
        assert "my-dataset" in out
        assert "(annotations)" in out


# ---------------------------------------------------------------------------
# WorkflowResult._report_serialized() — file paths
# ---------------------------------------------------------------------------


class TestReportSerialized:
    def test_json_to_directory(self, tmp_path: Path):
        """JSON written to a directory creates results.json."""
        result = _make_result()
        out = result.export(tmp_path)
        assert isinstance(out, Path)
        assert out.name == "results.json"
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert "metadata" in parsed

    def test_json_to_file_path(self, tmp_path: Path):
        """JSON written to a specific file path."""
        dest = tmp_path / "sub" / "output.json"
        result = _make_result()
        out = result.export(dest)
        assert isinstance(out, Path)
        assert out == dest
        assert out.exists()

    def test_yaml_to_directory(self, tmp_path: Path):
        """YAML written to a directory creates results.yaml."""
        result = _make_result()
        out = result.export(tmp_path, fmt="yaml")
        assert isinstance(out, Path)
        assert out.name == "results.yaml"
        assert out.exists()

    def test_yaml_to_file_path(self, tmp_path: Path):
        """YAML written to a specific file path."""
        dest = tmp_path / "deep" / "nested" / "out.yaml"
        result = _make_result()
        out = result.export(dest, fmt="yaml")
        assert isinstance(out, Path)
        assert out == dest
        assert out.exists()

    def test_json_no_path_returns_string(self):
        """path=None returns serialized string."""
        result = _make_result()
        out = result.export()
        assert isinstance(out, str)
        parsed = json.loads(out)
        assert "metadata" in parsed

    def test_yaml_no_path_returns_string(self):
        """path=None returns YAML string."""
        result = _make_result()
        out = result.export(fmt="yaml")
        assert isinstance(out, str)
        assert "metadata:" in out

    def test_directory_without_suffix(self, tmp_path: Path):
        """Path without suffix treated as directory."""
        dest = tmp_path / "no_suffix_dir"
        result = _make_result()
        out = result.export(dest)
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
        names = [w.name for w in workflows]
        assert "data-cleaning" in names
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
# _source_lines — multi-source continuation
# ---------------------------------------------------------------------------


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
