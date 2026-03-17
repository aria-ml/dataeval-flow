"""Tests for workflow/__init__.py — WorkflowResult.report() and discovery helpers."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from dataeval_flow.config import ResultMetadata
from dataeval_flow.workflow import WorkflowResult, get_workflow, list_workflows
from dataeval_flow.workflow.base import Reportable, WorkflowReportBase

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
        """data without .report returns 'no report available'."""
        result = _make_result(data=_DummyOutputNoReport())
        out = result.report(format="text")
        assert "no report available" in out

    def test_empty_findings(self):
        """Empty findings list shows 'No findings to report.'."""
        result = _make_result()
        out = result.report(format="text")
        assert "No findings to report." in out
        assert "TEST SUMMARY" in out  # summary is uppercased in the banner

    def test_findings_with_warnings(self):
        """Findings with warnings show count in health line."""
        findings = [
            Reportable(report_type="text", severity="warning", title="Bad Image", data="detail", description="desc1"),
            Reportable(report_type="text", severity="warning", title="Corrupt File", data="detail", description=None),
            Reportable(report_type="text", severity="ok", title="All Good", data="detail", description="ok desc"),
        ]
        report = _DummyReport(summary="Findings Test", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "2 warning(s)" in out
        assert "FINDINGS TEST" in out

    def test_findings_no_warnings(self):
        """Findings with no warnings show 'All checks passed' in health line."""
        findings = [
            Reportable(report_type="text", severity="ok", title="All Good", data="detail", description="fine"),
        ]
        report = _DummyReport(summary="Clean Report", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "All checks passed [ok]" in out

    def test_summary_section_present(self):
        """Report includes a SUMMARY section with dotted lines."""
        findings = [
            Reportable(report_type="text", severity="info", title="Check A", data="d", description="desc a"),
            Reportable(report_type="text", severity="info", title="Check B", data="d", description="desc b"),
        ]
        report = _DummyReport(summary="Summary Test", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "SUMMARY" in out
        assert "Check A" in out
        assert "Check B" in out

    def test_detail_sections_present(self):
        """Each finding gets a detail section with uppercased title."""
        findings = [
            Reportable(report_type="text", severity="info", title="My Finding", data="d", description="some detail"),
        ]
        report = _DummyReport(summary="S", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "MY FINDING" in out
        assert "some detail" in out

    def test_finding_without_description(self):
        """Finding with no description still renders its detail section."""
        findings = [
            Reportable(report_type="text", severity="info", title="NoDesc", data="d", description=None),
        ]
        report = _DummyReport(summary="S", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "NODESC" in out

    def test_outlier_per_metric_breakdown(self):
        """Image Outliers finding renders per-metric table in detail section."""
        findings = [
            Reportable(
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
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "IMAGE OUTLIERS" in out
        assert "brightness" in out
        assert "contrast" in out
        assert "multiple metrics" in out

    def test_duplicate_detail_renders(self):
        """Duplicate finding renders group details, methods, orientations."""
        findings = [
            Reportable(
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
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "DUPLICATES" in out
        assert "2 exact-duplicate groups (6 images)" in out
        assert "3 near-duplicate groups (10 images)" in out
        assert "dhash, phash" in out
        assert "rotated" in out

    def test_label_distribution_bar_chart(self):
        """Label Distribution finding renders bar chart with block characters."""
        findings = [
            Reportable(
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
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "LABEL DISTRIBUTION" in out
        assert "cat" in out
        assert "dog" in out
        assert "\u2588" in out  # bar chart block character
        assert "Imbalance ratio: 2.0" in out

    def test_health_line_with_warnings(self):
        """Health line shows warning count when warnings exist."""
        findings = [
            Reportable(report_type="text", severity="warning", title="Issue", data="d", description="bad"),
        ]
        report = _DummyReport(summary="S", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "1 warning(s)" in out

    def test_health_line_without_warnings(self):
        """Health line shows 'No issues detected' when no warnings."""
        findings = [
            Reportable(report_type="text", severity="info", title="Info", data="d", description="ok"),
        ]
        report = _DummyReport(summary="S", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "Health: All checks passed [ok]" in out

    def test_warning_marker_in_summary(self):
        """Warning findings get [!!] marker in summary line."""
        findings = [
            Reportable(report_type="text", severity="warning", title="Bad Thing", data="d", description="bad"),
        ]
        report = _DummyReport(summary="S", findings=findings)
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "[!!]" in out

    def test_duplicate_exact_only_no_methods_line(self):
        """Exact-only duplicates don't render Methods/Orientations lines."""
        findings = [
            Reportable(
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
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "1 exact-duplicate groups (3 images)" in out
        # No Methods/Orientations lines when near_groups=0
        assert "Methods:" not in out
        assert "Orientations:" not in out

    def test_label_distribution_balanced_no_imbalance_line(self):
        """Balanced labels (imbalance_ratio=1.0) suppress the imbalance line."""
        findings = [
            Reportable(
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
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "Imbalance" not in out

    def test_target_outlier_multiple_metrics_message(self):
        """Target outlier detail says 'targets' not 'images' in multiple-metrics message."""
        findings = [
            Reportable(
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
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        assert "Some targets trigger multiple metrics" in out
        assert "Some images" not in out

    def test_bar_chart_uses_left_filling_blocks(self):
        """Bar chart uses left-filling fractional blocks, not bottom-filling."""
        findings = [
            Reportable(
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
        data = _DummyOutputWithReport(report=report)
        result = _make_result(data=data)
        out = result.report(format="text")
        # Ensure no bottom-filling blocks are present (U+2581-U+2587)
        bottom_blocks = set("\u2581\u2582\u2583\u2584\u2585\u2586\u2587")
        assert not any(ch in bottom_blocks for ch in out), "Should use left-filling blocks, not bottom-filling"


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
        meta.model_id = None
        meta.preprocessor_id = None
        meta.selection_id = None
        meta.model_dump = MagicMock(return_value={})
        result = _make_result(metadata=meta)
        out = result.report(format="text")
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
        out = result.report(format="text")
        assert "Model:" in out
        assert "resnet50" in out

    def test_metadata_with_preprocessor(self):
        """Preprocessor ID appears in metadata block."""
        meta = ResultMetadata(preprocessor_id="resnet50_preprocessor")
        result = _make_result(metadata=meta)
        out = result.report(format="text")
        assert "Preprocessor:" in out
        assert "resnet50_preprocessor" in out

    def test_metadata_with_selection(self):
        """Selection ID appears in metadata block."""
        meta = ResultMetadata(selection_id="training_subset")
        result = _make_result(metadata=meta)
        out = result.report(format="text")
        assert "Selection:" in out
        assert "training_subset" in out


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
