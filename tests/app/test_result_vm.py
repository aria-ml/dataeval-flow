"""Tests for _app._viewmodel._result_vm — ResultViewModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from dataeval_flow._app._viewmodel._result_vm import FindingSummary, ResultViewModel, table_data
from dataeval_flow._blocks import Column, Fields, Paragraph, Section, Table
from dataeval_flow.workflows import Finding

pytestmark = pytest.mark.optional

# ---------------------------------------------------------------------------
# Helpers — build fake WorkflowResult with typed findings
# ---------------------------------------------------------------------------


@dataclass
class _FakeReport:
    summary: str = "Test Report"
    findings: list[Finding] = field(default_factory=list)


@dataclass
class _FakeOutput:
    report: _FakeReport = field(default_factory=_FakeReport)


@dataclass
class _FakeMetadata:
    timestamp: datetime | None = datetime(2025, 1, 1, tzinfo=UTC)
    execution_time_s: float | None = 1.23
    source_descriptions: list[str] = field(default_factory=lambda: ["src1 (ds1)"])
    model_id: str | None = "resnet (onnx)"
    preprocessor_id: str | None = "prep1"
    dataset_id: str | None = "ds1"
    selection_id: str | None = None
    label_source: str | None = None
    resolved_config: dict[str, Any] = field(default_factory=dict)
    tool: str = "dataeval-flow"
    tool_version: str = "0.0.0"


@dataclass
class _FakeResult:
    type: str = "test_task"
    success: bool = True
    output: Any = None
    metadata: Any = None
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.output is None and self.success:
            self.output = _FakeOutput()
        if self.metadata is None:
            self.metadata = _FakeMetadata()


def _make_finding(
    title: str = "Finding",
    severity: str = "ok",
    brief: str = "test",
    blocks: list[Any] | None = None,
    description: str | None = None,
) -> Finding:
    return Finding(
        severity=severity,  # type: ignore[arg-type]
        title=title,
        brief=brief,
        description=description,
        blocks=blocks or [],
    )


def _make_result(*findings: Finding) -> _FakeResult:
    report = _FakeReport(findings=list(findings))
    return _FakeResult(output=_FakeOutput(report=report))


# ---------------------------------------------------------------------------
# ResultViewModel basics
# ---------------------------------------------------------------------------


class TestResultViewModelBasics:
    def test_empty_findings(self) -> None:
        rvm = ResultViewModel(_make_result())
        assert rvm.finding_count() == 0
        assert rvm.warning_count() == 0
        assert "0 findings" in rvm.summary_line()

    def test_a_failed_run_has_no_findings_and_shows_failed(self) -> None:
        rvm = ResultViewModel(_FakeResult(success=False, errors=["boom"]))
        assert rvm.finding_count() == 0
        assert rvm.report_summary() == ""
        assert rvm.status_tag() == " [bold red][failed][/bold red]"

    def test_report_summary(self) -> None:
        rvm = ResultViewModel(_make_result())
        assert rvm.report_summary() == "Test Report"

    def test_finding_count(self) -> None:
        rvm = ResultViewModel(
            _make_result(
                _make_finding("A"),
                _make_finding("B"),
            )
        )
        assert rvm.finding_count() == 2

    def test_warning_count(self) -> None:
        rvm = ResultViewModel(
            _make_result(
                _make_finding("A", severity="ok"),
                _make_finding("B", severity="warning"),
                _make_finding("C", severity="warning"),
            )
        )
        assert rvm.warning_count() == 2


class TestSummaryLine:
    def test_with_warnings(self) -> None:
        rvm = ResultViewModel(
            _make_result(
                _make_finding("A", severity="warning"),
            )
        )
        line = rvm.summary_line()
        assert "1 finding" in line
        assert "1 warning" in line
        assert "1.2" in line

    def test_no_execution_time(self) -> None:
        result = _make_result(_make_finding("A"))
        result.metadata.execution_time_s = None
        rvm = ResultViewModel(result)
        line = rvm.summary_line()
        assert "1 finding" in line


class TestMetadataLines:
    def test_full_metadata(self) -> None:
        rvm = ResultViewModel(_make_result())
        lines = rvm.metadata_lines()
        assert any("Timestamp" in ln for ln in lines)
        assert any("Duration" in ln for ln in lines)
        assert any("Source" in ln for ln in lines)
        assert any("Model" in ln for ln in lines)
        assert any("Preprocessor" in ln for ln in lines)

    def test_minimal_metadata(self) -> None:
        result = _make_result()
        result.metadata = _FakeMetadata(
            timestamp=None,
            execution_time_s=None,
            source_descriptions=[],
            model_id=None,
            preprocessor_id=None,
        )
        rvm = ResultViewModel(result)
        assert rvm.metadata_lines() == []


class TestHealthLine:
    def test_all_ok(self) -> None:
        rvm = ResultViewModel(_make_result(_make_finding("A", severity="ok")))
        assert "All checks passed" in rvm.health_line()

    def test_with_warnings(self) -> None:
        rvm = ResultViewModel(_make_result(_make_finding("A", severity="warning")))
        assert "1 warning" in rvm.health_line()


class TestFindingSummaries:
    def test_returns_list(self) -> None:
        rvm = ResultViewModel(
            _make_result(
                _make_finding("Outliers", severity="warning"),
                _make_finding("Labels", severity="ok"),
            )
        )
        summaries = rvm.finding_summaries()
        assert len(summaries) == 2
        assert isinstance(summaries[0], FindingSummary)
        assert summaries[0].title == "Outliers"
        assert summaries[0].severity == "warning"
        assert summaries[1].brief == "test"


class TestFindingMarkup:
    def test_summary_markup(self) -> None:
        rvm = ResultViewModel(_make_result(_make_finding("Outliers")))
        markup = rvm.finding_summary_markup(0)
        assert "Outliers" in markup

    def test_summary_markup_out_of_range(self) -> None:
        rvm = ResultViewModel(_make_result())
        assert rvm.finding_summary_markup(0) == ""
        assert rvm.finding_summary_markup(-1) == ""

    def test_a_summary_too_long_for_one_line_keeps_its_brief_and_marker(self) -> None:
        """A wrapped title keeps every line, so the brief and the marker on the last one still show."""
        brief = "worst: motorcycle (12.5%), 3/10 classes over 3.0%"
        rvm = ResultViewModel(_make_result(_make_finding("Classwise Outliers", severity="warning", brief=brief)))
        lines = rvm.finding_summary_markup(0).split("\n")
        assert len(lines) > 1
        assert lines[-1].endswith(f"{brief}  [!!]")


# ---------------------------------------------------------------------------
# Finding blocks and the detail segments the modal draws
# ---------------------------------------------------------------------------


class TestFindingBlocks:
    def test_the_description_leads_the_blocks(self) -> None:
        values = Fields(items=[("Count", 3)])
        rvm = ResultViewModel(_make_result(_make_finding("X", description="3 flagged.", blocks=[values])))
        assert rvm.finding_blocks(0) == [Paragraph(text="3 flagged."), values]

    def test_no_description_adds_no_paragraph_and_an_index_out_of_range_has_nothing(self) -> None:
        rvm = ResultViewModel(_make_result(_make_finding("X")))
        assert rvm.finding_blocks(0) == []
        assert rvm.finding_blocks(3) == []
        assert rvm.finding_segments(-1) == []


class TestSegments:
    """A finding's detail as the modal draws it: data tables on their own, everything else as text runs."""

    _DATA = Table(
        columns=[Column(key="name", header="Class"), Column(key="n", header="Count")],
        rows=[{"name": "cat", "n": 10}, {"name": "dog", "n": 5}],
    )

    def _segments(self, *blocks: Any, description: str | None = None) -> list[Any]:
        finding = _make_finding("X", description=description, blocks=list(blocks))
        return ResultViewModel(_make_result(finding)).finding_segments(0)

    def test_a_data_table_is_a_segment_of_its_own_between_runs_of_text(self) -> None:
        values = Fields(items=[("Count", 3)])
        assert self._segments(self._DATA, values, description="3 flagged.") == [
            [Paragraph(text="3 flagged.")],
            self._DATA,
            [values],
        ]

    def test_a_table_with_a_chart_stays_in_the_text_so_its_bars_show(self) -> None:
        chart = Table(
            columns=[Column(key="name", header="Class"), Column(key="n", kind="bar")],
            rows=[{"name": "cat", "n": 10}],
        )
        assert self._segments(chart) == [[chart]]

    def test_a_table_inside_a_section_stays_with_its_section(self) -> None:
        section = Section(title="Group", blocks=[self._DATA])
        assert self._segments(section) == [[section]]

    def test_a_table_without_rows_is_not_a_segment_of_its_own(self) -> None:
        empty = Table(columns=[Column(key="k", header="K")], rows=[])
        assert self._segments(empty) == [[empty]]

    def test_image_outliers_draw_their_flags_as_text_and_their_limits_as_a_data_table(self) -> None:
        """A real cleaning finding: its lede and flags table as text, its limits table native, then its values."""
        from dataeval_flow.workflows.data_cleaning import DataCleaningHealthThresholds
        from dataeval_flow.workflows.data_cleaning._outputs import DataCleaningRawOutput
        from dataeval_flow.workflows.data_cleaning._report import build_findings

        issues = [
            {"item_index": 0, "metric_name": "brightness", "metric_value": 0.1},
            {"item_index": 0, "metric_name": "contrast", "metric_value": 0.2},
        ]
        raw = DataCleaningRawOutput(dataset_size=29, img_outliers={"count": 2, "issues": issues})  # type: ignore[typeddict-item]
        finding = next(
            f for f in build_findings(raw, None, DataCleaningHealthThresholds()) if f.title == "Image Outliers"
        )
        text, limits, rest = ResultViewModel(_make_result(finding)).finding_segments(0)
        assert isinstance(text, list)
        lede, flagged = text
        assert lede == Paragraph(text="1 images (3.4%) flagged as outliers.")
        assert isinstance(flagged, Table)
        assert flagged.columns[-1].kind == "flags"
        assert isinstance(limits, Table)
        assert limits.columns[0].header == "Metric"
        assert [type(block) for block in rest] == [Fields]


class TestTableData:
    def test_cells_read_as_the_text_report_prints_them(self) -> None:
        table = Table(
            columns=[
                Column(key="c", header="Class"),
                Column(key="pct", header="%", format="{:.1f}%"),
                Column(key="note", header="Note"),
            ],
            rows=[{"c": "cat", "pct": 50.0, "note": None}],
        )
        assert table_data(table) == (["Class", "%", "Note"], [["cat", "50.0%", ""]])


# ---------------------------------------------------------------------------
# Evaluator results — no findings, no health, no severity
# ---------------------------------------------------------------------------


class TestEvaluatorResults:
    def _result(self):
        from dataeval_flow.evaluators import EvaluatorResult
        from dataeval_flow.evaluators._result import EvaluatorMetadata

        return EvaluatorResult(
            type="quality.duplicates",
            success=True,
            output=object(),
            serialized={
                "shape": "table",
                "columns": ["group_id", "item_indices"],
                "rows": [{"group_id": 0, "item_indices": [0, 5]}],
            },
            metadata=EvaluatorMetadata(evaluator="quality.duplicates", execution_time_s=1.25),
        )

    def test_it_is_recognized(self):
        from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

        assert ResultViewModel(self._result()).is_evaluator

    def test_the_summary_counts_rows_not_findings(self):
        from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

        assert ResultViewModel(self._result()).summary_line() == "1 row, 1.2s"

    def test_there_is_no_health_and_no_status_tag(self):
        from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

        rvm = ResultViewModel(self._result())
        assert rvm.health_line() == ""
        assert rvm.status_tag() == ""
        assert rvm.finding_count() == 0

    def test_the_output_renders_as_text(self):
        from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

        text = ResultViewModel(self._result()).output_text()
        assert "group_id" in text
        assert "item_indices" in text

    def test_workflow_results_keep_their_tag(self):
        from unittest.mock import MagicMock

        from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

        result = MagicMock()
        result.output.report.findings = []
        rvm = ResultViewModel(result)
        assert not rvm.is_evaluator
        assert rvm.status_tag() == " [green][ok][/green]"

    def _failed_result(self):
        from dataeval_flow.evaluators import EvaluatorResult
        from dataeval_flow.evaluators._result import EvaluatorMetadata

        return EvaluatorResult(
            type="quality.duplicates",
            success=False,
            metadata=EvaluatorMetadata(evaluator="quality.duplicates", execution_time_s=1.25),
            errors=["boom: bad params"],
        )

    def test_a_failed_run_shows_failed_not_a_health_verdict(self):
        from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

        rvm = ResultViewModel(self._failed_result())
        assert "failed" in rvm.status_tag()
        assert rvm.summary_line().startswith("failed: boom")
        text = rvm.output_text()
        assert "FAILED" in text
        assert "boom: bad params" in text

    def test_report_summary_is_empty_for_an_evaluator(self):
        from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

        assert ResultViewModel(self._result()).report_summary() == ""
