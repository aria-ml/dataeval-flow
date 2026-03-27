"""Tests for _app._viewmodel._result_vm — ResultViewModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from dataeval_flow._app._viewmodel._result_vm import FindingSummary, ResultViewModel
from dataeval_flow.workflow.base import Reportable

# ---------------------------------------------------------------------------
# Helpers — build fake WorkflowResult with typed findings
# ---------------------------------------------------------------------------


@dataclass
class _FakeReport:
    summary: str = "Test Report"
    findings: list[Reportable] = field(default_factory=list)


@dataclass
class _FakeData:
    report: _FakeReport | None = None


@dataclass
class _FakeMetadata:
    timestamp: datetime | None = datetime(2025, 1, 1, tzinfo=timezone.utc)
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
    name: str = "test_task"
    success: bool = True
    data: Any = None
    metadata: Any = None
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = _FakeData()
        if self.metadata is None:
            self.metadata = _FakeMetadata()


def _make_finding(
    title: str = "Finding",
    severity: str = "ok",
    report_type: str = "key_value",
    data: dict[str, Any] | str | None = None,
    description: str | None = None,
) -> Reportable:
    return Reportable(
        report_type=report_type,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        title=title,
        data=data if data is not None else {"brief": "test"},
        description=description,
    )


def _make_result(*findings: Reportable) -> _FakeResult:
    report = _FakeReport(findings=list(findings))
    data = _FakeData(report=report)
    return _FakeResult(data=data)


# ---------------------------------------------------------------------------
# ResultViewModel basics
# ---------------------------------------------------------------------------


class TestResultViewModelBasics:
    def test_empty_findings(self) -> None:
        rvm = ResultViewModel(_make_result())
        assert rvm.finding_count() == 0
        assert rvm.warning_count() == 0
        assert "0 findings" in rvm.summary_line()

    def test_no_report(self) -> None:
        result = _FakeResult(data=_FakeData(report=None))
        rvm = ResultViewModel(result)
        assert rvm.finding_count() == 0
        assert rvm.report_summary() == ""

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
                _make_finding("Outliers", severity="warning", report_type="key_value"),
                _make_finding("Labels", severity="ok", report_type="table"),
            )
        )
        summaries = rvm.finding_summaries()
        assert len(summaries) == 2
        assert isinstance(summaries[0], FindingSummary)
        assert summaries[0].title == "Outliers"
        assert summaries[0].severity == "warning"
        assert summaries[0].has_table is False
        assert summaries[1].has_table is True

    def test_table_types_have_table_flag(self) -> None:
        for rt in ("table", "pivot_table", "classwise_table", "chunk_table"):
            rvm = ResultViewModel(_make_result(_make_finding("X", report_type=rt, data={"brief": "x"})))
            summaries = rvm.finding_summaries()
            assert summaries[0].has_table is True, f"{rt} should have has_table=True"


class TestFindingMarkup:
    def test_summary_markup(self) -> None:
        rvm = ResultViewModel(_make_result(_make_finding("Outliers")))
        markup = rvm.finding_summary_markup(0)
        assert "Outliers" in markup

    def test_summary_markup_out_of_range(self) -> None:
        rvm = ResultViewModel(_make_result())
        assert rvm.finding_summary_markup(0) == ""
        assert rvm.finding_summary_markup(-1) == ""

    def test_detail_markup(self) -> None:
        rvm = ResultViewModel(
            _make_result(_make_finding("Outliers", report_type="key_value", data={"brief": "3 flagged"}))
        )
        detail = rvm.finding_detail_markup(0)
        assert "OUTLIERS" in detail

    def test_detail_markup_out_of_range(self) -> None:
        rvm = ResultViewModel(_make_result())
        assert rvm.finding_detail_markup(0) == ""


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------


class TestFindingTableData:
    def test_out_of_range(self) -> None:
        rvm = ResultViewModel(_make_result())
        assert rvm.finding_table_data(0) is None

    def test_non_dict_data(self) -> None:
        rvm = ResultViewModel(_make_result(_make_finding("X", data="plain text")))
        assert rvm.finding_table_data(0) is None

    def test_simple_table(self) -> None:
        data = {"table_data": {"cat": 10, "dog": 5}, "table_headers": ("Class", "Count")}
        rvm = ResultViewModel(_make_result(_make_finding("Labels", report_type="table", data=data)))
        result = rvm.finding_table_data(0)
        assert result is not None
        headers, rows = result
        assert headers == ["Class", "Count"]
        assert len(rows) == 2
        assert rows[0] == ["cat", "10"]

    def test_simple_table_default_headers(self) -> None:
        data = {"table_data": {"a": 1}}
        rvm = ResultViewModel(_make_result(_make_finding("X", report_type="table", data=data)))
        result = rvm.finding_table_data(0)
        assert result is not None
        headers, _ = result
        assert headers == ["Name", "Value"]

    def test_simple_table_empty(self) -> None:
        data: dict[str, Any] = {"table_data": {}}
        rvm = ResultViewModel(_make_result(_make_finding("X", report_type="table", data=data)))
        assert rvm.finding_table_data(0) is None

    def test_pivot_table(self) -> None:
        data = {
            "table_headers": ["Class Name", "Count", "%"],
            "table_data": [
                {"class_name": "cat", "count": 10, "pct": 50.0},
                {"class_name": "dog", "count": 10, "pct": 50.0},
            ],
        }
        rvm = ResultViewModel(_make_result(_make_finding("X", report_type="pivot_table", data=data)))
        result = rvm.finding_table_data(0)
        assert result is not None
        headers, rows = result
        assert headers == ["Class Name", "Count", "%"]
        assert rows[0] == ["cat", "10", "50.0%"]

    def test_pivot_table_empty(self) -> None:
        data: dict[str, Any] = {"table_headers": [], "table_data": []}
        rvm = ResultViewModel(_make_result(_make_finding("X", report_type="pivot_table", data=data)))
        assert rvm.finding_table_data(0) is None

    def test_classwise_table(self) -> None:
        data = {
            "table_rows": [
                {"Class": "cat", "Distance": 0.1234, "Status": "OK"},
                {"Class": "dog", "Distance": 0.5678, "Status": "DRIFT"},
            ],
        }
        rvm = ResultViewModel(_make_result(_make_finding("X", report_type="classwise_table", data=data)))
        result = rvm.finding_table_data(0)
        assert result is not None
        headers, rows = result
        assert headers == ["Class", "Distance", "Status"]
        assert rows[0] == ["cat", "0.1234", "OK"]

    def test_chunk_table(self) -> None:
        data = {
            "table_rows": [
                {"Chunk": "1", "Distance": 0.5, "Status": "OK"},
            ],
        }
        rvm = ResultViewModel(_make_result(_make_finding("X", report_type="chunk_table", data=data)))
        result = rvm.finding_table_data(0)
        assert result is not None

    def test_row_table_empty(self) -> None:
        data: dict[str, Any] = {"table_rows": []}
        rvm = ResultViewModel(_make_result(_make_finding("X", report_type="classwise_table", data=data)))
        assert rvm.finding_table_data(0) is None

    def test_key_value_returns_none(self) -> None:
        rvm = ResultViewModel(_make_result(_make_finding("X", report_type="key_value")))
        assert rvm.finding_table_data(0) is None

    def test_text_returns_none(self) -> None:
        rvm = ResultViewModel(_make_result(_make_finding("X", report_type="text", data="hello")))
        assert rvm.finding_table_data(0) is None

    def test_pivot_table_none_value(self) -> None:
        data = {
            "table_headers": ["Name", "Count"],
            "table_data": [{"Name": "x", "count": None}],
        }
        rvm = ResultViewModel(_make_result(_make_finding("X", report_type="pivot_table", data=data)))
        result = rvm.finding_table_data(0)
        assert result is not None
        _, rows = result
        assert rows[0][1] == ""
