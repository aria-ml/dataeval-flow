"""Tests for _app._viewmodel._result_vm — ResultViewModel."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from dataeval_flow._app._viewmodel._result_vm import FindingSummary, ResultViewModel
from dataeval_flow.steps import ChainMetadata, ChainResult, Finding
from tests.workflow_toys import count_result

pytestmark = pytest.mark.optional

# ---------------------------------------------------------------------------
# Helpers — build a workflow's result with typed findings
# ---------------------------------------------------------------------------

_METADATA = {
    "timestamp": datetime(2025, 1, 1, tzinfo=UTC),
    "execution_time_s": 1.23,
    "source_descriptions": ["src1 (ds1)"],
    "model_id": "resnet (onnx)",
    "preprocessor_id": "prep1",
    "dataset_id": "ds1",
}


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


def _make_result(*findings: Finding) -> ChainResult:
    return count_result(*findings, metadata=ChainMetadata(**_METADATA))


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
        rvm = ResultViewModel(ChainResult.failed(type="test.count", errors=["boom"]))
        assert rvm.finding_count() == 0
        assert rvm.status_tag() == " [bold red][failed][/bold red]"

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

    def test_the_summary_line_states_the_count_the_result_made(self) -> None:
        with patch.object(ChainResult, "warning_count", new=7):
            rvm = ResultViewModel(_make_result(_make_finding()))
            assert rvm.warning_count() == 7
            assert "7 warnings" in rvm.summary_line()


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
        result.metadata = ChainMetadata.model_construct(
            timestamp=None,
            execution_time_s=None,
            source_descriptions=[],
            model_id=None,
            preprocessor_id=None,
        )
        rvm = ResultViewModel(result)
        assert rvm.metadata_lines() == []


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
# Evaluator results — no findings, no health, no severity
# ---------------------------------------------------------------------------


class TestEvaluatorResults:
    def _result(self):
        from dataeval_flow.evaluators import EvaluatorResult
        from dataeval_flow.evaluators._result import EvaluatorMetadata

        return EvaluatorResult(
            type="duplicates",
            success=True,
            output=object(),
            serialized={
                "shape": "table",
                "columns": ["group_id", "item_indices"],
                "rows": [{"group_id": 0, "item_indices": [0, 5]}],
            },
            metadata=EvaluatorMetadata(evaluator="duplicates", execution_time_s=1.25),
        )

    def test_the_summary_counts_rows_not_findings(self):
        from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

        assert ResultViewModel(self._result()).summary_line() == "1 row, 1.2s"

    def test_there_is_no_health_and_no_status_tag(self):
        from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

        rvm = ResultViewModel(self._result())
        assert rvm.status_tag() == ""
        assert rvm.finding_count() == 0

    def test_the_output_renders_as_text(self):
        from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

        text = ResultViewModel(self._result()).output_text()
        assert "group_id" in text
        assert "item_indices" in text

    def test_workflow_results_keep_their_tag(self):
        from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

        assert ResultViewModel(_make_result()).status_tag() == " [green][ok][/green]"

    def _failed_result(self):
        from dataeval_flow.evaluators import EvaluatorResult
        from dataeval_flow.evaluators._result import EvaluatorMetadata

        return EvaluatorResult(
            type="duplicates",
            success=False,
            metadata=EvaluatorMetadata(evaluator="duplicates", execution_time_s=1.25),
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


class TestChainResults:
    """A custom workflow's result holds its steps, not one report: its findings are its steps'."""

    def _result(self):
        from dataeval_flow._chain._run import ChainRun
        from dataeval_flow.steps import ChainResult

        return ChainResult.from_run("w", ChainRun(steps={}, nodes={}, lineage=[], label_space=[]))

    def test_it_reads_the_steps_findings(self) -> None:
        rvm = ResultViewModel(self._result())
        assert rvm.finding_count() == 0
        assert rvm.summary_line() == "0 findings"
        assert rvm.status_tag() == " [green][ok][/green]"

    def test_a_successful_chain_shows_each_step_and_its_status(self, chain_results: dict[str, Any]) -> None:
        rvm = ResultViewModel(chain_results["ok"])
        text = rvm.output_text()
        # A step that ran carries no marker of its own; the count above the sections gives every step's status.
        assert re.search(r"Steps:\s+2 ran\n", text)
        assert re.search(r"TOY-FIRST · FEW\n", text)
        assert re.search(r"DUPLICATES · DUPES\n", text)
        assert "Items:  6" in text
        assert rvm.status_tag() == " [green][ok][/green]"

    def test_a_partly_failed_chain_shows_the_error_and_the_completed_steps_findings(
        self, chain_results: dict[str, Any]
    ) -> None:
        result = chain_results["mixed"]
        rvm = ResultViewModel(result)
        text = rvm.output_text()
        assert re.search(r"TOY-EXPLODE · BOOM\s+failed\n", text)
        assert "RuntimeError: boom on a" in text
        cleaned = result.steps["clean/at-least"].output
        assert len(cleaned) == 1
        assert rvm.finding_count() == 1
        assert rvm.summary_line().startswith("1 finding,")
        assert rvm.status_tag() == " [bold red][failed][/bold red]"


def test_a_matrix_result_shows_its_report_and_its_health() -> None:
    from dataeval_flow import MatrixResult, ResultMetadata
    from dataeval_flow._app._viewmodel._result_vm import ResultViewModel
    from dataeval_flow._matrix._result import MatrixRun
    from dataeval_flow.evaluators._result import EvaluatorResult

    run = MatrixRun(number=1, label="k=1", values={"k": 1}, result=EvaluatorResult.failed(type="toy", errors=["no"]))
    vm = ResultViewModel(MatrixResult(type="toy", keys=["k"], runs=[run], metadata=ResultMetadata()))
    assert "k=1" in vm.output_text().lower()
    assert "failed" in vm.status_tag()
    assert vm.summary_line().startswith("1 runs, 0 warning(s)")
