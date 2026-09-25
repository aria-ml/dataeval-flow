"""Evaluator results: DataEval's output, the envelope, and a report with no verdict in it."""

import json
from pathlib import Path
from typing import Any

import dataeval

from dataeval_flow._text_report import _WIDTH
from dataeval_flow.evaluators import EvaluatorResult
from dataeval_flow.evaluators._report import ROW_LIMIT, render_result_body, render_rows
from dataeval_flow.evaluators._result import DataEvalExecution, EvaluatorMetadata


def _result(
    serialized: dict[str, Any], *, success: bool = True, errors: tuple[str, ...] = ()
) -> EvaluatorResult[object]:
    return EvaluatorResult(
        type="quality.duplicates",
        success=success,
        output=object() if success else None,
        serialized=serialized if success else None,
        metadata=EvaluatorMetadata(evaluator="quality.duplicates", dataeval=DataEvalExecution(version="1.1.1")),
        errors=list(errors),
    )


def _table(rows: int) -> dict[str, Any]:
    return {
        "shape": "table",
        "columns": ["group_id", "item_indices"],
        "rows": [{"group_id": i, "item_indices": [i, i + 100]} for i in range(rows)],
    }


class TestToDict:
    def test_it_is_an_evaluator_with_no_health(self):
        payload = _result(_table(1)).to_dict()
        assert payload["kind"] == "evaluator"
        assert "health" not in payload
        assert payload["output"] == _table(1)
        metadata = payload["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["evaluator"] == "quality.duplicates"
        dataeval_meta = metadata["dataeval"]
        assert isinstance(dataeval_meta, dict)
        assert dataeval_meta["version"] == "1.1.1"

    def test_export_writes_json(self, tmp_path: Path):
        path = _result(_table(1)).export(tmp_path)
        assert path == tmp_path / "results.json"
        assert json.loads(path.read_text())["kind"] == "evaluator"

    def test_export_returns_a_string_without_a_path(self):
        assert '"kind": "evaluator"' in _result(_table(1)).export()


class TestReport:
    def test_it_carries_no_verdict(self):
        text = _result(_table(2)).report()
        assert "QUALITY.DUPLICATES" in text
        assert "OUTPUT" in text
        assert "Health" not in text
        assert "SUMMARY" not in text
        assert "DataEval:     1.1.1" in text

    def test_the_console_form_caps_long_tables(self):
        text = _result(_table(ROW_LIMIT + 5)).report(detailed=False)
        assert "… and 5 more rows" in text

    def test_the_detailed_form_shows_every_row(self):
        text = _result(_table(ROW_LIMIT + 5)).report(detailed=True)
        assert "more rows" not in text

    def test_a_failure_lists_its_errors(self):
        text = _result({}, success=False, errors=("RuntimeError: boom",)).report()
        assert "FAILED" in text
        assert "boom" in text

    def test_a_mapping_renders_nested_tables(self):
        output = {"shape": "mapping", "data": {"drifted": True, "per_class": _table(1)}}
        text = _result(output).report()
        assert "drifted: True" in text
        assert "per_class:" in text
        assert "group_id" in text

    def test_an_array_shows_its_head(self):
        text = _result({"shape": "array", "data": list(range(15))}).report()
        assert "15 values" in text
        assert "… and 5 more" in text


class TestRenderResultBody:
    def test_a_success_renders_its_output(self):
        lines = render_result_body(_result(_table(1)), detailed=True)
        assert any("OUTPUT" in line for line in lines)
        assert any("group_id" in line for line in lines)

    def test_a_failure_renders_failed_plus_its_errors(self):
        lines = render_result_body(_result({}, success=False, errors=("RuntimeError: boom",)), detailed=True)
        assert lines[0] == "  FAILED"
        assert any("boom" in line for line in lines)


class TestRenderRows:
    def test_wide_cells_are_cut(self):
        rows = [{"a": ["xxxxxxxxxx"] * 50, "b": 1}]
        lines = render_rows(["a", "b"], rows, limit=None)
        assert all(len(line) <= _WIDTH for line in lines)
        assert "…" in lines[-1]

    def test_no_rows_says_so(self):
        assert render_rows(["a"], [], limit=None)[-1] == "  (no rows)"

    def test_a_contiguous_int_list_cell_is_not_compacted_to_a_range(self):
        """`_flow_repr`'s `range(...)` shorthand is for the workflow report. A table cell
        such as `dataset_indices: [0, 1]` must print as written."""
        rows = [{"dataset_indices": [0, 1]}]
        lines = render_rows(["dataset_indices"], rows, limit=None)
        assert "[0, 1]" in lines[-1]
        assert "range" not in lines[-1]


class TestDataEvalExecution:
    def test_without_a_record_it_names_the_installed_version(self):
        assert DataEvalExecution.from_meta(None).version == dataeval.__version__

    def test_an_empty_record_keeps_only_the_version(self):
        from dataeval.types import ExecutionMetadata

        execution = DataEvalExecution.from_meta(ExecutionMetadata._empty())
        assert execution.name == ""
        assert execution.execution_time is None
        assert execution.version


def test_workflow_results_say_what_they_are():
    from unittest.mock import MagicMock

    from dataeval_flow import ResultMetadata
    from dataeval_flow.workflows import WorkflowResult

    output = MagicMock()
    output.model_dump.return_value = {}
    result = WorkflowResult(type="data-cleaning", success=True, output=output, metadata=ResultMetadata())
    assert result.to_dict()["kind"] == "workflow"
