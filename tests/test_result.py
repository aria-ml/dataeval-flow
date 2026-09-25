"""The shared result shape: both kinds render and serialize the same way, and differ only inside."""

import json

import pytest

from dataeval_flow import Result, ResultMetadata
from dataeval_flow.evaluators import EvaluatorResult
from dataeval_flow.evaluators._result import EvaluatorMetadata
from dataeval_flow.workflows import WorkflowResult
from dataeval_flow.workflows.data_cleaning import DataCleaningResult
from dataeval_flow.workflows.data_cleaning._outputs import (
    DataCleaningMetadata,
    DataCleaningOutput,
    DataCleaningRawOutput,
    DataCleaningReport,
)


def _output() -> DataCleaningOutput:
    return DataCleaningOutput(
        raw=DataCleaningRawOutput(dataset_size=3),
        report=DataCleaningReport(summary="Data cleaning complete."),
    )


def _workflow(*, success: bool = True) -> DataCleaningResult:
    return DataCleaningResult(
        type="data-cleaning",
        success=success,
        output=_output() if success else None,
        metadata=DataCleaningMetadata(),
        errors=[] if success else ["boom"],
    )


def _evaluator(*, success: bool = True) -> EvaluatorResult[object]:
    return EvaluatorResult(
        type="quality.duplicates",
        success=success,
        output=object() if success else None,
        serialized={"shape": "mapping", "data": {"groups": 0}} if success else None,
        metadata=EvaluatorMetadata(evaluator="quality.duplicates"),
        errors=[] if success else ["boom", "bang"],
    )


def test_the_base_cannot_be_built_on_its_own():
    with pytest.raises(TypeError, match="abstract"):
        Result(type="r", success=False, metadata=ResultMetadata())  # type: ignore[abstract]


@pytest.mark.parametrize(("make", "kind"), [(_workflow, "workflow"), (_evaluator, "evaluator")])
class TestOneShape:
    def test_each_kind_subclasses_result(self, make, kind):
        result = make()
        assert isinstance(result, Result)
        assert result.kind == kind

    def test_type_names_what_ran(self, make, kind):
        result = make()
        assert result.type in {"data-cleaning", "quality.duplicates"}
        assert not hasattr(result, "name")

    def test_a_dict_leads_with_the_kind_and_the_envelope(self, make, kind):
        payload = make().to_dict()
        assert list(payload)[:2] == ["kind", "metadata"]
        assert payload["kind"] == kind

    def test_json_export_is_the_dict(self, make, kind):
        result = make()
        assert json.loads(result.export()) == result.to_dict()

    def test_a_failed_run_reports_its_errors(self, make, kind):
        assert "  FAILED\n    boom" in make(success=False).report()

    def test_a_failed_runs_output_raises_naming_each_error(self, make, kind):
        failed = make(success=False)
        with pytest.raises(RuntimeError) as raised:
            _ = failed.output
        assert all(error in str(raised.value) for error in failed.errors)

    def test_a_failed_runs_dict_is_its_kind_envelope_and_errors(self, make, kind):
        payload = make(success=False).to_dict()
        assert set(payload) == {"kind", "metadata", "errors"}
        assert payload["errors"] == make(success=False).errors

    def test_a_success_must_carry_its_output(self, make, kind):
        result = make()
        with pytest.raises(ValueError, match="output"):
            type(result)(type=result.type, success=True, metadata=result.metadata)

    def test_a_failure_cannot_carry_an_output(self, make, kind):
        result = make()
        with pytest.raises(ValueError, match="output"):
            type(result)(type=result.type, success=False, output=result.output, metadata=result.metadata)

    def test_output_raises_whenever_the_run_is_not_a_success(self, make, kind):
        """The guard reads ``success``, so a result marked failed never hands back what it holds."""
        result = make()
        result.success = False
        with pytest.raises(RuntimeError, match="did not complete"):
            _ = result.output


def test_isinstance_narrows_to_the_types_own_result():
    result: Result = _workflow()
    assert isinstance(result, DataCleaningResult)
    assert isinstance(result, WorkflowResult)
    assert result.output.raw.dataset_size == 3
    assert isinstance(result.metadata, DataCleaningMetadata)


def test_only_a_workflow_carries_health():
    assert "health" in _workflow().to_dict()
    assert "health" not in _evaluator().to_dict()
    assert not hasattr(_evaluator(), "health")


def test_a_failed_workflow_has_no_findings_and_a_failed_health():
    failed = _workflow(success=False)
    assert failed.findings == []
    assert failed.health["status"] == "failed"
    assert failed.warning_count == 0


def test_a_failed_result_of_a_class_carries_that_class_metadata():
    failed = DataCleaningResult.failed(type="data-cleaning", errors=["boom"])
    assert isinstance(failed, DataCleaningResult)
    assert isinstance(failed.metadata, DataCleaningMetadata)
    assert not failed.success
    assert failed.errors == ["boom"]


def test_every_argument_is_keyword_only():
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        WorkflowResult("data-cleaning", True, _output(), ResultMetadata())  # type: ignore[misc]
