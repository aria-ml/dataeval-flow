"""The shared result shape: both kinds render and serialize the same way, and differ only inside."""

import json

import pytest
from pydantic import BaseModel

from dataeval_flow import EvaluatorResult, Result, WorkflowResult
from dataeval_flow.config import ResultMetadata
from dataeval_flow.evaluator.result import EvaluatorMetadata


class _Outputs(BaseModel):
    """Workflow outputs with no report, the smallest a workflow can return."""

    count: int = 0


def _workflow(*, success: bool = True) -> WorkflowResult:
    return WorkflowResult(
        name="data-cleaning",
        success=success,
        data=_Outputs(count=3),
        metadata=ResultMetadata(),
        errors=[] if success else ["boom"],
    )


def _evaluator(*, success: bool = True) -> EvaluatorResult:
    return EvaluatorResult(
        name="quality.duplicates",
        success=success,
        output={"shape": "mapping", "values": {"groups": 0}},
        metadata=EvaluatorMetadata(evaluator="quality.duplicates"),
        errors=[] if success else ["boom"],
    )


def test_the_base_cannot_be_built_on_its_own():
    with pytest.raises(TypeError, match="abstract"):
        Result(name="r", success=True, metadata=ResultMetadata())  # type: ignore[abstract]


@pytest.mark.parametrize(("make", "kind"), [(_workflow, "workflow"), (_evaluator, "evaluator")])
class TestOneShape:
    def test_each_kind_is_a_result(self, make, kind):
        result = make()
        assert isinstance(result, Result)
        assert result.kind == kind

    def test_a_dict_leads_with_the_kind_and_the_envelope(self, make, kind):
        payload = make().to_dict()
        assert list(payload)[:2] == ["kind", "metadata"]
        assert payload["kind"] == kind

    def test_json_export_is_the_dict(self, make, kind):
        result = make()
        assert json.loads(result.export()) == result.to_dict()

    def test_a_failed_run_reports_its_errors(self, make, kind):
        assert "  FAILED\n    boom" in make(success=False).report()


def test_only_a_workflow_carries_health():
    assert "health" in _workflow().to_dict()
    assert "health" not in _evaluator().to_dict()
    assert not hasattr(_evaluator(), "health")


def test_fields_after_the_payload_are_keyword_only():
    """Dataclass inheritance puts the shared fields first; positional metadata must fail loudly."""
    with pytest.raises(TypeError):
        WorkflowResult("data-cleaning", True, _Outputs(), ResultMetadata())  # type: ignore[misc]
