"""A workflow or evaluator that raises yields a failed result of its own class, whose output refuses to be read."""

from typing import Any
from unittest.mock import patch

import pytest

from dataeval_flow import run_task
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesEvaluator, DuplicatesResult
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig, DataCleaningResult, DataCleaningWorkflow
from tests.evaluator_toys import toy_pipeline

_CLEAN = DataCleaningConfig(name="clean", outlier_method="zscore", outlier_flags=["pixel"])


def test_an_exception_becomes_a_failed_result_of_the_workflows_class() -> None:
    config = toy_pipeline(workflows=[_CLEAN])
    task = TaskConfig(name="t", workflow="clean", sources="src")
    with patch.object(DataCleaningWorkflow, "run", side_effect=RuntimeError("boom")):
        result = run_task(task, config)
    assert isinstance(result, DataCleaningResult)
    assert not result.success
    assert result.errors == ["RuntimeError: boom"]
    with pytest.raises(RuntimeError, match="boom"):
        _ = result.output


def test_an_evaluator_exception_becomes_a_failed_result_naming_the_evaluator() -> None:
    config = toy_pipeline(evaluators=[DuplicatesConfig(name="dupes")])
    task = TaskConfig(name="t", workflow="dupes", sources="src", kind="evaluator")
    with patch.object(DuplicatesEvaluator, "run", side_effect=RuntimeError("boom")):
        result = run_task(task, config)
    assert isinstance(result, DuplicatesResult)
    assert not result.success
    assert result.metadata.evaluator == "quality.duplicates"
    assert result.errors == ["RuntimeError: boom"], "the same form a failed workflow records"
    with pytest.raises(RuntimeError, match="boom"):
        _ = result.output


@pytest.mark.parametrize(
    ("entry", "kind"),
    [(_CLEAN, "workflow"), (DuplicatesConfig(name="clean", cluster_sensitivity=1.0), "evaluator")],
    ids=["workflow", "evaluator"],
)
def test_a_task_refused_by_its_inputs_carries_the_envelope(entry: Any, kind: str) -> None:
    """Refused before it runs, as a failed result, with the envelope every other failed result carries."""
    import dataeval_flow

    config = toy_pipeline(
        workflows=[entry] if kind == "workflow" else (),
        evaluators=[entry] if kind == "evaluator" else (),
        sources=("a", "b"),
    )
    task = TaskConfig(name="t", workflow="clean", kind=kind, sources=["a", "b"])  # type: ignore[arg-type]
    result = run_task(task, config)
    assert not result.success
    assert "runs " + kind in result.errors[0]
    assert result.metadata.tool_version == dataeval_flow.__version__
    assert result.metadata.dataset_id == "toy,toy"
    assert [source["name"] for source in result.metadata.resolved_config["sources"]] == ["a", "b"]
    assert result.metadata.resolved_config[kind]["name"] == "clean"
    assert result.sources is not None
