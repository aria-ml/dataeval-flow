"""The TUI's task form carries both references, because `PipelineConfig.tasks` is a union it cannot introspect."""

import pytest
from pydantic import ValidationError

from dataeval_flow._app._model._registry import SECTION_MODELS, TaskFormModel


def test_the_tasks_section_uses_the_form_model():
    assert SECTION_MODELS["tasks"] is TaskFormModel


def test_a_workflow_task_validates():
    form = TaskFormModel(name="t", workflow="clean", sources="a")
    assert form.workflow == "clean"


def test_an_evaluator_task_validates():
    form = TaskFormModel(name="t", evaluator="dupes", sources="a")
    assert form.evaluator == "dupes"


@pytest.mark.parametrize("targets", [{}, {"workflow": "clean", "evaluator": "dupes"}])
def test_exactly_one_target_is_required(targets: dict[str, str]):
    with pytest.raises(ValidationError, match="exactly one of `workflow` or `evaluator`"):
        TaskFormModel(name="t", sources="a", **targets)  # type: ignore
