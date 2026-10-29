"""Experimental parts of Flow warn once per process, and the step catalog marks experimental step types."""

import warnings
from typing import Any

import pytest
from dataeval.exceptions import ExperimentalWarning as DataEvalExperimentalWarning

from dataeval_flow import ExperimentalWarning, _experimental, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._experimental import experimental, warn_experimental
from dataeval_flow.steps import list_steps
from dataeval_flow.workflows.bias import BiasWorkflow
from tests.chain_toys import chain_pipeline
from tests.golden.coverage import CoverageImages

_TAIL = " is experimental: it may change or be removed in any release without a deprecation cycle."


@pytest.fixture(autouse=True)
def _nothing_warned_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_experimental, "_WARNED", set())


def _ours(record: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in record if w.category is ExperimentalWarning]


def _run_bias(*, as_step: bool = False) -> list[str]:
    """Run the bias preset, as a task or as a custom workflow's step, and return Flow's experimental warnings."""
    DatasetCache.clear_instances()
    outer = {"name": "outer", "inputs": ["data"], "steps": [{"name": "b", "workflow": "w", "input": "data"}]}
    workflows: list[Any] = [{"name": "w", "type": "bias"}, *([outer] if as_step else [])]
    config = chain_pipeline(
        workflows=workflows,
        tasks=[{"name": "t", "workflow": "outer" if as_step else "w", "sources": ["src"]}],
        datasets={"src": CoverageImages()},
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert run_tasks(config)["t"].success
    return _ours(record)


def test_a_filter_on_dataeval_s_warning_covers_flow_s() -> None:
    assert issubclass(ExperimentalWarning, DataEvalExperimentalWarning)


def test_a_subject_warns_once_per_process() -> None:
    with pytest.warns(ExperimentalWarning) as record:
        warn_experimental("'x'")
    assert [str(w.message) for w in record] == ["'x'" + _TAIL]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_experimental("'x'")


def test_a_marked_function_or_class_warns_on_its_first_use() -> None:
    @experimental
    def helper() -> int:
        return 1

    @experimental
    class Thing:
        def __init__(self, value: int) -> None:
            self.value = value

    with pytest.warns(ExperimentalWarning, match=r"\.helper'"):
        assert helper() == 1
    with pytest.warns(ExperimentalWarning, match=r"\.Thing'"):
        assert Thing(2).value == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert helper() == 1
        assert Thing(3).value == 3


def test_the_catalog_marks_parity_and_the_check_judging_it() -> None:
    marked = {(entry.kind, entry.type) for entry in list_steps(plugins=False).steps if entry.experimental}
    assert marked == {("evaluator", "parity"), ("check", "factor-parity")}


def test_running_an_experimental_step_warns_once_per_type() -> None:
    assert _run_bias() == ["The 'parity' evaluator" + _TAIL, "The 'factor-parity' check" + _TAIL]
    assert _run_bias() == []


@pytest.mark.parametrize("as_step", [False, True], ids=["task", "step"])
def test_an_experimental_preset_warns_run_as_a_task_or_a_step(monkeypatch: pytest.MonkeyPatch, as_step: bool) -> None:
    monkeypatch.setattr(BiasWorkflow, "experimental", True)
    assert "The 'bias' workflow" + _TAIL in _run_bias(as_step=as_step)
