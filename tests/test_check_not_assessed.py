"""Checks that cannot assess raise `StepSkipped`, and the engine records `StepResult.not_assessed` (audit spec §6.3)."""

from typing import Any
from unittest.mock import patch

import pytest

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult, Check, CheckConfig, DataType, Port
from tests.chain_toys import GroupCount, chain_pipeline
from tests.evaluator_toys import ToyFactors, ToyImages

_EVALUATORS = [{"name": "labels", "type": "label-health"}, {"name": "balance", "type": "balance"}]
_SPLITS = [{"name": "evals", "list": True}]


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _chain(steps: list[dict[str, Any]], inputs: list[Any], datasets: dict[str, Any]) -> ChainResult:
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": inputs, "steps": steps}],
        evaluators=_EVALUATORS,
        tasks=[{"name": "t", "workflow": "w", "sources": list(datasets)}],
        datasets=datasets,
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


_LABELS = [
    {"name": "train-labels", "evaluator": "labels", "input": "train"},
    {"name": "eval-labels", "evaluator": "labels", "input": "evals"},
]


def test_class_sufficiency_records_a_train_with_no_labelled_class_as_not_assessed() -> None:
    check = {"name": "judge", "check": "class-sufficiency", "input": "train-labels", "evals": "eval-labels"}
    result = _chain([*_LABELS, check], ["train", _SPLITS[0]], {"train": ToyImages(labeled=False), "v": ToyImages()})
    assert result.steps["judge"].status == "ok"
    assert result.steps["judge"].not_assessed == "train holds no labelled class"


def test_untrained_classes_records_evaluation_splits_with_no_labelled_class_as_not_assessed() -> None:
    check = {"name": "judge", "check": "untrained-classes", "input": "train-labels", "evals": "eval-labels"}
    result = _chain([*_LABELS, check], ["train", _SPLITS[0]], {"train": ToyImages(), "v": ToyImages(labeled=False)})
    assert result.steps["judge"].not_assessed == "no evaluation split holds a labelled class"


def test_shortcut_risk_records_no_factor_to_score_as_not_assessed() -> None:
    steps = [
        {"name": "bal", "evaluator": "balance", "input": "train"},
        {"name": "judge", "check": "shortcut-risk", "input": "bal"},
    ]
    # DataEval refuses a Balance with no factor, so what it scored is emptied where the check reads it.
    with patch("dataeval_flow.steps.combines._gaps.mi_from_balance", return_value={}):
        result = _chain(steps, ["train"], {"train": ToyFactors(60)})
    assert result.steps["judge"].not_assessed == "no factor to score"


def test_a_port_may_be_empty_only_where_it_takes_a_whole_list() -> None:
    class _Config(CheckConfig):
        input: str

    with pytest.raises(TypeError, match="may be empty only where it takes a whole list"):

        class _Bad(Check[_Config]):
            """A check whose single-item port claims it may be empty."""

            name = "bad-may-be-empty"
            description = "Refused."
            title = "Bad"
            inputs = (Port("input", DataType.OUTPUT, classes=(GroupCount,), may_be_empty=True),)

            def run(self, config, inputs, context):
                return []
