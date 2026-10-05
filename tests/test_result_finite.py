"""A result's JSON writes non-finite floats as null, which strict parsers require (ood-detection spec §6.4)."""

import json
from typing import Any

import pytest

from dataeval_flow import run, run_task
from dataeval_flow._result import finite_json
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult, StepResult
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages

_BODY: dict[str, Any] = {"nan": float("nan"), "list": [1.0, float("inf")], "tuple": (float("-inf"), 2)}


def test_finite_json_writes_every_non_finite_float_as_none() -> None:
    assert finite_json(_BODY) == {"nan": None, "list": [1.0, None], "tuple": [None, 2]}
    assert finite_json("NaN") == "NaN"


def test_a_result_s_json_holds_no_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    result = run(DuplicatesConfig(name="dupes"), ToyImages())
    monkeypatch.setattr(type(result), "_dict_body", lambda self: {"output": _BODY})  # noqa: ARG005
    payload = result.to_dict()
    assert payload["output"] == {"nan": None, "list": [1.0, None], "tuple": [None, 2]}
    json.dumps(payload, allow_nan=False)


def test_a_chain_s_json_holds_no_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = {"name": "w", "inputs": ["a"], "steps": [{"name": "d", "evaluator": "dupes", "input": "a"}]}
    config = chain_pipeline(workflows=[workflow], evaluators=[DuplicatesConfig(name="dupes")])
    result = run_task(config, TaskConfig(name="t", workflow="w", sources="src"))
    assert isinstance(result, ChainResult)
    monkeypatch.setattr(StepResult, "to_dict", lambda self: {"x": float("nan")})  # noqa: ARG005
    payload = result.to_dict()
    assert payload["steps"] == {"d": {"x": None}}
    json.dumps(payload, allow_nan=False)
