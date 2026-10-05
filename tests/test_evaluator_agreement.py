"""Each bias evaluator gives the same answer as DataEval's own call on the same metadata.

The evaluator runs as a task. DataEval's class then evaluates the metadata Flow builds for that source, under the
policy Flow resolves for the evaluator, with settings that mean the same call. DataEval and torch are seeded before
each, so a random estimate agrees too.
"""

import json
from collections.abc import Callable
from typing import Any

import pytest
import torch
from dataeval.bias import Balance, Diversity
from dataeval.config import set_seed

from dataeval_flow import run_task
from dataeval_flow._metadata import build_metadata
from dataeval_flow._orchestrator import _resolve_metadata_policy
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.bias import BalanceConfig, DiversityConfig
from tests.evaluator_toys import ToyFactors, output_json, toy_pipeline


def _seeded() -> None:
    set_seed(0)
    torch.manual_seed(0)


def _json(value: Any) -> Any:
    """`value` as JSON reads it back, so NumPy and Python numbers compare alike."""
    return json.loads(json.dumps(value, default=float))


@pytest.mark.parametrize(
    ("evaluator", "dataeval_call", "tables"),
    [
        (BalanceConfig(name="ev"), Balance, ("balance", "factors", "classwise")),
        (DiversityConfig(name="ev", method="shannon"), lambda: Diversity(method="shannon"), ("factors", "classwise")),
    ],
    ids=["balance", "diversity"],
)
def test_a_bias_evaluator_agrees_with_dataeval(
    evaluator: Any, dataeval_call: Callable[[], Any], tables: tuple[str, ...]
) -> None:
    task = TaskConfig(name="evaluator", workflow="ev", sources=["src"], kind="evaluator")
    config = toy_pipeline(evaluators=[evaluator], tasks=[task], dataset=ToyFactors())
    _seeded()
    result = run_task(config, task)
    assert result.success, result.errors

    metadata = build_metadata(ToyFactors(), _resolve_metadata_policy(evaluator, config, None))
    _seeded()
    expected = dataeval_call().evaluate(metadata)
    data = output_json(result)["data"]
    for table in tables:
        assert data[table]["rows"] == _json(getattr(expected, table).to_dicts()), table
