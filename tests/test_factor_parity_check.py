"""The `factor-parity` check: factors strongly and significantly associated with the class."""

from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from dataeval.bias import ParityOutput

from dataeval_flow import run
from dataeval_flow.evaluators.bias import ParityConfig
from dataeval_flow.steps import CheckContext, StepSkipped
from dataeval_flow.steps.checks import FactorParityCheck, FactorParityConfig
from tests.evaluator_toys import ToyFactors


def _judge(output: Any = None, **limits: Any) -> Any:
    if output is None:
        output = run(ParityConfig(), ToyFactors(60)).output  # `site` follows the class; `angle` does not
    node = SimpleNamespace(value=output, computed_on=(SimpleNamespace(address="train"),), address="parity")
    (finding,) = FactorParityCheck().run(
        FactorParityConfig(input="parity", **limits), {"input": node}, CheckContext("t", "s")
    )
    return finding


def test_a_factor_associated_with_the_class_warns() -> None:
    finding = _judge()
    assert finding.severity == "warning"
    assert finding.title == "Factor Parity"
    assert finding.brief.startswith("1 of 2 factors associated with the class: site (V=")


def test_the_limits_decide() -> None:
    assert _judge(warning=None).severity == "info"
    assert _judge(warning=1.0).severity == "ok"


def _output(score: float, p_value: float, *, sparse: bool = False) -> ParityOutput:
    factors = pl.DataFrame(
        {
            "factor_name": ["site"],
            "score": [score],
            "p_value": [p_value],
            "is_significant": [False],
            "has_insufficient_data": [sparse],
        }
    )
    return ParityOutput(factors=factors, insufficient_data={"site": {}} if sparse else {})


def test_a_strong_association_that_is_not_significant_does_not_warn() -> None:
    assert _judge(_output(0.9, 0.2)).severity == "ok"
    assert _judge(_output(0.9, 0.2), p_value=0.5).severity == "warning"


def test_sparse_tables_are_named_in_the_description() -> None:
    assert "unreliable: site." in _judge(_output(0.1, 0.5, sparse=True)).description


def test_no_factor_is_not_assessed() -> None:
    empty = pl.DataFrame(
        schema={
            "factor_name": pl.String,
            "score": pl.Float64,
            "p_value": pl.Float64,
            "is_significant": pl.Boolean,
            "has_insufficient_data": pl.Boolean,
        }
    )
    with pytest.raises(StepSkipped, match="no factor to score"):
        _judge(ParityOutput(factors=empty, insufficient_data={}))
