"""The `uncovered-items` check: the share of a Dataset's items coverage left uncovered (data-splitting spec §6.2)."""

import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps.checks import UncoveredItemsCheck, UncoveredItemsConfig
from tests.chain_toys import chain_pipeline, run_chain_task
from tests.evaluator_toys import ToyFactors


def _judge(uncovered: int, items: int, **settings: Any) -> Any:
    node = SimpleNamespace(value=SimpleNamespace(uncovered_indices=np.arange(uncovered)), items=items)
    (finding,) = UncoveredItemsCheck().run(UncoveredItemsConfig(input="cov", **settings), {"input": node}, None)  # type: ignore[arg-type]
    return finding


@pytest.mark.parametrize(("uncovered", "severity"), [(3, "info"), (4, "info"), (5, "warning")])
def test_past_rate_it_warns(uncovered: int, severity: str) -> None:
    finding = _judge(uncovered, 40)
    assert (finding.severity, finding.title) == (severity, "Uncovered Items")


def test_the_brief_counts_the_uncovered() -> None:
    assert _judge(3, 40, warning=5.0).brief == "3 of 40 uncovered (7.5%)"


def test_a_null_rate_judges_nothing() -> None:
    assert _judge(40, 40, warning=None).severity == "info"


def test_it_judges_naive_coverage_in_a_chain() -> None:
    DatasetCache.clear_instances()
    # naive coverage's critical value takes gamma(d / 2 + 1), which overflows past ~340 dimensions; 3 x 8 x 8 pixels
    # flatten to 192
    toys = ToyFactors(count=12)
    steps = [
        {"name": "cov", "evaluator": "cov", "input": "data"},
        {"name": "uncovered", "check": "uncovered-items", "input": "cov", "warning": 100.0},
    ]
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        evaluators=[{"name": "cov", "type": "coverage", "method": "naive", "num_observations": 3}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"], "extractor": "flat"}],
        datasets={"src": toys},
        extractor=True,
    )
    result = run_chain_task(config)
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    (finding,) = result.findings
    assert finding.severity == "info"
    assert re.fullmatch(r"\d+ of 12 uncovered \([\d.]+%\)", finding.brief or "")
