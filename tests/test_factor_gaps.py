"""`factor-gaps` and `coverage-gaps`: class-factor-value combinations under-represented among factors balance ties
to the class, as legacy data-coverage's Metadata Coverage Gaps (coverage spec §6.1, §6.2)."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps.checks import CoverageGapsCheck, CoverageGapsConfig
from dataeval_flow.steps.combines import FactorGap, FactorGapsOutput
from tests.chain_toys import chain_pipeline
from tests.golden.coverage import CoverageImages

_STEPS = [
    {"name": "balance", "evaluator": "balance", "input": "data"},
    {"name": "gaps", "combine": "factor-gaps", "input": "data", "balance": "balance"},
    {"name": "gaps-check", "check": "coverage-gaps", "input": "gaps"},
]


def _run(steps: list[dict[str, Any]], dataset: Any) -> ChainResult:
    DatasetCache.clear_instances()
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        evaluators=[{"name": "balance", "type": "balance"}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": dataset},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


def test_it_finds_the_under_represented_combinations() -> None:
    result = _run(_STEPS, CoverageImages())
    assert result.success, result.errors
    output = result.steps["gaps"].output
    assert isinstance(output, FactorGapsOutput)
    assert output.mutual_information["site"] >= 0.1
    assert len(output.gaps) == 5
    assert output.gaps == sorted(output.gaps, key=lambda gap: gap.deficit, reverse=True)
    (finding,) = result.findings
    assert (finding.severity, finding.title, finding.brief) == (
        "warning",
        "Metadata Coverage Gaps",
        "5 gaps identified",
    )


def test_a_balance_from_another_dataset_is_refused_at_load() -> None:
    steps = [
        {"name": "half", "transform": "split", "input": "data", "test_frac": 0.5},
        {"name": "balance", "evaluator": "balance", "input": "half.train"},
        {"name": "gaps", "combine": "factor-gaps", "input": "data", "balance": "balance"},
    ]
    with pytest.raises(ValidationError, match="reads `balance`, which was computed on `half.train`, not on `data`"):
        _run(steps, CoverageImages())


def _judge(gaps: list[FactorGap], **settings: Any) -> Any:
    node = SimpleNamespace(value=FactorGapsOutput(mutual_information={"site": 0.5}, gaps=gaps))
    (finding,) = CoverageGapsCheck().run(CoverageGapsConfig(input="gaps", **settings), {"input": node}, None)  # type: ignore[arg-type]
    return finding


_GAP = FactorGap(
    class_name="dog", factor_name="site", factor_value="site-0", class_count=1, expected_count=13.5, deficit=0.926
)


def test_no_gaps_is_ok() -> None:
    finding = _judge([])
    assert (finding.severity, finding.brief) == ("ok", "No significant gaps detected")


@pytest.mark.parametrize(("n", "severity"), [(2, "info"), (3, "warning")])
def test_count_gaps_or_more_warn(n: int, severity: str) -> None:
    assert _judge([_GAP] * n).severity == severity


def test_a_null_count_judges_nothing() -> None:
    assert _judge([_GAP] * 9, count=None).severity == "info"


def test_gaps_of_equal_deficit_come_in_value_order() -> None:
    import polars as pl

    from dataeval_flow.steps.combines._gaps import _factor_gaps

    angle = pl.Series("angle", ["4", "3"] * 20 + ["5"] * 40)
    labels = [0] * 40 + [1] * 40
    for _ in range(20):  # Polars orders equal counts by chance
        gaps = _factor_gaps("angle", angle, labels, {0: "cat", 1: "dog"}, len(labels), 5)
        assert [(gap.class_name, gap.factor_value) for gap in gaps] == [("cat", "5"), ("dog", "3"), ("dog", "4")]
