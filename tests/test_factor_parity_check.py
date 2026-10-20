"""The `factor-parity` check: factors strongly and significantly associated with the class, and the cells to collect."""

import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
from dataeval.bias import ParityOutput
from dataeval.protocols import DatasetMetadata

from dataeval_flow import run, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.evaluators.bias import ParityConfig
from dataeval_flow.steps import ChainResult, CheckContext, StepSkipped
from dataeval_flow.steps.checks import FactorParityCheck, FactorParityConfig
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyFactors

_FACTOR_STATUS = pl.Enum(["associated", "weak", "inconclusive", "not_detected", "untestable"])
_CELL_STATUS = pl.Enum(["underrepresented", "overrepresented", "too_thin", "no_finding", "untestable"])
_FACTORS = {
    "factor_name": pl.String,
    "status": _FACTOR_STATUS,
    "score": pl.Float64,
    "p_value": pl.Float64,
    "p_value_adjusted": pl.Float64,
    "is_associated": pl.Boolean,
    "has_insufficient_data": pl.Boolean,
    "design_effect": pl.Float64,
    "effective_groups": pl.Float64,
    "df_num": pl.Float64,
    "df_den": pl.Float64,
}
_CELLS = {
    "factor_name": pl.String,
    "level": pl.String,
    "class_name": pl.String,
    "status": _CELL_STATUS,
    "is_actionable": pl.Boolean,
    "observed": pl.Int64,
    "expected": pl.Float64,
    "deficit": pl.Float64,
    "lift": pl.Float64,
    "p_value": pl.Float64,
    "p_value_adjusted": pl.Float64,
    "is_deviant": pl.Boolean,
    "is_sparse": pl.Boolean,
}


def _factor(
    name: str = "site",
    status: str = "not_detected",
    score: float = 0.1,
    p_adjusted: float = 0.5,
    p_value: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    untestable = status == "untestable"
    return {
        "factor_name": name,
        "status": status,
        "score": math.nan if untestable else score,
        "p_value": math.nan if untestable else p_adjusted if p_value is None else p_value,
        "p_value_adjusted": math.nan if untestable else p_adjusted,
        "is_associated": status == "associated",
        "has_insufficient_data": status in ("inconclusive", "untestable"),
        "design_effect": 1.0,
        "effective_groups": 60.0,
        "df_num": 4.0,
        "df_den": math.inf,
        **extra,
    }


def _cell(
    status: str, level: str = "day", class_name: str = "cat", factor: str = "site", deficit: float = 0.0
) -> dict[str, Any]:
    return {
        "factor_name": factor,
        "level": level,
        "class_name": class_name,
        "status": status,
        "is_actionable": status in ("underrepresented", "overrepresented"),
        "observed": 0,
        "expected": deficit,
        "deficit": deficit,
        "lift": 0.0,
        "p_value": 0.001,
        "p_value_adjusted": 0.002,
        "is_deviant": status in ("underrepresented", "overrepresented"),
        "is_sparse": status == "too_thin",
    }


def _output(factors: list[dict[str, Any]], cells: list[dict[str, Any]] | None = None) -> ParityOutput:
    return ParityOutput(
        factors=pl.DataFrame(factors, schema=_FACTORS),
        cells=pl.DataFrame(cells or [], schema=_CELLS),
        group_source="rows",
        test="pearson",
        n_rows=60,
        n_groups=60,
    )


def _judge_all(output: Any = None, **limits: Any) -> list[Any]:
    if output is None:
        output = run(ParityConfig(), ToyFactors(60)).output  # `site` follows the class; `angle` does not
    node = SimpleNamespace(value=output, computed_on=(SimpleNamespace(address="train"),), address="parity")
    return FactorParityCheck().run(
        FactorParityConfig(input="parity", **limits), {"input": node}, CheckContext("t", "s")
    )


def _judge(output: Any = None, **limits: Any) -> Any:
    return _judge_all(output, **limits)[0]


def test_a_factor_associated_with_the_class_warns() -> None:
    finding = _judge()
    assert finding.severity == "warning"
    assert finding.title == "Factor Parity"
    assert finding.brief.startswith("1 of 2 factors associated with the class: site (V=")


def test_the_limits_decide() -> None:
    output = _output([_factor("site", "associated", 0.9, 0.001), _factor("angle", "not_detected", 0.0, 0.9)])
    assert _judge(output).severity == "warning"
    assert _judge(output, warning=None).severity == "info"
    assert _judge(output, warning=1.0).severity == "ok"


def test_a_strong_association_that_is_not_significant_does_not_warn() -> None:
    output = _output([_factor("site", "not_detected", 0.9, 0.2)])
    assert _judge(output).severity == "ok"
    assert _judge(output, p_value=0.5).severity == "warning"


def test_the_adjusted_p_value_decides_not_the_raw_one() -> None:
    output = _output([_factor("site", "not_detected", 0.9, 0.2, p_value=0.001)])
    assert _judge(output).severity == "ok"


@pytest.mark.parametrize("status", ["untestable", "inconclusive"])
def test_untestable_and_inconclusive_never_pass(status: str) -> None:
    output = _output([_factor("site", "not_detected"), _factor("angle", status)])
    finding = _judge(output)
    assert finding.severity == "info"
    assert finding.brief == "2 factors checked, 1 not assessed"
    assert f"{status}: angle" in finding.description


def test_a_warning_still_warns_beside_a_factor_that_could_not_be_assessed() -> None:
    output = _output([_factor("site", "associated", 0.9, 0.001), _factor("angle", "untestable")])
    finding = _judge(output)
    assert finding.severity == "warning"
    assert "untestable: angle" in finding.description


def test_a_factor_that_could_not_be_assessed_is_in_the_table_without_a_score() -> None:
    finding = _judge(_output([_factor("site", "not_detected", 0.2), _factor("angle", "untestable")]))
    rows = {row["factor"]: row for row in finding.blocks[0].rows}
    assert rows["angle"]["status"] == "untestable"
    assert rows["angle"]["score"] is None
    assert rows["site"]["score"] == 0.2


@pytest.mark.parametrize(
    ("statuses", "reason"),
    [
        (("untestable", "untestable"), r"untestable: site, angle \(too few levels"),
        (("inconclusive", "inconclusive"), r"inconclusive: site, angle \(not significant"),
        (("inconclusive", "untestable"), r"untestable: angle \(.*; inconclusive: site \("),
    ],
)
def test_no_factor_assessed_is_not_assessed(statuses: tuple[str, str], reason: str) -> None:
    output = _output([_factor("site", statuses[0]), _factor("angle", statuses[1])])
    with pytest.raises(StepSkipped, match=f"no factor could be assessed, .*{reason}"):
        _judge(output)


def _skip_reason(output: ParityOutput) -> str:
    with pytest.raises(StepSkipped) as skipped:
        _judge(output)
    return skipped.value.reason


def test_the_skip_summarizes_the_thin_cells_short_of_their_expected_count() -> None:
    cells = [
        _cell("too_thin", "a", "cat", deficit=0.7),
        _cell("too_thin", "b", "dog", deficit=2.5),
        _cell("too_thin", "c", "dog", "angle", deficit=1.0),
        _cell("too_thin", "d", "bird", deficit=0.2),
        _cell("too_thin", "e", "cat", deficit=-1.0),  # holds more than expected, so it cannot fall short
        _cell("no_finding", "f", "cat"),
    ]
    reason = _skip_reason(_output([_factor("site", "inconclusive"), _factor("angle", "inconclusive")], cells))
    assert "inconclusive: site, angle (" in reason  # the factors and why stay
    assert "4 thin cells hold fewer items than expected" in reason
    top = "site: b × dog (deficit 2.5), angle: c × dog (deficit 1.0), site: a × cat (deficit 0.7)"
    assert f"the largest deficits: {top}" in reason
    assert "d × bird" not in reason  # the top three only


def test_the_skip_says_nothing_of_cells_when_none_is_thin_and_short() -> None:
    cells = [
        _cell("too_thin", "a", "cat", deficit=0.0),
        _cell("too_thin", "b", "dog", deficit=-1.0),
        _cell("no_finding"),
    ]
    outputs = [_output([_factor("site", "inconclusive")], cells), _output([_factor("site", "inconclusive")])]
    assert not any("thin cell" in _skip_reason(output) for output in outputs)


def test_sparse_tables_are_named_in_the_description() -> None:
    output = _output([_factor("site", "weak", 0.1, 0.01, has_insufficient_data=True)])
    assert "unreliable: site." in _judge(output).description


def test_no_factor_is_not_assessed() -> None:
    with pytest.raises(StepSkipped, match="no factor to score"):
        _judge(_output([]))


def test_actionable_deficits_are_a_finding() -> None:
    cells = [
        _cell("underrepresented", "day", "cat", deficit=3.0),
        _cell("underrepresented", "night", "dog", deficit=9.0),
        _cell("overrepresented", "night", "cat", deficit=-9.0),
        _cell("no_finding", "day", "dog"),
    ]
    factors, gaps = _judge_all(_output([_factor("site", "associated", 0.9, 0.001)], cells))
    assert factors.title == "Factor Parity"
    assert gaps.title == "Parity Collection Gaps"
    assert gaps.severity == "info"  # a gap is a place to collect, which nothing judged
    assert gaps.brief == "2 under-represented cells"
    rows = gaps.blocks[0].rows
    assert [(row["level"], row["class"]) for row in rows] == [("night", "dog"), ("day", "cat")]  # strongest first


def test_too_thin_cells_are_listed_as_collect_to_find_out() -> None:
    cells = [
        _cell("too_thin", "a", "cat", deficit=1.0),
        _cell("too_thin", "b", "dog", deficit=2.5),
        _cell("too_thin", "a", "dog", "angle", deficit=0.5),
    ]
    _, thin = _judge_all(_output([_factor("site"), _factor("angle")], cells))
    assert thin.severity == "info"
    assert thin.brief == "3 cells too thin to judge"
    paragraph, table = thin.blocks
    assert "Collect more" in paragraph.text
    assert [(row["factor"], row["level"], row["class"]) for row in table.rows] == [
        ("site", "b", "dog"),
        ("site", "a", "cat"),
        ("angle", "a", "dog"),
    ]  # the largest deficit first
    assert "lift" not in table.rows[0]  # a thin cell's lift and p-value are not to be trusted


def test_a_thin_cell_holding_its_expected_share_cannot_fall_short() -> None:
    cells = [_cell("too_thin", "a", "cat", deficit=0.0), _cell("too_thin", "b", "dog", deficit=-1.0)]
    assert len(_judge_all(_output([_factor("site")], cells))) == 1
    cells.append(_cell("too_thin", "c", "dog", deficit=2.0))
    _, thin = _judge_all(_output([_factor("site")], cells))
    assert thin.brief == "1 cell too thin to judge"
    assert [row["level"] for row in thin.blocks[1].rows] == ["c"]


def test_a_cell_gap_and_a_thin_cell_share_a_finding() -> None:
    cells = [_cell("underrepresented", deficit=4.0), _cell("too_thin", "night", deficit=1.0)]
    _, gaps = _judge_all(_output([_factor("site")], cells))
    assert gaps.brief == "1 under-represented cell, 1 cell too thin to judge"
    assert [type(block).__name__ for block in gaps.blocks] == ["Table", "Paragraph", "Table"]


def test_nothing_to_collect_makes_one_finding() -> None:
    cells = [_cell("no_finding"), _cell("overrepresented", deficit=-5.0)]
    findings = _judge_all(_output([_factor("site", "associated", 0.9, 0.001)], cells))
    assert len(findings) == 1


class _Factors:
    """`count` images over three classes, each with the factors `factors` names, as a function of the item index."""

    def __init__(self, count: int, **factors: Callable[[int], Any]) -> None:
        rng = np.random.default_rng(0)
        self._images = [rng.integers(0, 255, (3, 8, 8), dtype=np.uint8) for _ in range(count)]
        self._factors = factors
        self.metadata: DatasetMetadata = {"id": f"factors-{count}", "index2label": {0: "cat", 1: "dog", 2: "bird"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(3, dtype=np.float32)
        target[index % 3] = 1.0
        return self._images[index], target, {name: make(index) for name, make in self._factors.items()}


def _verdict(dataset: Any) -> tuple[Any, list[Any]]:
    """The verdict of `parity` judged by `factor-parity` over `dataset`, and the findings."""
    DatasetCache.clear_instances()
    config = chain_pipeline(
        workflows=[
            {
                "name": "w",
                "inputs": ["data"],
                "steps": [
                    {"name": "parity", "evaluator": "parity", "input": "data"},
                    {"name": "factor-parity", "check": "factor-parity", "input": "parity"},
                ],
                "verdict": {},
            }
        ],
        evaluators=[ParityConfig(name="parity")],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": dataset},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert result.verdict is not None
    return result.verdict, result.findings


_SIDE = {"side": lambda i: (i // 3) % 2}  # balanced against the class, and tested on enough data to call independent


@pytest.mark.parametrize(
    ("dataset", "cells"),
    [
        (_Factors(12, angle=lambda i: float((i * 5) % 7)), True),  # inconclusive: too thin to call independent
        (_Factors(240, flat=lambda _: 1.0), False),  # untestable: one level, so its cells are untestable too
    ],
    ids=["inconclusive", "untestable"],
)
def test_a_run_where_no_factor_was_assessed_is_ready_with_caveats(dataset: Any, cells: bool) -> None:
    verdict, findings = _verdict(dataset)
    assert verdict.level == "ready-with-caveats"
    assert [(item.check, item.step) for item in verdict.not_assessed] == [("factor-parity", "factor-parity")]
    assert ("thin cells hold fewer items than expected" in verdict.not_assessed[0].reason) is cells
    assert [finding.brief for finding in findings] == ["not assessed"]


@pytest.mark.parametrize(
    "dataset",
    [
        _Factors(240, **_SIDE, angle=lambda i: float((i * 5) % 100)),  # beside one too thin to call independent
        _Factors(240, **_SIDE, flat=lambda _: 1.0),  # beside one with a single level
    ],
    ids=["inconclusive", "untestable"],
)
def test_a_factor_assessed_beside_one_not_is_ready_though_the_finding_says_so(dataset: Any) -> None:
    """A finding cannot move the verdict, only a skip can, so a partial run reads ready: the limit the docs state."""
    verdict, findings = _verdict(dataset)
    assert verdict.level == "ready"
    assert [finding.severity for finding in findings] == ["info"]
    assert findings[0].brief == "2 factors checked, 1 not assessed"


def test_the_gaps_of_a_real_run_are_the_underrepresented_site_cells() -> None:
    _, gaps = _judge_all()
    assert gaps.brief.startswith("6 under-represented cells")
    assert {row["factor"] for row in gaps.blocks[0].rows} == {"site"}
