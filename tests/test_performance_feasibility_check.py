"""The `performance-feasibility` check: a row per target with its answer in DataEval's word and the bounds behind it,
and what the answers rest on.

The outputs are the evaluator's own, run on `ToyClasses`: four classes a flatten extractor separates, which demonstrate
an easy target, and the same with 60% of the labels redrawn, which no model fits better than 0.55 and which DataEval
therefore cannot show to reach 0.95. Allowances of `{beta: 0, gamma: 0}` let it rule 0.95 out at the default scope.
"""

import dataclasses
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
from dataeval.performance import Feasibility, FeasibilityOutput
from pydantic import ValidationError

from dataeval_flow import run, run_tasks
from dataeval_flow._blocks import Fields, Table
from dataeval_flow.evaluators.performance import FeasibilityConfig
from dataeval_flow.steps import ChainResult, CheckContext
from dataeval_flow.steps.checks import PerformanceFeasibilityCheck, PerformanceFeasibilityConfig
from dataeval_flow.steps.checks._feasibility import _SYMBOLS, _WORDS, _pct
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import FLAT, ToyClasses, ToyCopies

_BOTH = {"beta": 0, "gamma": 0}


def _output(target: dict[str, float], data: ToyClasses | ToyCopies | None = None, **settings: Any) -> FeasibilityOutput:
    config = FeasibilityConfig(target=target, seed=0, **settings)
    result = run(config, data or ToyClasses(count=400, classes=4), extractor=FLAT)
    assert result.success, result.errors
    return result.output


def _noisy(target: dict[str, float], **settings: Any) -> FeasibilityOutput:
    return _output(target, ToyClasses(count=1000, classes=4, flip=0.6), **settings)


def _judge(output: FeasibilityOutput) -> Any:
    node = SimpleNamespace(value=output)
    (finding,) = PerformanceFeasibilityCheck().run(
        PerformanceFeasibilityConfig(input="f"), {"input": node}, CheckContext("t", "s")
    )
    return finding


def _table(finding: Any) -> Table:
    table, _ = finding.blocks
    assert isinstance(table, Table)
    return table


def _rows(finding: Any) -> list[dict[str, Any]]:
    return _table(finding).rows


def _bounds(row: dict[str, Any]) -> tuple[str, str]:
    """The (lower, upper) a row's bounds cell reads, ``—`` for one it gives none: ``≥42.4%``, ``41.6–75.1%*``."""
    text = row["bounds"]
    if text == "—":
        return "—", "—"
    if text.startswith("≥"):
        return text[1:], "—"
    if text.startswith("≤"):
        return "—", text[1:]
    lower, upper = text.split("–")
    return f"{lower}%", upper


def _basis(finding: Any) -> dict[str, Any]:
    _, fields = finding.blocks
    assert isinstance(fields, Fields)
    return dict(fields.items)


def _with(output: FeasibilityOutput, **columns: Any) -> FeasibilityOutput:
    """`output` with the first determination's `columns` set, as a run that decided otherwise would hold them."""
    schema = output.determinations.schema
    frame = output.determinations.with_columns(
        [pl.lit(value, dtype=schema[name]).alias(name) for name, value in columns.items()]
    )
    return dataclasses.replace(output, determinations=frame)


def test_a_target_demonstrated_is_ok_and_draws_its_bound() -> None:
    output = _output({"accuracy": 0.9})
    finding = _judge(output)
    assert (finding.severity, finding.title) == ("ok", "Performance Feasibility")
    assert finding.brief == "1 of 1 targets demonstrated"
    assert finding.description is None  # the explanation belongs in the tutorial, once
    (row,) = _rows(finding)
    determination = output.determinations.row(0, named=True)
    assert (row["answer"], row["target"]) == ("✓", "accuracy ≥ 90%")
    assert _bounds(row) == (_pct(determination["witness"], "lower"), "—")
    shown, open_, ruled_out = row["interval"]
    assert (shown, ruled_out) == (int(determination["witness"] * 1000) / 1000, 0)
    assert shown + open_ + ruled_out == pytest.approx(1)
    basis = _basis(finding)
    assert basis["Confidence"] == "95%, all bounds together, for new images like these 400"
    assert "Population" not in basis  # every item the source holds is labeled: the confidence line counts them
    assert "Allowances" not in basis


def test_the_interval_is_drawn_as_a_stacked_bar_with_its_series_named() -> None:
    column = next(column for column in _table(_judge(_output({"accuracy": 0.9}))).columns if column.key == "interval")
    assert (column.kind, column.header, column.series) == ("stacked", "", ["shown", "open", "ruled out"])


def test_a_target_with_insufficient_evidence_warns() -> None:
    output = _noisy({"accuracy": 0.95})
    finding = _judge(output)
    assert finding.severity == "warning"
    assert finding.brief == "0 of 1 targets demonstrated, 1 insufficient evidence"
    (row,) = _rows(finding)
    determination = output.determinations.row(0, named=True)
    assert (row["answer"], row["target"]) == ("?", "accuracy ≥ 95%")
    assert _bounds(row)[1] == "—"
    assert row["interval"][2] == 0  # nothing ruled out
    assert determination["limiting_factor"] == "scope_restricted"
    basis = _basis(finding)
    assert "Allowances" not in basis


def test_a_target_ruled_out_warns_says_under_what_and_shows_the_allowances_used() -> None:
    output = _noisy({"accuracy": 0.95}, allowances=_BOTH)
    finding = _judge(output)
    assert finding.severity == "warning"  # fails closed: a ruled-out target holds only under assumptions
    assert finding.brief == "0 of 1 targets demonstrated, 1 ruled out under the recorded assumptions"
    (row,) = _rows(finding)
    determination = output.determinations.row(0, named=True)
    assert (row["answer"], row["target"]) == ("✗", "accuracy ≥ 95%")
    assert _bounds(row)[1] == _pct(determination["ceiling"], "upper") + "*"  # marked: it holds under the allowances
    assert row["interval"][2] > 0  # part of the scale is ruled out
    basis = _basis(finding)
    assert basis["Allowances"] == "beta=0.0, gamma=0.0, asserted by you; * rests on them"


def test_allowances_are_shown_where_none_row_is_starred() -> None:
    output = _noisy({"accuracy": 0.95, "ap": 0.5}, allowances={"beta": 0})
    finding = _judge(output)
    assert [row["bounds"].endswith("*") for row in _rows(finding)] == [False, False]
    assert _basis(finding)["Allowances"] == "beta=0.0, asserted by you"


def test_a_target_not_ruled_out_warns() -> None:
    output = _noisy({"accuracy/partition_restricted": 0.5})
    (determination,) = output.determinations.to_dicts()
    assert determination["determination"] == "not_ruled_out"
    finding = _judge(output)
    assert finding.severity == "warning"
    (row,) = _rows(finding)
    assert (row["answer"], row["target"]) == ("~", "accuracy/partition_restricted ≥ 50%")
    assert _bounds(row)[1] == _pct(determination["ceiling"], "upper")  # no star: it is not ruled out


def test_an_evidence_conflict_warns_and_trusts_neither_bound() -> None:
    """A witness at 0.9568 over a ceiling at 0.9432: the lower bound rounds down and the upper up, and none is drawn."""
    output = _with(
        _output({"accuracy": 0.9}),
        determination="evidence_conflict",
        witness=0.9568,
        ceiling=0.9432,
        ceiling_scope="full_input",
        limiting_factor="conflict",
        recommendation="Check the assumptions.",
    )
    finding = _judge(output)
    assert finding.severity == "warning"
    (row,) = _rows(finding)
    assert (row["answer"], row["target"]) == ("!", "accuracy ≥ 90%")
    assert _bounds(row) == ("95.6%", "94.4%")  # not 95.7%, not 94.3%: never in the favorable direction
    assert row["interval"] == [0, 1, 0]  # all of it open
    assert _basis(finding)["Key"] == "! evidence conflict"


@pytest.mark.parametrize(
    ("value", "direction", "text"),
    [
        (0.9568, "lower", "95.6%"),
        (0.9568, "upper", "95.7%"),
        (0.95601, "lower", "95.6%"),
        (0.94301, "upper", "94.4%"),
        (0.957, "lower", "95.7%"),  # 0.957 * 1000 is 956.9999999999999: float error is not a reason to go down
        (0.943, "upper", "94.3%"),  # and 0.943 * 1000 is 943.0000000000001: nor to go up
        (1.0, "lower", "100.0%"),
        (0.0, "upper", "0.0%"),
    ],
)
def test_a_bound_is_never_rounded_in_the_favorable_direction(value: float, direction: Any, text: str) -> None:
    assert _pct(value, direction) == text


def test_the_bar_is_cut_where_the_rounded_bounds_are() -> None:
    output = _with(_output({"accuracy": 0.9}), witness=0.9568, ceiling=0.9831, ceiling_scope="full_input")
    (row,) = _rows(_judge(output))
    assert _bounds(row) == ("95.6%", "98.4%")
    assert row["interval"] == pytest.approx([0.956, 0.028, 0.016])  # 1 - 0.984 ruled out, the rest open


def test_several_targets_keep_the_evaluators_order_and_each_has_its_own_row() -> None:
    target = {"ap": 0.5, "accuracy": 0.9, "accuracy/fixed_representation": 0.5}
    finding = _judge(_output(target))
    assert [(row["answer"], row["target"]) for row in _rows(finding)] == [
        ("?", "ap (macro) ≥ 50%"),
        ("✓", "accuracy ≥ 90%"),
        ("✓", "accuracy/fixed_representation ≥ 50%"),
    ]
    assert _basis(finding)["Key"] == "✓ demonstrated  ? insufficient evidence"
    assert finding.severity == "warning"  # one short of a demonstration is enough
    assert finding.brief == "2 of 3 targets demonstrated, 1 insufficient evidence"


def test_a_class_and_an_aggregation_are_named_in_the_label() -> None:
    assert _rows(_judge(_output({"ap:c1": 0.5})))[0]["target"] == "ap:c1 ≥ 50%"


def test_a_threshold_is_written_in_percent_as_dataeval_writes_it() -> None:
    assert _rows(_judge(_output({"accuracy": 0.123456789})))[0]["target"] == "accuracy ≥ 12.3457%"


def test_the_population_names_the_declared_classes_that_have_no_labeled_item() -> None:
    output = _output({"accuracy": 0.9}, ToyClasses(count=400, classes=4, declared=6, unlabeled=range(10)))
    assert _basis(_judge(output))["Population"] == (
        "labeled items only, 390 of 400; classes with no labeled item are in no claim: c4, c5"
    )


def test_the_grouping_line_says_so_when_every_item_is_its_own_unit() -> None:
    assert _basis(_judge(_output({"accuracy": 0.9})))["Grouping"] == "each of the 400 items is its own unit"


def test_the_grouping_line_counts_the_units_and_the_ones_duplicates_removed() -> None:
    output = _output({"accuracy|sequence_mean": 0.9}, ToyCopies(bases=40, exact=1, near=1))
    assert _basis(_judge(output))["Grouping"] == (
        "40 independent units among 120 items, the largest of 3; exact and near duplicates account for 80 fewer units"
    )


def test_the_grouping_line_leaves_out_duplicates_when_none_joined_a_unit() -> None:
    data = ToyCopies(bases=40, site=lambda base, copy: f"s{base % 8}")  # noqa: ARG005
    output = _output({"accuracy|sequence_mean": 0.9}, data, groups="site")
    assert _basis(_judge(output))["Grouping"] == "8 independent units among 40 items, the largest of 5"


def test_dataevals_own_output_is_judged_without_the_population_flow_adds() -> None:
    data = ToyClasses(count=400, classes=4)
    embeddings = np.stack([np.ravel(data[i][0]) for i in range(len(data))])
    output = Feasibility(seed=0).evaluate(data, target={"accuracy": 0.9}, embeddings=embeddings)
    finding = _judge(output)
    assert finding.severity == "ok"
    assert "Population" not in _basis(finding)


def test_the_words_are_the_ones_the_printout_uses() -> None:
    """The key words each answer as DataEval's printout does, with its mark but for one: the printout marks "not
    ruled out" ``?`` as it does "insufficient evidence", and a column of marks alone has to tell the rows apart. A
    release that rewords an answer fails here."""
    assert set(_SYMBOLS) == set(_WORDS)
    assert len(set(_SYMBOLS.values())) == len(_SYMBOLS)
    for kind, symbol in _SYMBOLS.items():
        printed = "?" if kind == "not_ruled_out" else symbol
        assert f"{_WORDS[kind].split(' under ')[0]} {printed}" in str(
            _with(_output({"accuracy": 0.9}), determination=kind, limiting_factor=None)
        )
    assert "* ruled out under the recorded assumptions" in str(
        _with(_output({"accuracy": 0.9}), determination="infeasible_under_recorded_assumptions")
    )


def _printed(output: FeasibilityOutput) -> list[tuple[str, str]]:
    """The (lower, upper) cells of each row of DataEval's printout, less the source in brackets."""
    lines = str(output).splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Requirement")) + 1
    rows = []
    for line in lines[start : start + len(output.determinations)]:
        _, _, lower, upper = re.split(r"\s{3,}", line.strip())
        rows.append(tuple(re.match(r"(\d+\.\d%\*?|—)", cell)[0] for cell in (lower, upper)))  # type: ignore[index]
    return rows


@pytest.mark.parametrize(
    "output",
    [
        _output({"accuracy": 0.9, "ap": 0.5}),
        _noisy({"accuracy": 0.95, "ap": 0.5}),
        _noisy({"accuracy": 0.95, "ap": 0.5}, allowances=_BOTH),
        _noisy({"accuracy/partition_restricted": 0.5, "accuracy": 0.3}),
    ],
)
def test_the_tables_numbers_are_the_numbers_dataevals_printout_shows(output: FeasibilityOutput) -> None:
    rows = _rows(_judge(output))
    assert [_bounds(row) for row in rows] == _printed(output)


def test_the_config_takes_only_an_input() -> None:
    assert PerformanceFeasibilityConfig(input="f").input == "f"
    with pytest.raises(ValidationError, match="Extra inputs"):
        PerformanceFeasibilityConfig(input="f", targets=[])  # type: ignore[call-arg]


def _health(data: ToyClasses, target: dict[str, float], **settings: Any) -> Any:
    steps = [
        {"name": "feas", "evaluator": "feas", "input": "data"},
        {"name": "gate", "check": "performance-feasibility", "input": "feas"},
    ]
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        evaluators=[FeasibilityConfig(name="feas", target=target, seed=0, **settings)],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"], "extractor": "flat"}],
        datasets={"src": data},
        extractor=True,
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def test_a_workflow_runs_the_evaluator_and_the_check_and_the_health_follows_the_severity() -> None:
    easy = _health(ToyClasses(count=400, classes=4), {"accuracy": 0.9})
    (finding,) = easy.findings
    assert (finding.title, finding.severity, finding.step) == ("Performance Feasibility", "ok", "gate")
    hard = _health(ToyClasses(count=1000, classes=4, flip=0.6), {"accuracy": 0.95})
    assert [finding.severity for finding in hard.findings] == ["warning"]
    assert "PERFORMANCE FEASIBILITY" in hard.report()
    assert "0 of 1 targets demonstrated, 1 insufficient evidence" in hard.report()


def test_the_whole_text_report_says_each_thing_once_and_in_few_lines() -> None:
    result = _health(ToyClasses(count=1000, classes=4, flip=0.6), {"accuracy": 0.95, "ap": 0.5}, allowances=_BOTH)
    lines = result.report().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("  PERFORMANCE FEASIBILITY")) - 1
    stop = next(i for i, line in enumerate(lines) if line == "  STEPS") - 1
    assert stop - start <= 15  # the finding and the evaluator's section, for two targets and allowances
    report = "\n".join(lines)
    assert "Proof" not in report
    assert "From Feasibility" not in report  # an empty evidence section adds no heading
    assert "Feasibility of 2 requirements" not in report  # DataEval's printout is not repeated
    assert "Lower confidence bound" not in report
    assert "Recommendation" not in report
    assert "Assumes" not in report  # what the answers assume is documented, and kept in the output
    assert "Assumptions" not in report
    assert "Next" not in report
    assert "█" in report
    assert "░" in report
    assert "▒" in report


def test_a_target_is_one_line_with_its_mark_bar_and_bound_in_the_text_report() -> None:
    result = _health(
        ToyClasses(count=1000, classes=4, flip=0.6), {"accuracy": 0.3, "accuracy/fixed_representation": 0.9}
    )
    (row,) = (
        line for line in result.report().splitlines() if "accuracy/fixed_representation ≥ 90%" in line and "░" in line
    )
    assert re.match(r"  \? +accuracy/fixed_representation ≥ 90%  +█*░+ +≥\d+\.\d%$", row)  # the table fits 80 columns
    assert any(line.startswith("  ☐  Target") for line in result.report().splitlines())  # the marks' column heading


def test_the_html_report_draws_the_table_with_its_legend() -> None:
    page = _health(ToyClasses(count=1000, classes=4, flip=0.6), {"accuracy": 0.95}, allowances=_BOTH).to_html()
    for name in ("shown", "open", "ruled out"):
        assert re.search(rf'class="swatch s\d"></span>{name}', page), name
    assert "accuracy ≥ 95%" in page
    assert re.search(r"<td[^>]*>✗</td>", page)
    assert "✗ ruled out" in page  # the key
