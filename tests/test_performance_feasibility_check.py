"""The `performance-feasibility` check: a row per target with its determination in DataEval's name and the bounds behind
it, what the answers rest on, and the bounds that decide them.

The outputs are the evaluator's own, run on `ToyClasses`: four classes a flatten extractor separates, which demonstrate
an easy target, and the same with 60% of the labels redrawn, which no model fits better than 0.55 and which DataEval
therefore cannot show to reach 0.95. Allowances of `{beta: 0, gamma: 0}` let it rule 0.95 out at the default scope. The
one output that is constructed, not run, is a conflict: DataEval's own bounds cannot be made to disagree with Flow's
inputs.
"""

import dataclasses
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from dataeval.performance import Feasibility, FeasibilityOutput, TargetSpec
from pydantic import ValidationError

from dataeval_flow import run, run_tasks
from dataeval_flow._blocks import Fields, Table
from dataeval_flow.evaluators.performance import FeasibilityConfig, TargetConfig
from dataeval_flow.steps import ChainResult, CheckContext
from dataeval_flow.steps.checks import PerformanceFeasibilityCheck, PerformanceFeasibilityConfig
from dataeval_flow.steps.checks._feasibility import _SYMBOLS, _pct
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import FLAT, ToyClasses, ToyCopies

_BOTH = {"beta": 0, "gamma": 0}
_CHECKS_PAGE = Path(__file__).parents[1] / "docs" / "source" / "reference" / "checks.md"


def _accuracy(threshold: float, **fields: Any) -> TargetConfig:
    return TargetConfig(metric="accuracy", threshold=threshold, **fields)


def _ap(threshold: float, **fields: Any) -> TargetConfig:
    return TargetConfig(metric="ap", threshold=threshold, **fields)


def _output(
    target: list[TargetConfig], data: ToyClasses | ToyCopies | None = None, **settings: Any
) -> FeasibilityOutput:
    config = FeasibilityConfig(target=target, seed=0, **settings)
    result = run(config, data or ToyClasses(count=400, classes=4), extractor=FLAT)
    assert result.success, result.errors
    return result.output


def _noisy(target: list[TargetConfig], **settings: Any) -> FeasibilityOutput:
    return _output(target, ToyClasses(count=1000, classes=4, flip=0.6), **settings)


def _judge(output: FeasibilityOutput) -> Any:
    node = SimpleNamespace(value=output)
    (finding,) = PerformanceFeasibilityCheck().run(
        PerformanceFeasibilityConfig(input="f"), {"input": node}, CheckContext("t", "s")
    )
    return finding


def _table(finding: Any) -> Table:
    table, _, _ = finding.blocks
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
    _, fields, _ = finding.blocks
    assert isinstance(fields, Fields)
    return dict(fields.items)


def _deciding(finding: Any) -> list[dict[str, Any]]:
    _, _, table = finding.blocks
    assert isinstance(table, Table)
    return table.rows


def _with(output: FeasibilityOutput, **columns: Any) -> FeasibilityOutput:
    """`output` with the first determination's `columns` set, as a run that decided otherwise would hold them.

    `dataclasses.replace` builds a new output, which holds no record of the call, so the record is put back.
    """
    schema = output.determinations.schema
    frame = output.determinations.with_columns(
        [pl.lit(value, dtype=schema[name]).alias(name) for name, value in columns.items()]
    )
    made = dataclasses.replace(output, determinations=frame)
    object.__setattr__(made, "_meta", output.meta())
    return made


def test_a_target_demonstrated_is_ok_and_draws_its_bound() -> None:
    output = _output([_accuracy(0.9)])
    finding = _judge(output)
    assert (finding.severity, finding.title) == ("ok", "Performance Feasibility")
    assert finding.brief == "1 of 1 targets demonstrated_feasible"
    assert finding.description is None  # the explanation belongs in the reference, once
    (row,) = _rows(finding)
    determination = output.determinations.row(0, named=True)
    assert (row["answer"], row["requirement"], row["target"]) == ("✓", 0, "accuracy ≥0.9")
    assert _bounds(row) == (_pct(determination["witness"], "lower"), "—")
    shown, open_, ruled_out = row["interval"]
    assert (shown, ruled_out) == (int(determination["witness"] * 1000) / 1000, 0)
    assert shown + open_ + ruled_out == pytest.approx(1)
    basis = _basis(finding)
    assert basis["Key"] == "✓ demonstrated_feasible"
    assert basis["Confidence"] == "95%, all bounds together, for new images like these 400"
    # every item is labeled and its own unit, nothing is assumed, nothing is undecided
    assert set(basis) == {"Key", "Confidence", "Grouping"}


def test_the_interval_is_drawn_as_a_stacked_bar_with_its_series_named() -> None:
    column = next(column for column in _table(_judge(_output([_accuracy(0.9)]))).columns if column.key == "interval")
    assert (column.kind, column.header, column.series) == ("stacked", "", ["shown", "open", "ruled out"])


def test_an_unresolved_target_warns_and_names_its_limitation() -> None:
    output = _noisy([_accuracy(0.95)])
    finding = _judge(output)
    assert finding.severity == "warning"
    assert finding.brief == "0 of 1 targets demonstrated_feasible, 1 unresolved"
    (row,) = _rows(finding)
    assert (row["answer"], row["target"]) == ("?", "accuracy ≥0.95")
    assert _bounds(row)[1] == "—"
    assert row["interval"][2] == 0  # nothing ruled out
    basis = _basis(finding)
    assert basis["Limitation"] == "accuracy ≥0.95: scope_restricted"
    assert basis["Key"] == "? unresolved"
    assert "Allowances" not in basis
    assert "Suggested minimum" not in basis  # DataEval gives none: no bound would cross at any size


def test_a_suggested_minimum_is_named_with_the_bound_it_projects_from() -> None:
    """Separable data with a threshold just above what its 400 items show: DataEval projects when it would be shown."""
    output = _output([_accuracy(0.99), _accuracy(0.9)])
    finding = _judge(output)
    minimum = output.determinations.row(0, named=True)
    assert (minimum["determination"], minimum["limitation"]) == ("unresolved", "scope_restricted")
    assert minimum["suggested_minimum"] is not None
    assert _basis(finding)["Suggested minimum"] == (
        f"accuracy ≥0.99: {minimum['suggested_minimum']:,} independent units, from knn accuracy"
    )  # one line, for the one row that has one


def test_a_target_ruled_out_warns_says_under_what_and_shows_the_allowances_used() -> None:
    output = _noisy([_accuracy(0.95)], allowances=_BOTH)
    finding = _judge(output)
    assert finding.severity == "warning"  # fails closed: a ruled-out target holds only under assumptions
    assert finding.brief == "0 of 1 targets demonstrated_feasible, 1 infeasible_under_recorded_assumptions"
    (row,) = _rows(finding)
    determination = output.determinations.row(0, named=True)
    assert (row["answer"], row["target"]) == ("✗", "accuracy ≥0.95")
    assert _bounds(row)[1] == _pct(determination["envelope"], "upper") + "*"  # marked: it holds under the allowances
    assert row["interval"][2] > 0  # part of the scale is ruled out
    basis = _basis(finding)
    assert basis["Allowances"] == "beta=0.0, gamma=0.0, asserted by you; * rests on them"
    assert basis["Key"] == "✗ infeasible_under_recorded_assumptions"
    assert "Limitation" not in basis  # a final determination has none


def test_allowances_are_shown_where_no_row_is_starred() -> None:
    output = _noisy([_accuracy(0.95), _ap(0.5)], allowances={"beta": 0})
    finding = _judge(output)
    assert [row["bounds"].endswith("*") for row in _rows(finding)] == [False, False]
    assert _basis(finding)["Allowances"] == "beta=0.0, asserted by you"


def test_a_target_not_ruled_out_warns() -> None:
    output = _noisy([_accuracy(0.5, scope="partition_restricted")])
    (determination,) = output.determinations.to_dicts()
    assert determination["determination"] == "not_ruled_out"
    finding = _judge(output)
    assert finding.severity == "warning"
    (row,) = _rows(finding)
    assert (row["answer"], row["target"]) == ("~", "accuracy ≥0.5 @partition_restricted")
    assert _bounds(row)[1] == _pct(determination["envelope"], "upper")  # no star: it is not ruled out
    assert _basis(finding)["Limitation"] == "accuracy ≥0.5 @partition_restricted: no_eligible_witness"


def test_a_conflict_is_unresolved_warns_and_trusts_neither_bound() -> None:
    """Constructed: a witness at 0.9568 over an envelope at 0.9432. The lower bound rounds down and the upper up, and
    none is drawn."""
    output = _with(
        _output([_accuracy(0.9)]),
        determination="unresolved",
        limitation="conflict",
        witness=0.9568,
        envelope=0.9432,
        envelope_scope="full_input",
    )
    finding = _judge(output)
    assert finding.severity == "warning"
    assert finding.brief == "0 of 1 targets demonstrated_feasible, 1 unresolved"
    (row,) = _rows(finding)
    assert (row["answer"], row["target"]) == ("!", "accuracy ≥0.9")
    assert _bounds(row) == ("95.6%", "94.4%")  # not 95.7%, not 94.3%: never in the favorable direction
    assert row["interval"] == [0, 1, 0]  # all of it open
    assert _basis(finding)["Key"] == "! unresolved, conflict"
    assert _basis(finding)["Limitation"] == "accuracy ≥0.9: conflict"


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
    output = _with(_output([_accuracy(0.9)]), witness=0.9568, envelope=0.9831, envelope_scope="full_input")
    (row,) = _rows(_judge(output))
    assert _bounds(row) == ("95.6%", "98.4%")
    assert row["interval"] == pytest.approx([0.956, 0.028, 0.016])  # 1 - 0.984 ruled out, the rest open


def test_several_targets_keep_the_order_they_were_given_in_and_each_has_its_own_row() -> None:
    target = [_ap(0.5), _accuracy(0.9), _accuracy(0.5, scope="fixed_representation")]
    finding = _judge(_output(target))
    assert [(row["answer"], row["requirement"], row["target"]) for row in _rows(finding)] == [
        ("?", 0, "ap ≥0.5"),
        ("✓", 1, "accuracy ≥0.9"),
        ("✓", 2, "accuracy ≥0.5 @fixed_representation"),
    ]
    assert _basis(finding)["Key"] == "✓ demonstrated_feasible  ? unresolved"
    assert finding.severity == "warning"  # one short of a demonstration is enough
    assert finding.brief == "2 of 3 targets demonstrated_feasible, 1 unresolved"


def test_reordering_the_targets_reorders_the_table_and_changes_nothing_else() -> None:
    first, second = _accuracy(0.9), _ap(0.5)
    forward = _judge(_output([first, second]))
    backward = _judge(_output([second, first]))
    assert [row["target"] for row in _rows(backward)] == [row["target"] for row in reversed(_rows(forward))]
    # Each row keeps its marks, interval and bounds; only its index follows the order.
    by_target = {row["target"]: (row["answer"], row["interval"], row["bounds"]) for row in _rows(forward)}
    assert by_target == {row["target"]: (row["answer"], row["interval"], row["bounds"]) for row in _rows(backward)}
    assert [row["requirement"] for row in _rows(backward)] == [0, 1]
    assert _basis(forward)["Key"] == _basis(backward)["Key"] or set(_basis(forward)["Key"].split("  ")) == set(
        _basis(backward)["Key"].split("  ")
    )


def test_a_class_an_aggregation_and_a_scope_are_named_in_the_label_as_dataeval_names_them() -> None:
    assert _rows(_judge(_output([_ap(0.5, class_name="c1")])))[0]["target"] == "ap:c1 ≥0.5"
    spec = TargetSpec(metric="ap", threshold=0.5, class_name="c1", scope="fixed_representation")
    output = _output([_ap(0.5, class_name="c1", scope="fixed_representation")])
    assert output.requirements == (spec,)
    assert _rows(_judge(output))[0]["target"] == "ap:c1 ≥0.5 @fixed_representation"
    # DataEval's own printout names the requirement the same way
    assert "ap:c1 ≥0.5 @fixed_representation" in str(output)


def test_a_threshold_is_written_as_dataeval_writes_it() -> None:
    assert _rows(_judge(_output([_accuracy(0.123456789)])))[0]["target"] == "accuracy ≥0.123457"


def test_the_population_names_the_declared_classes_that_have_no_labeled_item() -> None:
    output = _output([_accuracy(0.9)], ToyClasses(count=400, classes=4, declared=6, unlabeled=range(10)))
    assert _basis(_judge(output))["Population"] == (
        "labeled items only, 390 of 400; classes with no labeled item are in no claim: c4, c5"
    )


def test_the_grouping_line_says_so_when_every_item_is_its_own_unit() -> None:
    assert _basis(_judge(_output([_accuracy(0.9)])))["Grouping"] == "each of the 400 items is its own unit"


def test_the_grouping_line_counts_the_units_and_the_ones_duplicates_removed() -> None:
    output = _output([_accuracy(0.9, aggregation="sequence_mean")], ToyCopies(bases=40, exact=1, near=1))
    assert _basis(_judge(output))["Grouping"] == (
        "40 independent units among 120 items, the largest of 3; exact and near duplicates account for 80 fewer units"
    )


def test_the_grouping_line_leaves_out_duplicates_when_none_joined_a_unit() -> None:
    data = ToyCopies(bases=40, site=lambda base, copy: f"s{base % 8}")  # noqa: ARG005
    output = _output([_accuracy(0.9, aggregation="sequence_mean")], data, groups="site")
    assert _basis(_judge(output))["Grouping"] == "8 independent units among 40 items, the largest of 5"


def test_dataevals_own_output_is_judged_without_the_population_flow_adds() -> None:
    data = ToyClasses(count=400, classes=4)
    embeddings = np.stack([np.ravel(data[i][0]) for i in range(len(data))])
    output = Feasibility(seed=0).evaluate(data, target=[TargetSpec("accuracy", 0.9)], embeddings=embeddings)
    finding = _judge(output)
    assert finding.severity == "ok"
    assert "Population" not in _basis(finding)


def test_an_output_with_no_recorded_alpha_is_refused_not_given_a_made_up_confidence() -> None:
    output = _output([_accuracy(0.9)])
    bare = dataclasses.replace(output)  # a new output holds no record of the call
    with pytest.raises(ValueError, match="records no `alpha`"):
        _judge(bare)


def test_the_marks_and_their_names_are_dataevals() -> None:
    """The key names each determination as DataEval's printout does, with a mark of its own for each. A release that
    renames one fails here."""
    assert len(set(_SYMBOLS.values())) == len(_SYMBOLS)
    output = _output([_accuracy(0.9)])
    for kind in _SYMBOLS:
        assert kind in str(_with(output, determination=kind, limitation=None))
    assert set(_SYMBOLS) == set(cast("pl.Enum", output.determinations.schema["determination"]).categories)


def test_every_limitation_dataeval_can_give_has_its_sentence_in_the_reference() -> None:
    """The report names a limitation; `reference/checks.md` says what each means. A new value fails here."""
    categories = cast("pl.Enum", _output([_accuracy(0.9)]).determinations.schema["limitation"]).categories.to_list()
    page = _CHECKS_PAGE.read_text()
    described = set(re.findall(r"^\| `(\w+)` \|", page, flags=re.MULTILINE))
    assert set(categories) <= described, sorted(set(categories) - described)


def _printed(output: FeasibilityOutput) -> list[tuple[str, str]]:
    """The (lower, upper) cells of each row of DataEval's printout: the witness's lower bound and the envelope's upper
    one, less the source that follows each."""
    rows = []
    for line in str(output).splitlines()[1 : 1 + len(output.determinations)]:
        cells = re.split(r"\s{3,}", line.strip())
        envelope, witness = cells[-3], cells[-2]
        rows.append(tuple(re.match(r"(\d+\.\d%|—)", cell)[0] for cell in (witness, envelope)))  # type: ignore[index]
    return rows


@pytest.mark.parametrize(
    "output",
    [
        _output([_accuracy(0.9), _ap(0.5)]),
        _noisy([_accuracy(0.95), _ap(0.5)]),
        _noisy([_accuracy(0.95), _ap(0.5)], allowances=_BOTH),
        _noisy([_accuracy(0.5, scope="partition_restricted"), _accuracy(0.3)]),
    ],
)
def test_the_tables_numbers_are_the_numbers_dataevals_printout_shows(output: FeasibilityOutput) -> None:
    rows = _rows(_judge(output))
    table = [tuple(part.rstrip("*") for part in _bounds(row)) for row in rows]
    assert table == _printed(output)


def test_the_deciding_table_lists_what_each_determination_rests_on() -> None:
    output = _noisy([_accuracy(0.95), _ap(0.5)], allowances=_BOTH)
    rows = _deciding(_judge(output))
    decided = [row for row in rows if row["decides"] == "yes"]
    bounds = output.bounds.filter(pl.col("is_determining")).to_dicts()
    assert [(row["requirement"], row["role"], row["source"], row["scope"]) for row in decided] == [
        (bound["requirement"], bound["role"], bound["source"], bound["information_scope"]) for bound in bounds
    ]
    assert {row["decides"] for row in rows} == {"yes", "no"}
    # A restricted-scope envelope cannot rule out a target judged at full input, and is listed as not deciding.
    undecided = [row for row in rows if row["decides"] == "no"]
    assert undecided
    assert all(row["role"] == "envelope" and row["scope"] != "full_input" for row in undecided)


def test_the_deciding_table_gives_each_bound_as_dataeval_rounds_it() -> None:
    output = _output([_accuracy(0.9)])
    (row,) = _deciding(_judge(output))
    assert row["bound"] == "≥" + _pct(output.bounds.filter(pl.col("is_determining"))["bound"][0], "lower")
    assert (row["role"], row["source"], row["scope"], row["decides"]) == (
        "witness",
        "knn",
        "fixed_representation",
        "yes",
    )


def test_the_config_takes_only_an_input() -> None:
    assert PerformanceFeasibilityConfig(input="f").input == "f"
    with pytest.raises(ValidationError, match="Extra inputs"):
        PerformanceFeasibilityConfig(input="f", targets=[])  # type: ignore[call-arg]


def _health(data: ToyClasses, target: list[TargetConfig], **settings: Any) -> Any:
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
    easy = _health(ToyClasses(count=400, classes=4), [_accuracy(0.9)])
    (finding,) = easy.findings
    assert (finding.title, finding.severity, finding.step) == ("Performance Feasibility", "ok", "gate")
    hard = _health(ToyClasses(count=1000, classes=4, flip=0.6), [_accuracy(0.95)])
    assert [finding.severity for finding in hard.findings] == ["warning"]
    assert "PERFORMANCE FEASIBILITY" in hard.report()
    assert "0 of 1 targets demonstrated_feasible, 1 unresolved" in hard.report()


def test_the_text_report_says_each_thing_once_and_in_few_lines() -> None:
    result = _health(ToyClasses(count=1000, classes=4, flip=0.6), [_accuracy(0.95), _ap(0.5)], allowances=_BOTH)
    lines = result.report().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("  PERFORMANCE FEASIBILITY")) - 1
    stop = next(i for i, line in enumerate(lines) if line == "  STEPS") - 1
    assert stop - start <= 25  # the finding's two tables and its basis, for two targets and allowances
    report = "\n".join(lines)
    assert "Proof" not in report
    assert "From Feasibility" not in report  # an empty evidence section adds no heading
    assert "joint confidence" not in report  # DataEval's printout is not repeated
    assert "Lower confidence bound" not in report
    assert "Recommendation" not in report
    assert "Assumes" not in report  # what the bounds assume is documented, and kept in the output
    assert "Assumptions" not in report
    assert "█" in report
    assert "░" in report
    assert "▒" in report


def test_a_target_is_one_line_with_its_mark_bar_and_bound_in_the_text_report() -> None:
    result = _health(ToyClasses(count=1000, classes=4, flip=0.6), [_accuracy(0.3), _accuracy(0.9)])
    (row,) = (line for line in result.report().splitlines() if "accuracy ≥0.9" in line and "░" in line)
    assert re.match(r"  \? +1 +accuracy ≥0.9 +█*░+ +≥\d+\.\d%$", row)  # the table fits 80 columns
    assert any(line.startswith("  ☐  #  Target") for line in result.report().splitlines())  # the marks' column heading


def test_a_target_with_a_scope_names_it_on_a_line_of_its_own_when_the_row_is_too_wide_for_80_columns() -> None:
    result = _health(ToyClasses(count=1000, classes=4, flip=0.6), [_accuracy(0.9, scope="fixed_representation")])
    lines = result.report().splitlines()
    row = next(i for i, line in enumerate(lines) if line.lstrip().startswith("? ") and "accuracy ≥0.9" in line)
    assert "@fixed_representation" in " ".join(lines[row : row + 2])


def test_the_html_report_draws_the_tables_with_the_legend() -> None:
    page = _health(ToyClasses(count=1000, classes=4, flip=0.6), [_accuracy(0.95)], allowances=_BOTH).to_html()
    for name in ("shown", "open", "ruled out"):
        assert re.search(rf'class="swatch s\d"></span>{name}', page), name
    assert "accuracy ≥0.95" in page
    assert re.search(r"<td[^>]*>✗</td>", page)
    assert "✗ infeasible_under_recorded_assumptions" in page  # the key
    assert "Decides" in page
