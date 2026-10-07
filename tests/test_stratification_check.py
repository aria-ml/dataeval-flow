"""The `class-stratification` check: the largest gap between a class's share of a part's labels and of the whole's, in
percentage points (data-splitting spec §6.1)."""

from types import SimpleNamespace
from typing import Any

import pytest

from dataeval_flow._blocks import Fields, Table
from dataeval_flow._cache import DatasetCache
from dataeval_flow.evaluators.quality import LabelHealthOutput
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps.checks import ClassStratificationCheck, ClassStratificationConfig
from tests.chain_toys import chain_pipeline, run_chain_task
from tests.evaluator_toys import ToyFactors


def _node(address: str, counts: dict[str, int]) -> Any:
    """A `label-health` Output's node, computed on the Dataset at `address`."""
    output = LabelHealthOutput({"label_counts_per_class": counts}, None)
    return SimpleNamespace(value=output, address=f"labels@{address}", computed_on=(SimpleNamespace(address=address),))


def _judge(parts: dict[str, dict[str, int]], *, shown: Any = None, **thresholds: Any) -> list[Any]:
    config = ClassStratificationConfig(input="labels", parts=list(parts), **thresholds)
    inputs = {
        "input": _node("data", {"a": 50, "b": 50}),
        "parts": [_node(address, counts) for address, counts in parts.items()],
        "shown": shown,
    }
    return list(ClassStratificationCheck().run(config, inputs, None))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("a", "severity", "deviation"),
    [(52, "ok", 2.0), (53, "info", 3.0), (60, "info", 10.0), (61, "warning", 11.0)],
)
def test_each_band_judges_the_largest_deviation(a: int, severity: str, deviation: float) -> None:
    (finding,) = _judge({"split.train": {"a": a, "b": 100 - a}})
    assert (finding.severity, finding.brief) == (severity, f"max deviation {deviation}pp (class 'a' in split.train)")


def test_the_deviation_is_judged_rounded_to_one_place() -> None:
    assert _judge({"split.train": {"a": 5204, "b": 4796}})[0].severity == "ok"
    assert _judge({"split.train": {"a": 5206, "b": 4794}})[0].severity == "info"


@pytest.mark.parametrize(
    ("thresholds", "a", "severity"),
    [
        ({"info": None}, 55, "ok"),
        ({"info": None}, 61, "warning"),
        ({"warning": None}, 61, "info"),
        ({"info": None, "warning": None}, 50, "info"),
    ],
)
def test_null_bands(thresholds: dict[str, Any], a: int, severity: str) -> None:
    assert _judge({"split.train": {"a": a, "b": 100 - a}}, **thresholds)[0].severity == severity


def test_an_even_split_strays_by_nothing() -> None:
    (finding,) = _judge({"split.train": {"a": 5, "b": 5}})
    assert (finding.severity, finding.brief) == ("ok", "max deviation 0.0pp")


def test_a_part_with_no_labels_is_left_out_and_named() -> None:
    (finding,) = _judge({"split.train": {"a": 50, "b": 50}, "split.val": {}})
    fields = finding.blocks[1]
    assert isinstance(fields, Fields)
    assert ("Left out, no labels", "split.val") in fields.items


def test_a_whole_with_no_labels_makes_no_finding() -> None:
    config = ClassStratificationConfig(input="labels", parts="p")
    inputs = {"input": _node("data", {}), "parts": [_node("split.train", {})], "shown": None}
    assert list(ClassStratificationCheck().run(config, inputs, None)) == []  # type: ignore[arg-type]


def test_the_table_heads_each_column_by_its_dataset_and_shows_counts() -> None:
    shown = _node("rebalance", {"a": 30, "b": 30})
    (finding,) = _judge({"split.train": {"a": 40, "b": 40}, "split.val": {"a": 5, "b": 5}}, shown=shown)
    table = finding.blocks[0]
    assert isinstance(table, Table)
    assert [column.header for column in table.columns] == ["Class", "split.train", "split.val", "rebalance", "data"]
    # Every judged part holds `a` at 50 %, so their cells are bare; the shown column and the whole carry the percent.
    assert table.rows[0] == {"class": "a", "c0": 40, "c1": 5, "c2": "30 (50%)", "c3": "50 (50%)"}


def test_parts_at_different_shares_carry_the_percent() -> None:
    (finding,) = _judge({"split.train": {"a": 60, "b": 40}, "split.val": {"a": 5, "b": 5}})
    assert finding.blocks[0].rows[0]["c0"] == "60 (60%)"


def test_over_twenty_classes_the_table_keeps_the_top_ten_and_bottom_five() -> None:
    counts = {f"k{index:02d}": 100 - index for index in range(25)}
    config = ClassStratificationConfig(input="labels", parts="p")
    inputs = {"input": _node("data", counts), "parts": [_node("split.train", counts)], "shown": None}
    (finding,) = ClassStratificationCheck().run(config, inputs, None)  # type: ignore[arg-type]
    table = finding.blocks[0]
    assert isinstance(table, Table)
    rows = table.rows
    assert len(rows) == 16
    assert rows[10]["class"] == "... 10 more ..."


def test_over_kfold_it_judges_each_fold() -> None:
    DatasetCache.clear_instances()
    steps = [
        {"name": "labels", "evaluator": "labels", "input": "data"},
        {"name": "split", "transform": "kfold", "input": "data", "folds": 2, "test_frac": 0.2, "stratify": True},
        {"name": "labels-train", "evaluator": "labels", "input": "split.train"},
        {"name": "labels-val", "evaluator": "labels", "input": "split.val"},
        {"name": "labels-test", "evaluator": "labels", "input": "split.test"},
        {
            "name": "class-stratification",
            "check": "class-stratification",
            "input": "labels",
            "parts": ["labels-train", "labels-val", "labels-test"],
        },
    ]
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        evaluators=[{"name": "labels", "type": "label-health"}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": ToyFactors(count=60)},
    )
    result = run_chain_task(config)
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    judged = [finding for finding in result.findings if finding.title == "Class Stratification"]
    assert [finding.step for finding in judged] == ["class-stratification[0]", "class-stratification[1]"]
    table = judged[0].blocks[0]
    assert isinstance(table, Table)
    headers = [column.header for column in table.columns]
    assert headers == ["Class", "split.train[0]", "split.val[0]", "split.test", "data"]
