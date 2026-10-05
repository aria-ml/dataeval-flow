"""Outlier evidence as report blocks: a row per flagged item with its flags, and a row per metric with its limits."""

import math
from collections.abc import Mapping
from typing import Any

import pytest

from dataeval_flow._blocks import Block, Cell, Column, Paragraph, Table
from dataeval_flow._blocks._html import render_html
from dataeval_flow._blocks._text import render_text
from dataeval_flow.evaluators.quality._report import (
    OutlierIssueRecord,
    flag_of,
    flagged_table,
    limits_table,
)

pytestmark = pytest.mark.required


def _issue(
    item: int,
    metric: str,
    *,
    value: float = 1.0,
    direction: str = "upper",
    bound: float = 0.5,
    percentile: float = 99.95,
    mean: float = 0.2,
    std: float = 0.1,
    target: int | None = None,
) -> OutlierIssueRecord:
    issue: OutlierIssueRecord = {
        "item_index": item,
        "metric_name": metric,
        "metric_value": value,
        "direction": direction,  # type: ignore[typeddict-item]
        "bound": bound,
        "percentile": percentile,
        "population_mean": mean,
        "population_std": std,
    }
    if target is not None:
        issue["target_index"] = target
    return issue


def _item(issue: Mapping[str, Any]) -> tuple[Cell, ...]:
    return (issue["item_index"],)


def _only_table(blocks: list[Block]) -> Table:
    (table,) = blocks
    assert isinstance(table, Table)
    return table


_ITEM = [Column(key="item", header="Item")]


class TestFlags:
    def test_a_flag_carries_the_issue_s_measurement_and_its_population(self):
        flag = flag_of(_issue(3, "brightness", value=0.99, bound=0.84, percentile=99.9, mean=0.52, std=0.11))
        assert (flag.name, flag.value, flag.direction, flag.bound) == ("brightness", 0.99, "upper", 0.84)
        assert (flag.percentile, flag.mean, flag.std) == (99.9, 0.52, 0.11)

    def test_an_issue_recorded_without_its_context_reads_as_unknown(self):
        """A DataEval that predates the context columns: the value still shows, its position does not."""
        flag = flag_of({"item_index": 0, "metric_name": "m", "metric_value": 0.1})
        assert flag.value == 0.1
        assert all(math.isnan(v) for v in (flag.bound, flag.percentile, flag.mean, flag.std))

    def test_an_issue_recorded_without_its_context_shows_its_value_and_names_no_limit(self):
        """No limit it never had, no crossing DataEval never recorded, and no `nan`, in HTML or text."""
        blocks = flagged_table([_OLD], key=_item, key_columns=_ITEM, classes=None, noun="images")
        page = render_html(blocks)
        text = "\n".join(render_text(blocks))
        assert (
            '<span class="tag" tabindex="0">brightness 0.9<span class="tip">'
            '<span class="tip-title">brightness · outside its limits</span>'
            '<span class="tip-row"><span>Percentile</span><span>p?</span></span></span></span>'
        ) in page
        assert text.splitlines()[-1] == "3         1  brightness 0.9"
        for drawn in (page, text):
            assert "nan" not in drawn
            assert "upper" not in drawn


# An issue from a DataEval that predates the context columns: its item, metric and value alone.
_OLD = {"item_index": 3, "metric_name": "brightness", "metric_value": 0.9}


class TestFlaggedTable:
    def test_one_row_per_item_with_every_flag_it_raised(self):
        issues = [_issue(0, "brightness"), _issue(0, "contrast", percentile=99.6), _issue(5, "entropy", percentile=0.3)]
        table = _only_table(flagged_table(issues, key=_item, key_columns=_ITEM, classes=None, noun="images"))
        assert isinstance(table, Table)
        assert [c.header for c in table.columns] == ["Item", "Flags", "Flagged by"]
        assert [c.kind for c in table.columns] == ["text", "text", "flags"]
        assert [(row["item"], row["flags"]) for row in table.rows] == [(0, 2), (5, 1)]
        by = table.rows[0]["by"]
        assert isinstance(by, list)
        assert [flag.name for flag in by] == ["brightness", "contrast"]  # type: ignore[union-attr]
        assert table.preview == 10

    def test_rows_run_in_the_order_of_their_keys(self):
        """No row ranks above another by how far past its limit a value lies."""
        issues = [_issue(1, "a"), _issue(2, "a", percentile=99.95), _issue(3, "a"), _issue(3, "b"), _issue(0, "a")]
        table = _only_table(flagged_table(issues, key=_item, key_columns=_ITEM, classes=None, noun="images"))
        assert [row["item"] for row in table.rows] == [0, 1, 2, 3]

    def test_a_class_column_appears_where_classes_are_known(self):
        table = _only_table(
            flagged_table([_issue(7, "a")], key=_item, key_columns=_ITEM, classes={(7,): "cat"}, noun="images")
        )
        assert [c.header for c in table.columns] == ["Item", "Class", "Flags", "Flagged by"]
        assert table.rows[0]["class"] == "cat"

    def test_a_long_table_keeps_the_first_500_and_says_how_many_it_left_out(self):
        issues = [_issue(i, "a") for i in reversed(range(503))]
        table, note = flagged_table(issues, key=_item, key_columns=_ITEM, classes=None, noun="images")
        assert isinstance(table, Table)
        assert [row["item"] for row in table.rows] == list(range(500))
        assert note == Paragraph(text="503 images flagged; the first 500 are listed, and every one is in `output.raw`.")

    def test_groups_run_in_the_order_given_and_a_small_group_s_spare_share_goes_to_the_rest(self):
        """10, 600 and 600 flagged: the 10 all list, and the other two split the remaining 490."""
        sizes = {"train": 10, "val": 600, "test": 600}
        issues = [{**_issue(i, "a"), "split": split} for split, size in sizes.items() for i in range(size)]
        table, note = flagged_table(
            issues,
            key=lambda issue: (issue["split"], issue["item_index"]),
            key_columns=[Column(key="split"), *_ITEM],
            classes=None,
            noun="images",
            groups=list(sizes),
        )
        assert isinstance(table, Table)
        splits = [row["split"] for row in table.rows]
        assert [(split, splits.count(split)) for split in dict.fromkeys(splits)] == [
            ("train", 10),
            ("val", 245),
            ("test", 245),
        ]
        assert note == Paragraph(
            text="1,210 images flagged; 500 are listed, leaving out 355 of 600 in val, 355 of 600 in test, "
            "and every one is in `output.raw`."
        )

    def test_nothing_flagged_is_no_table(self):
        assert flagged_table([], key=_item, key_columns=_ITEM, classes=None, noun="images") == []


class TestLimitsTable:
    def test_one_row_per_metric_with_the_limits_crossed_and_the_population(self):
        issues = [
            _issue(0, "brightness", direction="upper", bound=0.84, mean=0.52, std=0.11),
            _issue(1, "brightness", direction="lower", bound=0.2, mean=0.52, std=0.11),
            _issue(1, "brightness", direction="lower", bound=0.2, mean=0.52, std=0.11),
            _issue(2, "entropy", direction="lower", bound=3.1, mean=5.0, std=1.0),
        ]
        table = limits_table(issues, key=_item)
        assert [c.header for c in table.columns] == ["Metric", "Count", "Lower", "Upper", "Mean", "Std"]
        assert table.rows == [
            {"metric": "brightness", "count": 2, "lower": 0.2, "upper": 0.84, "mean": 0.52, "std": 0.11},
            {"metric": "entropy", "count": 1, "lower": 3.1, "upper": None, "mean": 5.0, "std": 1.0},
        ]

    def test_limits_that_differ_between_flags_say_they_vary(self):
        """As `cluster_distance`'s do, one cluster to the next; each flag's card holds its own."""
        issues = [_issue(0, "a", bound=0.8, mean=0.5), _issue(1, "a", bound=0.9, mean=0.6)]
        (row,) = limits_table(issues, key=_item).rows
        assert (row["upper"], row["mean"]) == ("varies", "varies")

    def test_metrics_run_from_the_most_flagged(self):
        issues = [_issue(0, "zeta"), _issue(1, "zeta"), _issue(2, "alpha")]
        assert [row["metric"] for row in limits_table(issues, key=_item).rows] == ["zeta", "alpha"]
