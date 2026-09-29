"""Evaluators report in their own sections: flagged items, duplicate groups, uncovered items, ranked factors (§9.6)."""

from collections.abc import Sequence
from typing import Any

from dataeval_flow import run_task
from dataeval_flow._blocks import Block, ItemRef, Section, Table
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators._result import EvaluatorMetadata
from dataeval_flow.evaluators.bias import BalanceConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesResult, OutliersConfig
from dataeval_flow.evaluators.scope import CoverageConfig
from tests.chain_toys import ToyDetections, chain_pipeline, run_chain_task
from tests.evaluator_toys import ToyFactors, ToyImages, toy_pipeline

_OUTLIERS = {"name": "e", "flags": ["pixel", "visual"], "outlier_threshold": "zscore"}


def _boxes() -> ToyDetections:
    """20 detection images with 28 boxes, two of them (item 3 box 0, item 8 box 1) drawn white."""
    return ToyDetections([[0, 1], [1], [0], [1, 1], [0]] * 4, {0: "car", 1: "van"}, bright={(3, 0), (8, 1)})


def _task(evaluator: Any, *, sources: Sequence[str] = ("src",), dataset: Any = None, extractor: bool = False, **result):
    task = TaskConfig(
        name="t", workflow="e", kind="evaluator", sources=list(sources), extractor="flat" if extractor else None
    )
    config = toy_pipeline(evaluators=[evaluator], tasks=[task], sources=sources, dataset=dataset, extractor=extractor)
    if result:
        config = config.model_copy(update={"result": config.result.model_copy(update=result)})
    outcome = run_task(task, config)
    assert outcome.success, outcome.errors
    return outcome


def _tables(blocks: Sequence[Block]) -> list[Table]:
    """Every table in `blocks`, at any depth, in order."""
    found: list[Table] = []
    for block in blocks:
        if isinstance(block, Table):
            found.append(block)
        elif isinstance(block, Section):
            found.extend(_tables(block.blocks))
    return found


def _section(blocks: Sequence[Block], title: str) -> Section:
    return next(block for block in blocks if isinstance(block, Section) and block.title == title)


def _refs(table: Table, column: str = "image") -> list[ItemRef]:
    refs: list[Any] = []
    for row in table.rows:
        cell = row.get(column)
        refs.extend(cell if isinstance(cell, list) else [cell])
    return [ref for ref in refs if isinstance(ref, ItemRef)]


def test_an_outliers_section_lists_each_flagged_image_with_its_flags_and_each_metric_s_limits() -> None:
    result = _task(OutliersConfig(**_OUTLIERS), dataset=ToyImages(count=24))
    flagged, limits = _tables(result._report_output(detailed=True))
    assert [column.key for column in flagged.columns] == ["image", "item", "flags", "by"]
    assert [row["item"] for row in flagged.rows] == [7]  # the white image
    assert [(ref.source, ref.index) for ref in _refs(flagged)] == [("src", 7)]
    assert [column.key for column in limits.columns] == ["metric", "count", "lower", "upper", "mean", "std"]


def test_an_outliers_section_lists_flagged_boxes_under_their_own_heading() -> None:
    result = _task(OutliersConfig(**_OUTLIERS, per_target=True), dataset=_boxes())
    flagged, _ = _tables(_section(result._report_output(detailed=True), "Flagged boxes").blocks)
    assert {(row["item"], row["box"]) for row in flagged.rows} == {(3, 0), (8, 1)}
    assert {(ref.index, ref.target) for ref in _refs(flagged)} == {(3, 0), (8, 1)}


def test_an_outliers_section_lists_no_more_rows_than_the_run_allowed() -> None:
    result = _task(OutliersConfig(**_OUTLIERS, per_target=True), dataset=_boxes(), max_rows=1)
    flagged, _ = _tables(_section(result._report_output(detailed=True), "Flagged boxes").blocks)
    assert len(flagged.rows) == 1


def test_a_duplicates_section_pictures_each_group_s_items() -> None:
    result = _task(DuplicatesConfig(name="e"))
    (table,) = _tables(result._report_output(detailed=True))
    assert [(row["kind"], row["count"], row["items"]) for row in table.rows] == [("exact", 2, "0, 5")]
    assert [(ref.source, ref.index) for ref in _refs(table)] == [("src", 0), ("src", 5)]


def test_a_duplicates_section_over_two_sources_names_each_item_s_own_source() -> None:
    result = _task(DuplicatesConfig(name="e"), sources=("a", "b"))
    (table,) = _tables(result._report_output(detailed=True))
    assert table.rows[0]["count"] == 4  # items 0 and 5, in each source
    assert sorted((ref.source, ref.index) for ref in table.rows[0]["image"]) == [
        ("a", 0),
        ("a", 5),
        ("b", 0),
        ("b", 5),
    ]
    assert all({ref.source for ref in row["image"]} == {"a", "b"} for row in table.rows)


def test_a_coverage_section_pictures_the_uncovered_items() -> None:
    config = CoverageConfig(name="e", method="adaptive", percent=0.1, num_observations=5, min_class_samples=5)
    result = _task(config, dataset=ToyImages(count=40), extractor=True)
    uncovered = [int(index) for index in result.output.uncovered_indices]
    assert uncovered
    (table,) = _tables(_section(result._report_output(detailed=True), "Uncovered images").blocks)
    assert sorted(ref.index for ref in _refs(table)) == sorted(uncovered)


def test_a_balance_section_ranks_each_factor_by_its_mutual_information_with_the_class() -> None:
    result = _task(BalanceConfig(name="e"), dataset=ToyFactors())
    (table, *_) = _tables(_section(result._report_output(detailed=True), "Balance").blocks)
    assert [row["name"] for row in table.rows] == ["site", "angle"]  # `site` follows the class; `angle` does not


def test_an_evaluator_step_s_section_names_its_items_by_the_node_it_read() -> None:
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["a"], "steps": [{"name": "dupes", "evaluator": "dupes", "input": "a"}]}],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
    )
    result = run_chain_task(config)
    refs = [ref for table in _tables(result._report_output(detailed=True)) for ref in _refs(table)]
    assert [(ref.source, ref.index) for ref in refs] == [("a", 0), ("a", 5)]


def test_a_result_that_does_not_know_its_sources_reports_dataeval_s_output_as_it_came() -> None:
    run = _task(DuplicatesConfig(name="e"))
    bare = DuplicatesResult(
        type="quality.duplicates",
        success=True,
        metadata=EvaluatorMetadata(),
        output=run.output,
        serialized=run.to_dict()["output"],  # type: ignore[arg-type]
    )
    (section,) = bare._report_output(detailed=True)
    assert isinstance(section, Section)
    assert section.title == "Output"
