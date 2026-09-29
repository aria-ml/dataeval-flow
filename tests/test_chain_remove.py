"""remove: plans from Duplicates and Outliers applied to the Dataset they were computed on (spec §6.1)."""

import re
from types import SimpleNamespace
from typing import Any

import pytest
from dataeval.quality import DuplicatesOutput
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.evaluators.quality import DuplicatesConfig, OutliersConfig
from dataeval_flow.steps import ChainResult, TransformContext
from dataeval_flow.steps.transforms import RemoveConfig, RemoveTransform
from tests.chain_toys import ToyDetections, chain_pipeline
from tests.evaluator_toys import ToyImages

_POOL = [
    DuplicatesConfig(name="dupes"),
    OutliersConfig(name="box_outliers", flags=["visual"], per_image=False, per_target=True, outlier_threshold="iqr"),
]


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _run(steps: list[dict[str, Any]], dataset: Any = None) -> ChainResult:
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["a"], "steps": steps}],
        evaluators=_POOL,
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": dataset if dataset is not None else ToyImages()},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


def _labels(dataset: Any) -> list[list[int]]:
    return [dataset[index][1].labels.tolist() for index in range(len(dataset))]


def test_removing_duplicates_drops_every_copy_but_the_first() -> None:
    result = _run(
        [
            {"name": "dupes", "evaluator": "dupes", "input": "a"},
            {"name": "clean", "transform": "remove", "input": "a", "plans": {"dupes": {"keep": "first"}}},
        ]
    )
    clean = result.steps["clean"]
    assert clean.status == "ok"
    assert len(clean.output) == 11  # ToyImages' item 5 copies item 0
    assert 5 not in list(clean.output.resolve_indices())
    assert clean.details == {"removed": {"items": 1, "detections": 0, "tracks": 0, "frames": 0}}


def test_removing_a_detection_keeps_its_item_and_changes_the_key() -> None:
    detections = ToyDetections([[0, 1]] * 12, {0: "car", 1: "person"}, bright={(4, 1)})
    result = _run(
        [
            {"name": "odd", "evaluator": "box_outliers", "input": "a"},
            {"name": "dupes", "evaluator": "dupes", "input": "a"},
            {"name": "clean", "transform": "remove", "input": "a", "plans": {"odd": {}}},
            {"name": "same", "transform": "remove", "input": "a", "plans": {"odd": {"min_flags": 1}}},
            {"name": "nothing", "transform": "remove", "input": "a", "plans": {"dupes": {}}},  # no two images alike
        ],
        dataset=detections,
    )
    clean = result.steps["clean"]
    assert clean.status == "ok"
    assert clean.details == {"removed": {"items": 0, "detections": 1, "tracks": 0, "frames": 0}}
    assert _labels(clean.output) == [[0, 1]] * 4 + [[0]] + [[0, 1]] * 7  # item 4 loses its white box
    assert list(clean.output.resolve_indices()) == list(range(12))
    assert list(result.steps["nothing"].output.resolve_indices()) == list(range(12))
    digests = {record.name: record.digest for record in result.metadata.lineage}
    assert digests["clean"] != digests["nothing"]  # the same items, one detection fewer
    assert digests["same"] == digests["clean"]  # equal plans key alike, whatever arguments made them


def test_plans_that_remove_nothing_keep_every_item() -> None:
    distinct = ToyDetections([[0]] * 8, {0: "car"})  # every image its own: nothing to deduplicate
    result = _run(
        [
            {"name": "dupes", "evaluator": "dupes", "input": "a"},
            {"name": "clean", "transform": "remove", "input": "a", "plans": {"dupes": {}}},
        ],
        dataset=distinct,
    )
    clean = result.steps["clean"]
    assert (clean.status, len(clean.output)) == ("ok", 8)
    assert clean.details == {"removed": {"items": 0, "detections": 0, "tracks": 0, "frames": 0}}


def test_plans_from_several_outputs_combine() -> None:
    result = _run(
        [
            {"name": "dupes", "evaluator": "dupes", "input": "a"},
            {"name": "odd", "evaluator": "box_outliers", "input": "a"},
            {"name": "clean", "transform": "remove", "input": "a", "plans": {"dupes": {}, "odd": {}}},
        ],
        dataset=ToyDetections([[0, 1]] * 12, {0: "car", 1: "person"}, duplicate_of={7: 2}, bright={(4, 1)}),
    )
    clean = result.steps["clean"]
    assert clean.details == {"removed": {"items": 1, "detections": 1, "tracks": 0, "frames": 0}}
    assert list(clean.output.resolve_indices()) == [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]  # item 7 copies item 2
    assert _labels(clean.output) == [[0, 1]] * 4 + [[0]] + [[0, 1]] * 6


def test_a_plan_argument_its_method_does_not_take_fails_the_load() -> None:
    with pytest.raises(ValidationError, match="passes min_flags, which DuplicatesOutput.deduplicate does not take"):
        _run(
            [
                {"name": "dupes", "evaluator": "dupes", "input": "a"},
                {"name": "clean", "transform": "remove", "input": "a", "plans": {"dupes": {"min_flags": 2}}},
            ]
        )


def test_a_plan_argument_value_its_method_does_not_take_fails_the_load() -> None:
    match = re.escape(
        "`plans: dupes` passes keep: 'middle', but DuplicatesOutput.deduplicate takes keep as Literal['first', 'last']."
    )
    with pytest.raises(ValidationError, match=match):
        _run(
            [
                {"name": "dupes", "evaluator": "dupes", "input": "a"},
                {"name": "clean", "transform": "remove", "input": "a", "plans": {"dupes": {"keep": "middle"}}},
            ]
        )


def test_an_output_without_a_removal_plan_fails_the_step() -> None:
    node = SimpleNamespace(value=object())
    config = RemoveConfig(input="a", plans={"r": {}})
    match = "object has no removal plan: `remove` reads a DuplicatesOutput or OutliersOutput"
    with pytest.raises(TypeError, match=match):
        RemoveTransform().run(config, {"input": node, "plans": node}, TransformContext(task="t", step="clean"))


def test_a_plan_computed_on_another_dataset_fails_the_load() -> None:
    steps = [
        {"name": "few", "transform": "view", "input": "a", "operations": [{"type": "Limit", "params": {"size": 6}}]},
        {"name": "dupes", "evaluator": "dupes", "input": "few"},
        {"name": "clean", "transform": "remove", "input": "a", "plans": {"dupes": {}}},
    ]
    with pytest.raises(ValidationError, match="reads `dupes`, which was computed on `few`, not on `a`"):
        _run(steps)


def test_a_plan_computed_on_more_datasets_than_its_input_fails_the_load() -> None:
    steps = [
        {"name": "dupes", "evaluator": "dupes", "input": ["a", "b"]},
        {"name": "clean", "transform": "remove", "input": "a", "plans": {"dupes": {}}},
    ]
    with pytest.raises(ValidationError, match="reads `dupes`, which was computed on `a`, `b`, not on `a`"):
        chain_pipeline(
            workflows=[{"name": "w", "inputs": ["a", "b"], "steps": steps}],
            evaluators=_POOL,
            tasks=[{"name": "t", "workflow": "w", "sources": ["src", "other"]}],
            datasets={"src": ToyImages(), "other": ToyImages(seed=1)},
        )


class _HintedForTypeCheckers(DuplicatesOutput):
    """A Duplicates Output whose plan method's hints name a type imported only for type checkers, as a future
    DataEval's might."""

    def deduplicate(
        self, *, dup_types: Any = "exact", keep: Any = "first", exclude_groups: Any = None, levels: Any = None
    ) -> Any:
        raise NotImplementedError


_HintedForTypeCheckers.deduplicate.__annotations__ = {
    "dup_types": "DupTypes",
    "keep": "Keep",
    "exclude_groups": "Groups",
    "levels": "Levels",
    "return": "RemovalPlan",
}


def test_plan_arguments_are_checked_by_name_alone_when_their_hints_do_not_resolve() -> None:
    classes = {"dupes": (_HintedForTypeCheckers,)}
    fits = RemoveConfig(input="a", plans={"dupes": {"keep": "last", "levels": ["item"]}})
    assert RemoveTransform.bound_problem(fits, classes) is None
    misspelled = RemoveConfig(input="a", plans={"dupes": {"kept": "last"}})
    assert RemoveTransform.bound_problem(misspelled, classes) == (
        "`plans: dupes` passes kept, which _HintedForTypeCheckers.deduplicate does not take; it takes dup_types, "
        "keep, exclude_groups, levels."
    )


_FOLDS = [
    {"name": "k", "transform": "kfold", "input": "a", "folds": 2},
    {"name": "dupes", "evaluator": "dupes", "input": "k.train"},
]


def test_a_plan_computed_on_one_element_applies_to_that_element() -> None:
    result = _run([*_FOLDS, {"name": "clean", "transform": "remove", "input": "k.train[1]", "plans": {"dupes[1]": {}}}])
    clean = result.steps["clean"]
    assert (clean.status, clean.inputs) == ("ok", ["k.train[1]", "dupes[1]"])
    assert list(result.steps["k"].output["train"]["1"].resolve_indices()) == [0, 1, 2, 3, 4, 5]
    assert list(clean.output.resolve_indices()) == [0, 1, 2, 3, 4]  # ToyImages' item 5 copies item 0
    assert clean.details == {"removed": {"items": 1, "detections": 0, "tracks": 0, "frames": 0}}


def test_a_plan_computed_on_another_element_fails_the_load() -> None:
    steps = [*_FOLDS, {"name": "clean", "transform": "remove", "input": "k.train[0]", "plans": {"dupes[1]": {}}}]
    match = re.escape("reads `dupes[1]`, which was computed on `k.train[1]`, not on `k.train[0]`")
    with pytest.raises(ValidationError, match=match):
        _run(steps)


def test_a_plan_list_applies_to_the_list_it_was_computed_on_element_by_element() -> None:
    result = _run([*_FOLDS, {"name": "clean", "transform": "remove", "input": "k.train", "plans": {"dupes": {}}}])
    clean = result.steps["clean"]
    elements = clean.elements or {}
    assert [(key, element.status, element.inputs) for key, element in elements.items()] == [
        ("0", "ok", ["k.train[0]", "dupes[0]"]),
        ("1", "ok", ["k.train[1]", "dupes[1]"]),
    ]
