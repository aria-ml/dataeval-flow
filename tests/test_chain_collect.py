"""`collect`: single Datasets gathered into one keyed list, each element its input handed on (spec §3)."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._graph import GraphError
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps.transforms import CollectConfig
from dataeval_flow.steps.transforms._collect import collect_keys
from tests.chain_toys import ToyDetections, chain_pipeline, run_chain_task
from tests.evaluator_toys import ToyImages


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _run(steps: list[dict[str, Any]], datasets: dict[str, Any] | None = None, inputs: list[Any] | None = None):
    datasets = datasets or {"src": ToyImages(count=20)}
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": inputs or ["a"], "steps": steps}],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "workflow": "w", "sources": list(datasets)}],
        datasets=datasets,
    )
    result = run_chain_task(config)
    assert isinstance(result, ChainResult)
    return result


_SPLIT = {"name": "s", "transform": "split", "input": "a", "test_frac": 0.2, "val_frac": 0.2}


def test_collect_gathers_its_inputs_into_one_list_keyed_by_each_output_name() -> None:
    result = _run([_SPLIT, {"name": "c", "transform": "collect", "input": ["s.val", "s.test"]}])
    record = result.steps["c"]
    assert record.status == "ok"
    assert list(record.output) == ["val", "test"]
    assert [len(record.output[key]) for key in ("val", "test")] == [
        len(result.steps["s"].output["val"]),
        len(result.steps["s"].output["test"]),
    ]


@pytest.mark.parametrize(
    ("inputs", "keys"),
    [(["a"], ("a",)), (["s.val", "s.test"], ("val", "test")), (["k.train[0]", "k.test"], ("0", "test"))],
)
def test_a_key_is_the_element_key_else_the_output_name_else_the_name(inputs: list[str], keys: tuple[str, ...]) -> None:
    assert collect_keys(CollectConfig(input=inputs)) == keys


def test_keys_name_each_element() -> None:
    assert collect_keys(CollectConfig(input=["a.clean", "b.clean"], keys=["day", "night"])) == ("day", "night")


def test_two_inputs_giving_one_key_are_refused() -> None:
    with pytest.raises(ValidationError, match=r"`a\.clean` and `b\.clean` both take key `clean`"):
        CollectConfig(input=["a.clean", "b.clean"])


def test_keys_of_another_length_are_refused() -> None:
    with pytest.raises(ValidationError, match="`keys` names 1 elements, but `input` gathers 2"):
        CollectConfig(input=["a", "b"], keys=["x"])


def test_one_input_makes_a_list_of_one() -> None:
    result = _run([_SPLIT, {"name": "c", "transform": "collect", "input": ["s.test"]}])
    assert list(result.steps["c"].output) == ["test"]


def test_a_list_input_is_refused_naming_one_element() -> None:
    steps = [
        {"name": "k", "transform": "kfold", "input": "a", "folds": 2, "test_frac": 0.2},
        {"name": "c", "transform": "collect", "input": ["k.train"]},
    ]
    with pytest.raises(ValidationError, match=r"lists do not nest: name one element, such as `k\.train\[0\]`"):
        chain_pipeline(
            workflows=[{"name": "w", "inputs": ["a"], "steps": steps}],
            tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
            datasets={"src": ToyImages(count=20)},
        )


def test_each_element_is_its_input_handed_on() -> None:
    result = _run([_SPLIT, {"name": "c", "transform": "collect", "input": ["s.val", "s.test"]}])
    records = {record.name: record for record in result.metadata.lineage}
    for key in ("val", "test"):
        element = records[f"c[{key}]"]
        assert element.inputs == [f"s.{key}"]
        assert element.digest == records[f"s.{key}"].digest


def test_a_step_over_the_list_runs_once_per_key() -> None:
    result = _run(
        [
            _SPLIT,
            {"name": "c", "transform": "collect", "input": ["s.val", "s.test"]},
            {"name": "d", "evaluator": "dupes", "input": "c"},
        ]
    )
    assert list(result.steps["d"].elements or {}) == ["val", "test"]


def test_inputs_of_different_kinds_are_refused_when_the_task_starts() -> None:
    datasets = {"x": ToyImages(count=8), "y": ToyDetections([[0], [1]], {0: "a", 1: "b"})}
    with pytest.raises(GraphError, match="kind"):
        _run([{"name": "c", "transform": "collect", "input": ["a", "b"]}], datasets, inputs=["a", "b"])
