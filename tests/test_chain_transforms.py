"""The built-in transforms that reshape Datasets: merge, split, kfold, wrap and select (spec §6.1)."""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from dataeval_flow._cache import DatasetCache
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.evaluators.scope import PrioritizeConfig
from dataeval_flow.steps import ChainResult
from tests.chain_toys import ToyDetections, chain_pipeline, run_chain_task
from tests.evaluator_toys import ToyImages


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _task(
    steps: list[dict[str, Any]],
    inputs: list[Any] | None = None,
    datasets: dict[str, Any] | None = None,
    **kwargs: Any,
):
    datasets = datasets or {"src": ToyImages(count=20)}
    workflow = {"name": "w", "inputs": inputs or ["a"], "steps": steps}
    task = {
        "name": "t",
        "workflow": "w",
        "sources": list(datasets),
        **({"extractor": "flat"} if kwargs.get("extractor") else {}),
    }
    evaluators = [DuplicatesConfig(name="dupes"), PrioritizeConfig(name="rank")]
    return chain_pipeline(workflows=[workflow], evaluators=evaluators, tasks=[task], datasets=datasets, **kwargs)


def test_merge_concatenates_its_inputs_in_order() -> None:
    task = _task(
        [{"name": "m", "transform": "merge", "input": ["a", "b"]}],
        inputs=["a", "b"],
        datasets={"x": ToyImages(), "y": ToyImages(seed=1)},
    )
    result = run_chain_task(task)
    assert isinstance(result, ChainResult)
    assert result.steps["m"].status == "ok"
    assert len(result.steps["m"].output) == 24


def test_merge_of_different_vocabularies_fails_the_step() -> None:
    datasets = {
        "x": ToyDetections([[0], [1]], {0: "a", 1: "b"}),
        "y": ToyDetections([[0], [1]], {0: "c", 1: "d"}, dataset_id="other"),
    }
    task = _task([{"name": "m", "transform": "merge", "input": ["a", "b"]}], inputs=["a", "b"], datasets=datasets)
    result = run_chain_task(task)
    assert isinstance(result, ChainResult)
    assert result.steps["m"].status == "failed"
    assert "index2label" in result.steps["m"].errors[0]


def test_split_makes_train_val_and_test_that_partition_the_input() -> None:
    task = _task([{"name": "s", "transform": "split", "input": "a", "test_frac": 0.2, "val_frac": 0.1}])
    result = run_chain_task(task)
    assert isinstance(result, ChainResult)
    parts = result.steps["s"].output
    sizes = {name: len(parts[name]) for name in ("train", "val", "test")}
    # DataEval may take val_frac of what the test split leaves, so val holds 1 or 2 of the 20.
    assert sum(sizes.values()) == 20
    assert sizes["test"] == 4
    assert sizes["val"] in (1, 2)
    indices = [set(np.asarray(parts[name].resolve_indices()).tolist()) for name in ("train", "val", "test")]
    assert not (indices[0] & indices[1])
    assert not (indices[0] & indices[2])
    assert not (indices[1] & indices[2])


def test_naming_a_split_left_empty_fails_the_load() -> None:
    with pytest.raises(ValidationError, match="reads `s.val`, which step 's' leaves empty"):
        _task(
            [
                {"name": "s", "transform": "split", "input": "a", "test_frac": 0.2},
                {"name": "d", "evaluator": "dupes", "input": "s.val"},
            ]
        )


def test_kfold_gives_one_train_and_val_per_fold_and_steps_run_per_fold() -> None:
    steps = [
        {"name": "k", "transform": "kfold", "input": "a", "folds": 3},
        {"name": "d", "evaluator": "dupes", "input": "k.train"},
    ]
    result = run_chain_task(_task(steps))
    assert isinstance(result, ChainResult)
    assert set(result.steps["k"].output["train"]) == {"0", "1", "2"}
    assert set(result.steps["d"].elements or {}) == {"0", "1", "2"}
    assert {element.status for element in (result.steps["d"].elements or {}).values()} == {"ok"}


def test_a_fold_that_does_not_exist_fails_the_load() -> None:
    with pytest.raises(ValidationError, match="has elements 0, 1, 2, not `5`"):
        _task(
            [
                {"name": "k", "transform": "kfold", "input": "a", "folds": 3},
                {"name": "d", "evaluator": "dupes", "input": "k.train[5]"},
            ]
        )


def test_wrap_crops_each_detection_into_a_classification_item() -> None:
    datasets = {"src": ToyDetections([[0, 1], [1], [0, 0]], {0: "a", 1: "b"})}
    task = _task([{"name": "c", "transform": "wrap", "input": "a", "wrapper": "DetectionCrops"}], datasets=datasets)
    result = run_chain_task(task)
    assert isinstance(result, ChainResult)
    assert len(result.steps["c"].output) == 5
    assert result.metadata.lineage[-1].name == "c"


def test_wrap_of_a_classification_dataset_fails_before_running() -> None:
    from dataeval_flow._chain._graph import GraphError

    match = "DetectionCrops takes a object_detection Dataset, but its input is classification"
    with pytest.raises(GraphError, match=match):
        run_chain_task(_task([{"name": "c", "transform": "wrap", "input": "a", "wrapper": "DetectionCrops"}]))


def test_select_keeps_the_first_n_of_a_ranking_of_the_same_dataset() -> None:
    steps = [
        {"name": "r", "evaluator": "rank", "input": "a"},
        {"name": "top", "transform": "select", "input": "a", "ranking": "r", "n": 5},
    ]
    result = run_chain_task(_task(steps, extractor=True))
    assert isinstance(result, ChainResult)
    ranking = np.asarray(result.steps["r"].output.indices)
    chosen = np.asarray(result.steps["top"].output.resolve_indices())
    assert chosen.tolist() == ranking[:5].tolist()


def test_select_takes_a_ranking_that_also_read_a_reference_set() -> None:
    steps = [
        {"name": "r", "evaluator": "rank", "input": ["a", "b"]},
        {"name": "top", "transform": "select", "input": "a", "ranking": "r", "n": 5},
    ]
    datasets = {"x": ToyImages(count=20), "y": ToyImages(count=8, seed=1)}
    result = run_chain_task(_task(steps, inputs=["a", "b"], datasets=datasets, extractor=True))
    assert isinstance(result, ChainResult)
    assert result.steps["top"].status == "ok"
    ranking = np.asarray(result.steps["r"].output.indices)
    chosen = np.asarray(result.steps["top"].output.resolve_indices())
    assert chosen.tolist() == ranking[:5].tolist()


def test_select_with_a_ranking_of_another_dataset_fails_the_load() -> None:
    steps = [
        {"name": "few", "transform": "view", "input": "a", "operations": [{"type": "Limit", "params": {"size": 10}}]},
        {"name": "r", "evaluator": "rank", "input": "few"},
        {"name": "top", "transform": "select", "input": "a", "ranking": "r", "n": 5},
    ]
    with pytest.raises(ValidationError, match="reads `r`, which was computed on `few`, not on `a`"):
        _task(steps, extractor=True)
