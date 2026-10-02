"""`split`, `kfold` and `view` record each output's indices into the dataset at the bottom of the views, and the
split steps show each part's size (data-splitting spec §5.4)."""

from typing import Any

import pytest

from dataeval_flow._blocks import Fields, Table
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps.transforms import KFoldTransform, SplitTransform
from tests.chain_toys import chain_pipeline, run_chain_task
from tests.evaluator_toys import ToyImages


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _run(steps: list[dict[str, Any]], count: int = 20) -> ChainResult:
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": ToyImages(count=count)},
    )
    result = run_chain_task(config)
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def _indices(result: ChainResult, step: str) -> dict[str, Any]:
    details = result.steps[step].details
    assert details is not None
    return details["indices"]


def _record_indices(record: Any) -> dict[str, Any]:
    assert record.details is not None
    return record.details["indices"]


_SPLIT = {"name": "split", "transform": "split", "input": "data", "test_frac": 0.2, "val_frac": 0.2}
_KFOLD = {"name": "split", "transform": "kfold", "input": "data", "folds": 2, "test_frac": 0.2}


def test_split_records_each_part_s_indices_partitioning_the_source() -> None:
    result = _run([_SPLIT])
    indices = _indices(result, "split")
    assert sorted(indices) == ["test", "train", "val"]
    assert sorted(i for part in indices.values() for i in part) == list(range(20))
    assert all(indices[part] for part in ("train", "val", "test"))


def test_indices_point_into_the_source_through_its_views() -> None:
    reverse = {"name": "flipped", "transform": "view", "input": "data", "operations": [{"type": "Reverse"}]}
    result = _run([reverse, {**_SPLIT, "input": "flipped"}])
    train = result.steps["split"].output["train"]
    indices = _indices(result, "split")["train"]
    assert [train[position][2]["id"] for position in range(len(train))] == indices


def test_kfold_records_each_fold_s_indices() -> None:
    result = _run([_KFOLD])
    indices = _indices(result, "split")
    assert list(indices["train"]) == ["0", "1"] == list(indices["val"])
    for key in ("0", "1"):
        assert sorted(indices["train"][key] + indices["val"][key] + indices["test"]) == list(range(20))


def test_a_view_records_its_items_indices() -> None:
    picked = {
        "name": "picked",
        "transform": "view",
        "input": "data",
        "operations": [{"type": "Indices", "params": {"indices": [4, 1, 1]}}],
    }
    assert _run([picked]).steps["picked"].details == {"indices": [4, 1, 1]}


def test_a_view_keeping_every_item_in_order_records_none() -> None:
    whole = {
        "name": "whole",
        "transform": "view",
        "input": "data",
        "operations": [{"type": "Indices", "params": {"indices": {"start": 0, "stop": 20}}}],
    }
    assert _run([whole]).steps["whole"].details is None


def test_split_s_section_gives_each_part_s_size() -> None:
    record = _run([_SPLIT]).steps["split"]
    (fields,) = SplitTransform().section(record)
    assert isinstance(fields, Fields)
    indices = _record_indices(record)
    assert fields.items == [(part.title(), len(indices[part])) for part in ("train", "val", "test")]


def test_kfold_s_section_gives_a_row_per_fold_and_each_part_s_spread() -> None:
    record = _run([_KFOLD]).steps["split"]
    table, spread = KFoldTransform().section(record)
    assert isinstance(table, Table)
    assert isinstance(spread, Fields)
    assert [row["fold"] for row in table.rows] == ["0", "1"]
    assert spread.items[-1] == ("Test", f"{len(_record_indices(record)['test'])} (shared across folds)")
