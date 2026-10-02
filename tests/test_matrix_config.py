"""A task's `matrix:` as config: lists, inclusive ranges and grids, and saving them as written (task-matrix spec §3)."""

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow.config import TaskConfig
from dataeval_flow.config._schemas._matrix import MatrixRange, grid_runs, run_label, show_value


def _task(matrix: object) -> TaskConfig:
    return TaskConfig.model_validate({"name": "t", "workflow": "w", "sources": "src", "matrix": matrix})


@pytest.mark.parametrize(
    ("bounds", "values"),
    [
        ({"from": 2.5, "to": 4.5, "step": 0.5}, [2.5, 3.0, 3.5, 4.0, 4.5]),
        ({"from": 2, "to": 4, "step": 1}, [2, 3, 4]),
        ({"from": 0, "to": 1, "step": 0.3}, [0.0, 0.3, 0.6, 0.9]),
        ({"from": 0.1, "to": 0.3, "step": 0.1}, [0.1, 0.2, 0.3]),
        ({"from": 3, "to": 3, "step": 1}, [3]),
    ],
)
def test_a_range_counts_from_from_to_to_in_decimal_steps(bounds: dict, values: list) -> None:
    got = MatrixRange.model_validate(bounds).values()
    assert got == values
    assert [type(value) for value in got] == [type(value) for value in values]


@pytest.mark.parametrize(
    "bounds",
    [
        {"from": 1, "to": 2, "step": 0},
        {"from": 1, "to": 2, "step": -1},
        {"from": 3, "to": 2, "step": 1},
        {"from": True, "to": 2, "step": 1},
        {"from": 1, "to": 2},
        {"from": 1, "to": 2, "step": 1, "by": 2},
        {"from": 1, "to": float("inf"), "step": 1},
        {"from": float("-inf"), "to": 2, "step": 1},
        {"from": float("nan"), "to": 2, "step": 1},
        {"from": 1, "to": 2, "step": float("inf")},
    ],
)
def test_a_range_that_cannot_count_is_refused(bounds: dict) -> None:
    with pytest.raises(ValidationError):
        MatrixRange.model_validate(bounds)


def test_one_grid_crosses_its_keys_last_key_fastest() -> None:
    task = _task({"a": [1, 2], "b": {"from": 10, "to": 20, "step": 10}})
    assert [pairs for _, pairs in grid_runs(task.matrix)] == [
        [("a", 1), ("b", 10)],
        [("a", 1), ("b", 20)],
        [("a", 2), ("b", 10)],
        [("a", 2), ("b", 20)],
    ]


def test_several_grids_run_in_order_each_crossed_alone() -> None:
    task = _task([{"m": ["zscore", "modzscore"], "t": [2, 3]}, {"m": ["iqr"], "t": [1.5]}])
    runs = grid_runs(task.matrix)
    assert [grid for grid, _ in runs] == [0, 0, 0, 0, 1]
    assert runs[-1][1] == [("m", "iqr"), ("t", 1.5)]


def test_a_list_of_lists_varies_a_list_setting() -> None:
    task = _task({"outlier_flags": [["pixel"], ["pixel", "visual"]]})
    assert [pairs for _, pairs in grid_runs(task.matrix)] == [
        [("outlier_flags", ["pixel"])],
        [("outlier_flags", ["pixel", "visual"])],
    ]


@pytest.mark.parametrize("matrix", [{}, [], [{}], {"a": []}, {"a": {"from": 1}}, {"": [1]}, {"a..b": [1]}])
def test_an_empty_or_malformed_matrix_is_refused(matrix: object) -> None:
    with pytest.raises(ValidationError):
        _task(matrix)


def test_a_label_writes_keys_as_written_and_values_as_yaml_does() -> None:
    pairs = [("outlier_threshold", None), ("sources", ["train", "jan"]), ("x", 2.5), ("on", True), ("m", "iqr")]
    assert run_label(pairs) == "outlier_threshold=null, sources=[train, jan], x=2.5, on=true, m=iqr"
    assert show_value({"k": 5}) == "{k: 5}"


def test_a_matrix_saves_back_as_written() -> None:
    text = (
        "name: t\nworkflow: w\nsources: src\nmatrix:\n"
        "  - outlier_threshold: {from: 2, to: 4, step: 1}\n    outlier_method: [zscore]\n"
        "  - outlier_method: [iqr]\n"
    )
    data = yaml.safe_load(text)
    dumped = TaskConfig.model_validate(data).model_dump()
    assert dumped["matrix"] == data["matrix"]
    assert list(dumped["matrix"][0]) == ["outlier_threshold", "outlier_method"]
    assert yaml.safe_load(yaml.safe_dump(dumped))["matrix"] == data["matrix"]


def test_a_task_without_a_matrix_saves_no_matrix_key() -> None:
    assert "matrix" not in TaskConfig.model_validate({"name": "t", "workflow": "w", "sources": "src"}).model_dump()
