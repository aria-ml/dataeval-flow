"""A task's `matrix:` as config: lists, inclusive ranges and grids, and saving them as written (task-matrix spec §3)."""

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow.config import TaskConfig
from dataeval_flow.config._schemas._matrix import MatrixRange, grid_runs, run_label, show_value


def _task(matrix: object) -> TaskConfig:
    return TaskConfig.model_validate({"name": "t", "workflow": "w", "sources": "src", "matrix": matrix})


def _runs(matrix: object) -> list[tuple[int, list[tuple[str, object]]]]:
    task = _task(matrix)
    assert task.matrix is not None
    return grid_runs(task.matrix)


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
    assert [pairs for _, pairs in _runs({"a": [1, 2], "b": {"from": 10, "to": 20, "step": 10}})] == [
        [("a", 1), ("b", 10)],
        [("a", 1), ("b", 20)],
        [("a", 2), ("b", 10)],
        [("a", 2), ("b", 20)],
    ]


def test_several_grids_run_in_order_each_crossed_alone() -> None:
    runs = _runs([{"m": ["zscore", "modzscore"], "t": [2, 3]}, {"m": ["iqr"], "t": [1.5]}])
    assert [grid for grid, _ in runs] == [0, 0, 0, 0, 1]
    assert runs[-1][1] == [("m", "iqr"), ("t", 1.5)]


def test_a_list_of_lists_varies_a_list_setting() -> None:
    assert [pairs for _, pairs in _runs({"outliers.flags": [["pixel"], ["pixel", "visual"]]})] == [
        [("outliers.flags", ["pixel"])],
        [("outliers.flags", ["pixel", "visual"])],
    ]


@pytest.mark.parametrize("matrix", [{}, [], [{}], {"a": []}, {"a": {"from": 1}}, {"": [1]}, {"a..b": [1]}])
def test_an_empty_or_malformed_matrix_is_refused(matrix: object) -> None:
    with pytest.raises(ValidationError):
        _task(matrix)


@pytest.mark.parametrize(
    ("matrix", "key", "value"),
    [
        ({"outliers.outlier_threshold": 3.0}, "outliers.outlier_threshold", "3.0"),
        ([{"outliers.outlier_threshold": "iqr"}], "outliers.outlier_threshold", "iqr"),
        ({"outliers.outlier_threshold": {"a": 1}}, "outliers.outlier_threshold", "{a: 1}"),
        ({"outliers.outlier_threshold": None}, "outliers.outlier_threshold", "null"),
    ],
)
def test_a_value_neither_a_list_nor_a_range_is_refused_in_one_plain_message(
    matrix: object, key: str, value: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        _task(matrix)
    (error,) = caught.value.errors()
    assert error["msg"] == (
        f"Value error, `{key}` takes a list of values, such as `[{value}]`, or a range `{{from, to, step}}`; "
        f"got {value}"
    )


@pytest.mark.parametrize(
    ("bounds", "reason"),
    [
        ({"from": 1, "to": 3}, "{from: 1, to: 3}: `step`: Field required"),
        (
            {"from": 1, "to": 2, "step": 1, "by": 2},
            "{from: 1, to: 2, step: 1, by: 2}: `by`: Extra inputs are not permitted",
        ),
        ({"from": 3, "to": 2, "step": 1}, "{from: 3, to: 2, step: 1}: a range's `from` (3) is past its `to` (2)"),
    ],
)
def test_a_range_that_cannot_count_is_refused_in_one_plain_message(bounds: dict, reason: str) -> None:
    with pytest.raises(ValidationError) as caught:
        _task({"a": bounds})
    (error,) = caught.value.errors()
    assert error["msg"] == f"Value error, `a` has a range that can't count, {reason}"


def test_a_label_writes_keys_as_written_and_values_as_yaml_does() -> None:
    pairs = [
        ("outliers.outlier_threshold", None),
        ("sources", ["train", "jan"]),
        ("x", 2.5),
        ("on", True),
        ("m", "iqr"),
    ]
    assert run_label(pairs) == "outliers.outlier_threshold=null, sources=[train, jan], x=2.5, on=true, m=iqr"
    assert show_value({"k": 5}) == "{k: 5}"


def test_a_matrix_saves_back_as_written() -> None:
    text = (
        "name: t\nworkflow: w\nsources: src\nmatrix:\n"
        "  - outliers.n_clusters: {from: 2, to: 4, step: 1}\n    outliers.outlier_threshold: [zscore]\n"
        "  - outliers.outlier_threshold: [iqr]\n"
    )
    data = yaml.safe_load(text)
    dumped = TaskConfig.model_validate(data).model_dump()
    assert dumped["matrix"] == data["matrix"]
    assert list(dumped["matrix"][0]) == ["outliers.n_clusters", "outliers.outlier_threshold"]
    assert yaml.safe_load(yaml.safe_dump(dumped))["matrix"] == data["matrix"]


def test_a_task_without_a_matrix_saves_no_matrix_key() -> None:
    assert "matrix" not in TaskConfig.model_validate({"name": "t", "workflow": "w", "sources": "src"}).model_dump()
