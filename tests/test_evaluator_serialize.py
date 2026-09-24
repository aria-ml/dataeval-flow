"""DataEval's three output shapes serialize to JSON-ready dicts."""

import dataclasses
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import polars as pl
import pytest
from dataeval.types import DataFrameOutput, DictOutput, Output

from dataeval_flow.evaluator._serialize import serialize_output


class _Dict(DictOutput):
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _Array(Output[np.ndarray]):
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def data(self) -> np.ndarray:
        return self._values


class _Odd(Output[object]):
    def data(self) -> object:
        return object()


@dataclasses.dataclass
class _ClassAxis:
    """Mock ClassAxis dataclass shaped like a real DataEval field."""

    name: str
    source: str
    level: Any
    groups: int
    rows_per_group_entity: float
    vocabulary: str


class _Source(Enum):
    """Test enum."""

    GROUND_TRUTH = "ground_truth"
    PREDICTED = "predicted"


class _Unsupported:
    """Unsupported class for error testing."""

    pass


class _ListOutput(Output[list]):
    """Mock output returning a plain Python list."""

    def __init__(self, values: list) -> None:
        self._values = values

    def data(self) -> list:
        return self._values


def test_a_frame_is_a_table():
    frame = pl.DataFrame({"group_id": [0], "item_indices": [[0, 5]], "dup_type": ["exact"]})
    out = serialize_output(DataFrameOutput(frame))
    assert out == {
        "shape": "table",
        "columns": ["group_id", "item_indices", "dup_type"],
        "rows": [{"group_id": 0, "item_indices": [0, 5], "dup_type": "exact"}],
    }


def test_an_empty_frame_keeps_its_columns():
    out = serialize_output(DataFrameOutput(pl.DataFrame(schema={"item_index": pl.Int64})))
    assert out == {"shape": "table", "columns": ["item_index"], "rows": []}


def test_a_dict_is_a_mapping_with_nested_frames_as_tables():
    output = _Dict(
        drifted=np.bool_(True),
        p_val=np.float32(0.01),
        distances=np.array([1.5, 2.5]),
        per_class=pl.DataFrame({"class": ["a"], "score": [0.2]}),
        when=datetime(2026, 9, 23, tzinfo=UTC),
    )
    out = serialize_output(output)
    assert out["shape"] == "mapping"
    data = out["data"]
    assert data["drifted"] is True
    assert data["p_val"] == pytest.approx(0.01)
    assert data["distances"] == [1.5, 2.5]
    expected_per_class = {
        "shape": "table",
        "columns": ["class", "score"],
        "rows": [{"class": "a", "score": 0.2}],
    }
    assert data["per_class"] == expected_per_class
    assert data["when"] == "2026-09-23T00:00:00+00:00"


def test_an_array_is_an_array():
    assert serialize_output(_Array(np.array([3, 1, 2]))) == {"shape": "array", "data": [3, 1, 2]}


def test_every_shape_is_json_ready():
    frame = DataFrameOutput(pl.DataFrame({"x": [np.int64(1)]}))
    for output in (frame, _Dict(a=np.int64(2)), _Array(np.array([1.0]))):
        serialized = serialize_output(output)
        assert json.loads(json.dumps(serialized)) == serialized


def test_an_unknown_shape_is_refused():
    with pytest.raises(TypeError, match="_Odd"):
        serialize_output(_Odd())


def test_a_dataclass_nested_in_a_mapping_is_serialized():
    """A frozen dataclass shaped like ClassAxis serializes to a plain dict."""
    output = _Dict(
        axis=_ClassAxis(
            name="class",
            source="ground_truth",
            level=None,
            groups=2,
            rows_per_group_entity=1.0,
            vocabulary="declared",
        ),
    )
    serialized = serialize_output(output)
    assert serialized["shape"] == "mapping"
    assert serialized["data"]["axis"] == {
        "name": "class",
        "source": "ground_truth",
        "level": None,
        "groups": 2,
        "rows_per_group_entity": 1.0,
        "vocabulary": "declared",
    }
    assert json.loads(json.dumps(serialized)) == serialized


def test_an_enum_nested_in_a_mapping_serializes_to_its_value():
    """An Enum member nested in a mapping serializes to its value."""
    output = _Dict(source=_Source.GROUND_TRUTH)
    serialized = serialize_output(output)
    assert serialized["shape"] == "mapping"
    assert serialized["data"]["source"] == "ground_truth"
    assert json.loads(json.dumps(serialized)) == serialized


def test_a_nested_unsupported_object_raises_typeerror():
    """An unsupported nested object raises TypeError with class name."""
    output = _Dict(unsupported=_Unsupported())
    with pytest.raises(TypeError, match="_Unsupported"):
        serialize_output(output)


def test_a_plain_list_from_data_is_an_array():
    """An Output whose data() returns a plain Python list is serialized as array."""
    output = _ListOutput([1, 2, 3])
    serialized = serialize_output(output)
    assert serialized == {"shape": "array", "data": [1, 2, 3]}
    assert json.loads(json.dumps(serialized)) == serialized
