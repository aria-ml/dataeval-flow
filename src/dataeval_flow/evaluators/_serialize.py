"""DataEval output to JSON: a table, a mapping or an array, exactly as DataEval returned it.

Only ``data()``, and the attributes an evaluator names as its extras, are serialized. Attributes that echo the
inputs back (``calculation_results``, ``cluster_result``) never are.
"""

__all__ = ["serialize_output"]

import dataclasses
import enum
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from pydantic import BaseModel

from dataeval_flow.workflows._common import to_serializable

if TYPE_CHECKING:
    from dataeval.types import Output


def serialize_output(output: "Output[Any]", extras: Sequence[str] = ()) -> dict[str, Any]:
    """Serialize a DataEval output by the shape of its ``data()``, with the attributes `extras` names beside it.

    Parameters
    ----------
    output : Output
        What a DataEval evaluator returned.
    extras : Sequence[str], optional
        Attributes of `output` that its ``data()`` leaves out, written under ``"extras"`` in this order; an attribute
        that is ``None`` is written as ``None``. Nothing is added when it is empty.

    Returns
    -------
    dict
        ``{"shape": "table", "columns", "rows"}``, ``{"shape": "mapping", "data"}`` or
        ``{"shape": "array", "data"}``, plus ``"extras"`` when `extras` names any.

    Raises
    ------
    TypeError
        When ``data()`` returns none of those shapes, or an extra cannot be made JSON.
    """
    serialized = _shaped(output)
    if extras:
        serialized["extras"] = {name: _plain(getattr(output, name)) for name in extras}
    return serialized


def _shaped(output: "Output[Any]") -> dict[str, Any]:
    data = output.data()
    if isinstance(data, pl.DataFrame):
        return _table(data)
    if isinstance(data, Mapping):
        return {"shape": "mapping", "data": _plain(data)}
    if isinstance(data, np.ndarray) or (isinstance(data, Sequence) and not isinstance(data, str | bytes)):
        return {"shape": "array", "data": _plain(data)}
    raise TypeError(f"Cannot serialize {type(output).__name__}: data() returned {type(data).__name__}")


def _table(frame: pl.DataFrame) -> dict[str, Any]:
    return {"shape": "table", "columns": frame.columns, "rows": [_plain(row) for row in frame.to_dicts()]}


def _plain(value: Any) -> Any:
    """Recursively turn a DataEval value into JSON-ready Python."""
    if isinstance(value, pl.DataFrame):
        return _table(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, datetime | date):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    result = to_serializable(value)
    if isinstance(result, str | int | float | bool | type(None)):
        return result
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")
