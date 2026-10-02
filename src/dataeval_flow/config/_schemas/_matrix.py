"""A task's `matrix:`: the values each run takes, as lists, inclusive ranges, and grids (task-matrix spec §3)."""

__all__ = ["Matrix", "MatrixGrid", "MatrixRange", "MatrixValues", "grid_runs", "run_label", "show_value"]

import itertools
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Annotated, Any, ClassVar, Self, TypeAlias

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

_KEY = re.compile(r"^[^.\s]+(\.[^.\s]+)*$")


class MatrixRange(BaseModel):
    """An inclusive range of numbers, ``{from, to, step}``: ``from + i·step`` for each ``i`` whose value is at most
    ``to``. Integers when all three bounds are, floats otherwise; counted in decimal, so ``0.1`` steps land exactly."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    from_: int | float = Field(alias="from", allow_inf_nan=False, description="The first value.")
    to: int | float = Field(
        allow_inf_nan=False, description="The last value allowed. It is a value when a step lands on it."
    )
    step: int | float = Field(allow_inf_nan=False, description="How far each value is from the one before; above 0.")

    @field_validator("from_", "to", "step", mode="before")
    @classmethod
    def _a_number(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("a range's bounds are numbers, not true or false")
        return value

    @model_validator(mode="after")
    def _counts_up(self) -> Self:
        if self.step <= 0:
            raise ValueError(f"a range's `step` must be above 0, not {self.step}")
        if self.from_ > self.to:
            raise ValueError(f"a range's `from` ({self.from_}) is past its `to` ({self.to})")
        return self

    def values(self) -> list[int | float]:
        """Every value the range holds, in order."""
        start, stop, step = (Decimal(str(bound)) for bound in (self.from_, self.to, self.step))
        integral = all(isinstance(bound, int) for bound in (self.from_, self.to, self.step))
        values: list[int | float] = []
        index = 0
        while (value := start + index * step) <= stop:
            values.append(int(value) if integral else float(value))
            index += 1
        return values


def _dotted_keys(grid: dict[str, Any]) -> dict[str, Any]:
    for key in grid:
        if not _KEY.match(key):
            raise ValueError(
                f"`{key}` is not a matrix key: a key is dotted names, such as `health_thresholds.ood.warning`"
            )
    return grid


MatrixValues: TypeAlias = Annotated[list[Any], Field(min_length=1)] | MatrixRange
MatrixGrid: TypeAlias = Annotated[dict[str, MatrixValues], Field(min_length=1), AfterValidator(_dotted_keys)]
Matrix: TypeAlias = MatrixGrid | Annotated[list[MatrixGrid], Field(min_length=1)]


def grid_runs(matrix: "Matrix") -> list[tuple[int, list[tuple[str, Any]]]]:
    """Every run `matrix` makes, in order: its grid's index and its ``(key, value)`` pairs in the order written.

    One grid crosses every key with every other, the last key varying fastest; several grids run in turn.
    """
    grids: Sequence[Mapping[str, Any]] = [matrix] if isinstance(matrix, Mapping) else matrix
    runs: list[tuple[int, list[tuple[str, Any]]]] = []
    for index, grid in enumerate(grids):
        keys = list(grid)
        columns = [values.values() if isinstance(values, MatrixRange) else list(values) for values in grid.values()]
        runs.extend((index, list(zip(keys, combination, strict=True))) for combination in itertools.product(*columns))
    return runs


def show_value(value: Any) -> str:
    """`value` as YAML writes it inline: ``null``, ``true``, ``[a, b]``, ``{k: v}``, a number or a bare string."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{key}: {show_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, Sequence) and not isinstance(value, str):
        return "[" + ", ".join(show_value(item) for item in value) + "]"
    return str(value)


def run_label(pairs: Sequence[tuple[str, Any]]) -> str:
    """A run's label: its grid's keys and values in the order written, ``key=value, key=value``."""
    return ", ".join(f"{key}={show_value(value)}" for key, value in pairs)
