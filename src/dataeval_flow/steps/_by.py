"""`by:`: a step run once per class, or per group of classes, inside one Output (spec §5.9)."""

__all__ = ["ByConfig", "ClassKeys"]

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator


class ClassKeys(BaseModel):
    """How `by: class` keys items: by class, or by named groups of classes, and the smallest key it runs."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    groups: dict[str, list[int | str]] | None = Field(
        default=None,
        description=(
            "Named groups of classes, each listed by index or by name, keying each group instead of each class. Groups "
            "may overlap; a class in no group is left out."
        ),
    )
    min_items: int = Field(
        default=2, ge=1, description="The fewest items of a key each input must hold for that key to run."
    )

    @field_validator("groups")
    @classmethod
    def _no_empty_group(cls, groups: dict[str, list[int | str]] | None) -> dict[str, list[int | str]] | None:
        empty = [name for name, members in (groups or {}).items() if not members]
        if empty:
            raise ValueError(f"Group {', '.join(f'`{name}`' for name in empty)} lists no class.")
        return groups


class ByConfig(BaseModel):
    """A step's `by:`. Written `by: class`, or `by: {class: {groups: ..., min_items: ...}}`."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    class_: ClassKeys = Field(default_factory=ClassKeys, alias="class", description="Key items by their class.")

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, value: Any) -> Any:
        if value == "class" or (isinstance(value, Mapping) and "class" in value and value["class"] is None):
            return {"class": {}}
        return value

    @model_serializer(mode="plain")
    def _as_written(self) -> Any:
        if self.class_ == ClassKeys():
            return "class"
        return {"class": self.class_.model_dump(exclude_defaults=True)}

    @property
    def label(self) -> Literal["class", "group"]:
        """What one key is: a class, or a group of classes."""
        return "group" if self.class_.groups is not None else "class"

    @property
    def plural(self) -> str:
        """`classes` or `groups`."""
        return "groups" if self.class_.groups is not None else "classes"
