"""`by:`: a step run once per class, or group of classes, by label or by a model's prediction.

See spec §5.9 and uncertainty-drift spec §5.
"""

__all__ = ["ByConfig", "ClassKeys", "PredictedKeys", "roll_up"]

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

from dataeval_flow._blocks import Paragraph
from dataeval_flow.workflows._base import Finding


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


class PredictedKeys(ClassKeys):
    """How `by: predicted` keys rows: by each class a model predicts, or by named groups of them."""

    threshold: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description=(
            "A row counts toward every class whose sigmoid score is at least this share of its largest, DataEval's "
            "rule; `1.0` takes the top class only."
        ),
    )


class ByConfig(BaseModel):
    """A step's `by:`. Written `by: class` or `by: predicted`, or with settings, `by: {class: {groups: ...}}`."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    class_: ClassKeys | None = Field(default=None, alias="class", description="Key items by their class.")
    predicted: PredictedKeys | None = Field(
        default=None, description="Key rows by the class a model predicts: needs an `uncertainty` extractor."
    )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        """The object form, or the bare `class` or `predicted` that `_shorthand` reads."""
        return {"anyOf": [{"enum": ["class", "predicted"], "type": "string"}, handler(core_schema)]}

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, value: Any) -> Any:
        if value in ("class", "predicted"):
            return {value: {}}
        if isinstance(value, Mapping):
            return {key: {} if inner is None else inner for key, inner in value.items()}
        return value

    @model_validator(mode="after")
    def _one_kind(self) -> "ByConfig":
        if (self.class_ is None) == (self.predicted is None):
            raise ValueError("`by:` keys by exactly one of `class` or `predicted`.")
        return self

    @model_serializer(mode="plain")
    def _as_written(self) -> Any:
        kind = "class" if self.class_ is not None else "predicted"
        keys = self.keys
        return kind if keys == type(keys)() else {kind: keys.model_dump(exclude_defaults=True)}

    @property
    def keys(self) -> ClassKeys:
        """The written kind's settings: its groups and `min_items`, and for `predicted`, its `threshold`."""
        return cast(ClassKeys, self.class_ if self.class_ is not None else self.predicted)

    @property
    def label(self) -> str:
        """What one key is: `class`, `group`, `predicted class` or `predicted group`."""
        noun = "group" if self.keys.groups is not None else "class"
        return f"predicted {noun}" if self.predicted is not None else noun

    @property
    def plural(self) -> str:
        """`classes`, `groups`, `predicted classes` or `predicted groups`."""
        noun = "groups" if self.keys.groups is not None else "classes"
        return f"predicted {noun}" if self.predicted is not None else noun


_ORDER: tuple[Literal["ok", "info", "warning"], ...] = ("ok", "info", "warning")


def roll_up(
    findings: Mapping[str, Sequence[Finding]], skipped: Mapping[str, str], *, title: str, by: ByConfig
) -> Finding:
    """One finding for a check run once per key: the worst severity, how many keys warn, and which."""
    heading = f"{next((f.title for found in findings.values() for f in found), title)} by {by.label}"
    not_assessed = "; ".join(f"{key} ({why})" for key, why in skipped.items())
    if not findings:
        return Finding(
            severity="info",
            title=heading,
            brief="not assessed",
            blocks=[Paragraph(text=f"Not assessed: {not_assessed}.")] if not_assessed else [],
        )
    worst: dict[str, Literal["ok", "info", "warning"]] = {
        key: max([f.severity for f in found] or [_ORDER[0]], key=_ORDER.index) for key, found in findings.items()
    }
    warned = [key for key, severity in worst.items() if severity == "warning"]
    noted = [key for key, severity in worst.items() if severity == "info"]
    lines = [f"Warned: {', '.join(warned)}." if warned else "", f"Noted: {', '.join(noted)}." if noted else ""]
    if not_assessed:
        lines.append(f"Not assessed: {not_assessed}.")
    return Finding(
        severity=max(worst.values(), key=_ORDER.index),
        title=heading,
        brief=f"{len(warned)}/{len(findings)} {by.plural} warn",
        blocks=[Paragraph(text=line) for line in lines if line],
    )
