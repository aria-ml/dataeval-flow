"""Stats policy schemas: which statistics over which views, defined once and shared."""

from collections.abc import Sequence
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["StatsMeasureConfig", "StatsPolicyConfig"]

#: Families that can be measured over the background. Hash and dimension are computed for
#: the image and its boxes as usual and skipped for the background, which has no meaningful
#: hash and no geometry of its own.
_BACKGROUND_FAMILIES = frozenset({"pixel", "visual"})

StatFamily = Literal["dimension", "pixel", "visual", "hash"]


def _render_view(view: str | None) -> str:
    """Render a view the way config authors write it: `~` for the whole image."""
    return "~" if view is None else view


class StatsMeasureConfig(BaseModel):
    """One view of the imagery and the statistic families measured over it."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    bands: str | None = Field(
        default=None,
        description=(
            "Channel group this entry measures, naming a group the dataset declares under "
            "`channel_groups`. Use `~` for the whole image, whose columns carry no prefix. "
            "A group's columns are named `<bands>_<statistic>`."
        ),
    )
    families: Sequence[StatFamily] = Field(
        min_length=1,
        description=(
            "Statistic families measured over this view. Families are groups, not "
            "individual statistics — declare `pixel` rather than `pixel_mean`."
        ),
    )


class StatsPolicyConfig(BaseModel):
    """A named statement of what is measured and which views each consumer reads.

    Define a policy once here and reference it by name so workflows meant to be compared
    measure the same things. Two workflows reading different views produce numbers you
    cannot compare.

    `measure` is a complete statement: a view with no entry is not measured, and nothing is
    inferred for a missing whole-image entry.

    YAML example::

        stats:
          - name: multispectral
            measure:
              - {bands: ~,   families: [dimension, visual, hash]}
              - {bands: rgb, families: [visual]}
              - {bands: ir,  families: [visual, pixel]}
            background: true
            outliers_from: [~]
            factors_from:  [~, rgb, ir]

        workflows:
          - name: clean
            type: data-cleaning
            stats: multispectral
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str = Field(description="Identifier for this policy")
    measure: Sequence[StatsMeasureConfig] = Field(
        min_length=1,
        description=(
            "Views to measure, and the families measured over each. Read as a complete "
            "statement: a view with no entry is not measured."
        ),
    )
    background: bool = Field(
        default=False,
        description=(
            "Measure every pixel the boxes do not cover, as `background_*` columns on the "
            "same rows. Describes the scene an item was captured in rather than the things "
            "annotated within it. Only `pixel` and `visual` families are measured for it. "
            "`background_fraction` — the share of the image left unmasked — comes with it, "
            "and you should read it before any other background number."
        ),
    )
    outliers_from: Sequence[str | None] = Field(
        default=(None,),
        description=(
            "Views whose columns drive outlier detection, named by the prefix those "
            "columns carry: `~`, a group name, `background`, or `background_<group>`. "
            "Defaults to the whole image alone, which is what every workflow flagged "
            "before band groups existed. Enabling a band group does not move a cleaning "
            "result until you name it here. Give an empty list to flag nothing."
        ),
    )
    factors_from: Sequence[str | None] = Field(
        default=(None,),
        description=(
            "Views whose columns become metadata factors, named the same way as "
            "`outliers_from`. Defaults to the whole image alone. Name a band group here to "
            "let `Balance` and `Diversity` see it. Which families are injected stays with "
            "`intrinsic_factors` on the metadata policy."
        ),
    )

    def produced_views(self) -> set[str | None]:
        """The column prefixes this policy's call emits.

        The background variant of a view exists only where that view measures a family the
        background is measured for, so a hash-only group contributes no `background_<group>`.
        """
        views: set[str | None] = set()
        for entry in self.measure:
            views.add(entry.bands)
            if self.background and _BACKGROUND_FAMILIES.intersection(entry.families):
                views.add("background" if entry.bands is None else f"background_{entry.bands}")
        return views

    @model_validator(mode="after")
    def _measure_is_coherent(self) -> "StatsPolicyConfig":
        """Refuse a `measure` that would compute a statistic nowhere, or twice."""
        seen: set[str | None] = set()
        for entry in self.measure:
            if entry.bands in seen:
                raise ValueError(
                    f"Stats policy {self.name!r} names view {entry.bands!r} twice. Give each "
                    "view one entry listing every family you want measured over it.",
                )
            seen.add(entry.bands)

        whole_image = next((entry for entry in self.measure if entry.bands is None), None)
        whole_families = set(whole_image.families) if whole_image is not None else set()
        for entry in self.measure:
            if entry.bands is None:
                continue
            if "dimension" in entry.families and "dimension" not in whole_families:
                raise ValueError(
                    f"Stats policy {self.name!r} asks `dimension` of group {entry.bands!r}, "
                    "but geometry does not vary with a band subset, so no "
                    f"`{entry.bands}_width` is produced and that family is computed "
                    "nowhere. Ask `dimension` of the whole image instead, with a "
                    "`{bands: ~, families: [dimension, ...]}` entry.",
                )
            if not set(entry.families) - {"dimension"}:
                raise ValueError(
                    f"Stats policy {self.name!r} asks only `dimension` of group "
                    f"{entry.bands!r}. Geometry does not vary with a band subset, so this "
                    f"entry produces no columns: no `{entry.bands}_*` column is ever "
                    "computed, whether or not the whole image also asks for `dimension`. "
                    "Add `pixel`, `visual` or `hash` to this entry, or drop it.",
                )

        if self.background and not any(_BACKGROUND_FAMILIES.intersection(e.families) for e in self.measure):
            raise ValueError(
                f"Stats policy {self.name!r} sets `background: true` but measures nothing "
                "the background is measured for. Add `pixel` or `visual` to a `measure` "
                "entry, or drop `background`.",
            )
        return self

    @model_validator(mode="after")
    def _consumers_name_produced_views(self) -> "StatsPolicyConfig":
        """Refuse a consumer naming a view this policy does not produce."""
        produced = self.produced_views()
        for field_name in ("outliers_from", "factors_from"):
            for view in getattr(self, field_name):
                if view not in produced:
                    valid = ", ".join(sorted(_render_view(v) for v in produced))
                    raise ValueError(
                        f"Stats policy {self.name!r} names view {_render_view(view)!r} in "
                        f"`{field_name}`, which `measure` does not produce. Views this "
                        f"policy produces: {valid}.",
                    )
        return self
