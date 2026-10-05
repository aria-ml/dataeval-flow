"""The report blocks: typed, serializable pieces of a report that every renderer draws its own way."""

from __future__ import annotations

__all__ = [
    "Asset",
    "Block",
    "BulletList",
    "Cell",
    "Code",
    "Column",
    "Distribution",
    "Fields",
    "Flag",
    "ItemRef",
    "Paragraph",
    "Proportion",
    "Quantiles",
    "Scalar",
    "Section",
    "Severity",
    "Summary",
    "SummaryItem",
    "Table",
    "Tree",
    "Verdict",
]

import math
from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    model_serializer,
)


def _native(value: Any) -> Any:
    """A numpy scalar as the Python value it prints as; anything else as it is.

    DataEval hands back numpy scalars, and left to pydantic an ``int64`` count becomes a float
    that an integer format then refuses.  A ``float32`` takes the digits it prints with rather
    than its float64 widening, so 0.1 stays 0.1.
    """
    if isinstance(value, list):
        return [_native(item) for item in value]
    if type(value).__module__ != "numpy" or not hasattr(value, "item"):
        return value
    if getattr(value, "ndim", 0):
        # An array, such as a sparkline's counts: its items, rather than an error from `item()`.
        return _native(value.tolist())
    native = value.item()
    return float(str(value)) if isinstance(native, float) else native


Scalar = Annotated[str | int | float | bool | None, BeforeValidator(_native)]
# A measured value a chart places: a drift threshold, a quantile.
Number = Annotated[float, BeforeValidator(_native)]
# A figure a run may not have recorded: NaN, which JSON writes as `null` and which reads back as NaN.
Unknown = Annotated[float, BeforeValidator(lambda value: math.nan if value is None else _native(value))]
Severity = Literal["ok", "info", "warning"]

_TYPE = "The block's kind, which says how its other fields read. Readers skip a kind they do not know."


class _Block(BaseModel):
    """Frozen and closed: a block is data a renderer reads, never something it fills in."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    @model_serializer(mode="wrap")
    def _without_defaults(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """The fields, less those at their defaults, which a reader fills back in; the ``type`` tag always stays."""
        fields = type(self).model_fields
        return {
            name: value
            for name, value in handler(self).items()
            if name == "type" or getattr(self, name) != fields[name].get_default(call_default_factory=True)
        }


class Flag(_Block):
    """One measurement against the population it was judged in: why an item was flagged, and the limit it crossed.

    A figure that wasn't recorded is NaN, which JSON writes as ``null`` and which reads back as NaN.
    """

    name: str = Field(description="What was measured, such as `brightness`.")
    value: Number = Field(description="This item's value.")
    direction: Literal["upper", "lower"] = Field(description="Which limit the value crossed.")
    bound: Unknown = Field(description="The limit it crossed; `null` when unknown.")
    percentile: Unknown = Field(
        description="Where the value ranks in its population, from 0 to 100; `null` when unknown."
    )
    mean: Unknown = Field(description="The population's mean; `null` when unknown.")
    std: Unknown = Field(description="The population's standard deviation; `null` when unknown.")


class ItemRef(_Block):
    """One dataset item as the run saw it: an index into one source, after the source's view.

    Frozen, so it is hashable: a result's assets are keyed by it, and one item named twice is one item.
    """

    source: str = Field(description="The source the item was read from, as the run's config or `run()` named it.")
    index: int = Field(description="The item's index in that source, after the source's view.")
    target: int | None = Field(
        default=None, description="A detection box, by its index in the item's annotation; `null` for the whole item."
    )

    def __hash__(self) -> int:
        # Pydantic hashes a frozen model already; spelled out so that a type checker sees it too.
        return hash((self.source, self.index, self.target))


class Asset(_Block):
    """A preview of one item, such as its thumbnail, carried in the result so that a report can show it."""

    item: ItemRef = Field(description="The item this previews.")
    media_type: str = Field(
        description="What `data` holds, such as `image/webp`. A reader that can't draw it names the item instead."
    )
    width: int = Field(description="The preview's width, in pixels.")
    height: int = Field(description="The preview's height, in pixels.")
    data: str = Field(description="The preview itself, base64-encoded.")


# A list cell holds a sparkline's counts or a stacked bar's segments, a flags column's flags, or an image
# column's group of items.
Cell = Annotated[
    str | int | float | bool | None | list[float] | list[Flag] | ItemRef | list[ItemRef], BeforeValidator(_native)
]


class Section(_Block):
    """A titled group of blocks. How the title is drawn depends on how deeply the section is nested.

    ``reference`` marks a section a reader opens when they need it, such as a run's configuration: HTML folds it
    away, apart from the findings.
    """

    type: Literal["section"] = Field(default="section", description=_TYPE)
    title: str = Field(description="The heading. A report's root title may hold `\\n` for a multi-line banner.")
    brief: str | None = Field(default=None, description="A short value shown beside the title.")
    severity: Severity | None = Field(default=None, description="The verdict this section carries, if any.")
    reference: bool = Field(
        default=False,
        description="Whether the section is reference a reader opens when needed, such as the configuration.",
    )
    blocks: list[Block] = Field(default_factory=list, description="The section's content, in order.")


class Paragraph(_Block):
    """Prose, wrapped to whatever width the renderer has."""

    type: Literal["paragraph"] = Field(default="paragraph", description=_TYPE)
    text: str = Field(description="The prose. Backticks mark inline code; a single `\\n` is a hard line break.")


class BulletList(_Block):
    """Short prose items, one bullet each."""

    type: Literal["bullet_list"] = Field(default="bullet_list", description=_TYPE)
    items: list[str] = Field(description="The items, in order, each wrapped under its own bullet.")


class Fields(_Block):
    """Labelled values, aligned."""

    type: Literal["fields"] = Field(default="fields", description=_TYPE)
    items: list[tuple[str, Scalar]] = Field(
        description=(
            "`[label, value]` pairs, in order. Pairs rather than a mapping, because JavaScript reorders "
            "integer-like object keys. A value's `\\n` starts a line aligned under the values."
        )
    )


class Column(_Block):
    """How one table column reads its cells, and how they are drawn."""

    key: str = Field(description="The row key this column reads. Two columns may read one key: a count and its bar.")
    header: str = Field(default="", description="The column's heading. A table whose headings are all empty has none.")
    kind: Literal["text", "bar", "stacked", "sparkline", "flags", "image"] = Field(
        default="text",
        description=(
            "`text` shows the cell; `bar` draws a number as a bar; `stacked` draws a list of numbers as one "
            "segmented bar; `sparkline` draws a list of counts as a small histogram; `flags` shows a list of "
            "flags, by name, each as its value against the limit it crossed; `image` shows an item's thumbnail, "
            "or a group's, from an item reference or a list of them."
        ),
    )
    align: Literal["left", "right"] | None = Field(
        default=None, description="Unset: the first column is left-aligned and the rest right-aligned."
    )
    format: str | None = Field(default=None, description='A `str.format` template for numeric cells, e.g. "{:.1f}%".')
    series: list[str] = Field(default_factory=list, description="A stacked column's segment names, in cell order.")
    markers: list[tuple[str, Number]] = Field(
        default_factory=list, description="A bar column's `[label, value]` reference lines, such as drift thresholds."
    )
    in_text: bool = Field(
        default=True,
        description=(
            "Whether the text report draws this column. HTML always does. False for a column text can do without, "
            "such as a title the section above already gives, where its room is scarce."
        ),
    )


class Table(_Block):
    """Rows of cells under typed columns."""

    type: Literal["table"] = Field(default="table", description=_TYPE)
    columns: list[Column] = Field(description="The columns, in display order.")
    rows: list[dict[str, Cell]] = Field(
        description="One mapping per row, from column key to cell. A string cell may hold `\\n` for several lines."
    )
    preview: int | None = Field(
        default=None,
        ge=0,
        description=(
            "How many rows a renderer with little room shows first, followed by a line counting the rest. "
            "Unset: every row."
        ),
    )


class Proportion(_Block):
    """How a whole splits into parts."""

    type: Literal["proportion"] = Field(default="proportion", description=_TYPE)
    parts: list[tuple[str, int]] = Field(
        description="`[label, count]` pairs. The bar shows the first part's share of the total."
    )


class Quantiles(_Block):
    """The order statistics a box plot is drawn from."""

    low: Number = Field(description="The smallest value.")
    q1: Number = Field(description="The 25th percentile.")
    median: Number = Field(description="The 50th percentile.")
    q3: Number = Field(description="The 75th percentile.")
    high: Number = Field(description="The largest value.")


class Distribution(_Block):
    """A histogram, drawn as a box plot when its quantiles are known and as a sparkline otherwise."""

    type: Literal["distribution"] = Field(default="distribution", description=_TYPE)
    histogram: list[int] = Field(description="Counts per cell, low to high.")
    quantiles: Quantiles | None = Field(default=None, description="The values' order statistics, when known.")


class Code(_Block):
    """Text printed exactly as given, never wrapped: YAML, a stanza to paste."""

    type: Literal["code"] = Field(default="code", description=_TYPE)
    text: str = Field(description="The text, lines separated by `\\n`.")
    language: str | None = Field(default=None, description="What the text is written in, such as `yaml`.")


class Tree(_Block):
    """A nested JSON value, such as a resolved configuration."""

    type: Literal["tree"] = Field(default="tree", description=_TYPE)
    value: JsonValue = Field(description="The value: mappings, lists and scalars, nested to any depth.")


class SummaryItem(_Block):
    """One line of a summary: a label, its value, and its verdict."""

    label: str = Field(description="What the line is about.")
    value: str = Field(default="", description="A short value, such as a count.")
    severity: Severity = Field(default="info", description="The line's verdict: `warning`, `ok` or `info`.")
    group: str = Field(
        default="",
        description="The sub-heading the line sits under, such as the split its finding judged; empty for none.",
    )


class Summary(_Block):
    """One line per finding, with a verdict marker each, then the health verdict: the warnings its result counted.

    ``warnings`` is the count the result made once, which every renderer states rather than recounting the lines.
    ``failed`` names the required steps whose failure failed the run: the verdict is then failed, whatever the
    warnings.
    """

    type: Literal["summary"] = Field(default="summary", description=_TYPE)
    items: list[SummaryItem] = Field(description="The lines, in order; those of one group are kept together.")
    warnings: int = Field(ge=0, description="How many findings are warnings, as the result counted them.")
    failed: list[str] = Field(
        default_factory=list,
        description="The required steps that failed, which fail the run whatever its warnings; empty where none did.",
    )


class Verdict(_Block):
    """Whether the data a report judged is ready: its level, as a word and as the line that gives its reasons.

    It stands in for a summary's health verdict, which it would contradict: a report with a verdict has no summary.
    """

    type: Literal["verdict"] = Field(default="verdict", description=_TYPE)
    level: Literal["not-ready", "ready-with-caveats", "ready"] = Field(description="How ready the data is.")
    label: str = Field(description='The level as a report writes it, such as "Ready with caveats".')
    line: str = Field(description='The label and its reasons, such as "Ready with caveats: 2 warnings".')

    @property
    def severity(self) -> Severity:
        """The marker the level is drawn with: a warning where the data is not ready, ok where it is ready."""
        return "warning" if self.level == "not-ready" else "ok" if self.level == "ready" else "info"


Block = Annotated[
    Section | Paragraph | BulletList | Fields | Table | Proportion | Distribution | Code | Tree | Summary | Verdict,
    Field(discriminator="type"),
]

Section.model_rebuild()
