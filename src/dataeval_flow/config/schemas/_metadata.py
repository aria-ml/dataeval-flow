"""Metadata schemas: the policy a run is given, and the record it hands back."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dataeval_flow.config._paths import validate_config_path
from dataeval_flow.config.schemas._task import AutoBinMethod, FactorSource


class ResultMetadata(BaseModel):
    """Base metadata envelope for workflow results.

    Contains JATIC-required fields (version, timestamp, tool info,
    dataset identifiers).
    """

    version: str = "1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_id: str | Sequence[str] = ""
    label_source: str | None = None
    model_id: str | None = None
    preprocessor_id: str | None = None
    selection_id: str | None = None
    source_descriptions: Sequence[str] = ()
    resolved_config: dict[str, Any] = Field(default_factory=dict)
    tool: str = "dataeval-flow"
    tool_version: str = ""
    execution_time_s: float | None = None

    #: How each metadata factor was encoded, and how well that encoding fits.
    #: Per factor: its type and level, the ``encoding`` applied (edges or
    #: vocabulary, who chose them, and how they were placed), the name each code
    #: reads as, and the ``fit`` this run's rows made against it — counts,
    #: occupied spans, and declared bins nothing reached.  ``None`` for workflows
    #: that build no metadata.  Recorded because the encoding decides what every
    #: evaluator reads, and it is otherwise reported only in logs no envelope
    #: references.
    metadata_binning: dict[str, Any] | None = None

    #: Fingerprint of the encoding every factor was read under, or ``None`` where
    #: the workflow built no metadata or its splits did not share one.  Comparing
    #: two runs is only sound if each can say which cuts produced it: without this,
    #: a bias score that moved is unattributable between *the override worked* and
    #: *the data changed*, which are the two readings a reader is trying to tell
    #: apart.  Same digest and same data means the numbers are comparable.
    encoding_digest: str | None = None

    #: Library diagnostics raised while the workflow ran — the decisions
    #: DataEval made on the caller's behalf and the ranges it could not resolve.
    #: Empty when the run raised none.
    diagnostics: Sequence[str] = ()


# --- Corrections -------------------------------------------------------------
#
# How a factor's values are *read*, decided before anything asks what code each one takes.
# The authoring counterpart to the `unusable` section of a binning record: that says a
# column mixes 16 numeric rows with 4 text ones and that a repair can reach it; these are
# how the repair gets written down.
#
# One model per DataEval correction type, discriminated on `kind` — the idiom `datasets`
# (`format`), `extractors` (`model`) and `workflows` (`type`) already use.
#
# Almost nothing is validated here. Every one of these types validates itself on
# construction with a message naming the factor — a backwards range, a bare-string `drop`,
# a `multiply` of zero, a `decimal` the rule also drops — and `resolve_policy` builds them,
# so a mistake is a config error carrying DataEval's own wording. What pydantic does is the
# part that is about YAML rather than about corrections: pick the model, and give a remap
# rule a shape that survives the trip.

DateTimeGranularity = Literal[
    "year", "quarter", "month", "week", "day", "hour", "month_of_year", "day_of_week", "hour_of_day"
]
"""Periods :class:`dataeval.types.ParseDateTime` buckets a timestamp into.

Pinned rather than left open so a misspelling is a config error instead of a failure after
the dataset walk. `DATETIME_GRANULARITIES` is not exported from `dataeval.types`, so a
registry-sync test holds this in step with it.
"""

EpochUnit = Literal["s", "ms", "us", "ns"]
"""Units a bare number is read as an offset in. Pinned for the reason above."""


class _CorrectionBase(BaseModel):
    """Fields every correction carries."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    factor: str = Field(description="Factor the rule applies to.")


class RemapRuleConfig(BaseModel):
    """One replacement: what is matched, and what it becomes.

    A rule rather than a mapping entry because DataEval matches a key with
    ``type(key) is type(value)`` and fires a range only on a real tuple — and YAML has no
    tuple literal and no way to keep ``1``, ``"1"`` and ``true`` apart once they are mapping
    keys. A key written ``[0, 100]`` would arrive as a list, match nothing, and leave the
    remap silently doing nothing. Naming the match kind removes the question rather than
    encoding around it, and lands on the same pair-array shape DataEval already writes to
    JSON.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    match: Any = Field(default=None, description="A value matched exactly as the dataset wrote it.")
    range: tuple[float | None, float | None] | None = Field(
        default=None,
        description=(
            "A half-open [low, high) range matching any number in it — how a sentinel band "
            "is retired to one value. null at either end is unbounded."
        ),
    )
    otherwise: Any = Field(
        default=None,
        description=(
            "The catch-all, applied to every value no other rule matched. What lets a "
            "recorded mapping survive a second dataset, which will bring values the first "
            "never held. A row that recorded nothing is never matched, this included."
        ),
    )
    to: Any = Field(default=None, description="What the matched values become. Not used with `otherwise`.")

    @model_validator(mode="after")
    def _exactly_one_match_kind(self) -> "RemapRuleConfig":
        """Refuse a rule that names two ways to match, or none.

        Checked on the fields that were *set* rather than on their values: `match: null` is
        a rule about the missing value and `to: null` is a legitimate replacement, so a
        None-based test would silently reclassify both.
        """
        given = {name for name in ("match", "range", "otherwise") if name in self.model_fields_set}
        if len(given) != 1:
            named = ", ".join(sorted(given)) if given else "none of them"
            raise ValueError(
                f"A remap rule names exactly one of `match`, `range` or `otherwise`; this one names {named}.",
            )
        if "otherwise" in given and "to" in self.model_fields_set:
            raise ValueError("A remap rule with `otherwise` carries its replacement there; drop the `to`.")
        if "otherwise" not in given and "to" not in self.model_fields_set:
            raise ValueError("A remap rule needs a `to` saying what the matched values become.")
        return self


class RemapCorrectionConfig(_CorrectionBase):
    """Replace named values outright, where the vocabulary is small and closed.

    YAML example::

        corrections:
          - kind: remap
            factor: direction
            rules:
              - match: "N"
                to: 0
              - range: [-1000, 0]
                to: -99
              - otherwise: -1
    """

    kind: Literal["remap"] = "remap"
    rules: Sequence[RemapRuleConfig] = Field(
        min_length=1,
        description="Replacements, applied in order. Exactly one may be a catch-all.",
    )

    @field_validator("rules")
    @classmethod
    def _at_most_one_catch_all(cls, rules: Sequence[RemapRuleConfig]) -> Sequence[RemapRuleConfig]:
        """A second catch-all is a mapping that silently loses one of them."""
        catch_alls = sum("otherwise" in rule.model_fields_set for rule in rules)
        if catch_alls > 1:
            raise ValueError(f"A remap has one catch-all at most; this one has {catch_alls}.")
        return rules


class RescaleCorrectionConfig(_CorrectionBase):
    """Convert values over a range by an affine rule — the units case.

    YAML example::

        corrections:
          - kind: rescale
            factor: altitude
            over: [0, 1000]
            multiply: 0.3048
    """

    kind: Literal["rescale"] = "rescale"
    over: tuple[float | None, float | None] = Field(
        default=(None, None),
        description="Half-open [low, high) range the rule applies over. null at either end is unbounded.",
    )
    multiply: float = Field(default=1.0, description="Factor every value in range is multiplied by.")
    add: float = Field(default=0.0, description="Offset added after multiplying.")


class ParseValueCorrectionConfig(_CorrectionBase):
    """Read text as a value by removing what is not part of it.

    YAML example::

        corrections:
          - kind: parse_value
            factor: weight
            drop: [" ", "kg"]
    """

    kind: Literal["parse_value"] = "parse_value"
    drop: Sequence[str] = Field(
        default=(),
        description=(
            "Substrings removed from every value, in the order given. Substrings rather "
            "than characters, so a rule removing 'kg' cannot also eat the 'k' of a value "
            "it was never meant to touch."
        ),
    )
    decimal: str = Field(
        default=".",
        description="The character this column separates a fraction with, swapped for '.' after the drops.",
    )


class ParseDateTimeCorrectionConfig(_CorrectionBase):
    """Read text as a timestamp, and optionally bucket it into a period.

    A timestamp holds a different value on nearly every row, so it names its rows rather
    than grouping them and is dropped for cardinality. Reading it as the month or the hour
    it falls in is what gives it a vocabulary.

    YAML example::

        corrections:
          - kind: parse_datetime
            factor: captured_at
            every: month_of_year
    """

    kind: Literal["parse_datetime"] = "parse_datetime"
    format: str | None = Field(
        default=None,
        description="Format the timestamps are written in. None reads them as ISO 8601 or as an epoch offset.",
    )
    every: DateTimeGranularity | None = Field(
        default=None,
        description="Period to bucket each timestamp into. None keeps the timestamp itself.",
    )
    epoch: EpochUnit = Field(default="s", description="Unit a bare number is read as an offset in.")


CorrectionConfig = Annotated[
    RemapCorrectionConfig | RescaleCorrectionConfig | ParseValueCorrectionConfig | ParseDateTimeCorrectionConfig,
    Field(discriminator="kind"),
]


# --- Aggregations ------------------------------------------------------------
#
# Rolling a factor up into a level above it: one value per destination row, by the name of
# a reduction. Declared here so a run's roll-ups are policy like its cuts are — two
# workflows over one dataset that roll up differently produce numbers that merge into one
# result file and cannot be compared.

Reduction = Literal[
    "all",
    "any",
    "changes",
    "count",
    "first",
    "last",
    "longest_run",
    "max",
    "mean",
    "median",
    "min",
    "mode",
    "n_unique",
    "std",
    "sum",
    "trend",
    "var",
    "variability",
]
"""Reductions :meth:`dataeval.Metadata.aggregate` knows, by name.

Pinned so a typo is a config error rather than a failure after the walk. `REDUCTIONS` is
not exported from `dataeval`, so a registry-sync test holds this in step with it.

Four of these are temporal (`variability`, `trend`, `changes`, `longest_run`) and need an
ordering column the source level carries. A flat list admits all eighteen, so a temporal
reduction on data with no ordering validates here and is refused at run time — which levels
carry an ordering is a property of the dataset, not of the name.
"""

FactorLevel = Literal["sequence", "unit", "track", "instance"]
"""The canonical levels a row can sit at, coarsest to finest."""

_Bounds = tuple[float | None, float | None]

ToleranceSpec = (
    str
    | float
    | _Bounds
    | tuple[str, float | _Bounds | None]
    | tuple[str, float | _Bounds | None, _Bounds]
    | tuple[float | _Bounds | None, _Bounds]
)
"""A threshold spec, in the tuples :func:`dataeval.utils.thresholds.resolve_threshold` reads.

Typed rather than left as a free mapping because that resolver branches on
``isinstance(value, tuple)`` and YAML has no tuple. Written as it arrives, a spec falls
through every branch to the adaptive default and then raises about a lower bound the author
never declared::

    ["iqr", [None, 1.5]]   as written   ->  AdaptiveThreshold   (silently the wrong one)
    ("iqr", (None, 1.5))   coerced      ->  IQRThreshold

Pydantic does the coercion for free. The `Threshold` member of ``ThresholdLike`` is
deliberately absent: it is a live object, and a config declares data, not instances.
"""


class ReductionOptionsConfig(BaseModel):
    """Parameters a particular reduction takes.

    Declared per reduction, and DataEval refuses one a reduction does not take rather than
    letting it sit there inert — which is why ``tolerance`` is not a field of the aggregator
    itself. ``longest_run`` is the only reduction declaring options today.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    tolerance: ToleranceSpec | None = Field(
        default=None,
        description=(
            "How far apart two consecutive readings may be and still count as unchanged, "
            "for `longest_run`. A bare number is that distance; a spec such as "
            '["iqr", [null, 1.5]] is fitted to the changes the factor actually shows.'
        ),
    )


class AggregatorConfig(BaseModel):
    """One roll-up: a reduction, the levels it moves between, and what it moves.

    YAML example::

        aggregations:
          - how: mean
            source: unit
            target: sequence
            factors: [brightness]
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    how: Reduction = Field(description="Name of the reduction.")
    target: FactorLevel = Field(description="Level receiving one value per row. Must sit above `source`.")
    source: FactorLevel | None = Field(
        default=None,
        description=("Level whose rows are rolled up. Omitted infers it per factor from where the factor is defined."),
    )
    factors: Sequence[str] = Field(
        default=(),
        description=(
            "Factors to roll up. Empty means every factor at `source` the reduction's value "
            "type admits, resolved against the dataset — which names a rule rather than a set."
        ),
    )
    unique_by: FactorLevel | None = Field(
        default=None,
        description=(
            "Count each entity at this level once within a group. Required by a reduction "
            "over a column defined above `source`, which repeats across the fan-out."
        ),
    )
    via: FactorLevel | None = Field(
        default=None,
        description=(
            "Roll up along routes through this level rather than every route. Only a diamond "
            "offers a choice, and a route is a different question, not a different spelling."
        ),
    )
    order_by: str | None = Field(
        default=None,
        description=(
            "Column a temporal reduction reads rows in the order of. Omitted infers it from the source level."
        ),
    )
    options: ReductionOptionsConfig | None = Field(default=None, description="Parameters specific to this reduction.")
    min_coverage: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Share of the rows beneath a destination that must carry a value for it to get "
            "an answer rather than a null. The default is all-or-nothing."
        ),
    )
    suffix: str | None = Field(
        default=None,
        description=(
            "Override for the output name's suffix, which otherwise derives from `how` and "
            "`via`. Required where two declarations would otherwise produce one name."
        ),
    )


class MetadataPolicyConfig(BaseModel):
    """A named metadata policy, referenced by the workflows that share it.

    A bin edge is a claim about the world — *below 0 °C is freezing* — so where the cuts
    fell, and who chose them, is part of what a result means.  That makes the encoding a
    decision worth writing down once and pointing several workflows at, rather than a
    setting each of them spells out again: two workflows over one dataset that cut it
    differently produce numbers that merge into one result file and cannot be compared.

    Defined once under ``metadata:`` and referenced by name, like ``datasets``, ``views``,
    ``sources`` and ``extractors``.

    YAML example::

        metadata:
          - name: standard
            encoding: policy/factor_bins.json
            strict: true
            exclude: [id, filename]

        workflows:
          - name: coverage_check
            type: data-coverage
            metadata: standard
    """

    name: str = Field(description="Identifier for this policy")

    encoding: str | None = Field(
        default=None,
        description=(
            "Path, under the data root, to a committed encoding descriptor — the artifact "
            "`dataeval-flow encoding` writes. Applies the recorded cuts and vocabularies to "
            "this run instead of deriving them from its own draw, which is what makes two "
            "runs over different data comparable. Factors it does not name are encoded "
            "normally."
        ),
    )
    factor_levels: Mapping[str, Sequence[Any]] | None = Field(
        default=None,
        description=(
            "Vocabularies declared ahead of the data, one per factor: code i means "
            "levels[i], so two datasets declared against the same list share an alphabet "
            "without either having been structured first. The categorical counterpart to "
            "`continuous_factor_bins`."
        ),
    )
    strict: bool = Field(
        default=False,
        description=(
            "Whether a value no declared vocabulary holds is an error. The default admits "
            "it, which is what extension wants; true is for a closed taxonomy that should "
            "report the data leaving it rather than be widened to fit. Bin edges are "
            "unaffected — an unseen magnitude lands in an end bin either way."
        ),
    )
    partial_factors: bool = Field(
        default=False,
        description=(
            "Whether to keep a factor only some rows declare, giving the rest a missing "
            "value. The default drops such a factor for every row, which is what a factor "
            "present for only part of a dataset can otherwise do to an analysis that does "
            "not know it is part absent. Set it when the values that were recorded are the "
            "point. Keys the metadata cache: it changes the factor set, and an archive "
            "restores it with `or`, so an entry built with it on can never be read back off."
        ),
    )
    corrections: Sequence[CorrectionConfig] | None = Field(
        default=None,
        description=(
            "How factors' values are read, applied in the order given and before anything "
            "asks what code each one takes. The repair for a factor `unusable` reports — a "
            "compass recorded sometimes in degrees and sometimes as a bearing, a sentinel "
            "standing for a bad reading — and equally the way to convert a factor that is "
            "readable but in the wrong units. Keys the metadata cache: a repair changes "
            "what the values are, so two runs differing only here computed their numbers "
            "from differently-read columns. Mutually exclusive per factor with the "
            "corrections a committed `encoding` descriptor carries."
        ),
    )
    aggregations: Sequence[AggregatorConfig] | None = Field(
        default=None,
        description=(
            "Roll factors up into a level above them, applied in the order given and after "
            "any corrections — a repair can make a column readable that a roll-up then "
            "needs. Keys the metadata cache: roll-ups add factors, and an archive carries "
            "the ones it was built with."
        ),
    )
    reference_split: str | None = Field(
        default=None,
        description=(
            "Which split's encoding the whole run uses, for a workflow that reads several. "
            "Defaults to the first split the task names. Splits encoded independently land "
            "on different cuts for the same factor, so their per-factor statistics are not "
            "comparable; one reference makes them so."
        ),
    )

    auto_bin_method: AutoBinMethod | None = Field(
        default=None,
        description=(
            "How a continuous factor no declaration reaches is cut. Governs only those: a "
            "factor named by `encoding` or `continuous_factor_bins` is cut as declared."
        ),
    )
    exclude: Sequence[str] = Field(
        default_factory=list,
        description="Factor names removed before any evaluator sees them.",
    )
    continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = Field(
        default=None,
        description=(
            "Bin count (int) or explicit edges (list) per factor. Edges carry meaning and "
            "travel with the configuration; a count only says how many, and where the cuts "
            "land is still read off this sample. Mutually exclusive with `encoding` per "
            "factor."
        ),
    )
    factor_source: FactorSource | None = Field(
        default=None,
        description=(
            "Which representation of each factor the bias statistics read — `coded`, "
            "`values`, or `auto`. It moves every number they report, so workflows meant to "
            "be compared want one answer."
        ),
    )
    intrinsic_factors: Sequence[str] = Field(
        default_factory=list,
        description=(
            "Statistic families computed from the imagery and injected as metadata "
            "factors, so bias analysis can read them: `visual`, `pixel`, `dimension`. "
            "Named as families rather than individual statistics because that is the "
            "granularity worth deciding — `visual` and `pixel` are the bias workhorses, "
            "`dimension` is meaningful only for variable-size imagery. Hashes are never "
            "injected: they are near-unique per item, so a factor made from one "
            "correlates with everything and describes nothing. Empty means inject "
            "nothing, which costs exactly what a run costs today."
        ),
    )

    @field_validator("encoding")
    @classmethod
    def _check_encoding_path(cls, value: str | None) -> str | None:
        """Keep the descriptor path portable, like every other config path."""
        return None if value is None else validate_config_path(value)
