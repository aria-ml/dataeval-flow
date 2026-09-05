"""The metadata policy a run is given, resolved and checked before the data is read.

A policy says how factors become codes: where a continuous factor is cut, what vocabulary
a categorical one takes, which of those a person actually chose, and which representation
the bias statistics then read.  It is a decision rather than a tuning knob — a bin edge is
a claim about the world — so it is defined once under the config's ``metadata:`` key and
referenced by the workflows that share it.

This module turns that configuration into something a workflow can use, and refuses the
combinations that would otherwise fail halfway through a run or, worse, quietly do
something other than what was asked.  Everything here happens **before** the dataset is
walked, so a misspelled factor or a descriptor that does not exist costs a config error
rather than an hour.
"""

__all__ = ["ResolvedPolicy", "policy_for", "policy_key", "resolve_policy"]

import json
import logging
import warnings
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataeval_flow.workflow.base import raw_field

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.config.schemas import AutoBinMethod, FactorSource, MetadataPolicyConfig
    from dataeval_flow.workflow.base import MetadataConfigMixin

_logger: logging.Logger = logging.getLogger(__name__)

# The legacy per-workflow spelling. Kept working, and refused alongside a `metadata:`
# reference: two sources disagreeing about one factor has no good resolution, and picking
# one silently is the failure this whole design exists to remove.
_LEGACY_FIELDS: tuple[str, ...] = (
    "metadata_auto_bin_method",
    "metadata_exclude",
    "metadata_continuous_factor_bins",
    "metadata_factor_source",
)

# dataeval's row levels: the prefixes a level-split statistic's name can carry (see
# metadata.expand_declared_bins, which is what actually produces a name like
# `unit_brightness` from a bare `brightness` declaration). Static rather than read off the
# dataset, because this check runs before the dataset is walked and must not need to know
# whether the data is IC or OD to catch the collision.
_ROW_LEVELS: tuple[str, ...] = ("sequence", "unit", "track", "instance")


@dataclass(frozen=True)
class ResolvedPolicy:
    """A metadata policy with its descriptor loaded and its contradictions ruled out.

    Frozen because it is a cache key as much as it is an argument: the same policy has to
    hash to the same entry however it was spelled in the config.
    """

    auto_bin_method: "AutoBinMethod | None" = None
    exclude: tuple[str, ...] = ()
    continuous_factor_bins: Mapping[str, Any] = field(default_factory=dict)
    encoding_path: Path | None = None
    """Where the descriptor lives. Handed to DataEval, which reads it itself."""
    encoding: Mapping[str, Any] | None = None
    """The descriptor's ``factors`` member, read here for the checks and the cache key.

    Read as well as passed, not instead: the path is what DataEval applies, and the
    contents are what decide whether a cached result is still the right one. A descriptor
    edited in place under an unchanged path is a different policy.
    """
    corrections: tuple[Mapping[str, Any], ...] = ()
    """The descriptor's ``corrections`` member, in the order they apply.

    Kept beside :attr:`encoding` and keyed with it, for a stronger reason than the cut is.
    A roll-up or a cut decides how values are *grouped*; a repair decides what they **are**,
    so two descriptors differing only here describe runs whose numbers came from
    differently-read columns.  Unkeyed they hash identically, and a stale hit serves one
    run's numbers to the other under the same digest.

    A tuple of plain mappings rather than DataEval's correction objects: the entries are
    hashed into :func:`policy_key`, and they are also what DataEval reads back off the path
    beside them, so they stay in the JSON-able form the descriptor wrote.
    """
    correction_specs: tuple[Any, ...] = ()
    """The same corrections as DataEval's own records, ready to hand to ``repair``.

    Built at resolve time so that a malformed rule is a config error rather than a failure
    after the dataset walk, and held beside the JSON-able :attr:`corrections` for the reason
    :attr:`encoding_specs` is held beside :attr:`encoding`: one of the pair is what gets
    applied and the other is what gets hashed, and only one of them can be hashed.

    Empty where the corrections came from a descriptor: there DataEval reads the file off
    the path itself and applies them, and nothing here has to.
    """
    aggregations: tuple[Mapping[str, Any], ...] = ()
    """Roll-ups the policy declares, in the order they replay, as plain mappings.

    Keyed because the archive persists rolled columns and ``load`` sets ``_is_structured``
    without replaying them: a ``.dem`` already carries the factors its roll-ups produced, so
    unkeyed, a run declaring them would be served one built without and the reverse.
    """
    aggregation_specs: tuple[Any, ...] = ()
    """The same roll-ups as DataEval :class:`~dataeval.types.Aggregator` records.

    Built at resolve time so a bad level pair is a config error, and held beside the
    JSON-able :attr:`aggregations` for the reason :attr:`correction_specs` is held beside
    :attr:`corrections`: one is applied, the other is hashed.
    """
    encoding_specs: Mapping[str, Any] | None = None
    """Records taken from an already-built ``Metadata``, applied to the next dataset.

    How one split's encoding reaches the others (see :func:`derive_from`).  Held as the
    objects DataEval hands back rather than as a file, because the point is to reuse a cut
    that was resolved from data rather than one somebody wrote down.  Excluded from the
    cache key, which reads the JSON-able ``encoding`` beside it — the two describe the same
    records and only one of them can be hashed.
    """
    factor_levels: Mapping[str, Sequence[Any]] | None = None
    strict: bool = False
    partial_factors: bool = False
    """Whether a factor only some rows declare is kept with a missing value.

    Keys the metadata cache for two reasons.  It changes the factor set, so an entry built
    without it answers a different question; and the archive restores it with ``or``, so a
    load can turn it **on** but never off — an entry built with it on would otherwise be
    served to a run that asked for the default and silently keep the partial factors.
    """
    factor_source: "FactorSource | None" = None
    reference_split: str | None = None
    intrinsic_factors: tuple[str, ...] = ()
    """Statistic families injected as factors.  Keys the metadata cache: it changes the
    factor set, so a run with injection on must not be served an entry built without it."""
    value_range: tuple[float, float] | None = None
    """The interval the imagery occupies, taken from the dataset config by the orchestrator.

    Authored on the dataset rather than here — it describes the data instead of deciding
    anything about it — but carried on the policy because it changes the injected values
    and therefore the codes, which is what ``policy_key`` is a rendering of.
    """

    def metadata_kwargs(self, *, for_load: bool = False) -> dict[str, Any]:
        """What to hand :class:`dataeval.Metadata`, omitting anything left unset.

        Omitted rather than passed as None so that DataEval's own defaults apply, and so
        that a release without one of these arguments still works with a policy that does
        not use it.

        Parameters
        ----------
        for_load : bool
            Whether these arguments are bound for :meth:`dataeval.Metadata.load` rather
            than the constructor.  ``factor_levels`` is a construction-time argument only:
            an archive persists the encoding record it was built under, so the declared
            vocabulary is already in the file, and the policy hash naming that file is
            what guarantees it is the one this policy asked for.  Passing it to ``load``
            is a ``TypeError``, which the cache would swallow as a permanent miss.

            ``partial_factors`` is withheld for the same reason and not the same cause:
            DataEval simply does not accept it on ``load`` — the constructor and
            ``from_factors`` both do — and the archive restores what it was structured
            under.  The asymmetry is upstream's; the guard here is what stops it becoming
            a cache that never hits.
        """
        kwargs: dict[str, Any] = {}
        if self.auto_bin_method is not None:
            kwargs["auto_bin_method"] = self.auto_bin_method
        if self.exclude:
            kwargs["exclude"] = list(self.exclude)
        if self.continuous_factor_bins:
            kwargs["continuous_factor_bins"] = dict(self.continuous_factor_bins)
        if self.encoding_specs is not None:
            # Records in hand beat a file to re-read, and are what a derived split gets.
            kwargs["encoding"] = dict(self.encoding_specs)
        elif self.encoding_path is not None:
            # The path, not the parsed contents: DataEval owns the descriptor format and
            # reads it itself, so a file written by one release is understood exactly as
            # that release meant it rather than reinterpreted here.
            kwargs["encoding"] = self.encoding_path
        if self.factor_levels and not for_load:
            kwargs["factor_levels"] = {name: list(levels) for name, levels in self.factor_levels.items()}
        if self.strict:
            kwargs["strict"] = True
        if self.partial_factors and not for_load:
            kwargs["partial_factors"] = True
        return kwargs


def policy_key(policy: ResolvedPolicy) -> str:
    """A stable, normalized rendering of everything that changes the codes.

    Normalized rather than hashed as given, so that a policy spelled as a mapping and one
    read back off a file key identically — a ``set`` and a ``list`` of the same exclusions
    describe the same policy and must not produce two cache entries.
    """
    bins = {
        name: int(value) if isinstance(value, int) else [float(edge) for edge in value]
        for name, value in (policy.continuous_factor_bins or {}).items()
    }
    return json.dumps(
        {
            "auto_bin_method": policy.auto_bin_method or "uniform_width",
            "exclude": sorted(policy.exclude, key=str),
            "continuous_factor_bins": bins,
            "encoding": policy.encoding,
            # As written, not sorted: corrections apply in sequence and one factor may take
            # several, so a reordering is a different reading of the same column.
            "corrections": [dict(entry) for entry in policy.corrections],
            # As written, for the same reason corrections are: roll-ups replay in order, and
            # one may read a column an earlier one wrote there.
            "aggregations": [dict(entry) for entry in policy.aggregations],
            "factor_levels": {name: list(levels) for name, levels in (policy.factor_levels or {}).items()},
            "strict": policy.strict,
            "partial_factors": policy.partial_factors,
            "intrinsic_factors": sorted(family.lower() for family in policy.intrinsic_factors),
            "value_range": list(policy.value_range) if policy.value_range else None,
        },
        sort_keys=True,
        default=str,
    )


# Each DataEval correction type, keyed by the `kind` its config model declares. The models
# are a discriminated union, so a kind that is not here cannot be constructed -- the mapping
# is the translation, not a second validation.
def _build_correction(entry: Any) -> Any:
    """Turn one config model into the DataEval record it describes.

    The record validates itself on construction -- a backwards range, a `multiply` of zero,
    a `decimal` the rule also drops -- so this deliberately checks nothing. Anything wrong
    raises here, carrying DataEval's own wording, and `resolve_policy` names the config
    entry that sent it.
    """
    from dataeval.types import ParseDateTime, ParseValue, Remap, Rescale

    if entry.kind == "remap":
        return Remap(entry.factor, dict(_remap_mapping(entry.rules)))
    if entry.kind == "rescale":
        return Rescale(entry.factor, over=tuple(entry.over), multiply=entry.multiply, add=entry.add)
    if entry.kind == "parse_value":
        return ParseValue(entry.factor, drop=list(entry.drop), decimal=entry.decimal)
    return ParseDateTime(entry.factor, format=entry.format, every=entry.every, epoch=entry.epoch)


def _remap_mapping(rules: "Sequence[Any]") -> list[tuple[Any, Any]]:
    """Rules to mapping entries, each key carrying the type its match kind implies.

    This is the step the rules list exists for. A range becomes a real tuple, which is what
    `_within` fires on; the catch-all becomes `None`; an exact value is passed through as
    written, because DataEval matches it with `type(key) is type(value)` and coercing it
    here would be the silent mismatch the list was chosen to avoid.
    """
    entries: list[tuple[Any, Any]] = []
    for rule in rules:
        given = rule.model_fields_set
        if "range" in given:
            entries.append((tuple(rule.range), rule.to))
        elif "otherwise" in given:
            entries.append((None, rule.otherwise))
        else:
            entries.append((rule.match, rule.to))
    return entries


def _correction_to_json(correction: Any) -> dict[str, Any]:
    """Render one record as the committed descriptor spells it.

    Byte-identical to what ``Metadata.export_encoding`` writes, and pinned by a test that
    compares the two -- because a run declaring a repair here and a later run referencing
    the descriptor exported from it describe one reading of the data, and keying them
    differently would rebuild the second for nothing.

    Rendered here rather than through DataEval's writer, which is private. The shapes agree
    because the remap pair-array is the rules list one for one; the test is what keeps them
    agreeing.
    """
    kind = type(correction).__name__
    if kind == "Remap":
        return {
            "factor": correction.factor,
            "kind": "remap",
            "map": [[list(key) if isinstance(key, tuple) else key, value] for key, value in correction.mapping.items()],
            "provenance": correction.provenance,
        }
    if kind == "Rescale":
        return {
            "add": correction.add,
            "factor": correction.factor,
            "kind": "rescale",
            "multiply": correction.multiply,
            "over": list(correction.over),
            "provenance": correction.provenance,
        }
    if kind == "ParseValue":
        return {
            "decimal": correction.decimal,
            "drop": list(correction.drop),
            "factor": correction.factor,
            "kind": "parse_value",
            "provenance": correction.provenance,
        }
    return {
        "epoch": correction.epoch,
        "every": correction.every,
        "factor": correction.factor,
        "format": correction.format,
        "kind": "parse_datetime",
        "provenance": correction.provenance,
    }


def _check_one_source_per_factor(
    declared: "Sequence[Any]",
    from_descriptor: "Sequence[Mapping[str, Any]]",
    source: str,
) -> None:
    """Refuse a factor whose values are read two ways, naming both places.

    Per factor rather than outright, exactly as `_check_no_double_declaration` treats a cut:
    a descriptor pinning one factor's vocabulary and a config repairing another are a
    longhand for one policy, not a conflict.
    """
    overlapping = sorted({c.factor for c in declared} & {str(entry.get("factor")) for entry in from_descriptor})
    if overlapping:
        raise ValueError(
            f"{source} has factors {overlapping} corrected by both `corrections` and the "
            "`encoding` descriptor, and two readings of one column have no good resolution. "
            "A descriptor's corrections are the record of a decision already made; drop the "
            "declaration here, or point at a descriptor that does not carry it.",
        )


# The full four-level relation, which every task's schema is a subset of. `resolve_policy`
# does not know whether the data is IC, OD or MOT -- that is why `_ROW_LEVELS` is a static
# tuple -- so a level pair is checked against the superset. Anything it rejects is wrong
# under every task; what it admits still has to survive the dataset, which is why this is a
# partial check and says so.
def _superset_schema() -> Any:
    """The MOT schema: a diamond, since an instance sits under both a unit and a track."""
    from dataeval.types import FactorLevelSchema

    return FactorLevelSchema(
        levels=("sequence", "unit", "track", "instance"),
        parents={"sequence": (), "unit": ("sequence",), "track": ("sequence",), "instance": ("unit", "track")},
    )


def _build_aggregator(entry: Any) -> Any:
    """Turn one config model into the DataEval record it describes.

    Options are unwrapped to a plain mapping with the unset ones dropped: DataEval refuses
    an option the reduction does not take, so passing `tolerance=None` to every reduction
    would turn a default into an error.
    """
    from dataeval.types import Aggregator

    options = {}
    if entry.options is not None:
        options = {name: value for name, value in entry.options.model_dump().items() if value is not None}
    return Aggregator(
        how=entry.how,
        source=entry.source,
        target=entry.target,
        factors=tuple(entry.factors),
        unique_by=entry.unique_by,
        via=entry.via,
        order_by=entry.order_by,
        options=options,
        min_coverage=entry.min_coverage,
        suffix=entry.suffix,
    )


def _aggregator_to_json(aggregator: Any) -> dict[str, Any]:
    """Render one roll-up as plain data, for the cache key.

    Not a descriptor section -- roll-ups are not part of the committed encoding artifact --
    so this answers only to `policy_key`, and its shape needs to be stable rather than to
    match anything upstream writes.
    """
    return {
        "how": aggregator.how,
        "source": aggregator.source,
        "target": aggregator.target,
        "factors": list(aggregator.factors),
        "unique_by": aggregator.unique_by,
        "via": aggregator.via,
        "order_by": aggregator.order_by,
        # Sorted: a mapping has no order of its own, so two spellings of one set of options
        # must not key differently.
        "options": {name: _plain(value) for name, value in sorted(aggregator.options.items())},
        "min_coverage": aggregator.min_coverage,
        "suffix": aggregator.suffix,
    }


def _plain(value: Any) -> Any:
    """A tuple renders as a list, so the key is JSON and a tuple never hashes as its repr."""
    return [_plain(item) for item in value] if isinstance(value, tuple | list) else value


def _check_output_names(aggregators: "Sequence[Any]", source: str) -> None:
    """Refuse two declarations that would produce one output name.

    DataEval renames the second to `<name>_agg`, then `<name>_agg_2` -- names that are not
    computable from the config, so `exclude`, `continuous_factor_bins` and a workflow's
    factor list all bind to something the author cannot predict. Asking for a `suffix`
    keeps every produced name derivable from what was written.

    A declaration with no `factors` names a *rule*, resolved against the dataset, so its
    outputs cannot be listed here. Two rules sharing a derived suffix and a destination
    would collide on any factor they both admit, so that pair is refused on the suffix
    alone.
    """
    seen: dict[tuple[str, str], int] = {}
    for position, aggregator in enumerate(aggregators):
        names = (
            [aggregator.name_for(factor) for factor in aggregator.factors]
            if aggregator.factors
            # The rule case: `name_for("")` is the bare suffix the outputs would all carry.
            else [aggregator.name_for("")]
        )
        for name in names:
            slot = (aggregator.target, name)
            if slot in seen:
                raise ValueError(
                    f"{source} declares aggregations {seen[slot]} and {position} that both produce "
                    f"{name or 'the same suffix'!r} at {aggregator.target!r}. DataEval would rename the "
                    "second to something no config can name, so `exclude` and "
                    "`continuous_factor_bins` could not bind to it. Give one of them a `suffix`.",
                )
            seen[slot] = position


def _read_descriptor(path: Path, source: str) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    """Read a committed descriptor's two halves, saying which config entry sent us here when it fails.

    Both halves, because a descriptor answers two questions about a factor and applying one
    without the other is what made the artifact lossy in the first place: the codes come
    back and the reading that produced them does not.

    The corrections are checked for shape only.  Which ``kind`` values exist is DataEval's
    vocabulary, not this module's, and DataEval reads the same file off the path beside
    these contents — so naming the kinds here would be a second copy of a list that has
    already grown once.  What is checked is what a reader of this file can see is wrong
    without knowing that vocabulary, and checking it here converts a failure that would
    otherwise land after the dataset walk into a config error.
    """
    if not path.exists():
        raise ValueError(
            f"{source} names encoding {str(path)!r}, which does not exist. A descriptor that "
            "matches nothing is not a no-op: every factor it was meant to pin falls back to a "
            "cut derived from this draw, which is the drift it exists to prevent.",
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} names encoding {str(path)!r}, which is not readable JSON: {exc}") from exc

    factors = document.get("factors") if isinstance(document, Mapping) else None
    if not isinstance(factors, Mapping):
        raise ValueError(
            f"{source} names encoding {str(path)!r}, which has no 'factors' member. Write one "
            "with `dataeval-flow encoding <result.json>`.",
        )

    # Absent is not malformed: version 1 predates corrections, and a descriptor written
    # from a run that declared none carries an empty array.  Both mean the same thing.
    written = document.get("corrections", [])
    if not isinstance(written, list):
        raise ValueError(
            f"{source} names encoding {str(path)!r}, whose 'corrections' member is "
            f"{type(written).__name__}, not an array. Corrections apply in order, so the "
            "descriptor records them as a list.",
        )
    for position, entry in enumerate(written):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"{source} names encoding {str(path)!r}, whose correction at position "
                f"{position} is {type(entry).__name__}, not an object.",
            )
        for member in ("kind", "factor"):
            if not isinstance(entry.get(member), str) or not entry[member]:
                raise ValueError(
                    f"{source} names encoding {str(path)!r}, whose correction at position "
                    f"{position} names no {member!r}. Rewrite it with "
                    "`dataeval-flow encoding <result.json>` rather than by hand.",
                )
    return factors, tuple(written)


def _check_no_double_declaration(
    factors: Mapping[str, Any],
    bins: Mapping[str, Any],
    levels: Mapping[str, Any],
    source: str,
    injectable_names: Collection[str] = (),
) -> None:
    """Refuse a factor declared through two channels rather than picking one.

    All three pairs, because DataEval refuses all three — and it refuses them once the
    dataset has been walked, which is the cost this check exists to convert into a config
    error.  ``encoding`` × ``continuous_factor_bins`` is a cut declared twice;
    ``factor_levels`` against either is a vocabulary declared twice.

    ``encoding`` × ``continuous_factor_bins`` is also checked one spelling down, but only
    for a name injection could actually have split: a descriptor factor named
    ``<level>_<name>`` collides with a bare ``continuous_factor_bins`` declaration of
    ``<name>`` when ``<name>`` is one of the statistics ``injectable_names`` lists, because
    injection's ``expand_declared_bins`` is what carries that bare declaration onto the
    level-prefixed name before it reaches ``Metadata``. Without that gate a dataset-native
    column that merely *looks* level-prefixed — ``unit_price``, ``instance_id`` — would
    collide with an unrelated bare declaration (``price``, ``id``) that injection was never
    going to touch, a false positive nothing produced. Gated on the actual injectable set,
    a real collision is refused exactly like the exact-name case above rather than letting
    the bare declaration silently overwrite the descriptor's committed record.
    """
    channels = (
        ("encoding", factors),
        ("continuous_factor_bins", bins),
        ("factor_levels", levels),
    )
    for (first_name, first), (second_name, second) in combinations(channels, 2):
        if both := sorted(set(first) & set(second)):
            raise ValueError(
                f"{source} declares {both} through both `{first_name}` and `{second_name}`. "
                "Two sources disagreeing about one factor has no good resolution — drop it "
                "from one of them.",
            )

    prefixed = sorted(
        (name, f"{level}_{name}")
        for name in bins
        if name in injectable_names
        for level in _ROW_LEVELS
        if f"{level}_{name}" in factors
    )
    if prefixed:
        both = sorted({spelling for pair in prefixed for spelling in pair})
        raise ValueError(
            f"{source} declares {both} through both `encoding` and `continuous_factor_bins`. "
            "`continuous_factor_bins` expands a bare declaration onto every row level it "
            "could apply to, so a level-prefixed name in `encoding` and its bare form in "
            "`continuous_factor_bins` are the same factor declared twice. Two sources "
            "disagreeing about one factor has no good resolution — drop it from one of "
            "them.",
        )


def _check_strict_is_earned(
    factors: Mapping[str, Any],
    strict: bool,
    source: str,
    *,
    declares_levels: bool = False,
) -> None:
    """Refuse to close a vocabulary nobody has reviewed.

    ``strict`` does not consult provenance: it is applied to any recorded vocabulary,
    including one DataEval derived from a draw and nobody looked at.  So the tempting rule
    — *a configured descriptor means a closed taxonomy* — would enforce a vocabulary nobody
    decided on and fail the run on the first new category, with an error calling it a
    "declared vocabulary" when nothing was declared.  The check goes the other way, and
    turns ``strict`` from a setting that is dangerous to default into one that is safe to
    set deliberately.

    Setting ``strict`` while declaring no vocabulary at all is the same mistake with
    nothing to point at: every vocabulary is then one DataEval derived from this draw, and
    closing them fails the run on the first category the sample happened to miss.
    """
    if not strict:
        return
    if not factors and not declares_levels:
        raise ValueError(
            f"{source} sets strict but declares no vocabulary, so the only vocabularies to "
            "close are the ones DataEval derives from this run's draw — nobody reviewed "
            "them, and the first category the sample missed fails the run. Declare the "
            "vocabularies with `factor_levels` or a committed `encoding`, or drop strict.",
        )
    derived = sorted(
        name
        for name, entry in factors.items()
        if isinstance(entry, Mapping) and entry.get("kind") == "levels" and entry.get("provenance") == "derived"
    )
    if derived:
        raise ValueError(
            f"{source} sets strict, which closes every vocabulary in its descriptor, but "
            f'{derived} still read provenance="derived" — nobody reviewed them. Ratify them '
            'in the descriptor (set provenance to "accepted" or "declared"), or drop strict.',
        )


def _named_policy(params: "MetadataConfigMixin", name: str, config: "PipelineConfig") -> "MetadataPolicyConfig":
    """Look up the referenced policy, refusing a config that also sets the old fields."""
    from dataeval_flow.workflow.orchestrator import _resolve_by_name

    if set_legacy := [field for field in _LEGACY_FIELDS if _is_set(params, field)]:
        raise ValueError(
            f"This workflow references metadata policy {name!r} and also sets "
            f"{set_legacy}. The `metadata_*` fields are the older spelling of the same "
            "settings; move them into the policy and remove them here.",
        )
    return _resolve_by_name(config.metadata, name, "metadata policy")


def _warn_deprecated_include_image_stats() -> None:
    """Name the field that replaces the flag, not just the fact that it is going away."""
    warnings.warn(
        "`include_image_stats` is deprecated and will be removed in the next minor "
        "version. Declare `intrinsic_factors: [visual, pixel]` on the metadata policy "
        "instead, where every workflow sharing that policy can see it.",
        DeprecationWarning,
        stacklevel=3,
    )


def _is_set(params: "MetadataConfigMixin", name: str) -> bool:
    """Whether a legacy field carries something, treating an empty list as unset."""
    value = getattr(params, name, None)
    return value is not None and value != [] and value != {}


def _resolve_corrections(
    declared: "Sequence[Any]",
    from_descriptor: tuple[Mapping[str, Any], ...],
    source: str,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Any, ...]]:
    """Build a policy's declared corrections, and merge them with a descriptor's.

    Built here rather than at first use so a malformed rule is a config error: every one of
    these types validates itself on construction, and this is the only place that can say
    which config entry the message belongs to.
    """
    if not declared:
        return from_descriptor, ()
    _check_one_source_per_factor(declared, from_descriptor, source)
    try:
        specs = tuple(_build_correction(entry) for entry in declared)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} declares a correction DataEval refuses: {exc}") from exc
    # Rendered into the same member the descriptor's fill, so `policy_key` needs no second
    # entry and a run keys alike however the repair was spelled.
    return (*from_descriptor, *(_correction_to_json(spec) for spec in specs)), specs


def _resolve_aggregations(
    declared: "Sequence[Any]",
    source: str,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Any, ...]]:
    """Build a policy's roll-ups, check their levels, and refuse an unnameable output."""
    if not declared:
        return (), ()
    try:
        specs = tuple(_build_aggregator(entry) for entry in declared)
        schema = _superset_schema()
        for aggregator in specs:
            aggregator.validate(schema)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} declares an aggregation DataEval refuses: {exc}") from exc
    _check_output_names(specs, source)
    return tuple(_aggregator_to_json(spec) for spec in specs), specs


def resolve_policy(
    params: "MetadataConfigMixin",
    config: "PipelineConfig | None" = None,
    data_dir: Path | None = None,
) -> ResolvedPolicy:
    """Resolve a workflow's metadata policy, from the pool or from its own fields.

    Parameters
    ----------
    params : MetadataConfigMixin
        The workflow's parameters, which either name a policy or carry the older
        per-workflow fields.
    config : PipelineConfig | None
        The pipeline the policy pool lives on.  Required only when a policy is named.
    data_dir : Path | None
        Data root that a descriptor path is resolved against, like every other config path.

    Returns
    -------
    ResolvedPolicy

    Raises
    ------
    ValueError
        When the reference names no policy, when the policy and the older fields are both
        set, when the descriptor is missing or unreadable, when a factor is declared twice,
        or when ``strict`` would close a vocabulary nobody reviewed.
    """
    from dataeval_flow.config._loader import resolve_path

    if params.metadata:
        if config is None:
            raise ValueError(
                f"This workflow references metadata policy {params.metadata!r}, which can only "
                "be resolved against a pipeline config.",
            )
        named = _named_policy(params, params.metadata, config)
        source = f"Metadata policy {named.name!r}"
        auto_bin_method, exclude = named.auto_bin_method, tuple(named.exclude or ())
        bins = dict(named.continuous_factor_bins or {})
        factor_levels, strict = named.factor_levels, named.strict
        partial_factors = named.partial_factors
        declared_corrections = tuple(named.corrections or ())
        declared_aggregations = tuple(named.aggregations or ())
        factor_source, reference_split = named.factor_source, named.reference_split
        intrinsic_factors = tuple(named.intrinsic_factors or ())
        descriptor_path = named.encoding
    else:
        source = "This workflow"
        auto_bin_method = params.metadata_auto_bin_method
        exclude = tuple(params.metadata_exclude or ())
        bins = dict(params.metadata_continuous_factor_bins or {})
        factor_levels, strict = None, False
        partial_factors = False
        declared_corrections = ()
        declared_aggregations = ()
        factor_source, reference_split = params.metadata_factor_source, None
        intrinsic_factors = ()
        descriptor_path = None

    factors: Mapping[str, Any] | None = None
    corrections: tuple[Mapping[str, Any], ...] = ()
    resolved_path: Path | None = None
    if descriptor_path:
        resolved_path = resolve_path(descriptor_path, data_dir)
        factors, corrections = _read_descriptor(resolved_path, source)
        _logger.info(
            "Applying encoding descriptor %s (%d factors, %d corrections)",
            descriptor_path,
            len(factors),
            len(corrections),
        )

    corrections, correction_specs = _resolve_corrections(declared_corrections, corrections, source)
    aggregations, aggregation_specs = _resolve_aggregations(declared_aggregations, source)

    # The legacy flag, folded in before the families are validated so that both spellings
    # meet the same check. Read out of the instance dict rather than off the attribute:
    # the field is marked deprecated, and pydantic warns on every read including this one,
    # which would scold every caller rather than the ones that set it.
    if raw_field(params, "include_image_stats", False):
        legacy = ("visual", "pixel")
        if intrinsic_factors and tuple(intrinsic_factors) != legacy:
            raise ValueError(
                f"{source} declares intrinsic_factors={list(intrinsic_factors)} and this "
                "workflow also sets `include_image_stats: true`, which means "
                f"{list(legacy)}. They are two spellings of one decision and they disagree "
                "— drop `include_image_stats` and keep the policy field.",
            )
        _warn_deprecated_include_image_stats()
        intrinsic_factors = legacy

    # Resolved before the double-declaration check, not after: that check needs to know
    # which bare names injection could actually turn into a level-prefixed descriptor
    # entry, and `stat_names_for` is only meaningful once the families are known to be
    # real ones. Modality is fixed at "image" until a second stat enum exists; validating
    # here rather than at injection is what makes a misspelled family a config error.
    injectable_names: frozenset[str] = frozenset()
    if intrinsic_factors:
        from dataeval_flow.metadata import resolve_families, stat_names_for

        try:
            flags = resolve_families("image", intrinsic_factors)
        except ValueError as exc:
            raise ValueError(f"{source} {exc}") from exc
        injectable_names = frozenset(stat_names_for(flags))

    # Outside the descriptor branch: `factor_levels` conflicts with `continuous_factor_bins`
    # whether or not a descriptor is named, and strict with nothing declared is the one
    # spelling that would otherwise reach DataEval unchecked.
    _check_no_double_declaration(factors or {}, bins, factor_levels or {}, source, injectable_names)
    _check_strict_is_earned(factors or {}, strict, source, declares_levels=bool(factor_levels))

    return ResolvedPolicy(
        auto_bin_method=auto_bin_method,
        exclude=exclude,
        continuous_factor_bins=bins,
        encoding_path=resolved_path,
        encoding=factors,
        corrections=corrections,
        correction_specs=correction_specs,
        aggregations=aggregations,
        aggregation_specs=aggregation_specs,
        factor_levels=factor_levels,
        strict=strict,
        partial_factors=partial_factors,
        factor_source=factor_source,
        reference_split=reference_split,
        intrinsic_factors=intrinsic_factors,
    )


def policy_for(context: Any, params: "MetadataConfigMixin") -> ResolvedPolicy:
    """The policy a workflow should read, however it was invoked.

    The orchestrator resolves the policy up front and puts it on the context, because a
    named policy needs the pipeline it is defined in and a descriptor path needs the data
    root — neither of which a workflow has.  But ``execute(context, params)`` is also a
    supported entry point on its own, and a context built by hand carries no policy.

    Falling back to the parameters is what keeps that path honest.  Reading only the
    context would silently drop a caller's configured bins and report numbers computed
    against cuts they did not choose — the same silent discard that
    ``metadata_*``-reaching-``Metadata`` was fixed for once already.
    """
    resolved = getattr(context, "metadata_policy", None)
    if resolved is not None:
        return resolved
    return resolve_policy(params)


def derive_from(policy: ResolvedPolicy, metadata: Any, descriptor: Mapping[str, Any] | None) -> ResolvedPolicy:
    """A policy that applies an already-built metadata's encoding to the next dataset.

    What makes several splits of one dataset comparable.  Encoded independently they land
    on different cuts for the same factor — the automatic bin count is derived from each
    draw — and their per-factor statistics then sit side by side under different alphabets,
    which a reader has every reason to compare and no way to know they should not.

    ``Metadata.new`` exists for exactly this and says so: *encoding this dataset against a
    record and the next one against its own draw is the drift the record exists to
    prevent*.  This is the same move, routed through the cache so a split that was built
    before is not walked again.

    Parameters
    ----------
    policy : ResolvedPolicy
        The run's policy, whose other settings carry over unchanged.
    metadata : Metadata
        The reference split's metadata, already built.
    descriptor : Mapping | None
        The same records rendered as the descriptor writes them, for the cache key.

    Returns
    -------
    ResolvedPolicy
        ``policy`` with the reference's records in place of its own encoding inputs.
    """
    encoding = getattr(metadata, "encoding", None)
    if encoding is None:
        return policy
    specs = {name: spec for name, spec in encoding().items() if spec is not None}
    if not specs:
        return policy

    # `strict` closes every vocabulary it is applied to, and the specs above include ones
    # DataEval derived from the reference split's draw.  Closing those fails the run on the
    # first category the reference happened not to contain — which is the same refusal
    # `_check_strict_is_earned` makes at config time, arriving too late to be a config
    # error.  So strict carries over only where every vocabulary here was reviewed.
    strict = policy.strict
    if strict:
        unreviewed = sorted(
            name
            for name, spec in specs.items()
            if getattr(spec, "levels", None) is not None and getattr(spec, "provenance", None) == "derived"
        )
        if unreviewed:
            strict = False
            _logger.warning(
                "Not applying strict to the encoding derived from the reference split: %s "
                'still read provenance="derived", so closing them would fail on the first '
                "category the reference split did not contain. Declare them with "
                "`factor_levels` or a committed `encoding` to close them.",
                unreviewed,
            )

    return replace(
        policy,
        encoding_specs=specs,
        encoding=dict(descriptor or {}),
        encoding_path=None,
        strict=strict,
        # Subsumed by the records above, which already say where every declared cut fell.
        # Passing both is what DataEval refuses per factor, and the records are the
        # resolved form of exactly these requests.
        continuous_factor_bins={},
        factor_levels=None,
    )
