"""Metadata convenience builder wrapping DataEval."""

__all__ = ["build_metadata", "expand_declared_bins", "inject_intrinsic_factors", "resolve_families", "stat_names_for"]

from collections.abc import Iterable, Mapping, Sequence
from enum import Flag
from typing import TYPE_CHECKING, Any, cast

from dataeval import Metadata
from dataeval.flags import ImageStats
from dataeval.protocols import AnnotatedDataset

if TYPE_CHECKING:
    from dataeval_flow.policy import ResolvedPolicy

# The only place a dataset modality maps to a statistics enum.  Adding VideoStats is an
# entry here, not a schema change: the config names families, which both enums share.
# The family set is explicit rather than derived, because `getattr(ImageStats, name)` also
# resolves individual statistics (`PIXEL_MEAN`) and the two degenerate wholes
# (`NONE`, `ALL`) — accepting those would make the config mean something it does not say.
_STAT_FAMILIES: dict[str, tuple[type[Flag], frozenset[str]]] = {
    "image": (ImageStats, frozenset({"VISUAL", "PIXEL", "DIMENSION", "HASH"})),
    # "video": (VideoStats, frozenset({"VISUAL", "PIXEL", "TEMPORAL"})),
}


def resolve_families(modality: str, families: Sequence[str]) -> Flag:
    """Resolve config-named statistic families to a flag set for *modality*.

    Parameters
    ----------
    modality : str
        The dataset's modality, which chooses the enum.
    families : Sequence[str]
        Family names as written in config, case-insensitively.

    Returns
    -------
    Flag
        The OR of the named families, or the enum's ``NONE`` when none are named.

    Raises
    ------
    ValueError
        When the modality has no enum, or a name is not one of its families.  The message
        names both sides: a silent empty injection is the failure this field exists to
        remove, so refusing has to say what was asked for and what exists.
    """
    entry = _STAT_FAMILIES.get(modality)
    if entry is None:
        known = ", ".join(sorted(_STAT_FAMILIES))
        raise ValueError(f"No statistics are defined for modality {modality!r}. Known modalities: {known}.")
    enum, allowed = entry
    flags = enum(0)
    for family in families:
        key = family.upper()
        if key not in allowed:
            valid = ", ".join(sorted(name.lower() for name in allowed))
            raise ValueError(
                f"{family!r} is not a statistic family for modality {modality!r}. "
                f"Valid families: {valid}. Families are groups, not individual statistics — "
                "declare `pixel` rather than `pixel_mean`."
            )
        flags |= cast(Flag, getattr(enum, key))
    return flags


def stat_names_for(flags: Flag) -> set[str]:
    """The statistic column names *flags* produces.

    Derived from the enum rather than listed here, so a statistic added upstream is picked
    up without an edit: every member is named ``<FAMILY>_<STATISTIC>`` and produces the
    lowercased second half.

    Only single-bit members are statistics; the multi-bit ones are the convenience groups
    (``PIXEL_BASIC``, ``HASH_DUPLICATES_D4``, ``NO_HASH``), which name no column. The
    ``bit_count`` test states that rather than relying on iteration to hide them: Python
    3.11 excludes composite members from ``iter(FlagClass)``, but 3.10 yields them, so
    without it the groups leak column names such as ``basic`` and ``hash`` on 3.10 alone.
    """
    return {
        member.name.split("_", 1)[1].lower()
        for member in type(flags)
        if member.name and "_" in member.name and member.value.bit_count() == 1 and member & flags
    }


def expand_declared_bins(
    declared: Mapping[str, Any],
    names: Iterable[str],
    levels: Iterable[str],
) -> dict[str, Any]:
    """Apply each declared bin to the factor names injection actually produced.

    ``add_factors`` names a statistic for the level it was measured at wherever the dataset
    has two — ``unit_brightness`` for the image, ``instance_brightness`` for the box — so a
    policy declaring ``brightness`` binds nothing on detection data unless its request is
    carried across. Each declared name is applied to itself and to ``<level>_<name>`` for
    every level this metadata has.

    Matched against *levels* rather than by bare suffix: ``endswith("_brightness")`` would
    also claim a dataset-native ``camera_brightness``, re-cutting a factor nobody named.

    A declaration matching nothing falls through unchanged, which is what keeps a
    misspelled factor visible in ``unmatched_bin_requests`` instead of silently absorbed.
    """
    available = set(names)
    prefixes = tuple(f"{level}_" for level in levels)
    expanded: dict[str, Any] = {}
    for name, spec in declared.items():
        candidates = {name, *(prefix + name for prefix in prefixes)}
        hits = sorted(candidates & available)
        for target in hits or [name]:
            expanded[target] = spec
    return expanded


def inject_intrinsic_factors(metadata: Metadata, calc_result: Mapping[str, Any]) -> set[str]:
    """Inject computed statistics into *metadata* as factors, returning the names added.

    The stats result labels every value with the entity it describes, so ``source_index``
    places each one at its own level: a whole-image measurement lands on the unit rows and a
    per-box measurement on the instance rows.  Unit-level values propagate down to instance
    rows, so both halves stay visible to the bias evaluators without being broadcast by hand.

    Where a statistic is measured at both levels the factor is split in two, named for the
    level it was measured at — ``unit_brightness`` for the image and ``instance_brightness``
    for the box.  Each is then binned over its own population rather than over the
    replicated copy.  The names are returned because a policy declares bins on the bare
    statistic, and :func:`expand_declared_bins` needs to know what those became.

    The only arrays withheld are the hashes.  They travel in the same result — one
    ``compute_stats`` pass serves both outlier and duplicate detection — and are near-unique
    per image, so digitizing them would yield a category per item: a factor that correlates
    with everything and describes nothing.

    Everything else is handed over as it comes, including the vector-valued statistics
    (``histogram``, ``percentiles``, ``center``).  Those have no single-column form and
    cannot become factors either, but dropping them *here* would drop them silently;
    ``add_factors`` records them in :attr:`~dataeval.Metadata.dropped_factors`, which is what
    lets the metadata summary report them as measured-but-not-representable rather than
    leaving them missing without explanation.

    *calc_result* is expected already restricted to the columns the policy names: the cache
    returns everything computed under one scope, so injecting it unrestricted would make the
    factor set a function of cache state rather than of the policy.
    """
    # Object, unicode, bytes and void dtypes are the hash columns.  Numeric and boolean
    # arrays are both usable — bool digitizes to a two-value category.
    usable = {name: arr for name, arr in calc_result["stats"].items() if arr.dtype.kind not in "OUSV"}
    if not usable:
        return set()
    before = set(metadata.factor_names)
    metadata.add_factors(usable, source_index=calc_result["source_index"])
    return set(metadata.factor_names) - before


def build_metadata(dataset: AnnotatedDataset[Any], policy: "ResolvedPolicy | None" = None) -> Metadata:
    """Build Metadata from a dataset under a resolved metadata policy.

    Parameters
    ----------
    dataset : AnnotatedDataset
        Input dataset.
    policy : ResolvedPolicy | None
        How factors become codes — the cuts, the vocabularies, and which of them somebody
        chose.  None takes DataEval's defaults, which derive everything from this draw.

    Returns
    -------
    Metadata
        DataEval Metadata instance, with the policy's ``intrinsic_factors`` injected.

    Notes
    -----
    Injection lives here rather than in each workflow because this is the one function every
    cached ``Metadata`` comes through, and it already receives the policy.  It happens
    *after* construction, which is sound because binning is lazy: a bin declared before its
    factor exists still binds when the factor arrives.

    Corrections and roll-ups are **declared on the constructor** rather than applied here.
    DataEval then owns their order — corrections before roll-ups, both before the factors
    are built — and one caller cannot get it subtly different from another.  Applying them
    here instead meant repeating that order on every path that builds a metadata, and
    ``repair`` replaces rather than accumulates, so a policy repairing one factor in YAML
    and another through its descriptor kept only the first.

    What is left here is injection, which is flow's own, and the bin re-expansion that has
    to follow it.  Re-expansion comes last because everything before it changes the factor
    set — injection adds measured factors, repairing a held-back column turns it *into* a
    factor, and a roll-up adds one per output — and ``expand_declared_bins`` matches
    declarations against the names that actually exist.
    """
    from dataeval_flow.policy import ResolvedPolicy

    resolved = policy or ResolvedPolicy()
    metadata = Metadata(dataset, **resolved.metadata_kwargs())
    injected = bool(resolved.intrinsic_factors) and _inject(metadata, dataset, resolved)
    # Corrections and roll-ups are declared on the constructor above, so DataEval has
    # already applied them in its own order by the time anything reads a factor here.
    declared = bool(resolved.correction_specs or resolved.aggregation_specs)
    if (injected or declared) and resolved.continuous_factor_bins:
        metadata.continuous_factor_bins = expand_declared_bins(
            resolved.continuous_factor_bins, metadata.factor_names, metadata.levels
        )
    return metadata


def _inject(
    metadata: Metadata,
    dataset: AnnotatedDataset[Any],
    policy: "ResolvedPolicy",
) -> bool:
    """Compute the policy's statistics and inject them.  True if any factor was produced.

    Bin re-expansion is the caller's, because a repair changes the factor set too and the
    declarations have to be matched against the names left once every step has run.
    """
    # Imported here, not at module scope: cache.py imports build_metadata from this module,
    # so a module-level import back would be circular.
    from dataeval_flow.cache import get_or_compute_stats
    from dataeval_flow.stats import ResolvedStatsPolicy, check_consumers, columns_for, restrict_columns

    flags = resolve_families(_modality_of(dataset), policy.intrinsic_factors)
    if not isinstance(flags, ImageStats):
        raise ValueError(f"Intrinsic factors are only supported for image datasets, not {flags}.")
    stats_policy = policy.stats
    if stats_policy is None:
        stats_policy = ResolvedStatsPolicy.of_flags(flags)
    else:
        check_consumers(
            stats_policy,
            outlier_flags=ImageStats.NONE,
            duplicate_flags=ImageStats.NONE,
            factor_flags=flags,
        )
    calc_result = get_or_compute_stats(
        stats_policy,
        dataset=dataset,
        per_image=True,
        # Detection data measures at both levels; asking for target statistics on
        # classification data would be a different cache scope for no extra factors.
        per_target=metadata.multi_target,
        value_range=policy.value_range,
    )
    allowed = columns_for(stats_policy.factors_from, flags)
    return bool(inject_intrinsic_factors(metadata, restrict_columns(calc_result, allowed)))


def _modality_of(dataset: AnnotatedDataset[Any]) -> str:
    """The modality whose statistics enum applies to *dataset*.

    Constant until a second enum exists.  A function rather than a literal so that adding
    ``VideoStats`` is a change here and in ``_STAT_FAMILIES``, and nowhere else.
    """
    del dataset
    return "image"
