"""Metadata convenience builder wrapping DataEval."""

__all__ = [
    "IMAGE_STAT_GROUPS",
    "build_metadata",
    "expand_declared_bins",
    "inject_intrinsic_factors",
    "resolve_families",
    "stat_names_for",
]

from collections.abc import Iterable, Mapping, Sequence
from enum import Flag
from typing import TYPE_CHECKING, Any

from dataeval import Metadata
from dataeval.flags import ImageStats
from dataeval.protocols import AnnotatedDataset

if TYPE_CHECKING:
    from dataeval_flow._policy import ResolvedPolicy

# The names config uses for groups of statistics, each mapped to its flags: the four families
# and the sub-groups DataEval defines inside them. Explicit rather than derived, because
# `getattr(ImageStats, name)` also resolves individual statistics (`PIXEL_MEAN`) and the two
# degenerate wholes (`NONE`, `ALL`) — accepting those would make the config mean something it
# does not say. The hash sub-groups keep the spelling `duplicates.flags` has always used.
IMAGE_STAT_GROUPS: dict[str, ImageStats] = {
    "dimension": ImageStats.DIMENSION,
    "dimension_basic": ImageStats.DIMENSION_BASIC,
    "dimension_box": ImageStats.DIMENSION_BOX,
    "dimension_offset": ImageStats.DIMENSION_OFFSET,
    "dimension_position": ImageStats.DIMENSION_POSITION,
    "hash": ImageStats.HASH,
    "hash_basic": ImageStats.HASH_DUPLICATES_BASIC,
    "hash_d4": ImageStats.HASH_DUPLICATES_D4,
    "pixel": ImageStats.PIXEL,
    "pixel_basic": ImageStats.PIXEL_BASIC,
    "pixel_distribution": ImageStats.PIXEL_DISTRIBUTION,
    "visual": ImageStats.VISUAL,
    "visual_basic": ImageStats.VISUAL_BASIC,
}

# The only place a dataset modality maps to its statistic groups. Adding VideoStats is an
# entry here, not a schema change: the config names groups, which both enums share.
_STAT_FAMILIES: "dict[str, Mapping[str, Flag]]" = {
    "image": IMAGE_STAT_GROUPS,
    # "video": VIDEO_STAT_GROUPS,
}


def resolve_families(modality: str, families: Sequence[str]) -> Flag:
    """Resolve config-named statistic families and sub-groups to a flag set for *modality*.

    Parameters
    ----------
    modality : str
        The dataset's modality, which chooses the enum.
    families : Sequence[str]
        Family or sub-group names as written in config, case-insensitively.

    Returns
    -------
    Flag
        The OR of the named groups, or the enum's ``NONE`` when none are named.

    Raises
    ------
    ValueError
        When the modality has no enum, or a name is not one of its groups.  The message
        states the requested name and the valid names: a silent empty injection is the
        failure this field exists to remove.
    """
    groups = _STAT_FAMILIES.get(modality)
    if groups is None:
        known = ", ".join(sorted(_STAT_FAMILIES))
        raise ValueError(f"No statistics are defined for modality {modality!r}. Known modalities: {known}.")
    flags = type(next(iter(groups.values())))(0)
    for family in families:
        group = groups.get(family.lower())
        if group is None:
            valid = ", ".join(sorted(groups))
            raise ValueError(
                f"{family!r} is not a statistic family for modality {modality!r}. "
                f"Valid families: {valid}. Families are groups, not individual statistics — "
                "declare `pixel` or `pixel_basic` rather than `pixel_mean`."
            )
        flags |= group
    return flags


def stat_names_for(flags: Flag) -> set[str]:
    """The statistic column names *flags* produces.

    Derived from the enum rather than listed here, so a statistic added upstream is picked
    up without an edit: every member is named ``<FAMILY>_<STATISTIC>`` and produces the
    lowercased second half.

    Only single-bit members are statistics; the multi-bit ones are the convenience groups
    (``PIXEL_BASIC``, ``HASH_DUPLICATES_D4``, ``NO_HASH``), which name no column. Iterating
    *flags* yields only the single-bit members it holds, so the groups never appear.
    """
    return {member.name.split("_", 1)[1].lower() for member in flags if member.name and "_" in member.name}


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

    Matched against *levels*, not by bare suffix: ``endswith("_brightness")`` would also
    claim a dataset-native ``camera_brightness``. A declaration matching nothing falls
    through unchanged, keeping a misspelled factor visible in ``unmatched_bin_requests``.
    A name declared exactly is never overwritten by another name's expansion, whichever comes first.
    """
    available = set(names)
    exact = set(declared) & available
    prefixes = tuple(f"{level}_" for level in levels)
    expanded: dict[str, Any] = {}
    for name, spec in declared.items():
        candidates = {name, *(prefix + name for prefix in prefixes)}
        hits = sorted(candidates & available)
        for target in hits or [name]:
            if target == name or target not in exact:
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
    for the box.  Each is then binned over its own population.  The names are returned
    because a policy declares bins on the bare statistic, and
    :func:`expand_declared_bins` needs to know what those became.

    The only arrays withheld are the hashes.  They travel in the same result — one
    ``compute_stats`` pass serves both outlier and duplicate detection — and are
    near-unique per image, so digitizing them would yield a category per item.

    Everything else is handed over as it comes, including the vector-valued statistics
    (``histogram``, ``percentiles``, ``center``).  Those cannot become factors;
    ``add_factors`` records them in :attr:`~dataeval.Metadata.dropped_factors`, so the
    metadata summary reports them as measured but not representable.

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
    Injection lives here because this is the one function every cached ``Metadata`` comes
    through, and it receives the policy.  It happens *after* construction, which is sound
    because binning is lazy: a bin declared before its factor exists still binds when the
    factor arrives.

    Corrections and roll-ups are **declared on the constructor** and not applied here.
    DataEval then owns their order — corrections before roll-ups, both before the factors
    are built — and ``repair`` replaces rather than accumulates, so a policy repairing one
    factor in YAML and another through its descriptor would keep only the first if they
    were applied here.

    What is left here is injection, which is flow's own, and the bin re-expansion that has
    to follow it.  Re-expansion comes last because everything before it changes the factor
    set — injection adds measured factors, repairing a held-back column turns it *into* a
    factor, and a roll-up adds one per output — and ``expand_declared_bins`` matches
    declarations against the names that actually exist.
    """
    from dataeval_flow._policy import ResolvedPolicy

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
    from dataeval_flow._cache import get_or_compute_stats
    from dataeval_flow._stats import ResolvedStatsPolicy, check_consumers, columns_for, restrict_columns

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
