"""Metadata convenience builder wrapping DataEval."""

__all__ = ["build_metadata", "expand_declared_bins", "resolve_families", "stat_names_for"]

from collections.abc import Iterable, Mapping, Sequence
from enum import Flag
from typing import TYPE_CHECKING, Any

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


def resolve_families(modality: str, families: Sequence[str]) -> Any:
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
        flags |= getattr(enum, key)
    return flags


def stat_names_for(flags: Any) -> set[str]:
    """The statistic column names *flags* produces.

    Derived from the enum rather than listed here, so a statistic added upstream is picked
    up without an edit: every member is named ``<FAMILY>_<STATISTIC>`` and produces the
    lowercased second half.
    """
    return {
        member.name.split("_", 1)[1].lower()
        for member in type(flags)
        if member.name and "_" in member.name and member & flags
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
        DataEval Metadata instance.
    """
    from dataeval_flow.policy import ResolvedPolicy

    return Metadata(dataset, **(policy or ResolvedPolicy()).metadata_kwargs())
