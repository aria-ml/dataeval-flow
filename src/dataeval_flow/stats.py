"""The stats policy a run is given, resolved and checked before the data is read.

A stats policy says which statistics are measured over which views of the imagery, and
which of those views each consumer reads. A view is the whole image, a named band group,
the background, or a band group's background, and it is named by the prefix its columns
carry.

Resolve a policy before the dataset is walked, so a group the dataset does not declare or a
family a consumer needs and nothing measures costs a config error rather than an hour.
"""

__all__ = [
    "ResolvedStatsPolicy",
    "check_consumers",
    "columns_for",
    "measurable_in",
    "resolve_stats_policy",
    "restrict_columns",
    "stats_policy_for",
]

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dataeval.core import StatsResult
from dataeval.flags import ImageStats

if TYPE_CHECKING:
    from dataeval_flow.config import PipelineConfig

#: Families the background is measured for. Hash and dimension are computed for the image
#: and its boxes as usual and skipped for the background.
_BACKGROUND_FAMILIES = ImageStats.PIXEL | ImageStats.VISUAL

#: Families a named band group can carry. Geometry does not vary with a band subset.
_GROUP_FAMILIES = ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.HASH

#: Always emitted with the background, and named for no family.
_BACKGROUND_FRACTION = "background_fraction"


def _is_background(view: str | None) -> bool:
    """Whether *view* names the background, of the whole image or of one band group."""
    return view is not None and (view == "background" or view.startswith("background_"))


def measurable_in(view: str | None, flags: ImageStats) -> ImageStats:
    """Return the part of *flags* that *view* can carry at all.

    Use this before demanding a family of a view. Asking geometry of a band group or of the
    background names a column that is never produced, and treating that as a missing
    measurement would refuse a config that is asking for the only thing available.
    """
    if view is None:
        return flags
    if _is_background(view):
        return flags & _BACKGROUND_FAMILIES
    return flags & _GROUP_FAMILIES


def columns_for(views: "Iterable[str | None]", flags: ImageStats) -> set[str]:
    """Return the column names *flags* produces across *views*.

    A view's columns are its prefix and the statistic name, so the whole image gives bare
    names. `background_fraction` comes with the background view whatever families are
    named, because it describes the view rather than measuring anything.
    """
    from dataeval_flow.metadata import stat_names_for

    names = stat_names_for(flags)
    view_list = list(views)
    columns = {name if view is None else f"{view}_{name}" for view in view_list for name in names}
    if "background" in view_list:
        columns.add(_BACKGROUND_FRACTION)
    return columns


def restrict_columns(calc_result: "Mapping[str, Any]", allowed: "set[str]") -> "StatsResult":
    """Return *calc_result* holding only the columns in *allowed*.

    Call this before handing a result to anything that reads its columns. The cache returns
    the union of everything computed under one scope, so a consumer reading the result
    directly reads whatever else shares the entry.

    The structural fields are carried through unchanged: restricting columns does not
    change which rows they describe.
    """
    return StatsResult(
        source_index=calc_result["source_index"],
        object_count=calc_result["object_count"],
        invalid_box_count=calc_result["invalid_box_count"],
        image_count=calc_result["image_count"],
        stats={name: array for name, array in calc_result["stats"].items() if name in allowed},
    )


@dataclass(frozen=True)
class ResolvedStatsPolicy:
    """A stats policy with its families resolved and its bands taken from the dataset.

    Frozen because it is a cache key as much as an argument: the same policy has to key the
    same entry however it was spelled in the config.
    """

    name: str | None = None
    measure: tuple[tuple[str | None, ImageStats], ...] = ()
    """What to compute, one entry per band view. `None` is the whole image."""
    channels: tuple[tuple[str, tuple[int, ...]], ...] = ()
    """The band groups `measure` names, with the bands the dataset gave them.

    Only the named ones. A dataset may declare groups a policy does not measure, and
    `compute_stats` refuses a `stats` mapping whose keys are not exactly `channels`'.
    """
    background: bool = False
    outliers_from: tuple[str | None, ...] = (None,)
    factors_from: tuple[str | None, ...] = (None,)

    @classmethod
    def of_flags(cls, flags: ImageStats) -> "ResolvedStatsPolicy":
        """The policy a config declaring none implies: *flags* over the whole image.

        `compute_stats` treats `{None: flags}` with no channels as identical to `flags`, so
        this is the same call flow issues without a policy.
        """
        return cls(measure=((None, flags),))

    @property
    def request(self) -> dict[str | None, ImageStats]:
        """What to ask `compute_stats` for, keyed by view."""
        return dict(self.measure)

    @property
    def channel_map(self) -> dict[str, list[int]] | None:
        """What to pass `compute_stats` as `channels`, or None where no group is measured."""
        return {name: list(bands) for name, bands in self.channels} or None

    def families_of(self, view: str | None) -> ImageStats:
        """The families measured for *view*, limited to what that view can carry.

        Read this rather than `request` directly. A view cannot carry every family
        `measure` names it: geometry does not vary with a band subset, and the background
        is measured for pixel and visual only. Reporting a family the view cannot carry
        would name a column `compute_stats` never produces.
        """
        base = view
        if _is_background(view):
            base = None if view == "background" else str(view).removeprefix("background_")
        return measurable_in(view, self.request.get(base, ImageStats.NONE))

    def scope_fragment(self) -> str:
        """What decides whether two results can share a stats cache entry.

        Only the band definitions and `background`. `measure` and the consumer view sets
        change which columns are computed, not what any column holds, and the cache merges
        columns on a partial hit — folding them in would fragment the cache and buy nothing.
        Empty for a policy that declares neither, so a config without band groups keys
        exactly as it did before this field existed.
        """
        if not self.channels and not self.background:
            return ""
        return json.dumps(
            {"channels": [[name, list(bands)] for name, bands in self.channels], "background": self.background},
            sort_keys=True,
        )

    def factor_identity(self) -> dict[str, Any]:
        """What decides the set of factors injection produces.

        Read into `policy_key`. `outliers_from` is deliberately absent: it moves no factor,
        and including it would invalidate every metadata archive on an edit that cannot
        change one.
        """
        return {
            "measure": sorted(["~" if view is None else view, flags.value] for view, flags in self.measure),
            "channels": sorted([name, list(bands)] for name, bands in self.channels),
            "background": self.background,
            "factors_from": sorted("~" if view is None else view for view in self.factors_from),
        }


def resolve_stats_policy(
    params: Any,
    config: "PipelineConfig | None",
    channel_groups: "Mapping[str, tuple[int, ...]] | None",
) -> ResolvedStatsPolicy | None:
    """Resolve the stats policy *params* names, or None where it names none.

    Parameters
    ----------
    params : Any
        Workflow parameters, whose `stats` field names a policy in the pool.
    config : PipelineConfig or None
        The pipeline the policy pool lives on. Required only when a policy is named.
    channel_groups : Mapping[str, tuple[int, ...]] or None
        Band groups the datasets declare, which `measure` selects from.

    Raises
    ------
    ValueError
        When the reference names no policy, when no pipeline is available to resolve it
        against, when a `measure` entry names a group no dataset declares, or when a family
        name is not a family.
    """
    from dataeval_flow.metadata import resolve_families
    from dataeval_flow.workflow.orchestrator import _resolve_by_name

    name: str | None = getattr(params, "stats", None)
    if name is None:
        return None
    if config is None:
        raise ValueError(
            f"This workflow references stats policy {name!r}, which can only be resolved against a pipeline config.",
        )

    declared = _resolve_by_name(config.stats, name, "stats policy")
    available = dict(channel_groups or {})

    measure: list[tuple[str | None, ImageStats]] = []
    selected: dict[str, tuple[int, ...]] = {}
    for entry in declared.measure:
        try:
            flags = resolve_families("image", list(entry.families))
        except ValueError as exc:
            raise ValueError(f"Stats policy {name!r} {exc}") from exc
        measure.append((entry.bands, ImageStats(flags)))
        if entry.bands is not None:
            if entry.bands not in available:
                known = ", ".join(sorted(available)) or "none"
                raise ValueError(
                    f"Stats policy {name!r} measures channel group {entry.bands!r}, but "
                    f"this workflow's dataset does not declare a channel group "
                    f"{entry.bands!r}. Groups it declares: {known}. Declare the group under "
                    "the dataset's `channel_groups`, or drop the `measure` entry.",
                )
            selected[entry.bands] = tuple(available[entry.bands])

    return ResolvedStatsPolicy(
        name=name,
        measure=tuple(measure),
        channels=tuple(sorted(selected.items())),
        background=declared.background,
        outliers_from=tuple(declared.outliers_from),
        factors_from=tuple(declared.factors_from),
    )


def stats_policy_for(
    context: Any,
    *,
    outlier_flags: ImageStats = ImageStats.NONE,
    duplicate_flags: ImageStats = ImageStats.NONE,
    factor_flags: ImageStats = ImageStats.NONE,
) -> ResolvedStatsPolicy:
    """The stats policy a caller should read, however it was invoked.

    Pass the flags this caller will actually consume, one argument per consumer. A declared
    policy is checked against them here, before the dataset is walked: `measure` is a
    complete statement, so a family the caller reads and no entry measures would otherwise
    surface as an empty column set. Without a declared policy the union of the three becomes
    the request over the whole image, which is the call flow issues today.

    The orchestrator resolves a named policy up front and puts it on the context, because
    resolving one needs the pipeline the pool lives on and the datasets that declare the
    bands — neither of which a workflow has. `execute(context, params)` is also a supported
    entry point on its own, and a context built by hand carries none.
    """
    resolved = getattr(context, "stats_policy", None)
    if resolved is None:
        return ResolvedStatsPolicy.of_flags(outlier_flags | duplicate_flags | factor_flags)
    check_consumers(
        resolved,
        outlier_flags=outlier_flags,
        duplicate_flags=duplicate_flags,
        factor_flags=factor_flags,
    )
    return resolved


def check_consumers(
    policy: ResolvedStatsPolicy,
    *,
    outlier_flags: ImageStats,
    duplicate_flags: ImageStats,
    factor_flags: ImageStats,
) -> None:
    """Refuse a policy whose `measure` does not produce what a consumer reads.

    `measure` is a complete statement, so a family a consumer needs and no entry measures is
    computed nowhere and that consumer reads an empty column set. Name the entry to add.

    Duplicate detection is checked against the whole image alone: DataEval reads the bare
    hash names and cannot see a prefixed one.
    """
    _check_views("outlier_flags", policy, policy.outliers_from, outlier_flags)
    _check_views("intrinsic_factors", policy, policy.factors_from, factor_flags)
    _check_views("duplicate_flags", policy, (None,), duplicate_flags)


def _check_views(
    declaration: str,
    policy: ResolvedStatsPolicy,
    views: "Sequence[str | None]",
    required: ImageStats,
) -> None:
    """Refuse where one consumer's views do not measure what it needs."""
    for view in views:
        wanted = columns_for([view], measurable_in(view, required))
        have = columns_for([view], policy.families_of(view))
        missing = sorted(wanted - have)
        if missing:
            spelled = "~" if view is None else view
            raise ValueError(
                f"Stats policy {policy.name!r} does not measure {', '.join(missing)}, which "
                f"`{declaration}` asks for on view {spelled!r}. `measure` is a complete "
                f"statement, so add those families to its `{{bands: {spelled}}}` entry, or "
                "stop asking for them.",
            )
