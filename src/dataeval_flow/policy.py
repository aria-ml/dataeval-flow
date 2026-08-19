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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    factor_source: "FactorSource | None" = None
    reference_split: str | None = None

    def metadata_kwargs(self) -> dict[str, Any]:
        """What to hand :class:`dataeval.Metadata`, omitting anything left unset.

        Omitted rather than passed as None so that DataEval's own defaults apply, and so
        that a release without one of these arguments still works with a policy that does
        not use it.
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
        if self.factor_levels:
            kwargs["factor_levels"] = {name: list(levels) for name, levels in self.factor_levels.items()}
        if self.strict:
            kwargs["strict"] = True
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
            "factor_levels": {name: list(levels) for name, levels in (policy.factor_levels or {}).items()},
            "strict": policy.strict,
        },
        sort_keys=True,
        default=str,
    )


def _read_descriptor(path: Path, source: str) -> Mapping[str, Any]:
    """Read a committed descriptor, saying which config entry sent us here when it fails."""
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
    return factors


def _check_no_double_declaration(factors: Mapping[str, Any], bins: Mapping[str, Any], source: str) -> None:
    """Refuse a factor declared through both channels rather than picking one."""
    if both := sorted(set(factors) & set(bins)):
        raise ValueError(
            f"{source} declares {both} through both `encoding` and `continuous_factor_bins`. "
            "Two sources disagreeing about one factor has no good resolution — drop it from "
            "one of them.",
        )


def _check_strict_is_earned(factors: Mapping[str, Any], strict: bool, source: str) -> None:
    """Refuse to close a vocabulary nobody has reviewed.

    ``strict`` does not consult provenance: it is applied to any recorded vocabulary,
    including one DataEval derived from a draw and nobody looked at.  So the tempting rule
    — *a configured descriptor means a closed taxonomy* — would enforce a vocabulary nobody
    decided on and fail the run on the first new category, with an error calling it a
    "declared vocabulary" when nothing was declared.  The check goes the other way, and
    turns ``strict`` from a setting that is dangerous to default into one that is safe to
    set deliberately.
    """
    if not strict:
        return
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


def _is_set(params: "MetadataConfigMixin", name: str) -> bool:
    """Whether a legacy field carries something, treating an empty list as unset."""
    value = getattr(params, name, None)
    return value is not None and value != [] and value != {}


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
        factor_source, reference_split = named.factor_source, named.reference_split
        descriptor_path = named.encoding
    else:
        source = "This workflow"
        auto_bin_method = params.metadata_auto_bin_method
        exclude = tuple(params.metadata_exclude or ())
        bins = dict(params.metadata_continuous_factor_bins or {})
        factor_levels, strict = None, False
        factor_source, reference_split = params.metadata_factor_source, None
        descriptor_path = None

    factors: Mapping[str, Any] | None = None
    resolved_path: Path | None = None
    if descriptor_path:
        resolved_path = resolve_path(descriptor_path, data_dir)
        factors = _read_descriptor(resolved_path, source)
        _check_no_double_declaration(factors, bins, source)
        _check_strict_is_earned(factors, strict, source)
        _logger.info("Applying encoding descriptor %s (%d factors)", descriptor_path, len(factors))

    return ResolvedPolicy(
        auto_bin_method=auto_bin_method,
        exclude=exclude,
        continuous_factor_bins=bins,
        encoding_path=resolved_path,
        encoding=factors,
        factor_levels=factor_levels,
        strict=strict,
        factor_source=factor_source,
        reference_split=reference_split,
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
    return replace(
        policy,
        encoding_specs=specs,
        encoding=dict(descriptor or {}),
        encoding_path=None,
        # Subsumed by the records above, which already say where every declared cut fell.
        # Passing both is what DataEval refuses per factor, and the records are the
        # resolved form of exactly these requests.
        continuous_factor_bins={},
        factor_levels=None,
    )
