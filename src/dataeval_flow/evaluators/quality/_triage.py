"""``factor-triage``: what a Dataset's metadata failed to read, and a policy that repairs it (spec §10.10).

Flow's own evaluator over DataEval's ``Metadata``: it describes how each factor was read, finds what the run could not
read as configured (``dataeval_flow._triage``), and, with ``verify``, reads the metadata back under the suggestions,
which ``Metadata.repair`` does without a second walk. It also recommends a policy: the suggestions completed, plus a
pin for every factor the run left unpinned, read back under them.
"""

__all__ = ["FactorTriageEvaluator", "describe", "places", "read_back", "verify"]

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, ClassVar

import polars as pl
from dataeval import Metadata

from dataeval_flow._binning import describe_binning
from dataeval_flow._blocks import ItemRef
from dataeval_flow._input_spec import InputKind
from dataeval_flow._metadata import expand_declared_bins
from dataeval_flow._policy import ResolvedPolicy, build_correction
from dataeval_flow._recommend import complete_stanza, recommend, render_recommendation
from dataeval_flow._tables import GROUP_SHOWN
from dataeval_flow._triage import TriageFinding, find_issues, incomplete_factors, render_stanza, to_policy_stanza
from dataeval_flow._triage_report import Places, minority_kind, summarize
from dataeval_flow.evaluators._core import execution
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.quality._config import FactorTriageConfig
from dataeval_flow.evaluators.quality._result import FactorTriageOutput, VerificationEntry

_logger: logging.Logger = logging.getLogger(__name__)


class FactorTriageEvaluator(Evaluator[FactorTriageConfig, FactorTriageOutput]):
    """``factor-triage``: what a Dataset's metadata failed to read, the policy stanza that repairs it, and what the
    repair recovers."""

    name: ClassVar[str] = "factor-triage"
    title: ClassVar[str] = "Factor Triage"
    description: ClassVar[str] = "What a Dataset's metadata failed to read, and a policy that repairs it"
    dataeval_class: ClassVar[Any] = Metadata
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.METADATA: "repair"}

    def run(self, config: FactorTriageConfig, inputs: Sequence[EvaluatorInputs]) -> FactorTriageOutput:
        """Describe the source's metadata, find what it failed to read, suggest repairs, verify them, and recommend a
        policy."""
        (source,) = inputs
        metadata = require(source.metadata, "metadata", source.source)
        policy = source.metadata_policy or ResolvedPolicy()
        started, clock = datetime.now(UTC), time.monotonic()
        record = describe(metadata, policy)
        findings = find_issues(
            record, min_missing_fraction=config.min_missing_fraction, default_bins=config.default_bins
        )
        stanza = to_policy_stanza(findings)
        verification: list[VerificationEntry] = []
        error: str | None = None
        if config.verify:
            try:
                verification = verify(metadata, policy, findings)
            except Exception as e:  # the findings are worth having without it
                _logger.warning("Verification unavailable", exc_info=True)
                error = str(e) or type(e).__name__
        recommended, recommended_yaml, recommendation_error = _recommendation(
            metadata, policy, findings, config.metadata
        )
        data = {
            "findings": findings,
            "suggested_policy": stanza,
            "suggested_policy_yaml": render_stanza(stanza, incomplete=incomplete_factors(findings)),
            "verification": verification,
            "verification_error": error,
            "recommended_policy": recommended,
            "recommended_policy_yaml": recommended_yaml,
            "recommendation_error": recommendation_error,
            "counts": summarize(findings),
            "factor_count": len(record.get("factors") or {}),
            "places": places(metadata, findings, source.source),
        }
        return FactorTriageOutput(data, execution("dataeval_flow.factor_triage", started, time.monotonic() - clock, {}))


def describe(metadata: Any, policy: Any) -> dict[str, Any]:
    """The binning record, built the way ``attach_binning`` builds it.

    The bins as applied, not as spelled: ``unmatched_bin_requests`` is a set difference
    against the factor names, and a bare declared name is not one of those.
    """
    declared = dict(policy.continuous_factor_bins) or None
    requested = expand_declared_bins(declared, metadata.factor_names, metadata.levels) if declared else None
    return describe_binning(
        metadata,
        excluded=list(policy.exclude) or None,
        requested_bins=requested,
        factor_source=policy.factor_source,
        declared_bins=declared,
    )


def read_back(metadata: Any, policy: Any, stanza: Mapping[str, Any]) -> dict[str, Any]:
    """The binning record under the policy plus `stanza`'s corrections and excludes: what a recommendation pins.

    Always read off the copy ``repair`` returns, never off `metadata` itself: the excludes are set on it, and setting
    them on the Metadata the chain holds would change what every later step reads. ``repair`` reuses the values the
    walk kept, so this costs no second walk, and it replaces rather than accumulates corrections, so the policy's own
    are passed through beside the stanza's.
    """
    from dataeval_flow.config._schemas import MetadataPolicyConfig

    validated = MetadataPolicyConfig.model_validate(
        {"name": "_recommended", "corrections": list(stanza.get("corrections") or ())}
    )
    built = [build_correction(entry) for entry in validated.corrections or ()]
    repaired = metadata.repair([*policy.correction_specs, *built])
    excluded = tuple(dict.fromkeys([*policy.exclude, *(stanza.get("exclude") or ())]))
    if excluded != tuple(policy.exclude):
        repaired.exclude = list(excluded)
    return describe(repaired, replace(policy, exclude=excluded))


def _recommendation(
    metadata: Any, policy: Any, findings: Sequence[TriageFinding], policy_name: str | None
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """The recommended policy, its YAML, and why it could not be made, or ``None`` for each that does not apply.

    The findings are worth having without it, so a read-back that raises costs the recommendation and says why,
    as a verification that raises does.

    A ``floor_mass`` suggestion awaiting an answer is left out of what the recommendation completes: its value may be
    a reading, such as a speed of zero, and dropping it could delete a large share of genuine rows. Only a value
    triage could not read, a mixed column's string, is dropped. The left-out values are passed to the render, which
    says to decide.
    """
    try:
        held: dict[str, list[Any]] = {}
        kept: list[TriageFinding] = []
        for finding in findings:
            if finding.category == "floor_mass" and finding.suggestion and not finding.suggestion.complete:
                for correction in finding.suggestion.corrections:
                    held.setdefault(correction["factor"], []).extend(r["match"] for r in correction["rules"])
            else:
                kept.append(finding)
        completed, dropped = complete_stanza(to_policy_stanza(kept))
        after = read_back(metadata, policy, completed)
        recommended = recommend(after, completed, skip=set(policy.encoding or {}))
        if recommended is None:
            return None, None, None
        text = render_recommendation(
            recommended, after, dropped, held=held, name=policy_name or "standard", merge_into=policy_name
        )
    except Exception as e:  # the findings are worth having without it
        _logger.warning("Recommendation unavailable", exc_info=True)
        return None, None, str(e) or type(e).__name__
    return recommended, text, None


def verify(
    metadata: Any,
    policy: Any,
    findings: "list[TriageFinding]",
) -> "list[VerificationEntry]":
    """Read the metadata back under the complete suggestions, and say what they recovered.

    Costs no second dataset walk: ``repair`` applies corrections to the values the walk
    already kept and returns a derived copy sharing the immutable store.  The policy's
    own corrections are passed through alongside the suggested ones because ``repair``
    **replaces** rather than accumulates.

    A correction and a bin suggestion claim different things, so ``recovered`` is
    checked differently for each — see :func:`_factor_recovered`, which the shape of
    each finding's suggestion (corrections vs. a bin count) routes to the right check.

    Incomplete suggestions are never applied, so this can never report recovery from a
    placeholder.  A suggestion can also be well-formed, run cleanly and still not
    recover what it claims — which is the case this exists to catch before a stanza is
    committed to a config.
    """
    from dataeval_flow.config._schemas import MetadataPolicyConfig

    entries: list[VerificationEntry] = []
    runnable: list[Any] = []
    bins: dict[str, Any] = {}
    correction_factors: set[str] = set()
    bin_factors: set[str] = set()
    for finding in findings:
        suggestion = finding.suggestion
        if suggestion is None:
            continue
        if not suggestion.complete:
            entries.append(_unapplied(finding))
            continue
        factor_bins = suggestion.policy.get("continuous_factor_bins") or {}
        bins.update(factor_bins)
        bin_factors.update(factor_bins)
        runnable.extend(suggestion.corrections)
        correction_factors.update(c["factor"] for c in suggestion.corrections)

    if not runnable and not bins:
        return entries

    # Validated as a policy first, so a malformed suggestion fails here naming the field
    # rather than inside DataEval naming a constructor argument.
    validated = MetadataPolicyConfig.model_validate({"name": "_triage", "corrections": runnable})
    built = [build_correction(entry) for entry in validated.corrections or ()]
    repaired = metadata.repair([*policy.correction_specs, *built])
    described_policy = policy
    if bins:
        merged_bins = {**dict(policy.continuous_factor_bins), **bins}
        repaired.continuous_factor_bins = merged_bins
        # `describe` reads `continuous_factor_bins` off the policy, not off `repaired`,
        # so it has to see the bins actually assigned here — otherwise the re-described
        # record's `requested_bins`/`bin_expansion` would describe the policy this
        # verification started from rather than what it just ran.
        described_policy = replace(policy, continuous_factor_bins=merged_bins)
    after = describe(repaired, described_policy)
    factors = after.get("factors") or {}

    for name in sorted(correction_factors | bin_factors):
        # A name a correction also names wins the correction check: that is the more
        # fundamental claim (the column exists at all), and no finding actually
        # produces both for one factor today.
        pinned = name in bin_factors and name not in correction_factors
        recovered = _factor_recovered(after, name, pinned=pinned)
        entries.append(
            VerificationEntry(
                factor=name,
                applied=True,
                recovered=recovered,
                detail=_recovery_detail(factors.get(name), recovered, pinned=pinned),
            )
        )
    return entries


def places(metadata: Any, findings: "list[TriageFinding]", source: str) -> Places:
    """Where each mixed column's problem values sit: its minority kind's values, most rows first, with their items.

    Up to eight items per value, each named in *source*, or by its box where the column sits below the item.
    A column dropped for naming its rows has none: every value is distinct, and none is a problem.
    """
    places: dict[str, list[tuple[str, int, list[ItemRef]]]] = {}
    for finding in findings:
        kind = minority_kind(finding.detail.get("counts") or {})
        if finding.category != "unreadable" or not finding.repairable or kind is None:
            continue
        if "cardinality_over_budget" in finding.reasons:
            continue
        try:
            frame = metadata.unusable_rows(finding.factor).filter(pl.col("kind") == kind)
        except Exception as error:  # noqa: BLE001 - the places are a sample; failing to find them never costs the run
            _logger.warning(
                "Could not find where %r's problem values sit, so none are pictured: %s", finding.factor, error
            )
            continue
        keys = ["item_index", *(["target_index"] if "target_index" in frame.columns else [])]
        values = frame.group_by("value", maintain_order=True).agg(
            pl.len().alias("count"), *(pl.col(key).head(GROUP_SHOWN) for key in keys)
        )
        places[finding.factor] = [
            (
                row["value"],
                row["count"],
                [
                    ItemRef(source=source, index=index, target=target)
                    for index, target in zip(
                        row["item_index"], row.get("target_index") or [None] * len(row["item_index"]), strict=True
                    )
                ],
            )
            for row in values.sort("count", descending=True, maintain_order=True).iter_rows(named=True)
        ]
    return places


def _unapplied(finding: "TriageFinding") -> "VerificationEntry":
    """A verification entry for a suggestion left incomplete, never applied.

    Pulled out of `verify` so that method does not trip C901
    — this is the one branch that does not touch the dataset at all.
    """
    suggestion = finding.suggestion
    corrections = suggestion.corrections if suggestion is not None else ()
    holes = sum(1 for c in corrections for rule in c.get("rules", ()) if rule.get("to") is None)
    return VerificationEntry(
        factor=finding.factor,
        applied=False,
        recovered=False,
        detail=f"not applied; {holes} values still need codes",
    )


def _factor_recovered(after: "dict[str, Any]", name: str, *, pinned: bool) -> bool:
    """Whether one factor's suggestion actually did what it claimed.

    A correction (an ``unreadable`` finding) claims a held-back column becomes a factor:
    recovered means present in ``factors`` and absent from ``unusable``.

    A bin suggestion (an ``unbinned`` finding) is only ever raised for a factor already
    present and already readable — what it lacks is a pinned cut, not existence, per
    ``triage._encodings`` — so that same check would be true before the suggestion runs
    and after it regardless of what the suggested count did, and could never say no.
    Recovered there instead means the factor is still present *and* its cut no longer reads
    ``provenance="derived"`` in the re-described record — presence alone is not enough, or a
    factor that vanished from the re-described record entirely (absent from ``factors``, and
    so absent from ``unreviewed`` too) would satisfy this by having disappeared rather than
    by having been pinned.
    """
    factors = after.get("factors") or {}
    if pinned:
        return name in factors and name not in (after.get("unreviewed") or ())
    unusable = after.get("unusable") or {}
    return name in factors and name not in unusable


def _recovery_detail(info: "dict[str, Any] | None", recovered: bool, *, pinned: bool) -> str:
    """One line saying what the reading actually produced.

    Worded for whichever claim :func:`_factor_recovered` checked: a bin suggestion reports
    the cut it produced, a correction reports whether the column became one at all.
    """
    fit = (info or {}).get("fit") or {}
    bin_buckets = fit.get("bins")
    buckets = bin_buckets if bin_buckets is not None else fit.get("levels") or []
    kind = "bins" if bin_buckets is not None else "levels"
    if pinned:
        if not recovered:
            return "applied, but bin cut remains derived rather than pinned"
        return f"{len(buckets)} {kind}, {len(fit.get('empty') or ())} empty"
    if not recovered:
        return "applied, but factor remains unreadable"
    return f"became a factor, {len(buckets)} {kind}"
