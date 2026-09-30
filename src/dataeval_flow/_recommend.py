"""The recommended policy: every factor the policy left unpinned, pinned as this data reads, beside the suggested fixes.

Pure over the binning record and the suggested stanza, as :mod:`dataeval_flow._triage` is: nothing here imports
``dataeval``. The read-back that produces the record is
:func:`dataeval_flow.evaluators.quality._triage.read_back`.
"""

__all__ = ["CAVEAT", "complete_stanza", "pinned_count", "recommend", "render_recommendation"]

import copy
import math
import textwrap
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import yaml

CAVEAT = (
    "Every cut and vocabulary below was read from this data. Pinning them keeps later runs comparable, and it also "
    "commits them to this data's assumptions. If this data does not represent the data you expect, results computed "
    "under this policy can be invalid or misleading. Review each factor before you commit it."
)

# Provenances nobody decided: DataEval chose the cut or the vocabulary, or placed a declared count's edges.
_UNPINNED = frozenset({"derived", "count"})

# Stands in for a completed placeholder while the stanza is dumped, so its line can take a comment no YAML writer
# emits. A null byte cannot appear in a factor's values, and YAML escapes it, so the dumped marker is unambiguous.
_DROP_MARK = "\x00DROPPED{}\x00"

# Wide enough that a pin's edges or levels never wrap, so each factor is one line its comment can follow.
_WIDTH = 1_000_000


def complete_stanza(stanza: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """The suggested stanza with every placeholder answered by a drop, and, by factor, the values it dropped.

    A placeholder is a remap rule whose ``to`` is ``None``. ``None`` is not a drop: ``Remap`` reads it as the
    catch-all *key*, and as a target it leaves the column mixed. So each becomes NaN, the target that means "no
    reading", which a sentinel's rule already is. A rule already NaN was answered by triage and is not listed.
    """
    completed = copy.deepcopy(dict(stanza))
    dropped: dict[str, list[Any]] = {}
    for correction in completed.get("corrections") or ():
        for rule in correction.get("rules") or ():
            if rule.get("to") is None:
                rule["to"] = math.nan
                dropped.setdefault(correction["factor"], []).append(rule["match"])
    return completed, dropped


def recommend(
    record: Mapping[str, Any], completed: Mapping[str, Any], *, skip: Collection[str] = ()
) -> dict[str, Any] | None:
    """The completed stanza plus a pin for each factor `record` holds unpinned, or ``None`` where there is none to add.

    A cut is pinned by its edges in ``continuous_factor_bins``, replacing any bin count the stanza suggested for it,
    and a vocabulary by its levels in ``factor_levels``. `skip` names the factors the policy's descriptor pins: they
    are never named again, whatever their provenance and not even as a bin count the stanza suggested, since an
    exported descriptor still says ``derived`` and a factor named by both the descriptor and
    ``continuous_factor_bins`` is refused.
    """
    edges: dict[str, list[float]] = {}
    levels: dict[str, list[Any]] = {}
    for name, info in sorted((record.get("factors") or {}).items()):
        encoding = info.get("encoding") or {}
        if name in skip or encoding.get("provenance") not in _UNPINNED:
            continue
        if encoding.get("kind") == "levels":
            levels[name] = list(encoding.get("levels") or ())
        else:
            edges[name] = [float(edge) for edge in encoding.get("edges") or ()]
    counts = {k: v for k, v in (completed.get("continuous_factor_bins") or {}).items() if k not in skip}
    kept = {k: v for k, v in completed.items() if k != "continuous_factor_bins"}
    if not edges and not levels and not counts and not kept:
        return None
    stanza = dict(kept)
    if bins := {**counts, **edges}:
        stanza["continuous_factor_bins"] = bins
    if levels:
        stanza["factor_levels"] = levels
    return stanza


def pinned_count(stanza: Mapping[str, Any]) -> int:
    """How many factors `stanza` pins: explicit edges and vocabularies, not bin counts."""
    bins = stanza.get("continuous_factor_bins") or {}
    return sum(1 for value in bins.values() if not isinstance(value, int)) + len(stanza.get("factor_levels") or {})


def render_recommendation(
    stanza: Mapping[str, Any],
    record: Mapping[str, Any],
    dropped: Mapping[str, Sequence[Any]],
    *,
    name: str = "standard",
    merge_into: str | None = None,
) -> str:
    """The recommendation as a YAML block to paste under ``metadata:``, headed by the caveat, commented per factor.

    Dumped with ``yaml.safe_dump``, so every value reads back as written, edges exactly and ``.inf`` and ``.nan``
    included, then commented line by line, since no YAML writer emits comments:

    - a pinned cut says the span this data covered, beyond which rows join the open outer bins;
    - a cut placed from a declared count says it was one;
    - a vocabulary says how many levels this data held;
    - a value dropped by default says so, where a sentinel's rule, answered by triage, says nothing;
    - a factor the record holds but cannot pin, with no encoding or still held back, is named at the end.
    """
    factors = record.get("factors") or {}
    pins = ("continuous_factor_bins", "factor_levels")
    body = copy.deepcopy({key: value for key, value in stanza.items() if key not in pins})
    notes = _mark_drops(body, dropped)
    text = yaml.safe_dump({"metadata": [{"name": name, **body}]}, sort_keys=False, width=_WIDTH)
    for mark, replacement in notes.items():
        text = text.replace(yaml.safe_dump(mark, width=_WIDTH).splitlines()[0], replacement)

    lines = [f"# {line}" for line in textwrap.wrap(CAVEAT, 116)]
    if merge_into:
        lines.append(f"# Merge these into your policy '{merge_into}'.")
    lines.extend(text.rstrip("\n").splitlines())
    for key in pins:
        if entries := stanza.get(key):
            lines.append(f"  {key}:")
            for factor, value in entries.items():
                lines.append(f"    {_pin_line(factor, value, factors.get(factor) or {})}")
    excluded = set(record.get("excluded") or ())
    unpinnable = {factor for factor, info in factors.items() if not info.get("encoding")}
    lines.extend(
        f"  # not pinned: {factor} has no encoding to read"
        for factor in sorted((unpinnable | set(record.get("unusable") or ())) - excluded)
    )
    return "\n".join(lines) + "\n"


def _mark_drops(body: dict[str, Any], dropped: Mapping[str, Sequence[Any]]) -> dict[str, str]:
    """Swap each dropped rule's NaN in `body` for a marker, and say what each marker's line should read instead."""
    notes: dict[str, str] = {}
    for correction in body.get("corrections") or ():
        for rule in correction.get("rules") or ():
            to = rule.get("to")
            if not (isinstance(to, float) and math.isnan(to)):
                continue
            if rule.get("match") in dropped.get(correction["factor"], ()):
                mark = _DROP_MARK.format(len(notes))
                notes[mark] = f".nan  # dropped by default: decide what '{rule['match']}' means"
                rule["to"] = mark
    return notes


def _pin_line(factor: str, value: Any, info: Mapping[str, Any]) -> str:
    """One pin as a YAML line, with what it assumes of this data where it assumes anything."""
    line = yaml.safe_dump({factor: value}, default_flow_style=None, sort_keys=False, width=_WIDTH).rstrip("\n")
    encoding = info.get("encoding") or {}
    if isinstance(value, int):
        return line
    if encoding.get("kind") == "levels":
        return f"{line}  # {len(value)} levels seen; a category this data never saw takes a new code"
    if encoding.get("provenance") == "count":
        return f"{line}  # was a count of {len(value) - 1}; these are the edges DataEval placed from this data"
    spans = [bucket for bucket in (info.get("fit") or {}).get("bins") or () if "min" in bucket and "max" in bucket]
    if not spans:
        return line
    low, high = min(b["min"] for b in spans), max(b["max"] for b in spans)
    return f"{line}  # seen {low:g} to {high:g}; the outer bins are open, so values beyond this range join them"
