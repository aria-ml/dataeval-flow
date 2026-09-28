"""Findings builders for the OOD detection workflow."""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Literal

from dataeval_flow._blocks import Block, Cell, Column, Fields, ItemRef, Paragraph, Section, Table
from dataeval_flow.workflows._base import Finding
from dataeval_flow.workflows._tables import ranked_table, table_limits
from dataeval_flow.workflows.ood_detection._config import (
    OODDetectionConfig,
    OODDetectionHealthThresholds,
)
from dataeval_flow.workflows.ood_detection._outputs import (
    DetectorOODResultDict,
    FactorDeviationDict,
    OODDetectionRawOutput,
)


def locator(parts: Sequence[tuple[str, int]]) -> Callable[[int], ItemRef]:
    """Where an index into the test sources, joined end to end in order, falls: its source, and its index there.

    *parts* are each test source's name and length, in the order the run joined them.
    """
    starts = list(itertools.accumulate((length for _, length in parts), initial=0))

    def locate(index: int) -> ItemRef:
        at = bisect.bisect_right(starts, index) - 1
        return ItemRef(source=parts[at][0], index=index - starts[at])

    return locate


def _samples_blocks(
    indices: Collection[int],
    normalized_scores: Mapping[int, float],
    locate: Callable[[int], ItemRef],
    factors: Mapping[int, str] | None = None,
) -> list[Block]:
    """The samples, most out of distribution first: each one's thumbnail, item, source and score, and its factors.

    At most 500, with a paragraph counting the rest.
    """
    ranked = sorted(indices, key=lambda index: (-normalized_scores.get(index, 0.0), index))
    if not ranked:
        return []
    limits = table_limits()
    rows: list[dict[str, Cell]] = []
    for index in ranked[: limits.rows]:
        ref = locate(index)
        row: dict[str, Cell] = {
            "image": ref,
            "item": ref.index,
            "source": ref.source,
            "score": normalized_scores.get(index, 0.0),
        }
        if factors is not None:
            row["factors"] = factors.get(index, "")
        rows.append(row)
    columns = [
        Column(key="image", kind="image"),
        Column(key="item", header="Item"),
        Column(key="source", header="Source", align="left"),
        Column(key="score", header="Score", format="{:.2f}x"),
        *([Column(key="factors", header="Top factors", align="left")] if factors is not None else []),
    ]
    blocks: list[Block] = [Table(columns=columns, rows=rows, preview=limits.preview)]
    if limits.rows is not None and len(ranked) > limits.rows:
        blocks.append(
            Paragraph(
                text=f"{len(ranked):,} samples; the {limits.rows:,} most out of distribution are listed, and every one "
                "is in `output.raw`."
            )
        )
    return blocks


def _severity_for_ood(
    ood_pct: float,
    thresholds: OODDetectionHealthThresholds,
) -> Literal["ok", "info", "warning"]:
    """Determine severity for an OOD detector result based on OOD percentage."""
    if ood_pct >= thresholds.ood_pct_warning:
        return "warning"
    if ood_pct >= thresholds.ood_pct_info:
        return "info"
    return "ok"


def _score_histogram_blocks(det_result: DetectorOODResultDict, n_bins: int = 10) -> list[Block]:
    """A detector's score distribution: in-dist and OOD counts per bin, the threshold's bin marked."""
    samples = det_result.get("samples", [])
    if not samples:
        return []

    in_scores = [s["score"] for s in samples if not s["is_ood"]]
    ood_scores = [s["score"] for s in samples if s["is_ood"]]
    all_scores = [s["score"] for s in samples]

    lo = min(all_scores)
    hi = max(all_scores)
    if hi == lo:
        return [Paragraph(text=f"All scores = {lo:.4f}")]

    bin_w = (hi - lo) / n_bins
    threshold = det_result["threshold_score"]

    rows: list[dict[str, Cell]] = []
    for i in range(n_bins):
        b_lo = lo + i * bin_w
        b_hi = b_lo + bin_w
        ic = sum(1 for s in in_scores if (b_lo <= s < b_hi) or (i == n_bins - 1 and s == b_hi))
        oc = sum(1 for s in ood_scores if (b_lo <= s < b_hi) or (i == n_bins - 1 and s == b_hi))
        # The threshold's bin is found by the bounds as printed, to three places.
        at_threshold = float(f"{b_lo:.3f}") <= threshold < float(f"{b_hi:.3f}")
        rows.append(
            {
                "range": f"{b_lo:.3f}-{b_hi:.3f}",
                "in": ic,
                "ood": oc,
                "bar": [ic, oc],
                "marker": "\u2190 threshold" if at_threshold else "",
            }
        )

    columns = [
        Column(key="range", header="Range", align="right"),
        Column(key="in", header="In"),
        Column(key="ood", header="OOD"),
        Column(key="bar", kind="stacked", series=["in-dist", "OOD"]),
        Column(key="marker", align="left"),
    ]
    return [Table(columns=columns, rows=rows)]


def _build_detector_finding(
    name: str,
    result: DetectorOODResultDict,
    thresholds: OODDetectionHealthThresholds,
) -> Finding:
    """Build a finding for a single OOD detector."""
    ood_pct = result["ood_percentage"]
    severity = _severity_for_ood(ood_pct, thresholds)

    counts = Fields(
        items=[
            ("OOD count", result["ood_count"]),
            ("Total count", result["total_count"]),
            ("OOD percentage", f"{ood_pct:.1f}%"),
            ("Threshold score", round(result["threshold_score"], 6)),
        ]
    )

    description = f"{name}: {result['ood_count']}/{result['total_count']} samples OOD ({ood_pct:.1f}%)"

    return Finding(
        severity=severity,
        title=name,
        description=description,
        blocks=[*_score_histogram_blocks(result), counts],
    )


def _build_factor_predictors_finding(
    predictors: dict[str, float],
) -> Finding:
    """Build a table finding showing mutual information per factor."""
    table = ranked_table({k: round(v, 4) for k, v in predictors.items()}, headers=("Factor", "MI (bits)"))

    return Finding(
        severity="info",
        title="OOD Factor Predictors",
        description="Mutual information between metadata factors and OOD status (higher = stronger association)",
        blocks=[table],
    )


def _compute_normalized_scores(
    detectors: dict[str, DetectorOODResultDict],
) -> tuple[dict[int, float], set[int], dict[str, set[int]]]:
    """Normalize OOD scores across detectors and find mutually agreed OOD samples.

    Scores are normalized by dividing by the detector's threshold, giving a
    unitless ratio where 1.0 = at threshold and >1.0 = OOD.  This makes scores
    comparable across detectors with different scales (e.g. distance-based
    KNeighbors vs probability-based DomainClassifier).

    Returns
    -------
    normalized_scores
        Mapping of sample index to mean normalized score across all detectors.
    mutual_ood
        Set of sample indices flagged as OOD by *every* detector.
    unique_ood
        Per-detector sets of OOD indices unique to that detector (not in mutual).
    """
    # Collect per-detector OOD index sets and normalized scores
    per_detector_ood: dict[str, set[int]] = {}
    per_sample_norm: dict[int, list[float]] = {}

    for method, det_result in detectors.items():
        threshold = det_result["threshold_score"]
        if threshold <= 0:
            continue

        ood_set: set[int] = set()
        for s in det_result.get("samples", []):
            norm = s["score"] / threshold
            per_sample_norm.setdefault(s["index"], []).append(norm)
            if s["is_ood"]:
                ood_set.add(s["index"])
        per_detector_ood[method] = ood_set

    # Mutual agreement: intersection of all detectors' OOD sets
    ood_sets = list(per_detector_ood.values())
    mutual_ood = ood_sets[0].intersection(*ood_sets[1:]) if ood_sets else set()

    # Per-detector unique OOD (flagged by this detector only, not in mutual)
    unique_ood = {method: ood - mutual_ood for method, ood in per_detector_ood.items()}

    # Average normalized score across detectors
    normalized_scores = {idx: sum(vals) / len(vals) for idx, vals in per_sample_norm.items()}

    return normalized_scores, mutual_ood, unique_ood


def _build_factor_deviations_finding(
    deviations: list[FactorDeviationDict],
    normalized_scores: dict[int, float],
    mutual_ood: set[int],
    locate: Callable[[int], ItemRef],
) -> Finding:
    """Build a finding for per-sample metadata deviations.

    Only includes samples that all detectors agree are OOD, most out of distribution first, each with
    its three most deviating factors.
    """
    factors = {
        d["index"]: ", ".join(f"{k}={v:.2f}" for k, v in list(d["deviations"].items())[:3])
        for d in deviations
        if d["index"] in mutual_ood
    }
    return Finding(
        severity="info",
        title="OOD Sample Metadata Deviations",
        description=(
            f"{len(factors)}/{len(deviations)} OOD samples agreed by all detectors, most out of distribution first"
        ),
        blocks=_samples_blocks(factors, normalized_scores, locate, factors),
    )


def _build_aggregate_finding(
    mutual_ood: set[int],
    normalized_scores: dict[int, float],
    total_ood: int,
    test_size: int,
    thresholds: OODDetectionHealthThresholds,
    locate: Callable[[int], ItemRef],
) -> Finding:
    """Build a finding for the aggregate (mutually agreed) OOD result."""
    n_mutual = len(mutual_ood)
    ood_pct = (n_mutual / test_size * 100) if test_size else 0.0
    severity = _severity_for_ood(ood_pct, thresholds)

    return Finding(
        severity=severity,
        title="Aggregate OOD (all detectors agree)",
        description=(
            f"{n_mutual}/{total_ood} OOD samples agreed by all detectors ({ood_pct:.1f}%), most out of distribution "
            "first. A score is a multiple of the detector's threshold, averaged over the detectors."
        ),
        blocks=_samples_blocks(mutual_ood, normalized_scores, locate),
    )


def _build_unique_ood_finding(
    unique_ood: dict[str, set[int]],
    normalized_scores: dict[int, float],
    detector_names: dict[str, str],
    locate: Callable[[int], ItemRef],
) -> Finding:
    """Build a single finding listing OOD samples unique to each detector."""
    groups: list[Block] = []
    for method_key, unique_indices in unique_ood.items():
        if not unique_indices:
            continue
        name = detector_names.get(method_key, method_key)
        samples = _samples_blocks(unique_indices, normalized_scores, locate)
        groups.append(Section(title=name, brief=f"{len(unique_indices)} unique sample(s)", blocks=samples))

    total_unique = sum(len(v) for v in unique_ood.values())

    return Finding(
        severity="info",
        title="Unique OOD Samples (single-detector only)",
        description=f"{total_unique} sample(s) flagged by only one detector",
        blocks=groups,
    )


def build_findings(
    raw: OODDetectionRawOutput,
    params: OODDetectionConfig,
    detector_names: dict[str, str],
    *,
    parts: Sequence[tuple[str, int]],
) -> list[Finding]:
    """Build all report findings from raw results.

    *parts* are the test sources' names and lengths, in the order the run joined them, which name
    each sample's item for its thumbnail.
    """
    locate = locator(parts)
    findings: list[Finding] = []
    multi_detector = len(raw.detectors) > 1

    # Compute normalized scores for cross-detector comparison
    normalized_scores, mutual_ood, unique_ood = _compute_normalized_scores(raw.detectors)

    # Per-detector findings. A lone detector has no aggregate to list its samples, so its own finding does.
    for method_key, result in raw.detectors.items():
        name = detector_names.get(method_key, method_key)
        finding = _build_detector_finding(name, result, params.health_thresholds)
        if not multi_detector:
            finding.blocks.extend(_samples_blocks(mutual_ood, normalized_scores, locate))
        findings.append(finding)

    # Aggregate + unique findings (only when multiple detectors)
    if multi_detector:
        total_ood = len(raw.ood_indices)
        findings.append(
            _build_aggregate_finding(
                mutual_ood, normalized_scores, total_ood, raw.test_size, params.health_thresholds, locate
            )
        )
        if any(unique_ood.values()):
            findings.append(_build_unique_ood_finding(unique_ood, normalized_scores, detector_names, locate))

    # Metadata insights findings
    if raw.factor_predictors:
        findings.append(_build_factor_predictors_finding(raw.factor_predictors))

    if raw.factor_deviations:
        findings.append(_build_factor_deviations_finding(raw.factor_deviations, normalized_scores, mutual_ood, locate))

    return findings
