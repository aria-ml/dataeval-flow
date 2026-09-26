"""Findings builders for the OOD detection workflow."""

from __future__ import annotations

from typing import Literal

from dataeval_flow._blocks import Block, BulletList, Cell, Column, Fields, Paragraph, Section, Table
from dataeval_flow.workflows._base import Finding
from dataeval_flow.workflows._tables import ranked_table
from dataeval_flow.workflows.ood_detection._config import (
    OODDetectionConfig,
    OODDetectionHealthThresholds,
)
from dataeval_flow.workflows.ood_detection._outputs import (
    DetectorOODResultDict,
    FactorDeviationDict,
    OODDetectionRawOutput,
)


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
) -> Finding:
    """Build a finding for per-sample metadata deviations.

    Only includes samples that all detectors agree are OOD, sorted by
    normalized OOD score (descending).
    """
    # Filter to mutually agreed OOD samples
    agreed_devs = [d for d in deviations if d["index"] in mutual_ood]

    # Sort by normalized score (most OOD first)
    agreed_devs.sort(key=lambda d: normalized_scores.get(d["index"], 0.0), reverse=True)

    detail_lines: list[str] = []
    for dev in agreed_devs[:10]:  # Cap display at 10 samples
        norm = normalized_scores.get(dev["index"], 0.0)
        top_factors = list(dev["deviations"].items())[:3]
        factors_str = ", ".join(f"{k}={v:.2f}" for k, v in top_factors)
        detail_lines.append(f"Sample {dev['index']:4d} (score={norm:.2f}x): {factors_str}")

    n_agreed = len(agreed_devs)
    n_total = len(deviations)

    return Finding(
        severity="info",
        title="OOD Sample Metadata Deviations",
        description=(
            f"{n_agreed}/{n_total} OOD samples agreed by all detectors "
            f"(sorted by normalized score, showing top {min(10, n_agreed)})"
        ),
        blocks=[BulletList(items=detail_lines)] if detail_lines else [],
    )


def _build_aggregate_finding(
    mutual_ood: set[int],
    normalized_scores: dict[int, float],
    total_ood: int,
    test_size: int,
    thresholds: OODDetectionHealthThresholds,
) -> Finding:
    """Build a finding for the aggregate (mutually agreed) OOD result."""
    sorted_indices = sorted(mutual_ood, key=lambda i: normalized_scores.get(i, 0.0), reverse=True)
    detail_lines: list[str] = []
    for idx in sorted_indices[:10]:
        norm = normalized_scores.get(idx, 0.0)
        detail_lines.append(f"Sample {idx:4d} (score={norm:.2f}x)")

    n_mutual = len(mutual_ood)
    ood_pct = (n_mutual / test_size * 100) if test_size else 0.0
    severity = _severity_for_ood(ood_pct, thresholds)

    return Finding(
        severity=severity,
        title="Aggregate OOD (all detectors agree)",
        description=(
            f"{n_mutual}/{total_ood} OOD samples agreed by all detectors ({ood_pct:.1f}%) "
            f"(sorted by normalized score, showing top {min(10, n_mutual)})"
        ),
        blocks=[BulletList(items=detail_lines)] if detail_lines else [],
    )


def _build_unique_ood_finding(
    unique_ood: dict[str, set[int]],
    normalized_scores: dict[int, float],
    detector_names: dict[str, str],
) -> Finding:
    """Build a single finding listing OOD samples unique to each detector."""
    groups: list[Block] = []
    for method_key, unique_indices in unique_ood.items():
        if not unique_indices:
            continue
        name = detector_names.get(method_key, method_key)
        sorted_indices = sorted(unique_indices, key=lambda i: normalized_scores.get(i, 0.0), reverse=True)
        items = [f"Sample {idx:4d} (score={normalized_scores.get(idx, 0.0):.2f}x)" for idx in sorted_indices[:10]]
        groups.append(
            Section(title=name, brief=f"{len(unique_indices)} unique sample(s)", blocks=[BulletList(items=items)])
        )

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
) -> list[Finding]:
    """Build all report findings from raw results."""
    findings: list[Finding] = []
    multi_detector = len(raw.detectors) > 1

    # Per-detector findings
    for method_key, result in raw.detectors.items():
        name = detector_names.get(method_key, method_key)
        findings.append(_build_detector_finding(name, result, params.health_thresholds))

    # Compute normalized scores for cross-detector comparison
    normalized_scores, mutual_ood, unique_ood = _compute_normalized_scores(raw.detectors)

    # Aggregate + unique findings (only when multiple detectors)
    if multi_detector:
        total_ood = len(raw.ood_indices)
        findings.append(
            _build_aggregate_finding(mutual_ood, normalized_scores, total_ood, raw.test_size, params.health_thresholds)
        )
        if any(unique_ood.values()):
            findings.append(_build_unique_ood_finding(unique_ood, normalized_scores, detector_names))

    # Metadata insights findings
    if raw.factor_predictors:
        findings.append(_build_factor_predictors_finding(raw.factor_predictors))

    if raw.factor_deviations:
        findings.append(_build_factor_deviations_finding(raw.factor_deviations, normalized_scores, mutual_ood))

    return findings
