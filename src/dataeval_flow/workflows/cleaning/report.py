"""Findings builders for the data cleaning workflow."""

from __future__ import annotations

from typing import Any, Literal

from dataeval_flow.workflow.base import Reportable
from dataeval_flow.workflows.cleaning.outputs import (
    DataCleaningRawOutputs,
    IndexValue,
)
from dataeval_flow.workflows.cleaning.params import DataCleaningHealthThresholds


def _duplicate_finding(raw: DataCleaningRawOutputs, thresholds: DataCleaningHealthThresholds) -> Reportable | None:
    """Build a Duplicates finding from raw results, or None if no duplicates."""
    exact_groups = raw.duplicates.get("items", {}).get("exact", [])
    near_groups = raw.duplicates.get("items", {}).get("near", [])
    if not exact_groups and not near_groups:
        return None

    exact_affected = sum(len(g) for g in exact_groups)
    near_affected = sum(len(g["indices"]) for g in near_groups)
    # Collect methods and orientations from near groups
    all_methods: set[str] = set()
    orientations: dict[str, int] = {}
    for g in near_groups:
        all_methods.update(g.get("methods", []))
        orient = g.get("orientation")
        if orient is not None:
            orientations[orient] = orientations.get(orient, 0) + 1
    detail_lines: list[str] = []
    if exact_groups:
        detail_lines.append(f"{len(exact_groups)} exact-duplicate groups ({exact_affected} images)")
    if near_groups:
        detail_lines.append(f"{len(near_groups)} near-duplicate groups ({near_affected} images)")
        if all_methods:
            detail_lines.append(f"  Methods: {', '.join(sorted(all_methods))}")
        if orientations:
            parts = [f"{c} {o}" for o, c in sorted(orientations.items())]
            detail_lines.append(f"  Orientations: {', '.join(parts)}")
    # Determine severity from thresholds
    exact_pct = (exact_affected / raw.dataset_size) * 100 if raw.dataset_size else 0.0
    near_pct = (near_affected / raw.dataset_size) * 100 if raw.dataset_size else 0.0
    severity: Literal["ok", "info", "warning"] = "info"
    if exact_pct > thresholds.exact_duplicates or near_pct > thresholds.near_duplicates:
        severity = "warning"

    return Reportable(
        report_type="key_value",
        severity=severity,
        title="Duplicates",
        data={
            "brief": (f"{exact_affected} exact ({round(exact_pct, 1)}%), {near_affected} near ({round(near_pct, 1)}%)"),
            "detail_lines": detail_lines,
            "exact_groups": len(exact_groups),
            "near_groups": len(near_groups),
            "exact_affected": exact_affected,
            "near_affected": near_affected,
            "near_methods": sorted(all_methods),
            "near_orientations": orientations,
        },
        description=(f"{len(exact_groups)} exact duplicate groups, {len(near_groups)} near-duplicate groups found."),
    )


def _label_distribution_finding(
    raw: DataCleaningRawOutputs,
    thresholds: DataCleaningHealthThresholds,
    label_source: str | None = None,
) -> Reportable | None:
    """Build a Label Distribution finding from raw results, or None if no label stats."""
    if not raw.label_stats:
        return None

    class_count = raw.label_stats.get("class_count", 0)
    if class_count == 0:
        return None  # No labels — suppress finding

    label_counts = raw.label_stats.get("label_counts_per_class", {})
    item_count = raw.label_stats.get("item_count", 0)
    counts_list = list(label_counts.values()) if label_counts else []
    has_empty_class = bool(counts_list) and min(counts_list) == 0
    imbalance_ratio = round(max(counts_list) / min(counts_list), 1) if counts_list and not has_empty_class else 0.0
    footer_lines: list[str] = []
    if label_source:
        footer_lines.append(f"Labels {label_source}")
    if has_empty_class:
        footer_lines.append("Warning: one or more classes have zero items")
    elif imbalance_ratio == 1.0:
        footer_lines.append("Balanced: all classes have equal counts")
    elif imbalance_ratio != 0.0:
        footer_lines.append(f"Imbalance ratio: {imbalance_ratio} (max/min)")
    severity: Literal["ok", "info", "warning"] = "info"
    if has_empty_class or imbalance_ratio > thresholds.class_label_imbalance:
        severity = "warning"

    return Reportable(
        report_type="table",
        severity=severity,
        title=("Label/Directory_Name Distribution" if label_source == "filepath" else "Label Distribution"),
        data={
            "brief": f"{class_count} classes, {item_count} items, imbalance {imbalance_ratio}:1",
            "table_data": label_counts,
            "table_headers": ("Class", "Count"),
            "footer_lines": footer_lines,
            # Keep existing keys for JSON/YAML consumers
            "label_counts": label_counts,
            "class_count": class_count,
            "item_count": item_count,
            "imbalance_ratio": imbalance_ratio,
            "label_source": label_source,
        },
        description=(f"{class_count} classes, {item_count} items."),
    )


def _classwise_finding(raw: DataCleaningRawOutputs, thresholds: DataCleaningHealthThresholds) -> Reportable:
    """Build a Classwise Outliers finding from raw results."""
    pivot = raw.classwise_outliers
    rows = pivot.get("rows", None) if pivot else None

    if not rows:
        return Reportable(
            report_type="pivot_table",
            severity="ok",
            title="Classwise Outliers",
            data={
                "brief": "no outliers detected",
                "level": "image",
                "table_data": [],
                "table_headers": ["Class Name", "Count", "%"],
                "worst_class": None,
                "worst_pct": 0.0,
                "classes_over_threshold": 0,
            },
            description="No outliers detected — classwise breakdown not applicable.",
        )

    level = pivot.get("level", "image")  # type: ignore[union-attr]

    # The last row is the "Total" row
    total_row = rows[-1] if rows else {}
    class_rows = rows[:-1] if len(rows) > 1 else rows
    total_pct = total_row.get("pct", 0.0)

    severity: Literal["ok", "info", "warning"] = "info"
    if total_pct > thresholds.classwise_outliers:
        severity = "warning"

    # Identify the worst class and how many classes exceed the threshold
    class_pcts = [r.get("pct", 0.0) for r in class_rows]
    worst_row = max(class_rows, key=lambda r: r.get("pct", 0.0))
    worst_name = worst_row.get("class_name", "?")
    worst_pct = worst_row.get("pct", 0.0)
    classes_over = sum(1 for p in class_pcts if p > thresholds.classwise_outliers)
    brief_prefix = f"worst: {worst_name} ({worst_pct}%), "

    # Brief: focus on concentration — which class is worst and how many are over threshold
    if classes_over > 0:
        brief = f"{brief_prefix}{classes_over}/{len(class_rows)} classes over {thresholds.classwise_outliers}%"
    else:
        brief = f"{brief_prefix}all classes within {thresholds.classwise_outliers}%"

    return Reportable(
        report_type="pivot_table",
        severity=severity,
        title="Classwise Outliers",
        data={
            "brief": brief,
            "level": level,
            "table_data": rows,
            "table_headers": ["Class Name", "Count", "%"],
            "worst_class": worst_name,
            "worst_pct": worst_pct,
            "classes_over_threshold": classes_over,
        },
        description=(
            f"Most outliers in {worst_name} ({worst_pct}%). "
            f"{classes_over}/{len(class_rows)} classes exceed {thresholds.classwise_outliers}% threshold."
        ),
    )


def build_findings(
    raw: DataCleaningRawOutputs,
    metadata: Any,  # noqa: ARG001 - reserved for future metadata-based findings
    thresholds: DataCleaningHealthThresholds,
    label_source: str | None = None,
) -> list[Reportable]:
    """Generate human-readable findings from raw results."""
    findings: list[Reportable] = []

    # Outlier findings — count distinct images, not total flags
    outlier_issues = raw.img_outliers.get("issues", [])
    outlier_image_count = len({issue["item_index"] for issue in outlier_issues})
    pct = (outlier_image_count / raw.dataset_size) * 100 if raw.dataset_size else 0
    # Per-metric breakdown: count distinct images per metric
    _per_metric_sets: dict[str, set[int]] = {}
    for issue in outlier_issues:
        _per_metric_sets.setdefault(issue["metric_name"], set()).add(issue["item_index"])
    per_metric = {k: len(v) for k, v in _per_metric_sets.items()}
    if outlier_image_count > 0:
        img_severity: Literal["ok", "info", "warning"] = "warning" if pct > thresholds.image_outliers else "info"
        img_description = f"{outlier_image_count} images ({pct:.1f}%) flagged as outliers."
    else:
        img_severity = "ok"
        img_description = "No images flagged as outliers."
    findings.append(
        Reportable(
            report_type="key_value",
            severity=img_severity,
            title="Image Outliers",
            data={
                "brief": f"{outlier_image_count} images ({round(pct, 1)}%)",
                "multi_metric_subject": "images",
                "count": outlier_image_count,
                "percentage": round(pct, 1),
                "per_metric": per_metric,
                "total_flags": len(outlier_issues),
                "dataset_size": raw.dataset_size,
            },
            description=img_description,
        )
    )

    # Target outlier findings — count distinct (item, target) pairs
    target_issues = raw.target_outliers.get("issues", []) if raw.target_outliers else []
    target_pair_count = len({(issue["item_index"], issue.get("target_index")) for issue in target_issues})
    if target_pair_count > 0:
        # Total target count from label stats for percentage
        total_targets = sum(raw.label_stats.get("label_counts_per_class", {}).values()) if raw.label_stats else 0
        target_pct = round((target_pair_count / total_targets) * 100, 1) if total_targets > 0 else 0.0
        # Per-metric breakdown for targets
        _target_metric_sets: dict[str, set[tuple[int, int | None]]] = {}
        for issue in target_issues:
            key = (issue["item_index"], issue.get("target_index"))
            _target_metric_sets.setdefault(issue["metric_name"], set()).add(key)
        target_per_metric = {k: len(v) for k, v in _target_metric_sets.items()}
        tgt_severity: Literal["ok", "info", "warning"] = (
            "warning" if target_pct > thresholds.target_outliers else "info"
        )
        findings.append(
            Reportable(
                report_type="key_value",
                severity=tgt_severity,
                title="Target Outliers",
                data={
                    "brief": f"{target_pair_count} targets ({target_pct}%)",
                    "multi_metric_subject": "targets",
                    "count": target_pair_count,
                    "percentage": target_pct,
                    "per_metric": target_per_metric,
                    "total_flags": len(target_issues),
                    "total_targets": total_targets,
                },
                description=f"{target_pair_count} bounding-box targets ({target_pct}%) flagged as outliers.",
            )
        )

    # Classwise outlier pivot — right after image/target outliers
    findings.append(_classwise_finding(raw, thresholds))

    # Duplicate findings
    dup_finding = _duplicate_finding(raw, thresholds)
    if dup_finding:
        findings.append(dup_finding)

    # Label distribution finding
    label_finding = _label_distribution_finding(raw, thresholds, label_source=label_source)
    if label_finding:
        findings.append(label_finding)

    return findings


def _item_id_of(idx: IndexValue) -> int:
    """Extract the item ID from an :class:`IndexValue`.

    Returns the ``int`` directly for image-level indices, or the ``"item"``
    field from a :class:`SourceIndexDict` for target-level indices.
    """
    if isinstance(idx, dict):
        return idx["item"]
    return idx


def collect_flagged_indices(raw: DataCleaningRawOutputs) -> set[int]:
    """Collect all unique item indices flagged by outlier or duplicate detection."""
    flagged: set[int] = set()

    # Outlier-flagged items
    for issue in raw.img_outliers.get("issues", []):
        flagged.add(issue["item_index"])

    # Duplicate-flagged items (keep first in each group, flag the rest)
    for group in raw.duplicates.get("items", {}).get("exact", []):
        for idx in group[1:]:  # keep first, flag rest
            flagged.add(_item_id_of(idx))
    for group in raw.duplicates.get("items", {}).get("near", []):
        for idx in group["indices"][1:]:
            flagged.add(_item_id_of(idx))

    return flagged
