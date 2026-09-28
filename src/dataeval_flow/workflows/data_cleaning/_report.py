"""Findings builders for the data cleaning workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from dataeval_flow._blocks import Block, Cell, Column, Fields, ItemRef, Paragraph, Scalar, Section, Table
from dataeval_flow._blocks._items import item_name
from dataeval_flow.workflows._base import Finding, render_label_source
from dataeval_flow.workflows._outliers import (
    OutlierIssueRecord,
    flagged_table,
    limits_sentence,
    limits_table,
    warn_if_unrecorded,
)
from dataeval_flow.workflows._tables import PREVIEW, ROW_CAP, ranked_table
from dataeval_flow.workflows.data_cleaning._config import DataCleaningHealthThresholds
from dataeval_flow.workflows.data_cleaning._outputs import (
    DataCleaningRawOutput,
    DetectionDict,
    IndexValue,
)

# A duplicate group's cell shows at most this many of its items.
_GROUP_SHOWN = 8


def _item(issue: Mapping[str, Any]) -> tuple[Cell, ...]:
    return (issue["item_index"],)


def _box(issue: Mapping[str, Any]) -> tuple[Cell, ...]:
    return (issue["item_index"], issue.get("target_index"))


def _classes(metadata: Any) -> tuple[dict[tuple[Cell, ...], str] | None, dict[tuple[Cell, ...], str] | None]:
    """Each item's class and each box's, keyed as the outlier tables key their rows; ``None`` where unknown.

    A classification item has one class. A detection image has one per box, so only its boxes are
    named, and its image rows have no Class column. A dataset without labels has none either.
    """
    if metadata is None:
        return None, None
    index2label = metadata.index2label
    if metadata.multi_target:
        rows = metadata.rows_at(metadata.label_level).select("item_index", "target_index", "class_label")
        return None, {(int(i), int(t)): index2label.get(int(c), str(c)) for i, t, c in rows.iter_rows()} or None
    labels = zip(metadata.item_indices, metadata.class_labels, strict=True)
    return {(int(i),): index2label.get(int(c), str(c)) for i, c in labels} or None, None


def _namer(source: str) -> Callable[[tuple[Cell, ...]], ItemRef]:
    """Names a flagged row's item, or its box where the row's key holds one, in *source*."""

    def ref(key: tuple[Cell, ...]) -> ItemRef:
        return ItemRef.model_validate({"source": source, "index": key[0], "target": key[1] if len(key) > 1 else None})

    return ref


def _outlier_blocks(
    issues: Sequence[OutlierIssueRecord],
    *,
    boxes: bool,
    classes: Mapping[tuple[Cell, ...], str] | None,
    noun: str,
    pairs: list[tuple[str, Scalar]],
    source: str,
) -> list[Block]:
    """Each flagged image or box with every flag it raised, each metric with its limits, then the labelled values."""
    key = _box if boxes else _item
    key_columns = [Column(key="item", header="Item"), *([Column(key="box", header="Box")] if boxes else [])]
    blocks: list[Block] = flagged_table(
        issues, key=key, key_columns=key_columns, classes=classes, noun=noun, ref=_namer(source)
    )
    if issues:
        blocks.append(limits_table(issues, key=key))
    blocks.append(Fields(items=pairs))
    return blocks


def _member(source: str, member: IndexValue) -> ItemRef:
    """A duplicate group's member as an item reference: an image by its index, or a box by its item and target."""
    if isinstance(member, dict):
        return ItemRef(source=source, index=member["item"], target=member["target"])
    return ItemRef(source=source, index=member)


def _groups_blocks(detection: DetectionDict, source: str, noun: str) -> list[Block]:
    """Each duplicate group, largest first: its kind, its size, and up to eight of its items, named and pictured.

    A group is numbered by its place among its kind in ``output.raw``, where every one of its items is.
    """
    groups = [("exact", number, members) for number, members in enumerate(detection.get("exact", []))]
    groups += [("near", number, group["indices"]) for number, group in enumerate(detection.get("near", []))]
    if not groups:
        return []
    # Stable, so groups of one size keep exact before near, and each kind its own order.
    groups.sort(key=lambda group: -len(group[2]))
    rows: list[dict[str, Cell]] = []
    for kind, number, members in groups[:ROW_CAP]:
        shown = [_member(source, member) for member in members[:_GROUP_SHOWN]]
        names = ", ".join(item_name(ref) for ref in shown)
        more = len(members) - len(shown)
        items = f"{names}, … {more:,} more" if more else names
        rows.append({"group": number, "kind": kind, "count": len(members), "items": items, "image": shown})
    columns = [
        Column(key="group", header="Group"),
        Column(key="kind", header="Kind", align="left"),
        Column(key="count", header="Count"),
        Column(key="items", header="Items", align="left"),
        Column(key="image", kind="image"),
    ]
    blocks: list[Block] = [Table(columns=columns, rows=rows, preview=PREVIEW)]
    if len(groups) > ROW_CAP:
        blocks.append(
            Paragraph(
                text=f"{len(groups):,} groups of {noun}; the {ROW_CAP:,} largest are listed, and every one is in "
                "`output.raw`."
            )
        )
    return blocks


def _duplicate_finding(
    raw: DataCleaningRawOutput, thresholds: DataCleaningHealthThresholds, source: str
) -> Finding | None:
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
    blocks: list[Block] = []
    if exact_groups:
        blocks.append(Paragraph(text=f"{len(exact_groups)} exact-duplicate groups ({exact_affected} images)"))
    if near_groups:
        near_line = f"{len(near_groups)} near-duplicate groups ({near_affected} images)"
        near_pairs: list[tuple[str, Scalar]] = []
        if all_methods:
            near_pairs.append(("Methods", ", ".join(sorted(all_methods))))
        if orientations:
            near_pairs.append(("Orientations", ", ".join(f"{c} {o}" for o, c in sorted(orientations.items()))))
        blocks.append(
            Section(title=near_line, blocks=[Fields(items=near_pairs)]) if near_pairs else Paragraph(text=near_line)
        )
    blocks.extend(_groups_blocks(raw.duplicates.get("items", {}), source, "images"))
    box_groups = raw.duplicates.get("targets", {})
    if box_count := len(box_groups.get("exact", [])) + len(box_groups.get("near", [])):
        boxes = _groups_blocks(box_groups, source, "boxes")
        blocks.append(Section(title="Duplicate boxes", brief=f"{box_count} groups", blocks=boxes))
    blocks.append(
        Fields(
            items=[
                ("Exact groups", len(exact_groups)),
                ("Near groups", len(near_groups)),
                ("Exact affected", exact_affected),
                ("Near affected", near_affected),
            ]
        )
    )
    # Determine severity from thresholds
    exact_pct = (exact_affected / raw.dataset_size) * 100 if raw.dataset_size else 0.0
    near_pct = (near_affected / raw.dataset_size) * 100 if raw.dataset_size else 0.0
    severity: Literal["ok", "info", "warning"] = "info"
    if exact_pct > thresholds.exact_duplicates or near_pct > thresholds.near_duplicates:
        severity = "warning"

    return Finding(
        severity=severity,
        title="Duplicates",
        brief=f"{exact_affected} exact ({round(exact_pct, 1)}%), {near_affected} near ({round(near_pct, 1)}%)",
        description=(f"{len(exact_groups)} exact duplicate groups, {len(near_groups)} near-duplicate groups found."),
        blocks=blocks,
    )


def _label_distribution_finding(
    raw: DataCleaningRawOutput,
    thresholds: DataCleaningHealthThresholds,
    label_source: str | Sequence[str] | None = None,
) -> Finding | None:
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
        footer_lines.append(f"Labels {render_label_source(label_source)}")
    if has_empty_class:
        footer_lines.append("Warning: one or more classes have zero items")
    elif imbalance_ratio == 1.0:
        footer_lines.append("Balanced: all classes have equal counts")
    elif imbalance_ratio != 0.0:
        footer_lines.append(f"Imbalance ratio: {imbalance_ratio} (max/min)")
    severity: Literal["ok", "info", "warning"] = "info"
    if has_empty_class or imbalance_ratio > thresholds.class_label_imbalance:
        severity = "warning"

    # The footer annotates the counts table, so with no counts to rank neither shows.
    blocks: list[Block] = []
    if label_counts:
        footer = [Paragraph(text=line) for line in footer_lines]
        blocks = [ranked_table(label_counts, headers=("Class", "Count")), *footer]
    return Finding(
        severity=severity,
        title=("Label/Directory_Name Distribution" if label_source == "filepath" else "Label Distribution"),
        brief=f"{class_count} classes, {item_count} items, imbalance {imbalance_ratio}:1",
        description=(f"{class_count} classes, {item_count} items."),
        blocks=blocks,
    )


def _classwise_finding(raw: DataCleaningRawOutput, thresholds: DataCleaningHealthThresholds) -> Finding:
    """Build a Classwise Outliers finding from raw results."""
    pivot = raw.classwise_outliers
    rows = pivot.get("rows", None) if pivot else None

    if not rows:
        return Finding(
            severity="ok",
            title="Classwise Outliers",
            brief="no outliers detected",
            description="No outliers detected — classwise breakdown not applicable.",
        )

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

    columns = [
        Column(key="class_name", header="Class Name"),
        Column(key="count", header="Count"),
        Column(key="pct", header="%", format="{:.1f}%"),
    ]
    cells: list[dict[str, Cell]] = [
        {"class_name": row["class_name"], "count": row["count"], "pct": row["pct"]} for row in rows
    ]
    return Finding(
        severity=severity,
        title="Classwise Outliers",
        brief=brief,
        description=(
            f"Most outliers in {worst_name} ({worst_pct}%). "
            f"{classes_over}/{len(class_rows)} classes exceed {thresholds.classwise_outliers}% threshold."
        ),
        blocks=[Table(columns=columns, rows=cells)],
    )


def build_findings(
    raw: DataCleaningRawOutput,
    metadata: Any,
    thresholds: DataCleaningHealthThresholds,
    label_source: str | Sequence[str] | None = None,
    *,
    source: str,
    outlier_method: str | None = None,
    outlier_threshold: float | None = None,
) -> list[Finding]:
    """Generate human-readable findings from raw results.

    *metadata* names each flagged item's class where it has labels, and *source* the items themselves,
    for their thumbnails. *outlier_method* and *outlier_threshold* are what the outliers were detected
    with, which the findings state.
    """
    findings: list[Finding] = []
    item_classes, box_classes = _classes(metadata)
    limits = limits_sentence(outlier_method, outlier_threshold)

    # Outlier findings — count distinct images, not total flags
    outlier_issues = raw.img_outliers.get("issues", [])
    outlier_image_count = len({issue["item_index"] for issue in outlier_issues})
    pct = (outlier_image_count / raw.dataset_size) * 100 if raw.dataset_size else 0
    if outlier_image_count > 0:
        img_severity: Literal["ok", "info", "warning"] = "warning" if pct > thresholds.image_outliers else "info"
        img_description = f"{outlier_image_count} images ({pct:.1f}%) flagged as outliers."
        if limits:
            img_description = f"{img_description} {limits}"
    else:
        img_severity = "ok"
        img_description = "No images flagged as outliers."
    findings.append(
        Finding(
            severity=img_severity,
            title="Image Outliers",
            brief=f"{outlier_image_count} images ({round(pct, 1)}%)",
            description=img_description,
            blocks=_outlier_blocks(
                outlier_issues,
                boxes=False,
                classes=item_classes,
                noun="images",
                pairs=[("Percentage", round(pct, 1)), ("Dataset size", raw.dataset_size)],
                source=source,
            ),
        )
    )

    # Target outlier findings — count distinct (item, target) pairs
    target_issues = raw.target_outliers.get("issues", []) if raw.target_outliers else []
    warn_if_unrecorded([*outlier_issues, *target_issues])
    target_pair_count = len({(issue["item_index"], issue.get("target_index")) for issue in target_issues})
    if target_pair_count > 0:
        # Total target count from label stats for percentage
        total_targets = sum(raw.label_stats.get("label_counts_per_class", {}).values()) if raw.label_stats else 0
        target_pct = round((target_pair_count / total_targets) * 100, 1) if total_targets > 0 else 0.0
        target_description = f"{target_pair_count} bounding-box targets ({target_pct}%) flagged as outliers."
        tgt_severity: Literal["ok", "info", "warning"] = (
            "warning" if target_pct > thresholds.target_outliers else "info"
        )
        findings.append(
            Finding(
                severity=tgt_severity,
                title="Target Outliers",
                brief=f"{target_pair_count} targets ({target_pct}%)",
                description=f"{target_description} {limits}" if limits else target_description,
                blocks=_outlier_blocks(
                    target_issues,
                    boxes=True,
                    classes=box_classes,
                    noun="targets",
                    pairs=[("Percentage", target_pct), ("Total targets", total_targets)],
                    source=source,
                ),
            )
        )

    # Classwise outlier pivot — right after image/target outliers
    findings.append(_classwise_finding(raw, thresholds))

    # Duplicate findings
    dup_finding = _duplicate_finding(raw, thresholds, source)
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


def collect_flagged_indices(raw: DataCleaningRawOutput) -> set[int]:
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
