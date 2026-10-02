"""Outliers per class: the pivot the `classwise-outliers` combine builds, and the split of outlier issues into images
and boxes that it and the outlier checks read."""

__all__ = ["class_labels_frame", "classwise_pivot", "split_outlier_issues"]

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from dataeval import Metadata


def split_outlier_issues(issues_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Outlier issues split into image-level rows and target-level rows; the second is ``None`` where no row names a
    target."""
    if "target_index" in issues_df.columns:
        img_issues = issues_df.filter(issues_df["target_index"].is_null())
        target_issues = issues_df.filter(issues_df["target_index"].is_not_null())
    else:
        img_issues = issues_df
        target_issues = None
    return img_issues, target_issues


def class_labels_frame(metadata: "Metadata") -> tuple[pl.DataFrame, list[str], dict[str, int]]:
    """Each item's class, or each box's, as a frame; the columns that identify a row; and the label count per class."""
    index2label = metadata.index2label
    has_targets = metadata.multi_target

    label_counts: dict[str, int] = {}
    for lbl in metadata.class_labels:
        name = index2label.get(lbl, str(lbl))
        label_counts[name] = label_counts.get(name, 0) + 1

    if has_targets:
        td = metadata.rows_at(metadata.label_level).select("item_index", "target_index", "class_label")
        names = [index2label.get(int(c), str(c)) for c in td["class_label"].to_list()]
        labels_df = td.with_columns(pl.Series("class_name", names)).select("item_index", "target_index", "class_name")
        id_cols = ["item_index", "target_index"]
    else:
        item_ids = getattr(metadata, "item_indices", list(range(len(metadata.class_labels))))
        names = [index2label.get(lbl, str(lbl)) for lbl in metadata.class_labels]
        labels_df = pl.DataFrame({"item_index": item_ids, "class_name": names})
        id_cols = ["item_index"]

    return labels_df, id_cols, label_counts


def classwise_pivot(
    target_issues: pl.DataFrame | None, img_issues: pl.DataFrame, metadata: "Metadata"
) -> dict[str, Any] | None:
    """How many of each class's items, or boxes, were flagged, most first, then a ``Total`` row.

    ``None`` where nothing was flagged at the level the labels sit at: boxes for detection, images otherwise. Raises
    where the labels cannot be joined to the issues, which fails the `classwise-outliers` step that asked.
    """
    has_targets = metadata.multi_target
    if has_targets:
        if target_issues is None or target_issues.shape[0] == 0:
            return None
        issues_df = target_issues
    else:
        if img_issues.shape[0] == 0:
            return None
        issues_df = img_issues

    labels_df, id_cols, label_counts = class_labels_frame(metadata)
    total_labels = sum(label_counts.values())

    for col in id_cols:
        if col in issues_df.columns and issues_df[col].dtype != labels_df[col].dtype:
            labels_df = labels_df.with_columns(pl.col(col).cast(issues_df[col].dtype))

    # Count unique outlier items/targets per class (not per metric flag)
    unique_per_class = (
        issues_df.join(labels_df, on=id_cols, how="left")
        .select(id_cols + ["class_name"])
        .unique()
        .group_by("class_name")
        .len()
        # Tied classes by name: `group_by` keeps no order, so a tie would otherwise name a different worst class.
        .sort(["len", "class_name"], descending=[True, False])
    )

    rows: list[Any] = []
    grand_total = 0
    for row_dict in unique_per_class.to_dicts():
        name = str(row_dict.get("class_name", ""))
        count = int(row_dict.get("len", 0))
        grand_total += count
        denom = label_counts.get(name, 0)
        pct = round((count / denom) * 100, 1) if denom > 0 else 0.0
        rows.append({"class_name": name, "count": count, "pct": pct})

    total_pct = round((grand_total / total_labels) * 100, 1) if total_labels > 0 else 0.0
    rows.append({"class_name": "Total", "count": grand_total, "pct": total_pct})

    return {
        # Flow's own label for what a row counts, deliberately not named
        # "level" — DataEval uses that key for metadata levels (unit,
        # instance, track, sequence) and duplicate levels (item, target).
        "count_basis": "annotation" if has_targets else "image",
        "rows": rows,
    }
