"""Findings builders for the dataset splitting workflow."""

from __future__ import annotations

from typing import Any, Literal

from dataeval_flow.workflow.base import Reportable
from dataeval_flow.workflows.splitting.outputs import DataSplittingRawOutputs, SplitInfo


def _format_factor_table(
    rows: list[dict[str, Any]],
    value_key: str,
    value_label: str,
) -> list[str]:
    """Format a list of factor dicts as a text table for detail_lines."""
    if not rows:
        return []
    w_name = max(6, *(len(str(r.get("factor_name", ""))) for r in rows))
    lines = [
        f"{'Factor':<{w_name}}  {value_label:>10}  Flag",
        f"{'-' * w_name}  {'-' * 10}  ----",
    ]
    for r in rows:
        name = str(r.get("factor_name", ""))
        val = r.get(value_key, 0.0)
        flag_key = "is_imbalanced" if "is_imbalanced" in r else "is_low_diversity"
        flagged = r.get(flag_key, False)
        flag_str = " [!!]" if flagged else ""
        lines.append(f"{name:<{w_name}}  {val:>10.4f}{flag_str}")
    return lines


def _normalize_label_counts(label_counts: dict[str, int] | list[int] | None) -> dict[str, int]:
    """Normalize label_counts_per_class to ``{str_key: count}``."""
    if not label_counts:
        return {}
    if isinstance(label_counts, dict):
        return {str(k): v for k, v in label_counts.items()}
    return {str(i): c for i, c in enumerate(label_counts)}


# ---------------------------------------------------------------------------
# Split sizes (consolidated for multi-fold)
# ---------------------------------------------------------------------------


def _build_split_sizes(raw: DataSplittingRawOutputs) -> list[Reportable]:
    """Build split-size findings — consolidated pivot table for multi-fold."""
    if not raw.folds:
        return []

    test_size = len(raw.test_indices)

    # Single fold: keep the original key_value format
    if len(raw.folds) == 1:
        fold = raw.folds[0]
        return [
            Reportable(
                report_type="key_value",
                severity="info",
                title=f"Fold {fold.fold} split sizes",
                data={
                    "train": len(fold.train_indices),
                    "val": len(fold.val_indices),
                    "test": test_size,
                },
            )
        ]

    # Multi-fold: consolidated pivot table
    rows: list[dict[str, Any]] = []
    train_sizes: list[int] = []
    val_sizes: list[int] = []
    for fold_info in raw.folds:
        t = len(fold_info.train_indices)
        v = len(fold_info.val_indices)
        train_sizes.append(t)
        val_sizes.append(v)
        rows.append({"Fold": str(fold_info.fold), "Train": t, "Val": v, "Test": test_size})

    footer_lines: list[str] = []
    for name, sizes in [("Train", train_sizes), ("Val", val_sizes)]:
        lo, hi = min(sizes), max(sizes)
        footer_lines.append(f"{name}: {lo}-{hi} (range {hi - lo})" if lo != hi else f"{name}: {lo}")
    footer_lines.append(f"Test: {test_size} (shared across folds)")

    return [
        Reportable(
            report_type="pivot_table",
            severity="info",
            title="Split sizes across folds",
            data={
                "brief": f"{len(raw.folds)} folds, test={test_size}",
                "table_data": rows,
                "table_headers": ["Fold", "Train", "Val", "Test"],
                "footer_lines": footer_lines,
            },
            description="Split sizes per fold. Test set is shared across folds.",
        )
    ]


# ---------------------------------------------------------------------------
# Cross-split class distribution
# ---------------------------------------------------------------------------

_MAX_CLASSES_DISPLAY = 20
_TOP_CLASSES = 10
_BOTTOM_CLASSES = 5


def _make_distribution_row(
    cls: str,
    splits: dict[str, dict[str, int]],
    split_totals: dict[str, int],
) -> dict[str, Any]:
    """Build one row of the cross-split distribution table."""
    row: dict[str, Any] = {"Class": cls}
    pcts: dict[str, int] = {}
    raw_counts: dict[str, int] = {}
    for sn, counts_map in splits.items():
        count = counts_map.get(cls, 0)
        total = split_totals.get(sn, 0)
        pcts[sn] = round(count / total * 100) if total else 0
        raw_counts[sn] = count

    split_names = [sn for sn in splits if sn != "Full"]
    all_same = len({pcts[sn] for sn in split_names}) == 1

    for sn in splits:
        if all_same and sn != "Full":
            row[sn] = str(raw_counts[sn])
        else:
            row[sn] = f"{raw_counts[sn]} ({pcts[sn]}%)"
    return row


def _truncate_classes(all_classes: list[str]) -> tuple[list[str], list[str], int]:
    """Return (top, bottom, omitted) after truncation if needed."""
    if len(all_classes) > _MAX_CLASSES_DISPLAY:
        return (
            all_classes[:_TOP_CLASSES],
            all_classes[-_BOTTOM_CLASSES:],
            len(all_classes) - _TOP_CLASSES - _BOTTOM_CLASSES,
        )
    return all_classes, [], 0


def _build_distribution_rows(
    all_classes: list[str],
    splits: dict[str, dict[str, int]],
    split_totals: dict[str, int],
) -> list[dict[str, Any]]:
    """Build the full row list including placeholder for omitted classes."""
    top, bottom, omitted = _truncate_classes(all_classes)
    rows: list[dict[str, Any]] = [_make_distribution_row(cls, splits, split_totals) for cls in top]
    if omitted:
        placeholder: dict[str, Any] = {"Class": f"... {omitted} more ..."}
        for sn in splits:
            placeholder[sn] = ""
        rows.append(placeholder)
        rows.extend(_make_distribution_row(cls, splits, split_totals) for cls in bottom)
    return rows


def _build_cross_split_distribution(
    raw: DataSplittingRawOutputs,
) -> list[Reportable]:
    """Build cross-split class distribution pivot table(s)."""
    full_counts = _normalize_label_counts(raw.label_stats_full.get("label_counts_per_class"))
    if not full_counts:
        return []

    folds_with_stats = [f for f in raw.folds if f.label_stats_train]
    if not folds_with_stats:
        return []

    test_counts = _normalize_label_counts(raw.label_stats_test.get("label_counts_per_class"))
    has_test = bool(test_counts)

    findings: list[Reportable] = []
    folds_to_show = folds_with_stats[:1] if len(folds_with_stats) > 1 else folds_with_stats

    for fold_info in folds_to_show:
        train_counts = _normalize_label_counts(fold_info.label_stats_train.get("label_counts_per_class"))
        val_counts = _normalize_label_counts(fold_info.label_stats_val.get("label_counts_per_class"))

        splits: dict[str, dict[str, int]] = {"Train": train_counts, "Val": val_counts}
        if has_test:
            splits["Test"] = test_counts
        splits["Full"] = full_counts

        split_totals = {name: sum(c.values()) for name, c in splits.items()}
        all_classes = sorted(full_counts.keys(), key=lambda c: full_counts.get(c, 0), reverse=True)
        rows = _build_distribution_rows(all_classes, splits, split_totals)

        max_dev, worst_class, worst_split = _max_proportion_deviation(splits, full_counts, split_totals)

        headers = ["Class", "Train", "Val"]
        if has_test:
            headers.append("Test")
        headers.append("Full")

        footer_lines: list[str] = []
        if max_dev > 0:
            footer_lines.append(
                f"Max proportion deviation from full dataset: {max_dev:.1f}pp ({worst_class} in {worst_split})"
            )

        num_folds = len(folds_with_stats)
        if num_folds > 1:
            title = f"Class distribution across splits (fold {fold_info.fold} of {num_folds})"
        else:
            title = "Class distribution across splits"

        findings.append(
            Reportable(
                report_type="pivot_table",
                severity="info",
                title=title,
                data={
                    "table_data": rows,
                    "table_headers": headers,
                    "footer_lines": footer_lines,
                },
                description="Per-class counts and proportions across splits.",
            )
        )

    return findings


def _max_proportion_deviation(
    splits: dict[str, dict[str, int]],
    full_counts: dict[str, int],
    split_totals: dict[str, int],
) -> tuple[float, str, str]:
    """Find the maximum absolute proportion deviation from the full dataset.

    Returns ``(max_dev_pct, worst_class, worst_split)``.
    """
    full_total = split_totals.get("Full", 0)
    if full_total == 0:
        return 0.0, "", ""

    max_dev = 0.0
    worst_class = ""
    worst_split = ""

    for cls, full_count in full_counts.items():
        full_pct = full_count / full_total * 100
        for sn, counts in splits.items():
            if sn == "Full":
                continue
            total = split_totals.get(sn, 0)
            if total == 0:
                continue
            split_pct = counts.get(cls, 0) / total * 100
            dev = abs(split_pct - full_pct)
            if dev > max_dev:
                max_dev = dev
                worst_class = cls
                worst_split = sn

    return round(max_dev, 1), worst_class, worst_split


# ---------------------------------------------------------------------------
# Stratification quality health check
# ---------------------------------------------------------------------------


def _worst_deviation_across_folds(
    folds_with_stats: list[SplitInfo],
    full_counts: dict[str, int],
    full_total: int,
    test_counts: dict[str, int],
) -> tuple[float, str, str, int]:
    """Find the worst proportion deviation across all folds.

    Returns ``(max_dev, worst_class, worst_split, worst_fold)``.
    """
    max_dev = 0.0
    worst_class = ""
    worst_split = ""
    worst_fold = 0

    for fold_info in folds_with_stats:
        train_counts = _normalize_label_counts(fold_info.label_stats_train.get("label_counts_per_class"))
        val_counts = _normalize_label_counts(fold_info.label_stats_val.get("label_counts_per_class"))

        splits: dict[str, dict[str, int]] = {"train": train_counts, "val": val_counts}
        if test_counts:
            splits["test"] = test_counts

        split_totals = {name: sum(c.values()) for name, c in splits.items()}

        for cls, full_count in full_counts.items():
            full_pct = full_count / full_total * 100
            for sn, counts in splits.items():
                total = split_totals.get(sn, 0)
                if total == 0:
                    continue
                split_pct = counts.get(cls, 0) / total * 100
                dev = abs(split_pct - full_pct)
                if dev > max_dev:
                    max_dev = dev
                    worst_class = cls
                    worst_split = sn
                    worst_fold = fold_info.fold

    return round(max_dev, 1), worst_class, worst_split, worst_fold


def _build_stratification_check(raw: DataSplittingRawOutputs) -> list[Reportable]:
    """Build a stratification quality health-check finding."""
    full_counts = _normalize_label_counts(raw.label_stats_full.get("label_counts_per_class"))
    if not full_counts:
        return []

    folds_with_stats = [f for f in raw.folds if f.label_stats_train]
    if not folds_with_stats:
        return []

    test_counts = _normalize_label_counts(raw.label_stats_test.get("label_counts_per_class"))
    full_total = sum(full_counts.values())
    if full_total == 0:
        return []

    global_max_dev, global_worst_class, global_worst_split, global_worst_fold = _worst_deviation_across_folds(
        folds_with_stats, full_counts, full_total, test_counts
    )

    # Severity thresholds
    severity: Literal["ok", "info", "warning"]
    if global_max_dev <= 2.0:
        severity = "ok"
        status = "OK"
    elif global_max_dev <= 10.0:
        severity = "info"
        status = "OK"
    else:
        severity = "warning"
        status = "WARNING"

    brief = f"{status} - max deviation {global_max_dev}pp"

    detail_lines: list[str] = [f"Max proportion deviation: {global_max_dev}pp"]
    if global_max_dev > 0:
        full_pct = round(full_counts.get(global_worst_class, 0) / full_total * 100, 1)
        fold_label = f"fold {global_worst_fold} " if len(folds_with_stats) > 1 else ""
        detail_lines.append(
            f"  Worst: class '{global_worst_class}' in {fold_label}{global_worst_split} "
            f"(deviation {global_max_dev}pp from {full_pct}% in full)"
        )
    detail_lines.append("")
    detail_lines.append(f"Folds checked: {len(folds_with_stats)}")
    detail_lines.append(f"Classes checked: {len(full_counts)}")

    return [
        Reportable(
            report_type="key_value",
            severity=severity,
            title="Stratification quality",
            data={
                "brief": brief,
                "detail_lines": detail_lines,
            },
            description="Checks whether class proportions in each split match the full dataset.\n"
            "  Train proportions may differ if rebalancing was applied.",
        )
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_findings(
    raw: DataSplittingRawOutputs,
) -> list[Reportable]:
    """Build human-readable findings from raw outputs."""
    findings: list[Reportable] = []

    # --- 1. Full-dataset label distribution ---
    label_counts = raw.label_stats_full.get("label_counts_per_class")
    if label_counts:
        normalized = _normalize_label_counts(label_counts)
        counts = list(normalized.values())
        max_count = max(counts) if counts else 0
        min_count = min(counts) if counts else 0
        ratio = max_count / min_count if min_count > 0 else float("inf")
        severity: Literal["ok", "info", "warning"] = "warning" if ratio > 10 else "info"
        findings.append(
            Reportable(
                report_type="table",
                severity=severity,
                title="Class distribution (full dataset)",
                data={
                    "table_data": normalized,
                    "table_headers": ("Class", "Count"),
                },
                description=f"Max/min class ratio: {ratio:.1f}:1" if min_count > 0 else "Some classes have 0 samples",
            )
        )

    # --- 2. Split sizes (consolidated for multi-fold) ---
    findings.extend(_build_split_sizes(raw))

    # --- 3. Cross-split class distribution ---
    findings.extend(_build_cross_split_distribution(raw))

    # --- 4. Stratification quality health check ---
    findings.extend(_build_stratification_check(raw))

    # --- 5. Balance scores ---
    balance_data = raw.pre_split_balance.get("balance")
    if balance_data and isinstance(balance_data, list):
        findings.append(
            Reportable(
                report_type="key_value",
                severity="info",
                title="Pre-split balance (mutual information)",
                data={"detail_lines": _format_factor_table(balance_data, "mi_value", "MI Score")},
                description="Higher MI = stronger correlation between factor and class label.",
            )
        )

    # --- 6. Diversity scores ---
    diversity_data = raw.pre_split_diversity.get("factors")
    if diversity_data and isinstance(diversity_data, list):
        findings.append(
            Reportable(
                report_type="key_value",
                severity="info",
                title="Pre-split diversity",
                data={"detail_lines": _format_factor_table(diversity_data, "diversity_value", "Diversity")},
                description="Values near 1.0 = high diversity. Low diversity factors are flagged.",
            )
        )

    # --- 7. Per-split coverage ---
    for fold_info in raw.folds:
        for split_name, coverage in [("train", fold_info.coverage_train), ("val", fold_info.coverage_val)]:
            if coverage:
                uncovered = coverage.get("uncovered_indices", [])
                split_size = len(fold_info.train_indices) if split_name == "train" else len(fold_info.val_indices)
                pct = (len(uncovered) / split_size * 100) if split_size > 0 else 0
                cov_severity: Literal["ok", "info", "warning"] = "warning" if pct > 5 else "info"
                findings.append(
                    Reportable(
                        report_type="key_value",
                        severity=cov_severity,
                        title=f"Coverage: fold {fold_info.fold} {split_name}",
                        data={
                            "uncovered_count": len(uncovered),
                            "split_size": split_size,
                            "uncovered_pct": round(pct, 2),
                            "coverage_radius": coverage.get("coverage_radius"),
                        },
                    )
                )

    if raw.coverage_test:
        uncovered = raw.coverage_test.get("uncovered_indices", [])
        test_size = len(raw.test_indices)
        pct = (len(uncovered) / test_size * 100) if test_size > 0 else 0
        cov_severity = "warning" if pct > 5 else "info"
        findings.append(
            Reportable(
                report_type="key_value",
                severity=cov_severity,
                title="Coverage: test",
                data={
                    "uncovered_count": len(uncovered),
                    "split_size": test_size,
                    "uncovered_pct": round(pct, 2),
                    "coverage_radius": raw.coverage_test.get("coverage_radius"),
                },
            )
        )

    return findings
