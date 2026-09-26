"""Findings builders for the dataset splitting workflow."""

from __future__ import annotations

from typing import Any, Literal

from dataeval_flow._blocks import Block, Cell, Column, Fields, Paragraph, Scalar, Table
from dataeval_flow.workflows._base import Finding
from dataeval_flow.workflows._tables import ranked_table
from dataeval_flow.workflows.data_splitting._outputs import DataSplittingRawOutput, SplitInfo


def _factor_table(
    rows: list[dict[str, Any]],
    value_key: str,
    value_header: str,
    flag_key: str,
) -> Table:
    """Each factor's score to four places, marked ``[!!]`` where DataEval flagged it."""
    cells: list[dict[str, Cell]] = [
        {
            "factor": str(row.get("factor_name", "")),
            "value": row.get(value_key, 0.0),
            "flag": "[!!]" if row.get(flag_key, False) else "",
        }
        for row in rows
    ]
    columns = [
        Column(key="factor", header="Factor"),
        Column(key="value", header=value_header, format="{:.4f}"),
        Column(key="flag", header="Flag", align="left"),
    ]
    return Table(columns=columns, rows=cells)


def _normalize_label_counts(
    label_counts: dict[Any, int] | list[int] | None,
    index2label: dict[Any, str] | None = None,
) -> dict[str, int]:
    """Normalize label_counts_per_class to ``{class_name: count}``.

    ``label_counts_per_class`` is keyed by class index; ``index2label`` names those
    indices when the dataset carries a class-name table. Indices with no name keep
    their stringified index, which is what unnamed label spaces report. Keys are
    compared as strings because a JSON round-trip turns the integer indices into
    strings on both sides.
    """
    if not label_counts:
        return {}
    names = {str(index): name for index, name in (index2label or {}).items()}
    pairs = label_counts.items() if isinstance(label_counts, dict) else enumerate(label_counts)
    return {names.get(str(index), str(index)): count for index, count in pairs}


# ---------------------------------------------------------------------------
# Split sizes (consolidated for multi-fold)
# ---------------------------------------------------------------------------


def _size_range(sizes: list[int]) -> int | str:
    """The one size every fold shares, or the spread of sizes across the folds."""
    lo, hi = min(sizes), max(sizes)
    return lo if lo == hi else f"{lo}-{hi} (range {hi - lo})"


def _build_split_sizes(raw: DataSplittingRawOutput) -> list[Finding]:
    """Build split-size findings — one table for multi-fold."""
    if not raw.folds:
        return []

    test_size = len(raw.test_indices)

    # Single fold: labelled sizes
    if len(raw.folds) == 1:
        fold = raw.folds[0]
        return [
            Finding(
                severity="info",
                title=f"Fold {fold.fold} split sizes",
                blocks=[
                    Fields(
                        items=[("Train", len(fold.train_indices)), ("Val", len(fold.val_indices)), ("Test", test_size)]
                    )
                ],
            )
        ]

    # Multi-fold: one table, then each split's spread across the folds
    rows: list[dict[str, Cell]] = [
        {"fold": f.fold, "train": len(f.train_indices), "val": len(f.val_indices), "test": test_size} for f in raw.folds
    ]
    columns = [
        Column(key="fold", header="Fold"),
        Column(key="train", header="Train"),
        Column(key="val", header="Val"),
        Column(key="test", header="Test"),
    ]
    ranges = Fields(
        items=[
            ("Train", _size_range([len(f.train_indices) for f in raw.folds])),
            ("Val", _size_range([len(f.val_indices) for f in raw.folds])),
            ("Test", f"{test_size} (shared across folds)"),
        ]
    )

    return [
        Finding(
            severity="info",
            title="Split sizes across folds",
            brief=f"{len(raw.folds)} folds, test={test_size}",
            description="Split sizes per fold. Test set is shared across folds.",
            blocks=[Table(columns=columns, rows=rows), ranges],
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
) -> dict[str, Cell]:
    """Build one row of the cross-split distribution table, keyed by each split's lowercased name.

    A split's cell is its bare count when every split other than the full dataset holds the class at
    the same whole-number percentage, and ``"count (pct%)"`` otherwise. The full dataset's cell always
    shows its percentage.
    """
    row: dict[str, Cell] = {"class": cls}
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
            row[sn.lower()] = raw_counts[sn]
        else:
            row[sn.lower()] = f"{raw_counts[sn]} ({pcts[sn]}%)"
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
) -> list[dict[str, Cell]]:
    """Build the full row list including placeholder for omitted classes."""
    top, bottom, omitted = _truncate_classes(all_classes)
    rows: list[dict[str, Cell]] = [_make_distribution_row(cls, splits, split_totals) for cls in top]
    if omitted:
        placeholder: dict[str, Cell] = {"class": f"... {omitted} more ..."}
        for sn in splits:
            placeholder[sn.lower()] = ""
        rows.append(placeholder)
        rows.extend(_make_distribution_row(cls, splits, split_totals) for cls in bottom)
    return rows


def _build_cross_split_distribution(
    raw: DataSplittingRawOutput,
) -> list[Finding]:
    """Build cross-split class distribution table(s)."""
    # One name table for every split, so the per-split keys line up with the full ones.
    index2label = raw.label_stats_full.get("index2label")
    full_counts = _normalize_label_counts(raw.label_stats_full.get("label_counts_per_class"), index2label)
    if not full_counts:
        return []

    folds_with_stats = [f for f in raw.folds if f.label_stats_train]
    if not folds_with_stats:
        return []

    test_counts = _normalize_label_counts(raw.label_stats_test.get("label_counts_per_class"), index2label)
    has_test = bool(test_counts)

    findings: list[Finding] = []
    folds_to_show = folds_with_stats[:1] if len(folds_with_stats) > 1 else folds_with_stats

    for fold_info in folds_to_show:
        train_counts = _normalize_label_counts(fold_info.label_stats_train.get("label_counts_per_class"), index2label)
        val_counts = _normalize_label_counts(fold_info.label_stats_val.get("label_counts_per_class"), index2label)

        splits: dict[str, dict[str, int]] = {"Train": train_counts, "Val": val_counts}
        if has_test:
            splits["Test"] = test_counts
        splits["Full"] = full_counts

        split_totals = {name: sum(c.values()) for name, c in splits.items()}
        all_classes = sorted(full_counts.keys(), key=lambda c: full_counts.get(c, 0), reverse=True)
        rows = _build_distribution_rows(all_classes, splits, split_totals)

        max_dev, worst_class, worst_split = _max_proportion_deviation(splits, full_counts, split_totals)

        columns = [Column(key="class", header="Class"), *(Column(key=sn.lower(), header=sn) for sn in splits)]
        blocks: list[Block] = [Table(columns=columns, rows=rows)]
        if max_dev > 0:
            blocks.append(
                Paragraph(
                    text=f"Max proportion deviation from full dataset: {max_dev:.1f}pp ({worst_class} in {worst_split})"
                )
            )

        num_folds = len(folds_with_stats)
        if num_folds > 1:
            title = f"Class distribution across splits (fold {fold_info.fold} of {num_folds})"
        else:
            title = "Class distribution across splits"

        findings.append(
            Finding(
                severity="info",
                title=title,
                description="Per-class counts and proportions across splits.",
                blocks=blocks,
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
    index2label: dict[Any, str] | None = None,
) -> tuple[float, str, str, int]:
    """Find the worst proportion deviation across all folds.

    Returns ``(max_dev, worst_class, worst_split, worst_fold)``.
    """
    max_dev = 0.0
    worst_class = ""
    worst_split = ""
    worst_fold = 0

    for fold_info in folds_with_stats:
        train_counts = _normalize_label_counts(fold_info.label_stats_train.get("label_counts_per_class"), index2label)
        val_counts = _normalize_label_counts(fold_info.label_stats_val.get("label_counts_per_class"), index2label)

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


def _build_stratification_check(raw: DataSplittingRawOutput) -> list[Finding]:
    """Build a stratification quality health-check finding."""
    index2label = raw.label_stats_full.get("index2label")
    full_counts = _normalize_label_counts(raw.label_stats_full.get("label_counts_per_class"), index2label)
    if not full_counts:
        return []

    folds_with_stats = [f for f in raw.folds if f.label_stats_train]
    if not folds_with_stats:
        return []

    test_counts = _normalize_label_counts(raw.label_stats_test.get("label_counts_per_class"), index2label)
    full_total = sum(full_counts.values())
    if full_total == 0:
        return []

    global_max_dev, global_worst_class, global_worst_split, global_worst_fold = _worst_deviation_across_folds(
        folds_with_stats, full_counts, full_total, test_counts, index2label
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

    deviation: list[tuple[str, Scalar]] = [("Max proportion deviation", f"{global_max_dev}pp")]
    if global_max_dev > 0:
        full_pct = round(full_counts.get(global_worst_class, 0) / full_total * 100, 1)
        fold_label = f"fold {global_worst_fold} " if len(folds_with_stats) > 1 else ""
        worst = (
            f"class '{global_worst_class}' in {fold_label}{global_worst_split} "
            f"(deviation {global_max_dev}pp from {full_pct}% in full)"
        )
        deviation.append(("Worst", worst))
    checked = Fields(items=[("Folds checked", len(folds_with_stats)), ("Classes checked", len(full_counts))])

    return [
        Finding(
            severity=severity,
            title="Stratification quality",
            brief=brief,
            description="Checks whether class proportions in each split match the full dataset. "
            "Train proportions may differ if rebalancing was applied.",
            blocks=[Fields(items=deviation), checked],
        )
    ]


# ---------------------------------------------------------------------------
# Per-split coverage
# ---------------------------------------------------------------------------


def _coverage_finding(title: str, coverage: dict[str, Any], split_size: int) -> Finding:
    """One split's coverage: a warning when more than 5% of the split is uncovered."""
    uncovered = coverage.get("uncovered_indices", [])
    pct = (len(uncovered) / split_size * 100) if split_size > 0 else 0
    severity: Literal["ok", "info", "warning"] = "warning" if pct > 5 else "info"
    return Finding(
        severity=severity,
        title=title,
        blocks=[
            Fields(
                items=[
                    ("Uncovered count", len(uncovered)),
                    ("Split size", split_size),
                    ("Uncovered %", round(pct, 2)),
                    ("Coverage radius", coverage.get("coverage_radius")),
                ]
            )
        ],
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_findings(
    raw: DataSplittingRawOutput,
) -> list[Finding]:
    """Build human-readable findings from raw outputs."""
    findings: list[Finding] = []

    # --- 1. Full-dataset label distribution ---
    label_counts = raw.label_stats_full.get("label_counts_per_class")
    if label_counts:
        normalized = _normalize_label_counts(label_counts, raw.label_stats_full.get("index2label"))
        counts = list(normalized.values())
        max_count = max(counts) if counts else 0
        min_count = min(counts) if counts else 0
        ratio = max_count / min_count if min_count > 0 else float("inf")
        severity: Literal["ok", "info", "warning"] = "warning" if ratio > 10 else "info"
        findings.append(
            Finding(
                severity=severity,
                title="Class distribution (full dataset)",
                description=f"Max/min class ratio: {ratio:.1f}:1" if min_count > 0 else "Some classes have 0 samples",
                blocks=[ranked_table(normalized, headers=("Class", "Count"))],
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
            Finding(
                severity="info",
                title="Pre-split balance (mutual information)",
                description="Higher MI = stronger correlation between factor and class label.",
                blocks=[_factor_table(balance_data, "mi_value", "MI Score", "is_imbalanced")],
            )
        )

    # --- 6. Diversity scores ---
    diversity_data = raw.pre_split_diversity.get("factors")
    if diversity_data and isinstance(diversity_data, list):
        findings.append(
            Finding(
                severity="info",
                title="Pre-split diversity",
                description="Values near 1.0 = high diversity. Low diversity factors are flagged.",
                blocks=[_factor_table(diversity_data, "diversity_value", "Diversity", "is_low_diversity")],
            )
        )

    # --- 7. Per-split coverage ---
    for fold_info in raw.folds:
        for split_name, coverage in [("train", fold_info.coverage_train), ("val", fold_info.coverage_val)]:
            if coverage:
                split_size = len(fold_info.train_indices) if split_name == "train" else len(fold_info.val_indices)
                findings.append(
                    _coverage_finding(f"Coverage: fold {fold_info.fold} {split_name}", coverage, split_size)
                )

    if raw.coverage_test:
        findings.append(_coverage_finding("Coverage: test", raw.coverage_test, len(raw.test_indices)))

    return findings
