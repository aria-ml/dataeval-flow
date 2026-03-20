"""Findings builders for the dataset splitting workflow."""

from __future__ import annotations

from typing import Any

from dataeval_flow.workflow.base import Reportable
from dataeval_flow.workflows.splitting.outputs import DataSplittingRawOutputs


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


def build_findings(
    raw: DataSplittingRawOutputs,
) -> list[Reportable]:
    """Build human-readable findings from raw outputs."""
    findings: list[Reportable] = []

    # --- Full-dataset label distribution ---
    label_counts = raw.label_stats_full.get("label_counts_per_class")
    if label_counts:
        # label_counts may be a list (from NDArray.tolist()) or a dict
        counts = list(label_counts.values()) if isinstance(label_counts, dict) else list(label_counts)
        max_count = max(counts) if counts else 0
        min_count = min(counts) if counts else 0
        ratio = max_count / min_count if min_count > 0 else float("inf")
        severity = "warning" if ratio > 10 else "info"
        table_data = (
            {str(k): v for k, v in label_counts.items()}
            if isinstance(label_counts, dict)
            else {str(i): c for i, c in enumerate(label_counts)}
        )
        findings.append(
            Reportable(
                report_type="table",
                severity=severity,
                title="Class distribution (full dataset)",
                data={
                    "table_data": table_data,
                    "table_headers": ("Class", "Count"),
                },
                description=f"Max/min class ratio: {ratio:.1f}:1" if min_count > 0 else "Some classes have 0 samples",
            )
        )

    # --- Split sizes ---
    findings.extend(
        Reportable(
            report_type="key_value",
            severity="info",
            title=f"Fold {fold_info.fold} split sizes",
            data={
                "train": len(fold_info.train_indices),
                "val": len(fold_info.val_indices),
                "test": len(raw.test_indices),
            },
        )
        for fold_info in raw.folds
    )

    # --- Balance scores ---
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

    # --- Diversity scores ---
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

    # --- Per-split coverage ---
    for fold_info in raw.folds:
        for split_name, coverage in [("train", fold_info.coverage_train), ("val", fold_info.coverage_val)]:
            if coverage:
                uncovered = coverage.get("uncovered_indices", [])
                split_size = len(fold_info.train_indices) if split_name == "train" else len(fold_info.val_indices)
                pct = (len(uncovered) / split_size * 100) if split_size > 0 else 0
                severity = "warning" if pct > 5 else "info"
                findings.append(
                    Reportable(
                        report_type="key_value",
                        severity=severity,
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
        severity = "warning" if pct > 5 else "info"
        findings.append(
            Reportable(
                report_type="key_value",
                severity=severity,
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
