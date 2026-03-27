"""Findings builders for the data-prioritization workflow."""

from __future__ import annotations

from typing import Any, Literal

from dataeval_flow.workflow.base import Reportable
from dataeval_flow.workflows.prioritization.outputs import (
    CleaningSummaryDict,
    DataPrioritizationRawOutputs,
    PerDatasetPrioritizationDict,
)
from dataeval_flow.workflows.prioritization.params import (
    DataPrioritizationHealthThresholds,
    DataPrioritizationParameters,
)


def _severity_for_cleaning(
    removed_pct: float,
    thresholds: DataPrioritizationHealthThresholds,
) -> Literal["ok", "info", "warning"]:
    """Determine severity based on percentage of items removed by cleaning."""
    if removed_pct >= thresholds.cleaning_removed_pct_warning:
        return "warning"
    if removed_pct > 0:
        return "info"
    return "ok"


def _build_cleaning_finding(
    summary: CleaningSummaryDict,
    thresholds: DataPrioritizationHealthThresholds,
) -> Reportable:
    """Build a finding for the cleaning step."""
    total = summary["total_combined"]
    removed = summary["total_removed"]
    removed_pct = (removed / total * 100) if total > 0 else 0.0
    severity = _severity_for_cleaning(removed_pct, thresholds)

    data: dict[str, Any] = {
        "brief": f"{removed} items ({removed_pct:.1f}%)",
        "total_combined": total,
        "outliers_flagged": summary["outliers_flagged"],
        "duplicates_flagged": summary["duplicates_flagged"],
        "total_removed": removed,
        "removed_pct": f"{removed_pct:.1f}%",
    }

    description = (
        f"Pruning removed {removed}/{total} items ({removed_pct:.1f}%): "
        f"{summary['outliers_flagged']} outliers, {summary['duplicates_flagged']} duplicates"
    )

    return Reportable(
        report_type="key_value",
        severity=severity,
        title="Pruning",
        data=data,
        description=description,
    )


def _build_prioritization_finding(
    result: PerDatasetPrioritizationDict,
    method: str,
    order: str,
    policy: str,
) -> Reportable:
    """Build a finding for a single dataset's prioritization results."""
    n_items = result["cleaned_size"]
    top_n = min(10, len(result["prioritized_indices"]))
    top_indices = result["prioritized_indices"][:top_n]

    data: dict[str, Any] = {
        "brief": f"{n_items} items",
        "source": result["source_name"],
        "original_size": result["original_size"],
        "cleaned_size": result["cleaned_size"],
        "method": method,
        "order": order,
        "policy": policy,
        "top_indices": top_indices,
    }

    if result["scores"] is not None:
        top_scores = result["scores"][:top_n]
        data["top_scores"] = [round(s, 4) for s in top_scores]

    description = f"{result['source_name']}: {n_items} items prioritized via {method} ({order}, {policy})"

    return Reportable(
        report_type="key_value",
        severity="info",
        title=f"Prioritization: {result['source_name']}",
        data=data,
        description=description,
    )


def build_findings(
    raw: DataPrioritizationRawOutputs,
    params: DataPrioritizationParameters,
) -> list[Reportable]:
    """Build all report findings from raw results."""
    findings: list[Reportable] = []

    # Cleaning finding
    if raw.cleaning_summary is not None:
        findings.append(_build_cleaning_finding(raw.cleaning_summary, params.health_thresholds))

    # Per-dataset prioritization findings
    findings.extend(
        _build_prioritization_finding(result, raw.method, raw.order, raw.policy) for result in raw.prioritizations
    )

    return findings
