"""Findings builders for the data-prioritization workflow."""

from __future__ import annotations

from typing import Literal

from dataeval_flow._blocks import Block, Cell, Column, Fields, ItemRef, Section, Table
from dataeval_flow.workflows._base import Finding
from dataeval_flow.workflows._tables import table_limits
from dataeval_flow.workflows.data_prioritization._config import (
    DataPrioritizationConfig,
    DataPrioritizationHealthThresholds,
)
from dataeval_flow.workflows.data_prioritization._outputs import (
    CleaningSummaryDict,
    DataPrioritizationRawOutput,
    PerDatasetPrioritizationDict,
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
) -> Finding:
    """Build a finding for the cleaning step."""
    total = summary["total_combined"]
    removed = summary["total_removed"]
    removed_pct = (removed / total * 100) if total > 0 else 0.0
    severity = _severity_for_cleaning(removed_pct, thresholds)

    description = (
        f"Pruning removed {removed}/{total} items ({removed_pct:.1f}%): "
        f"{summary['outliers_flagged']} outliers, {summary['duplicates_flagged']} duplicates"
    )
    counts = Fields(
        items=[
            ("Total combined", total),
            ("Outliers flagged", summary["outliers_flagged"]),
            ("Duplicates flagged", summary["duplicates_flagged"]),
            ("Total removed", removed),
            ("Removed", f"{removed_pct:.1f}%"),
        ]
    )

    return Finding(
        severity=severity,
        title="Pruning",
        brief=f"{removed} items ({removed_pct:.1f}%)",
        description=description,
        blocks=[counts],
    )


# The ranking's first and last this many items are listed.
_ENDS = 25


def _ranked_blocks(result: PerDatasetPrioritizationDict) -> list[Block]:
    """The ranking's 25 highest-priority and 25 lowest-priority items: each one's rank, thumbnail, item and score.

    Rank is the position in the ranking, which a stratified or class-balanced policy doesn't keep in score
    order. Score shows where the method gives one. A ranking of 50 or fewer is split between the two.
    """
    indices = result["prioritized_indices"]
    scores = result["scores"]
    columns = [
        Column(key="rank", header="Rank"),
        Column(key="image", kind="image"),
        Column(key="item", header="Item"),
        *([Column(key="score", header="Score", format="{:.4g}")] if scores is not None else []),
    ]
    ends = [("Highest priority", range(min(_ENDS, len(indices))))]
    ends.append(("Lowest priority", range(max(_ENDS, len(indices) - _ENDS), len(indices))))
    blocks: list[Block] = []
    for title, positions in ends:
        rows: list[dict[str, Cell]] = [
            {
                "rank": position + 1,
                "image": ItemRef(source=result["source_name"], index=indices[position]),
                "item": indices[position],
                "score": None if scores is None else scores[position],
            }
            for position in positions
        ]
        if rows:
            blocks.append(
                Section(title=title, blocks=[Table(columns=columns, rows=rows, preview=table_limits().preview)])
            )
    return blocks


def _build_prioritization_finding(
    result: PerDatasetPrioritizationDict,
    method: str,
    order: str,
    policy: str,
) -> Finding:
    """Build a finding for a single dataset's prioritization results."""
    n_items = result["cleaned_size"]
    description = f"{result['source_name']}: {n_items} items prioritized via {method} ({order}, {policy})"
    run = Fields(
        items=[
            ("Source", result["source_name"]),
            ("Original size", result["original_size"]),
            ("Cleaned size", result["cleaned_size"]),
            ("Method", method),
            ("Order", order),
            ("Policy", policy),
        ]
    )

    return Finding(
        severity="info",
        title=f"Prioritization: {result['source_name']}",
        brief=f"{n_items} items",
        description=description,
        blocks=[run, *_ranked_blocks(result)],
    )


def build_findings(
    raw: DataPrioritizationRawOutput,
    params: DataPrioritizationConfig,
) -> list[Finding]:
    """Build all report findings from raw results."""
    findings: list[Finding] = []

    # Cleaning finding
    if raw.cleaning_summary is not None:
        findings.append(_build_cleaning_finding(raw.cleaning_summary, params.health_thresholds))

    # Per-dataset prioritization findings
    findings.extend(
        _build_prioritization_finding(result, raw.method, raw.order, raw.policy) for result in raw.prioritizations
    )

    return findings
