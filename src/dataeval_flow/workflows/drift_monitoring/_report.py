"""Findings builders for the drift monitoring workflow."""

from __future__ import annotations

from typing import Literal

import numpy as np

from dataeval_flow._blocks import Cell, Column, Fields, Scalar, Table
from dataeval_flow.workflows._base import Finding
from dataeval_flow.workflows.drift_monitoring._config import (
    DriftMonitoringConfig,
    DriftMonitoringHealthThresholds,
)
from dataeval_flow.workflows.drift_monitoring._outputs import (
    ChunkResultDict,
    ClasswiseDriftRowDict,
    DetectorResultDict,
    DriftMonitoringRawOutput,
)


def _severity_for_detector(
    drifted: bool,
    thresholds: DriftMonitoringHealthThresholds,
) -> Literal["ok", "info", "warning"]:
    """Determine severity for a non-chunked detector result."""
    if drifted and thresholds.any_drift_is_warning:
        return "warning"
    return "info" if drifted else "ok"


def _severity_for_chunks(
    chunks: list[ChunkResultDict],
    thresholds: DriftMonitoringHealthThresholds,
) -> Literal["ok", "info", "warning"]:
    """Determine severity for chunked results."""
    if not chunks:
        return "ok"
    n_drifted = sum(1 for c in chunks if c["drifted"])
    pct = 100.0 * n_drifted / len(chunks) if chunks else 0.0

    # Check consecutive drift window
    max_consecutive = _max_consecutive_drifted(chunks)

    if pct >= thresholds.chunk_drift_pct_warning:
        return "warning"
    if max_consecutive >= thresholds.consecutive_chunks_warning:
        return "warning"
    return "info" if n_drifted > 0 else "ok"


def _max_consecutive_drifted(chunks: list[ChunkResultDict]) -> int:
    """Count the longest run of consecutive drifted chunks."""
    max_run = 0
    current_run = 0
    for c in chunks:
        if c["drifted"]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _classwise_table(rows: list[ClasswiseDriftRowDict]) -> Table:
    """Per class: its distance, its p-value when any class has one, a bar of the distance's size, and its status."""
    columns = [Column(key="class_name", header="Class"), Column(key="distance", header="Distance", format="{:.4f}")]
    if any(row["p_val"] is not None for row in rows):
        # Two significant figures, as data analysis prints its p-values: a small one reads 0.0003, not 0.00.
        columns.append(Column(key="p_val", header="PVal", format="{:.2g}"))
    columns += [Column(key="abs_distance", kind="bar"), Column(key="status", header="Status", align="left")]
    cells: list[dict[str, Cell]] = [
        {
            "class_name": row["class_name"],
            "distance": round(row["distance"], 4),
            "p_val": row["p_val"],
            "abs_distance": abs(round(row["distance"], 4)),
            "status": "DRIFT" if row["drifted"] else "ok",
        }
        for row in rows
    ]
    return Table(columns=columns, rows=cells)


def _chunk_table(chunks: list[ChunkResultDict]) -> Table:
    """Per chunk: its distance, drawn as a bar against the drift thresholds, and its status."""
    lowers = [c["lower_threshold"] for c in chunks if c["lower_threshold"] is not None]
    uppers = [c["upper_threshold"] for c in chunks if c["upper_threshold"] is not None]
    markers = [("Threshold", round(value, 4)) for value in (*lowers[:1], *uppers[:1])]
    columns = [
        Column(key="chunk", header="Chunk"),
        Column(key="distance", header="Distance", format="{:.4f}"),
        Column(key="distance", kind="bar", format="{:.4f}", markers=markers),
        Column(key="status", header="Status", align="left"),
    ]
    cells: list[dict[str, Cell]] = [
        {"chunk": c["key"], "distance": round(c["value"], 4), "status": "DRIFT" if c["drifted"] else "ok"}
        for c in chunks
    ]
    return Table(columns=columns, rows=cells)


def _build_detector_finding(
    name: str,
    result: DetectorResultDict,
    thresholds: DriftMonitoringHealthThresholds,
    classwise_rows: list[ClasswiseDriftRowDict] | None = None,
) -> Finding:
    """Build a finding for a single detector (non-chunked)."""
    drifted = result["drifted"]
    severity = _severity_for_detector(drifted, thresholds)

    # Classwise breakdown → a per-class table in place of the detector's own values
    if classwise_rows:
        drifted_classes = [r["class_name"] for r in classwise_rows if r["drifted"]]
        n_cls_drifted = len(drifted_classes)
        n_total = len(classwise_rows)
        description = f"Classes drifted: {', '.join(drifted_classes)}" if drifted_classes else "No classes drifted"
        if n_cls_drifted > 0 and thresholds.classwise_any_drift_is_warning:
            severity = "warning"

        return Finding(
            severity=severity,
            title=name,
            brief=f"{n_cls_drifted}/{n_total} classes drifted",
            description=description,
            blocks=[_classwise_table(classwise_rows)],
        )

    distance = round(result["distance"], 4)
    threshold = round(result["threshold"], 4)
    values: list[tuple[str, Scalar]] = [
        ("Distance", distance),
        ("Threshold", threshold),
        ("Metric", result["metric_name"]),
    ]
    description = f"{name}: distance={distance}, threshold={threshold}"

    # Add p_val from details if available
    details = result.get("details", {})
    if isinstance(details, dict) and "p_val" in details:
        p_val = round(float(details["p_val"]), 6)
        values.append(("p-value", p_val))
        description += f", p={p_val}"

    # Univariate: summarize feature drift
    if isinstance(details, dict) and "feature_drift" in details:
        fd = details["feature_drift"]
        if isinstance(fd, list):
            n_drifted = sum(fd)
            n_total = len(fd)
        else:
            n_drifted = int(np.sum(fd))
            n_total = len(fd)
        values.append(("Features drifted", f"{n_drifted} / {n_total}"))

    return Finding(
        severity=severity,
        title=name,
        description=description,
        blocks=[Fields(items=values)],
    )


def _build_chunked_finding(
    name: str,
    result: DetectorResultDict,
    thresholds: DriftMonitoringHealthThresholds,
) -> Finding:
    """Build a table finding for chunked detector results."""
    chunks = result.get("chunks", [])
    if not chunks:
        return _build_detector_finding(name, result, thresholds)

    severity = _severity_for_chunks(chunks, thresholds)
    n_drifted = sum(1 for c in chunks if c["drifted"])
    pct = 100.0 * n_drifted / len(chunks) if chunks else 0.0
    max_consec = _max_consecutive_drifted(chunks)

    description = f"{n_drifted}/{len(chunks)} chunks drifted ({pct:.0f}%) | max consecutive: {max_consec}"

    return Finding(
        severity=severity,
        title=name,
        description=description,
        blocks=[_chunk_table(chunks)],
    )


def build_findings(
    raw: DriftMonitoringRawOutput,
    params: DriftMonitoringConfig,
    detector_names: dict[str, str],
) -> list[Finding]:
    """Build all report findings from raw results."""
    findings: list[Finding] = []

    # Index classwise results by detector display name for per-detector lookup
    classwise_by_detector: dict[str, list[ClasswiseDriftRowDict]] = {}
    if raw.classwise:
        for cw in raw.classwise:
            classwise_by_detector[cw["detector"]] = cw["rows"]

    for method_key, result in raw.detectors.items():
        name = detector_names.get(method_key, method_key)
        cw_rows = classwise_by_detector.get(name)
        if result.get("chunks"):
            findings.append(_build_chunked_finding(name, result, params.health_thresholds))
        else:
            findings.append(_build_detector_finding(name, result, params.health_thresholds, cw_rows))

    return findings
