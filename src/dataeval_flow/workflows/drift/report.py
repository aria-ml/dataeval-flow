"""Findings builders for the drift monitoring workflow."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from dataeval_flow.workflow.base import Reportable
from dataeval_flow.workflows.drift.outputs import (
    ChunkResultDict,
    ClasswiseDriftRowDict,
    DetectorResultDict,
    DriftMonitoringRawOutputs,
)
from dataeval_flow.workflows.drift.params import (
    DriftHealthThresholds,
    DriftMonitoringParameters,
)


def _severity_for_detector(
    drifted: bool,
    thresholds: DriftHealthThresholds,
) -> Literal["ok", "info", "warning"]:
    """Determine severity for a non-chunked detector result."""
    if drifted and thresholds.any_drift_is_warning:
        return "warning"
    return "info" if drifted else "ok"


def _severity_for_chunks(
    chunks: list[ChunkResultDict],
    thresholds: DriftHealthThresholds,
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


def _build_detector_finding(
    name: str,
    result: DetectorResultDict,
    thresholds: DriftHealthThresholds,
    classwise_rows: list[ClasswiseDriftRowDict] | None = None,
) -> Reportable:
    """Build a finding for a single detector (non-chunked)."""
    drifted = result["drifted"]
    severity = _severity_for_detector(drifted, thresholds)
    data: dict[str, Any] = {
        "distance": round(result["distance"], 4),
        "threshold": round(result["threshold"], 4),
        "metric": result["metric_name"],
    }

    # Add p_val from details if available
    details = result.get("details", {})
    if isinstance(details, dict) and "p_val" in details:
        data["p_val"] = round(float(details["p_val"]), 6)

    # Univariate: summarize feature drift
    if isinstance(details, dict) and "feature_drift" in details:
        fd = details["feature_drift"]
        if isinstance(fd, list):
            n_drifted = sum(fd)
            n_total = len(fd)
        else:
            n_drifted = int(np.sum(fd))
            n_total = len(fd)
        data["features_drifted"] = f"{n_drifted} / {n_total}"

    description = f"{name}: distance={data['distance']}, threshold={data['threshold']}"
    if "p_val" in data:
        description += f", p={data['p_val']}"

    # Classwise breakdown → render as a classwise_table instead of key_value
    if classwise_rows:
        drifted_classes = [r["class_name"] for r in classwise_rows if r["drifted"]]
        n_cls_drifted = len(drifted_classes)
        n_total = len(classwise_rows)
        description = f"Classes drifted: {', '.join(drifted_classes)}" if drifted_classes else "No classes drifted"
        if n_cls_drifted > 0 and thresholds.classwise_any_drift_is_warning:
            severity = "warning"

        table_rows = [
            {
                "Class": r["class_name"],
                "Distance": round(r["distance"], 4),
                "PVal": round(r["p_val"], 6) if r["p_val"] is not None else None,
                "Status": "DRIFT" if r["drifted"] else "ok",
            }
            for r in classwise_rows
        ]

        data["table_rows"] = table_rows
        data["brief"] = f"{n_cls_drifted}/{n_total} classes drifted"

        return Reportable(
            report_type="classwise_table",
            severity=severity,
            title=name,
            data=data,
            description=description,
        )

    return Reportable(
        report_type="key_value",
        severity=severity,
        title=name,
        data=data,
        description=description,
    )


def _build_chunked_finding(
    name: str,
    result: DetectorResultDict,
    thresholds: DriftHealthThresholds,
) -> Reportable:
    """Build a table finding for chunked detector results."""
    chunks = result.get("chunks", [])
    if not chunks:
        return _build_detector_finding(name, result, thresholds)

    severity = _severity_for_chunks(chunks, thresholds)
    n_drifted = sum(1 for c in chunks if c["drifted"])
    pct = 100.0 * n_drifted / len(chunks) if chunks else 0.0
    max_consec = _max_consecutive_drifted(chunks)

    rows: list[dict[str, Any]] = [
        {
            "Chunk": c["key"],
            "Distance": round(c["value"], 4),
            "UpperThreshold": round(c["upper_threshold"], 4) if c["upper_threshold"] is not None else None,
            "LowerThreshold": round(c["lower_threshold"], 4) if c["lower_threshold"] is not None else None,
            "Status": "DRIFT" if c["drifted"] else "ok",
        }
        for c in chunks
    ]

    description = f"{n_drifted}/{len(chunks)} chunks drifted ({pct:.0f}%) | max consecutive: {max_consec}"

    return Reportable(
        report_type="chunk_table",
        severity=severity,
        title=name,
        data={
            "table_rows": rows,
            "drift_flags": [c["drifted"] for c in chunks],
        },
        description=description,
    )


def build_findings(
    raw: DriftMonitoringRawOutputs,
    params: DriftMonitoringParameters,
    detector_names: dict[str, str],
) -> list[Reportable]:
    """Build all report findings from raw results."""
    findings: list[Reportable] = []

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
