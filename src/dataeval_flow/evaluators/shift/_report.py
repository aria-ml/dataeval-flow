"""The drift evaluators' report section: the verdict as fields, or one row per chunk."""

__all__ = ["drift_section"]

from collections.abc import Mapping
from typing import Any

from dataeval_flow._blocks import Block, Cell, Column, Fields, Scalar, Table


def drift_section(output: Mapping[str, Any]) -> list[Block]:
    """A drift Output's JSON as a section: one row per chunk when chunked, else its verdict and statistics."""
    data = output["data"]
    details = data.get("details")
    if isinstance(details, Mapping) and details.get("shape") == "table":
        return [_chunk_table(details["rows"])]
    items: list[tuple[str, Scalar]] = [
        ("Drifted", "yes" if data["drifted"] else "no"),
        ("Distance", round(float(data["distance"]), 4)),
        ("Threshold", round(float(data["threshold"]), 4)),
        ("Metric", data["metric_name"]),
    ]
    if isinstance(details, Mapping) and details.get("p_val") is not None:
        items.append(("p-value", round(float(details["p_val"]), 6)))
    if isinstance(details, Mapping) and details.get("feature_drift") is not None:
        flags = list(details["feature_drift"])
        items.append(("Features drifted", f"{sum(bool(flag) for flag in flags)} / {len(flags)}"))
    return [Fields(items=items)]


def _chunk_table(rows: list[Mapping[str, Any]]) -> Table:
    """Per chunk: its distance, drawn as a bar against the drift thresholds, and its status."""
    lowers = [row["lower_threshold"] for row in rows if row.get("lower_threshold") is not None]
    uppers = [row["upper_threshold"] for row in rows if row.get("upper_threshold") is not None]
    markers = [("Threshold", round(value, 4)) for value in (*lowers[:1], *uppers[:1])]
    columns = [
        Column(key="chunk", header="Chunk"),
        Column(key="distance", header="Distance", format="{:.4f}"),
        Column(key="distance", kind="bar", format="{:.4f}", markers=markers),
        Column(key="status", header="Status", align="left"),
    ]
    cells: list[dict[str, Cell]] = [
        {"chunk": row["key"], "distance": round(row["value"], 4), "status": "DRIFT" if row["drifted"] else "ok"}
        for row in rows
    ]
    return Table(columns=columns, rows=cells)
