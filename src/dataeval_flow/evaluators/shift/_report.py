"""The drift evaluators' report section: the verdict as fields, or one row per chunk."""

__all__ = ["drift_section"]

from collections.abc import Mapping
from typing import Any

from dataeval_flow._blocks import Block, Cell, Column, Fields, Paragraph, Scalar, Table


def drift_section(output: Mapping[str, Any]) -> list[Block]:
    """A drift Output's JSON as a section: one row per chunk when chunked, else its verdict and statistics. Rows that
    are detections first say what was compared, and chunked, list the chunks left out."""
    data = output["data"]
    details = data.get("details")
    rows = data.get("rows") if isinstance(data.get("rows"), Mapping) else None
    blocks: list[Block] = [Paragraph(text=_compared(rows))] if rows is not None else []
    if isinstance(details, Mapping) and details.get("shape") == "table":
        blocks.append(_chunk_table(details["rows"], (rows or {}).get("chunk_images")))
        if rows is not None and rows.get("unassessed"):
            blocks.append(Paragraph(text=_unassessed(rows["unassessed"])))
        return blocks
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
    return [*blocks, Fields(items=items)]


def _compared(rows: Mapping[str, Any]) -> str:
    """What was compared: each source's detections, and the images they came from."""
    per_source = "; ".join(
        f"`{source}` {count:,} in {rows['images'][source]:,} images" for source, count in rows["compared"].items()
    )
    return f"Compared detections at confidence ≥ {rows['confidence']}: {per_source}."


def _unassessed(entries: list[Mapping[str, Any]]) -> str:
    listed = "; ".join(
        f"images {entry['images'][0]}–{entry['images'][1]} in `{entry['source']}` ({entry['reason']})"
        for entry in entries
    )
    return f"Not assessed: {listed}."


def _chunk_table(rows: list[Mapping[str, Any]], images: list[list[int]] | None = None) -> Table:
    """Per chunk: its distance, drawn as a bar against the drift thresholds, and its status. A chunk of whole images
    is labelled by its image range."""
    lowers = [row["lower_threshold"] for row in rows if row.get("lower_threshold") is not None]
    uppers = [row["upper_threshold"] for row in rows if row.get("upper_threshold") is not None]
    markers = [("Threshold", round(value, 4)) for value in (*lowers[:1], *uppers[:1])]
    columns = [
        Column(key="chunk", header="Chunk"),
        Column(key="distance", header="Distance", format="{:.4f}"),
        Column(key="distance", kind="bar", format="{:.4f}", markers=markers),
        Column(key="status", header="Status", align="left"),
    ]
    labels = [f"images {first}–{last}" for first, last in images] if images else [row["key"] for row in rows]
    cells: list[dict[str, Cell]] = [
        {"chunk": label, "distance": round(row["value"], 4), "status": "DRIFT" if row["drifted"] else "ok"}
        for label, row in zip(labels, rows, strict=True)
    ]
    return Table(columns=columns, rows=cells)
