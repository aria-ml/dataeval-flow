"""The drift evaluators' report section: the verdict as fields, or one row per chunk."""

__all__ = ["derived_threshold", "drift_section", "ood_section", "score_histogram"]

import math
from collections.abc import Mapping, Sequence
from typing import Any

from dataeval_flow._blocks import Block, Cell, Column, Fields, Paragraph, Scalar, Table


def drift_section(output: Mapping[str, Any]) -> list[Block]:
    """A drift Output's JSON as a section: one row per chunk when chunked, else its verdict and statistics. Rows that
    are detections first say what was compared, and chunked, list the chunks left out."""
    data = output["data"]
    details = data.get("details")
    rows = data.get("rows") if isinstance(data.get("rows"), Mapping) else None
    chunked = isinstance(details, Mapping) and details.get("shape") == "table"
    blocks: list[Block] = [Paragraph(text=_compared(rows))] if rows is not None and chunked else []
    if chunked:
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
    if rows is not None:
        items.append(("Compared", _compared_item(rows)))
    return [*blocks, Fields(items=items)]


def _per_source(rows: Mapping[str, Any]) -> str:
    return "; ".join(
        f"`{source}` {count:,} in {rows['images'][source]:,} images" for source, count in rows["compared"].items()
    )


def _compared(rows: Mapping[str, Any]) -> str:
    """What was compared: each source's detections, and the images they came from."""
    return f"Compared detections at confidence ≥ {rows['confidence']}: {_per_source(rows)}."


def _compared_item(rows: Mapping[str, Any]) -> str:
    """The same, as one field, where a per-class table lists it beside each class's verdict."""
    return f"{_per_source(rows)} (confidence ≥ {rows['confidence']})"


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


def _assessed(scores: Sequence[float | None], flags: Sequence[bool]) -> list[tuple[float, bool]]:
    """Each assessed image's score and flag: those whose score is finite, ``None`` or NaN being not assessed."""
    return [
        (float(score), bool(flag))
        for score, flag in zip(scores, flags, strict=True)
        if score is not None and math.isfinite(score)
    ]


def derived_threshold(scores: Sequence[float | None], flags: Sequence[bool]) -> float | None:
    """A detector's threshold, derived as legacy derived it from what it flagged: the highest score of an assessed
    image it did not flag, or the lowest score where it flagged every assessed one. ``None`` where it assessed none.
    DataEval keeps the true percentile private."""
    assessed = _assessed(scores, flags)
    if not assessed:
        return None
    unflagged = [score for score, flag in assessed if not flag]
    return max(unflagged) if unflagged else min(score for score, _ in assessed)


def score_histogram(
    scores: Sequence[float | None], flags: Sequence[bool], threshold: float | None, n_bins: int = 10
) -> list[Block]:
    """The assessed images' scores in `n_bins` bins: in-distribution and OOD counts per bin, the threshold's bin
    marked. Legacy's histogram, over assessed images."""
    pairs = _assessed(scores, flags)
    if not pairs:
        return []
    low, high = min(score for score, _ in pairs), max(score for score, _ in pairs)
    if high == low:
        return [Paragraph(text=f"All scores = {low:.4f}")]
    width = (high - low) / n_bins
    rows: list[dict[str, Cell]] = []
    for index in range(n_bins):
        start = low + index * width
        stop = start + width
        inside = [flag for score, flag in pairs if start <= score < stop or (index == n_bins - 1 and score == stop)]
        ood = sum(inside)
        # The threshold's bin is found by the bounds as printed, to three places.
        at = threshold is not None and float(f"{start:.3f}") <= threshold < float(f"{stop:.3f}")
        rows.append(
            {
                "range": f"{start:.3f}-{stop:.3f}",
                "in": len(inside) - ood,
                "ood": ood,
                "bar": [len(inside) - ood, ood],
                "marker": "← threshold" if at else "",
            }
        )
    columns = [
        Column(key="range", header="Range", align="right"),
        Column(key="in", header="In"),
        Column(key="ood", header="OOD"),
        Column(key="bar", kind="stacked", series=["in-dist", "OOD"]),
        Column(key="marker", align="left"),
    ]
    return [Table(columns=columns, rows=rows)]


def ood_section(output: Mapping[str, Any]) -> list[Block]:
    """An OOD Output's JSON as a section: its score histogram, and how many images it flagged of those it assessed,
    against the threshold derived from its flags. Rows that are detections add what was compared, and list the
    images not assessed."""
    data = output["data"]
    scores = list(data["instance_score"])
    flags = [bool(flag) for flag in data["is_ood"]]
    rows = data.get("rows") if isinstance(data.get("rows"), Mapping) else None
    threshold = derived_threshold(scores, flags)
    assessed = len(_assessed(scores, flags))
    shown: Scalar = round(threshold, 6) if threshold is not None else "—"
    items: list[tuple[str, Scalar]] = [("Flagged", sum(flags)), ("Assessed", assessed), ("Threshold", shown)]
    if rows is not None:
        items.append(("Compared", _compared_item(rows)))
    blocks: list[Block] = [*score_histogram(scores, flags, threshold), Fields(items=items)]
    if rows is not None and rows.get("unassessed"):
        listed = ", ".join(str(image) for image in rows["unassessed"])
        blocks.append(
            Paragraph(text=f"Not assessed, with no detection at confidence ≥ {rows['confidence']}: images {listed}.")
        )
    return blocks
