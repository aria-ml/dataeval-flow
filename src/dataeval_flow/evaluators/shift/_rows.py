"""Drift over rows that are detections: chunks of whole images, chunks without detections left out, and what was
compared (uncertainty-drift spec §4.3, §4.4, §7)."""

__all__ = ["DriftRowsOutput", "detect_drift_by_image", "image_chunks"]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from dataeval.shift import DriftOutput

from dataeval_flow.evaluators._fields import require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.shift._config import ChunkedDriftConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from dataeval_flow._predictions import Predictions


@dataclass(frozen=True, repr=False)
class DriftRowsOutput(DriftOutput[Any]):
    """DataEval's ``DriftOutput`` for rows that are detections, with what was compared under ``rows``.

    ``rows`` holds ``unit`` (``"detections"``), ``compared`` and ``images`` (per source, its rows and the images they
    came from), ``confidence``, and, chunked, ``chunk_images`` (each assessed chunk's first and last image, in
    ``details`` order) and ``unassessed`` (each chunk left out: its source, first and last image, and why).
    """

    rows: Mapping[str, Any] | None = None


def image_chunks(n_images: int, chunking: ChunkedDriftConfig) -> "list[NDArray[np.intp]]":
    """`chunking`'s chunks over `n_images` images, cut as DataEval cuts rows.

    ``chunk_size`` makes chunks of that many images, its remainder kept, dropped or appended as ``incomplete`` says
    (``keep`` unset); otherwise ``chunk_count`` makes that many near-equal chunks.
    """
    if chunking.chunk_size is None:
        count = cast(int, chunking.chunk_count)
        return [chunk.astype(np.intp) for chunk in np.array_split(np.arange(n_images), count)]
    size = chunking.chunk_size
    chunks = [np.arange(start, start + size, dtype=np.intp) for start in range(0, n_images - size + 1, size)]
    left = size * (n_images // size)
    if left < n_images and (chunking.incomplete or "keep") == "keep":
        chunks.append(np.arange(left, n_images, dtype=np.intp))
    elif left < n_images and chunking.incomplete == "append" and chunks:
        chunks[-1] = np.arange(chunks[-1][0], n_images, dtype=np.intp)
    return chunks


def detect_drift_by_image(
    detector: Any, chunking: ChunkedDriftConfig | None, inputs: Sequence[EvaluatorInputs]
) -> DriftRowsOutput:
    """Fit `detector` on the first source's rows and predict on the last's, chunked by whole images where `chunking`
    says so. A middle source is ``drift-wasserstein``'s validation set.

    Raises
    ------
    ValueError
        When a source holds no detections, or when fewer than 3 reference chunks do and the threshold is derived from
        their spread.
    """
    from dataeval_flow.evaluators.shift._evaluator import chunked_arguments

    predictions = [require(prepared.predictions, "predictions", prepared.source) for prepared in inputs]
    embeddings = [require(prepared.embeddings, "embeddings", prepared.source) for prepared in inputs]
    confidence = predictions[0].confidence
    for prepared, made in zip(inputs, predictions, strict=True):
        if len(made.scores) == 0:
            raise ValueError(f"No detections at `confidence` ≥ {confidence} in `{prepared.source}`.")
    facts: dict[str, Any] = {
        "unit": "detections",
        "compared": {prepared.source: len(made.scores) for prepared, made in zip(inputs, predictions, strict=True)},
        "images": {prepared.source: made.items for prepared, made in zip(inputs, predictions, strict=True)},
        "confidence": confidence,
    }
    if chunking is None:
        return _with_rows(detector.fit(*embeddings[:-1]).predict(embeddings[-1]), facts)
    reference_chunks = image_chunks(predictions[0].items, chunking)
    if not reference_chunks:
        raise ValueError(
            f"`chunking` cuts `{inputs[0].source}`'s {predictions[0].items} images into no chunk: use a smaller "
            "`chunk_size`, or `incomplete: keep`."
        )
    # DataEval cuts the data to test at the reference's first chunk's size, its remainder appended; here, in images.
    test_size = max(1, len(reference_chunks[0]))
    test_chunks = image_chunks(predictions[-1].items, ChunkedDriftConfig(chunk_size=test_size, incomplete="append"))
    if not test_chunks:
        raise ValueError(
            f"`{inputs[-1].source}` holds {predictions[-1].items} images, fewer than one chunk of {test_size}: "
            "chunked drift needs at least one whole chunk."
        )
    reference_kept, reference_left = _groups(inputs[0].source, predictions[0], reference_chunks)
    test_kept, test_left = _groups(inputs[-1].source, predictions[-1], test_chunks)
    threshold = {key: value for key, value in chunked_arguments(chunking).items() if key == "threshold"}
    fitted = detector.chunked(chunker=lambda _: [rows for _, rows in reference_kept], **threshold)
    try:
        fitted.fit(*embeddings[:-1])
    except ValueError as error:
        if len(reference_kept) >= 3:
            raise
        raise ValueError(
            f"`chunking` cuts `{inputs[0].source}`'s {predictions[0].items} images into {len(reference_chunks)} "
            f"chunks, {len(reference_kept)} with detections, and a threshold derived from their spread needs at least "
            "3: use a smaller `chunk_size` or a larger `chunk_count`."
        ) from error
    output = fitted.predict(embeddings[-1], chunk_indices=[rows.tolist() for _, rows in test_kept])
    facts["chunk_images"] = [[int(chunk[0]), int(chunk[-1])] for chunk, _ in test_kept]
    facts["unassessed"] = reference_left + test_left
    return _with_rows(output, facts)


def _groups(
    source: str, predictions: "Predictions", chunks: "Sequence[NDArray[np.intp]]"
) -> "tuple[list[tuple[NDArray[np.intp], NDArray[np.intp]]], list[dict[str, Any]]]":
    """Each chunk holding detections, with its rows, in order; and each chunk holding none, as an unassessed entry."""
    rows = cast("NDArray[np.intp]", predictions.rows)
    kept: list[tuple[NDArray[np.intp], NDArray[np.intp]]] = []
    left: list[dict[str, Any]] = []
    for chunk in chunks:
        if not len(chunk):
            continue
        own = np.flatnonzero(np.isin(rows, chunk))
        if len(own):
            kept.append((chunk, own))
        else:
            left.append({"source": source, "images": [int(chunk[0]), int(chunk[-1])], "reason": "no detections"})
    return kept, left


def _with_rows(output: DriftOutput[Any], facts: Mapping[str, Any]) -> DriftRowsOutput:
    """`output`, with what was compared under ``rows`` and DataEval's record of the call."""
    made = DriftRowsOutput(**output.data(), rows=dict(facts))
    object.__setattr__(made, "_meta", output.meta())
    return made
