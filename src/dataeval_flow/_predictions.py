"""A model's predictions over a source, for the `uncertainty` extractor: each row's class scores, and the item each
row came from (uncertainty-drift spec §3)."""

__all__ = ["Predictions", "compute_predictions", "membership", "runs_model", "uncertainty_rows"]

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from dataeval.models import ModelIOSpec
    from numpy.typing import NDArray

    from dataeval_flow.config.extractors import UncertaintyExtractorConfig

_EPSILON = 1e-6
"""How far sigmoid scores are clipped from 0 and 1 before they are turned back into logits."""


@dataclass(frozen=True)
class Predictions:
    """A model's predictions over a source, as the `uncertainty` extractor reads them.

    ``scores`` holds each row's class scores, ``(n_rows, n_classes)``, as DataEval reads them: ``preds_type`` says
    whether they are ``logits`` or ``probs``, and sigmoid scores arrive as the logits recovered from them. ``rows``
    holds each row's item index, its position in the source after the source's view, or is ``None`` where each row is
    one item, as a classifier's are. ``items`` counts the source's items. ``confidence`` is the threshold a detector's
    boxes met, ``None`` for a classifier.
    """

    scores: "NDArray[np.float32]"
    rows: "NDArray[np.intp] | None"
    items: int
    preds_type: Literal["logits", "probs"]
    confidence: float | None

    def sliced(self, mask: "NDArray[np.bool_]") -> "Predictions":
        """These predictions narrowed to the rows `mask` selects."""
        rows = None if self.rows is None else self.rows[mask]
        return dataclasses.replace(self, scores=self.scores[mask], rows=rows)

    def to_arrays(self) -> "dict[str, NDArray[Any]]":
        """The arrays a cache stores; the rest follows from the config that made them."""
        arrays = {"scores": self.scores, "items": np.asarray(self.items)}
        return arrays if self.rows is None else arrays | {"rows": self.rows}

    @classmethod
    def from_arrays(cls, arrays: "Mapping[str, NDArray[Any]]", config: "UncertaintyExtractorConfig") -> "Predictions":
        """The predictions :meth:`to_arrays` stored, under the config that made them."""
        return cls(
            scores=np.asarray(arrays["scores"], dtype=np.float32),
            rows=np.asarray(arrays["rows"], dtype=np.intp) if "rows" in arrays else None,
            items=int(arrays["items"]),
            preds_type=_read_as(config),
            confidence=config.confidence,
        )


def runs_model(extractor: object) -> bool:
    """Whether `extractor` runs a model, whose rows may be detections: an ``uncertainty`` entry.

    Load cannot read the model's metadata, so a classifier's entry counts too.
    """
    from dataeval_flow.config.extractors import UncertaintyExtractorConfig

    return isinstance(extractor, UncertaintyExtractorConfig)


def compute_predictions(
    dataset: Any,
    config: "UncertaintyExtractorConfig",
    transforms: Callable[[Any], Any] | None,
    batch_size: int | None,
) -> Predictions:
    """Run `config`'s model over `dataset`, `batch_size` images at a time, keeping each row's scores and item.

    A detector keeps each box whose top raw score is at least ``confidence``. A model whose metadata fixes its batch
    size runs at that size, its last batch padded with its last image, whose predictions are dropped.

    Raises
    ------
    ValueError
        Before inference: metadata for a task other than classification or detection (DataEval's message), a model
        with fewer than 2 classes, ``confidence`` missing on a detector or set on a classifier, or a `batch_size` the
        metadata's fixed batch size disagrees with. After it: ``probs`` scores that do not sum to 1.
    """
    from dataeval.config import resolve_batch_size
    from dataeval.models import OnnxImageClassifier, OnnxObjectDetector, read_model_metadata

    spec = read_model_metadata(config.metadata_path)
    detector = spec.task == "IMAGE_OBJECT_DETECTION"
    _refuse(config, spec, detector=detector, batch_size=batch_size)
    fixed = spec.batch_size > 0
    size = spec.batch_size if fixed else resolve_batch_size(batch_size)
    height, width = config.image_height, config.image_width
    image_size = (height, width) if height is not None and width is not None else None
    model_type = OnnxObjectDetector if detector else OnnxImageClassifier
    model = model_type(config.model_path, config.metadata_path, image_size=image_size)
    threshold = config.confidence or 0.0
    scores: list[NDArray[np.float32]] = []
    rows: list[NDArray[np.intp]] = []
    for start in range(0, len(dataset), size):
        indices = range(start, min(start + size, len(dataset)))
        images = [_image(dataset[index][0], transforms) for index in indices]
        padded = images + [images[-1]] * (size - len(images)) if fixed else images
        for index, prediction in zip(indices, model(padded)[: len(images)], strict=True):
            own = np.asarray(prediction.scores if detector else prediction, dtype=np.float32).reshape(  # type: ignore[union-attr]
                -1, spec.n_classes
            )
            if detector:
                own = own[own.max(axis=1) >= threshold]
            scores.append(own)
            rows.append(np.full(len(own), index, dtype=np.intp))
    stacked = np.concatenate(scores) if scores else np.empty((0, spec.n_classes), dtype=np.float32)
    # DataEval's own test, so the refusal can say what the scores more likely are.
    if config.preds_type == "probs" and len(stacked) and np.abs(1 - np.nansum(stacked, axis=-1)).mean() > 1e-6:
        raise ValueError(
            f"`{config.name}` says its model emits `probs`, but its scores do not sum to 1: a detector's per-class "
            "scores are usually `sigmoid`, and raw outputs `logits`."
        )
    if config.preds_type == "sigmoid":
        clipped = np.clip(stacked, _EPSILON, 1 - _EPSILON)
        stacked = np.log(clipped / (1 - clipped)).astype(np.float32)
    return Predictions(
        scores=stacked,
        rows=(np.concatenate(rows) if rows else np.empty(0, dtype=np.intp)) if detector else None,
        items=len(dataset),
        preds_type=_read_as(config),
        confidence=config.confidence,
    )


def uncertainty_rows(predictions: Predictions) -> "NDArray[np.float32]":
    """Each row's normalized entropy, ``(n_rows, 1)``: DataEval's ``UncertaintyExtractor`` on the scores."""
    from dataeval.extractors import UncertaintyExtractor

    return UncertaintyExtractor(lambda scores: scores, preds_type=predictions.preds_type)(predictions.scores)


def membership(scores: "NDArray[np.float32]", threshold: float) -> "NDArray[np.bool_]":
    """Which classes each row counts toward, ``(n_rows, n_classes)``: DataEval's rule from
    ``ClasswiseUncertaintyExtractor``, every class whose sigmoid score is at least `threshold` of the row's largest."""
    from scipy.special import expit

    sigmoid = expit(scores)
    return sigmoid / sigmoid.max(axis=1, keepdims=True) >= threshold


def _read_as(config: "UncertaintyExtractorConfig") -> Literal["logits", "probs"]:
    """How DataEval reads the scores `config`'s model emits: sigmoid scores arrive as logits."""
    return "probs" if config.preds_type == "probs" else "logits"


def _image(image: Any, transforms: Callable[[Any], Any] | None) -> Any:
    return transforms(image) if transforms is not None else image


def _refuse(
    config: "UncertaintyExtractorConfig", spec: "ModelIOSpec", *, detector: bool, batch_size: int | None
) -> None:
    """Refuse a model run that cannot work, before inference."""
    if spec.n_classes < 2:
        raise ValueError(f"`{config.name}`'s model has {spec.n_classes} class: normalized entropy needs at least 2.")
    if detector and config.confidence is None:
        raise ValueError(
            f"`{config.name}` runs a detector, which returns a fixed number of boxes per image, padding included: set "
            "`confidence` (`0` keeps every box)."
        )
    if not detector and config.confidence is not None:
        raise ValueError(f"`{config.name}` runs a classifier, which has no boxes to filter: remove `confidence`.")
    if spec.batch_size > 0 and batch_size is not None and batch_size != spec.batch_size:
        raise ValueError(
            f"`{config.name}`'s model takes batches of {spec.batch_size} exactly, and `batch_size` is {batch_size}: "
            f"set it to {spec.batch_size}, or remove it."
        )
