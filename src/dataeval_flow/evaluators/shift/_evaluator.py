"""The shift evaluators: DataEval's drift and out-of-distribution detectors.

Each fits on the first source's embeddings and predicts on the last's. ``run`` is the only code here that calls
DataEval.
"""

__all__ = [
    "DivergenceEvaluator",
    "DriftDomainClassifierEvaluator",
    "DriftKNeighborsEvaluator",
    "DriftMMDEvaluator",
    "DriftUnivariateEvaluator",
    "DriftWassersteinEvaluator",
    "OODDomainClassifierEvaluator",
    "OODKNeighborsEvaluator",
    "chunked_arguments",
    "detect_drift",
    "detect_ood",
]

import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar

import numpy as np
from dataeval.core import divergence_fnn, divergence_mst
from dataeval.shift import (
    DriftDomainClassifier,
    DriftKNeighbors,
    DriftMMD,
    DriftOutput,
    DriftUnivariate,
    DriftWasserstein,
    OODDomainClassifier,
    OODKNeighbors,
    OODOutput,
)

from dataeval_flow._input_spec import InputKind
from dataeval_flow.evaluators._core import execution
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import dataeval_arguments, require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.shift._config import (
    ChunkedDriftConfig,
    DivergenceConfig,
    DriftDomainClassifierConfig,
    DriftKNeighborsConfig,
    DriftMMDConfig,
    DriftUnivariateConfig,
    DriftWassersteinConfig,
    OODDomainClassifierConfig,
    OODKNeighborsConfig,
)
from dataeval_flow.evaluators.shift._result import DivergenceOutput

_PREDICT: Mapping[InputKind, str] = {InputKind.EMBEDDINGS: "predict"}


def chunked_arguments(chunking: ChunkedDriftConfig) -> dict[str, Any]:
    """``chunked()``'s arguments, the threshold resolved to the ``Threshold`` object it takes."""
    from dataeval.utils.thresholds import resolve_threshold

    arguments = dataeval_arguments(chunking)
    if "threshold" in arguments:
        arguments["threshold"] = resolve_threshold(arguments["threshold"])
    return arguments


def detect_drift(
    detector: Any, chunking: ChunkedDriftConfig | None, inputs: Sequence[EvaluatorInputs]
) -> DriftOutput[Any]:
    """Fit `detector` on the first source and predict on the last, chunked where `chunking` says so.

    A middle source, which only ``drift-wasserstein`` takes, is its validation set, fitted beside the reference. Rows
    that are detections, from a detector's predictions, are chunked by whole images instead (``_rows``).
    """
    made = inputs[0].predictions
    if made is not None and made.rows is not None:
        from dataeval_flow.evaluators.shift._rows import detect_drift_by_image

        return detect_drift_by_image(detector, chunking, inputs)
    reference, *validation, test = (require(i.embeddings, "embeddings", i.source) for i in inputs)
    fitted = detector.chunked(**chunked_arguments(chunking)) if chunking is not None else detector
    return fitted.fit(reference, *validation).predict(test)


class DriftUnivariateEvaluator(Evaluator[DriftUnivariateConfig, DriftOutput[Any]]):
    """``drift-univariate``: whether the data drifted, one dimension at a time, per DataEval's DriftUnivariate."""

    name: ClassVar[str] = "drift-univariate"
    title: ClassVar[str] = "Drift (Univariate)"
    description: ClassVar[str] = "Per-dimension statistical tests for drift (DataEval DriftUnivariate)"
    dataeval_class: ClassVar[type] = DriftUnivariate
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _PREDICT

    def run(self, config: DriftUnivariateConfig, inputs: Sequence[EvaluatorInputs]) -> DriftOutput[Any]:
        """Fit on the reference's embeddings, and test the second source's."""
        return detect_drift(DriftUnivariate(**dataeval_arguments(config)), config.chunking, inputs)


class DriftMMDEvaluator(Evaluator[DriftMMDConfig, DriftOutput[Any]]):
    """``drift-mmd``: whether the data drifted, by maximum mean discrepancy, per DataEval's DriftMMD."""

    name: ClassVar[str] = "drift-mmd"
    title: ClassVar[str] = "Drift (MMD)"
    description: ClassVar[str] = "Maximum mean discrepancy between reference and test (DataEval DriftMMD)"
    dataeval_class: ClassVar[type] = DriftMMD
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _PREDICT

    def run(self, config: DriftMMDConfig, inputs: Sequence[EvaluatorInputs]) -> DriftOutput[Any]:
        """Fit on the reference's embeddings, and test the second source's."""
        return detect_drift(DriftMMD(**dataeval_arguments(config)), config.chunking, inputs)


class DriftKNeighborsEvaluator(Evaluator[DriftKNeighborsConfig, DriftOutput[Any]]):
    """``drift-kneighbors``: whether the data drifted, by neighbor distances, per DataEval's DriftKNeighbors."""

    name: ClassVar[str] = "drift-kneighbors"
    title: ClassVar[str] = "Drift (K-Neighbors)"
    description: ClassVar[str] = (
        "Nearest-neighbor distances to the reference, tested for drift (DataEval DriftKNeighbors)"
    )
    dataeval_class: ClassVar[type] = DriftKNeighbors
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _PREDICT

    def run(self, config: DriftKNeighborsConfig, inputs: Sequence[EvaluatorInputs]) -> DriftOutput[Any]:
        """Fit on the reference's embeddings, and test the second source's."""
        return detect_drift(DriftKNeighbors(**dataeval_arguments(config)), config.chunking, inputs)


class DriftWassersteinEvaluator(Evaluator[DriftWassersteinConfig, DriftOutput[Any]]):
    """``drift-wasserstein``: whether the data drifted past a validation baseline, per DataEval's
    DriftWasserstein."""

    name: ClassVar[str] = "drift-wasserstein"
    title: ClassVar[str] = "Drift (Wasserstein)"
    description: ClassVar[str] = (
        "Per-dimension Wasserstein distance against a validation baseline (DataEval DriftWasserstein)"
    )
    dataeval_class: ClassVar[type] = DriftWasserstein
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _PREDICT

    def run(self, config: DriftWassersteinConfig, inputs: Sequence[EvaluatorInputs]) -> DriftOutput[Any]:
        """Fit on the reference and validation embeddings, and test the third source's."""
        return detect_drift(DriftWasserstein(**dataeval_arguments(config)), config.chunking, inputs)


class DriftDomainClassifierEvaluator(Evaluator[DriftDomainClassifierConfig, DriftOutput[Any]]):
    """``drift-domain-classifier``: whether a classifier tells the data apart, per DataEval's
    DriftDomainClassifier."""

    name: ClassVar[str] = "drift-domain-classifier"
    title: ClassVar[str] = "Drift (Domain Classifier)"
    description: ClassVar[str] = "A classifier's ability to tell reference from test (DataEval DriftDomainClassifier)"
    dataeval_class: ClassVar[type] = DriftDomainClassifier
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _PREDICT

    def run(self, config: DriftDomainClassifierConfig, inputs: Sequence[EvaluatorInputs]) -> DriftOutput[Any]:
        """Fit on the reference's embeddings, and test the second source's."""
        return detect_drift(DriftDomainClassifier(**dataeval_arguments(config)), config.chunking, inputs)


def detect_ood(detector: Any, inputs: Sequence[EvaluatorInputs]) -> OODOutput:
    """Fit `detector` on the first source's embeddings, and flag the second source's items. Rows that are detections,
    from a detector's predictions, are judged per test image instead (``_rows``)."""
    made = inputs[0].predictions
    if made is not None and made.rows is not None:
        from dataeval_flow.evaluators.shift._rows import detect_ood_by_image

        return detect_ood_by_image(detector, inputs)
    reference, test = (require(i.embeddings, "embeddings", i.source) for i in inputs)
    return detector.fit(reference).predict(test)


class OODKNeighborsEvaluator(Evaluator[OODKNeighborsConfig, OODOutput]):
    """``ood-kneighbors``: which test items sit far from the reference, per DataEval's OODKNeighbors."""

    name: ClassVar[str] = "ood-kneighbors"
    title: ClassVar[str] = "OOD (K-Neighbors)"
    description: ClassVar[str] = "Test items far from their nearest reference neighbors (DataEval OODKNeighbors)"
    dataeval_class: ClassVar[type] = OODKNeighbors
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _PREDICT

    def run(self, config: OODKNeighborsConfig, inputs: Sequence[EvaluatorInputs]) -> OODOutput:
        """Fit on the reference's embeddings, and score the second source's items."""
        return detect_ood(OODKNeighbors(**dataeval_arguments(config)), inputs)


class OODDomainClassifierEvaluator(Evaluator[OODDomainClassifierConfig, OODOutput]):
    """``ood-domain-classifier``: which test items a classifier tells apart, per DataEval's
    OODDomainClassifier."""

    name: ClassVar[str] = "ood-domain-classifier"
    title: ClassVar[str] = "OOD (Domain Classifier)"
    description: ClassVar[str] = "Test items a classifier tells apart from the reference (DataEval OODDomainClassifier)"
    dataeval_class: ClassVar[type] = OODDomainClassifier
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _PREDICT

    def run(self, config: OODDomainClassifierConfig, inputs: Sequence[EvaluatorInputs]) -> OODOutput:
        """Fit on the reference's embeddings, and score the second source's items."""
        return detect_ood(OODDomainClassifier(**dataeval_arguments(config)), inputs)


class DivergenceEvaluator(Evaluator[DivergenceConfig, DivergenceOutput]):
    """``divergence``: how far apart two sources' embeddings sit, per DataEval's divergence_mst or divergence_fnn."""

    name: ClassVar[str] = "divergence"
    title: ClassVar[str] = "Divergence"
    description: ClassVar[str] = "How far apart two sources' embeddings sit (DataEval divergence_mst, divergence_fnn)"
    dataeval_class: ClassVar[Any] = divergence_mst
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.EMBEDDINGS: "__call__"}
    reads_factors: ClassVar[bool] = False

    def run(self, config: DivergenceConfig, inputs: Sequence[EvaluatorInputs]) -> DivergenceOutput:
        """The divergence of the first source's embeddings from the second's."""
        first, second = inputs
        a = np.asarray(require(first.embeddings, "embeddings", first.source))
        b = np.asarray(require(second.embeddings, "embeddings", second.source))
        for source, rows in ((first.source, a), (second.source, b)):
            if len(rows) == 0:  # DataEval divides by each source's size
                raise ValueError(f"`divergence` needs embeddings from both sources; '{source}' has none.")
        function = divergence_mst if config.method == "mst" else divergence_fnn
        started, clock = datetime.now(UTC), time.monotonic()
        result = function(a, b)
        meta = execution(
            f"dataeval.core.divergence_{config.method}", started, time.monotonic() - clock, {"method": config.method}
        )
        return DivergenceOutput(
            {"divergence": float(result["divergence"]), "errors": int(result["errors"]), "method": config.method},
            meta,
        )
