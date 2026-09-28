"""The shift evaluators: DataEval's drift and out-of-distribution detectors.

Each fits on the first source's embeddings and predicts on the last's. ``run`` is the only code here that calls
DataEval.
"""

__all__ = [
    "DriftDomainClassifierEvaluator",
    "DriftKNeighborsEvaluator",
    "DriftMMDEvaluator",
    "DriftUnivariateEvaluator",
    "DriftWassersteinEvaluator",
    "chunked_arguments",
    "detect_drift",
]

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from dataeval.shift import (
    DriftDomainClassifier,
    DriftKNeighbors,
    DriftMMD,
    DriftOutput,
    DriftUnivariate,
    DriftWasserstein,
)

from dataeval_flow._input_spec import InputKind
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import dataeval_arguments, require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.shift._config import (
    ChunkedDriftConfig,
    DriftDomainClassifierConfig,
    DriftKNeighborsConfig,
    DriftMMDConfig,
    DriftUnivariateConfig,
    DriftWassersteinConfig,
)

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

    A middle source, which only ``shift.drift-wasserstein`` takes, is its validation set, fitted beside the reference.
    """
    reference, *validation, test = (require(i.embeddings, "embeddings", i.source) for i in inputs)
    fitted = detector.chunked(**chunked_arguments(chunking)) if chunking is not None else detector
    return fitted.fit(reference, *validation).predict(test)


class DriftUnivariateEvaluator(Evaluator[DriftUnivariateConfig, DriftOutput[Any]]):
    """``shift.drift-univariate``: whether the data drifted, one dimension at a time, per DataEval's DriftUnivariate."""

    name: ClassVar[str] = "shift.drift-univariate"
    description: ClassVar[str] = "Per-dimension statistical tests for drift (DataEval DriftUnivariate)"
    dataeval_class: ClassVar[type] = DriftUnivariate
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _PREDICT

    def run(self, config: DriftUnivariateConfig, inputs: Sequence[EvaluatorInputs]) -> DriftOutput[Any]:
        """Fit on the reference's embeddings, and test the second source's."""
        return detect_drift(DriftUnivariate(**dataeval_arguments(config)), config.chunking, inputs)


class DriftMMDEvaluator(Evaluator[DriftMMDConfig, DriftOutput[Any]]):
    """``shift.drift-mmd``: whether the data drifted, by maximum mean discrepancy, per DataEval's DriftMMD."""

    name: ClassVar[str] = "shift.drift-mmd"
    description: ClassVar[str] = "Maximum mean discrepancy between reference and test (DataEval DriftMMD)"
    dataeval_class: ClassVar[type] = DriftMMD
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _PREDICT

    def run(self, config: DriftMMDConfig, inputs: Sequence[EvaluatorInputs]) -> DriftOutput[Any]:
        """Fit on the reference's embeddings, and test the second source's."""
        return detect_drift(DriftMMD(**dataeval_arguments(config)), config.chunking, inputs)


class DriftKNeighborsEvaluator(Evaluator[DriftKNeighborsConfig, DriftOutput[Any]]):
    """``shift.drift-kneighbors``: whether the data drifted, by neighbor distances, per DataEval's DriftKNeighbors."""

    name: ClassVar[str] = "shift.drift-kneighbors"
    description: ClassVar[str] = (
        "Nearest-neighbor distances to the reference, tested for drift (DataEval DriftKNeighbors)"
    )
    dataeval_class: ClassVar[type] = DriftKNeighbors
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _PREDICT

    def run(self, config: DriftKNeighborsConfig, inputs: Sequence[EvaluatorInputs]) -> DriftOutput[Any]:
        """Fit on the reference's embeddings, and test the second source's."""
        return detect_drift(DriftKNeighbors(**dataeval_arguments(config)), config.chunking, inputs)


class DriftWassersteinEvaluator(Evaluator[DriftWassersteinConfig, DriftOutput[Any]]):
    """``shift.drift-wasserstein``: whether the data drifted past a validation baseline, per DataEval's
    DriftWasserstein."""

    name: ClassVar[str] = "shift.drift-wasserstein"
    description: ClassVar[str] = (
        "Per-dimension Wasserstein distance against a validation baseline (DataEval DriftWasserstein)"
    )
    dataeval_class: ClassVar[type] = DriftWasserstein
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _PREDICT

    def run(self, config: DriftWassersteinConfig, inputs: Sequence[EvaluatorInputs]) -> DriftOutput[Any]:
        """Fit on the reference and validation embeddings, and test the third source's."""
        return detect_drift(DriftWasserstein(**dataeval_arguments(config)), config.chunking, inputs)


class DriftDomainClassifierEvaluator(Evaluator[DriftDomainClassifierConfig, DriftOutput[Any]]):
    """``shift.drift-domain-classifier``: whether a classifier tells the data apart, per DataEval's
    DriftDomainClassifier."""

    name: ClassVar[str] = "shift.drift-domain-classifier"
    description: ClassVar[str] = "A classifier's ability to tell reference from test (DataEval DriftDomainClassifier)"
    dataeval_class: ClassVar[type] = DriftDomainClassifier
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _PREDICT

    def run(self, config: DriftDomainClassifierConfig, inputs: Sequence[EvaluatorInputs]) -> DriftOutput[Any]:
        """Fit on the reference's embeddings, and test the second source's."""
        return detect_drift(DriftDomainClassifier(**dataeval_arguments(config)), config.chunking, inputs)
