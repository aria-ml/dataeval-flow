"""The cases the OOD golden records: the data, the legacy config, the preset config, and the preset step that answers
each legacy detector.

Configs are plain dicts, so this module imports no legacy class and survives the legacy workflow's deletion.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dataeval_flow import PipelineConfig
from tests.evaluator_toys import toy_pipeline
from tests.ood_toys import FactorImages


@dataclass(frozen=True)
class Case:
    """One OOD scenario. `steps` maps each legacy detector key to the preset step that answers it, in legacy's
    detector order."""

    datasets: Callable[[], dict[str, Any]]
    legacy: dict[str, Any]
    preset: dict[str, Any]
    steps: dict[str, str]


_SHIFTED = range(0, 40, 5)


def _shifted() -> dict[str, Any]:
    return {"reference": FactorImages(40), "test": FactorImages(40, seed=1, shifted=_SHIFTED)}


def _same() -> dict[str, Any]:
    return {"reference": FactorImages(40), "test": FactorImages(40)}


def _two_tests() -> dict[str, Any]:
    first, second = FactorImages(40, seed=1, shifted=_SHIFTED), FactorImages(40, seed=2)
    return {"reference": FactorImages(40), "first": first, "second": second}


_KNN = {"k": 5, "distance_metric": "euclidean", "threshold_perc": 95.0}
# Legacy always passed `threshold_perc`, which overrides `n_std` in DataEval, so the preset's entry writes it too.
_DC = {"n_folds": 3, "n_repeats": 2, "n_std": 2.0, "threshold_perc": 95.0}
_LEGACY_KNN, _LEGACY_DC = {"method": "kneighbors", **_KNN}, {"method": "domain_classifier", **_DC}
_PRESET_KNN, _PRESET_DC = {"type": "ood-kneighbors", **_KNN}, {"type": "ood-domain-classifier", **_DC}
_KNN_STEP = {"kneighbors": "ood-kneighbors"}
_DC_STEP = {"domain_classifier": "ood-domain-classifier"}

CASES: dict[str, Case] = {
    "kneighbors": Case(_shifted, {"detectors": [_LEGACY_KNN]}, {"detectors": [_PRESET_KNN]}, _KNN_STEP),
    "domain_classifier": Case(_shifted, {"detectors": [_LEGACY_DC]}, {"detectors": [_PRESET_DC]}, _DC_STEP),
    "both": Case(
        _shifted,
        {"detectors": [_LEGACY_KNN, _LEGACY_DC]},
        {"detectors": [_PRESET_KNN, _PRESET_DC]},
        _KNN_STEP | _DC_STEP,
    ),
    "insights_off": Case(
        _shifted,
        {"detectors": [_LEGACY_KNN], "metadata_insights": False},
        {"detectors": [_PRESET_KNN], "metadata_insights": False},
        _KNN_STEP,
    ),
    "nothing_flagged": Case(_same, {"detectors": [_LEGACY_KNN]}, {"detectors": [_PRESET_KNN]}, _KNN_STEP),
    "two_tests": Case(_two_tests, {"detectors": [_LEGACY_KNN]}, {"detectors": [_PRESET_KNN]}, _KNN_STEP),
}

SINGLE_SOURCE: tuple[str, ...] = tuple(name for name, case in CASES.items() if len(case.datasets()) == 2)


def pipeline(name: str) -> PipelineConfig:
    """Case `name`'s datasets and flatten extractor with seed 0. Tests compute on the CPU, as a CPU-only runner does."""
    config = toy_pipeline(datasets=CASES[name].datasets(), extractor=True)
    return config.model_copy(update={"seed": 0})
