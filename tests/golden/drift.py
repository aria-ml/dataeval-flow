"""The cases the drift golden records: the data, the legacy config, the preset config and the steps that correspond.

Configs are plain dicts, so this module imports no legacy class and survives the legacy workflow's deletion.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dataeval_flow import PipelineConfig
from tests.drift_toys import BoxImages, ClassImages
from tests.evaluator_toys import ToyImages, toy_pipeline


@dataclass(frozen=True)
class Case:
    """One drift scenario. `steps` maps each legacy detector key to the preset step answering it; `custom`, when set,
    is a workflow run in place of the preset."""

    datasets: Callable[[], dict[str, Any]]
    legacy: dict[str, Any]
    preset: dict[str, Any]
    steps: dict[str, str]
    custom: dict[str, Any] | None = None


def _shifted() -> dict[str, Any]:
    return {"reference": ToyImages(40), "test": ToyImages(40, seed=1, bright=True)}


def _same() -> dict[str, Any]:
    return {"reference": ToyImages(40), "test": ToyImages(40, seed=1)}


def _two_tests() -> dict[str, Any]:
    return {
        "reference": ToyImages(40),
        "first": ToyImages(40, seed=1, bright=True),
        "second": ToyImages(40, seed=3),
    }


def _classes() -> dict[str, Any]:
    return {
        "reference": ClassImages({0: 15, 1: 15, 2: 15}),
        "test": ClassImages({0: 15, 1: 15, 2: 1}, seed=1, bright_classes={1}),
    }


def _boxes() -> dict[str, Any]:
    return {"reference": BoxImages(40), "test": BoxImages(40, seed=1, bright=True)}


_ZSCORE = ["zscore", 3.0]
_UNIVARIATE = {"p_val": 0.05, "correction": "bonferroni", "alternative": "two-sided"}
_MMD = {"p_val": 0.05, "n_permutations": 50}
_KNN = {"distance_metric": "euclidean", "p_val": 0.05}

_MERGED = {
    "name": "merged",
    "inputs": ["reference", "first", "second"],
    "steps": [
        {"name": "test", "transform": "merge", "input": ["first", "second"]},
        {"name": "drift-mmd", "evaluator": "drift-mmd", "input": ["reference", "test"]},
        {"name": "drift-mmd-check", "check": "drift", "input": "drift-mmd"},
    ],
}

CASES: dict[str, Case] = {
    "univariate": Case(
        _shifted,
        {"detectors": [{"method": "univariate"}]},
        {"detectors": [{"type": "drift-univariate", "method": "ks", **_UNIVARIATE}]},
        {"univariate": "drift-univariate"},
    ),
    "mmd": Case(
        _shifted,
        {"detectors": [{"method": "mmd", "n_permutations": 50}]},
        {"detectors": [{"type": "drift-mmd", **_MMD}]},
        {"mmd": "drift-mmd"},
    ),
    "kneighbors_same": Case(
        _same,
        {"detectors": [{"method": "kneighbors", "k": 5}]},
        {"detectors": [{"type": "drift-kneighbors", "k": 5, **_KNN}]},
        {"kneighbors": "drift-kneighbors"},
    ),
    "domain_classifier": Case(
        _shifted,
        {"detectors": [{"method": "domain_classifier", "n_folds": 3}]},
        {"detectors": [{"type": "drift-domain-classifier", "n_folds": 3, "threshold": 0.55}]},
        {"domain_classifier": "drift-domain-classifier"},
    ),
    "two_tests": Case(
        _two_tests,
        {"detectors": [{"method": "mmd", "n_permutations": 50}]},
        {"detectors": [{"type": "drift-mmd", **_MMD}]},
        {"mmd": "drift-mmd"},
        custom=_MERGED,
    ),
    "chunk_count": Case(
        _shifted,
        {"detectors": [{"method": "kneighbors", "k": 5, "chunking": {"chunk_count": 4}}]},
        {
            "detectors": [
                {"type": "drift-kneighbors", "k": 5, **_KNN, "chunking": {"chunk_count": 4, "threshold": _ZSCORE}}
            ]
        },
        {"kneighbors": "drift-kneighbors"},
    ),
    "chunk_size": Case(
        _shifted,
        {"detectors": [{"method": "kneighbors", "k": 5, "chunking": {"chunk_size": 9, "incomplete": "append"}}]},
        {
            "detectors": [
                {
                    "type": "drift-kneighbors",
                    "k": 5,
                    **_KNN,
                    "chunking": {"chunk_size": 9, "incomplete": "append", "threshold": _ZSCORE},
                }
            ]
        },
        {"kneighbors": "drift-kneighbors"},
    ),
    "classwise": Case(
        _classes,
        {"detectors": [{"method": "mmd", "n_permutations": 50, "classwise": True}, {"method": "kneighbors", "k": 3}]},
        {
            "detectors": [{"type": "drift-mmd", **_MMD}, {"type": "drift-kneighbors", "k": 3, **_KNN}],
            "classwise": ["drift-mmd"],
        },
        {"mmd": "drift-mmd", "kneighbors": "drift-kneighbors"},
    ),
    "classwise_chunked": Case(
        _classes,
        {"detectors": [{"method": "kneighbors", "k": 3, "classwise": True, "chunking": {"chunk_count": 3}}]},
        {
            "detectors": [
                {"type": "drift-kneighbors", "k": 3, **_KNN, "chunking": {"chunk_count": 3, "threshold": _ZSCORE}}
            ],
            "classwise": ["drift-kneighbors"],
        },
        {"kneighbors": "drift-kneighbors"},
    ),
    "classwise_boxes": Case(
        _boxes,
        {"detectors": [{"method": "kneighbors", "k": 3, "classwise": True}]},
        {"detectors": [{"type": "drift-kneighbors", "k": 3, **_KNN}], "classwise": ["drift-kneighbors"]},
        {"kneighbors": "drift-kneighbors"},
    ),
}


def pipeline(name: str) -> PipelineConfig:
    """Case `name`'s datasets and flatten extractor on the CPU with seed 0, so the golden holds on a CPU-only runner."""
    config = toy_pipeline(datasets=CASES[name].datasets(), extractor=True)
    return config.model_copy(update={"seed": 0, "device": "cpu"})
