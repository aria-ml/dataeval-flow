"""The taxonomy runs the agreement golden records: one pipeline per case, in the preset's settings.

The generator ran each case once on legacy data-coverage, with `ontology:` set, and recorded its ontology findings
and what they were computed from; the agreement test runs the preset's settings (coverage spec §8.1). scope
now refuses `ontology:`, so legacy's settings no longer build.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import PipelineConfig
from tests.chain_toys import chain_pipeline


class Vehicles:
    """3x8x8 images of `names`, the first three classes 12, 8 and 4 times; any further name is declared and unseen."""

    def __init__(self, names: list[str]) -> None:
        rng = np.random.default_rng(0)
        counts = [12, 8, 4] + [0] * (len(names) - 3)
        self._labels = [index for index, count in enumerate(counts) for _ in range(count)]
        self._images = [rng.integers(0, 255, (3, 8, 8), dtype=np.uint8) for _ in self._labels]
        self._classes = len(names)
        self.metadata = {"id": "vehicles-" + "-".join(names), "index2label": dict(enumerate(names))}

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(self._classes, dtype=np.float32)
        target[self._labels[index]] = 1.0
        return self._images[index], target, {}


def _concept(id_: str, label: str, parents: tuple[str, ...] = (), synonyms: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"id": id_, "label": label, "parents": list(parents), "synonyms": list(synonyms)}


@dataclass(frozen=True)
class Case:
    """One golden case: the class names, the ontology (inline, or declared `concepts` under the pool name `vocab`),
    and the settings."""

    names: list[str]
    ontology: dict[str, Any] | None = None
    concepts: list[dict[str, Any]] = field(default_factory=list)
    expected: dict[str, float] | None = None
    label_pattern: str | None = None


_VEHICLE = {"vehicle": {"car": None, "truck": None, "bus": None}}

CASES: dict[str, Case] = {
    "conforms": Case(["car", "truck", "bus"], _VEHICLE),
    "unmatched": Case(["car", "truk", "bus"], _VEHICLE),
    "ambiguous": Case(
        ["car", "truck", "bus"],
        concepts=[
            _concept("vehicle", "vehicle"),
            _concept("car1", "car", ("vehicle",)),
            _concept("car2", "car", ("vehicle",)),
            _concept("truck", "truck", ("vehicle",)),
            _concept("bus", "bus", ("vehicle",)),
        ],
    ),
    "lossy": Case(
        ["car", "automobile", "bus"],
        concepts=[
            _concept("vehicle", "vehicle"),
            _concept("car", "car", ("vehicle",), ("automobile",)),
            _concept("bus", "bus", ("vehicle",)),
        ],
    ),
    "partial": Case(["car", "truck", "boat"], {"vehicle": {"car": None, "truck": None}}),
    "expected": Case(["car", "truck", "bus"], _VEHICLE, expected={"bus": 0.5, "plane": 0.1}),
    "pattern": Case(
        ["car", "truck", "bus"], {"vehicle": {"car": None, "Truck": None, "bus": None}}, label_pattern="^[a-z]+$"
    ),
    "empty_branch": Case(
        ["car", "truck", "bus"],
        {"land": {"car": None, "truck": None, "bus": None}, "air": {"plane": None, "drone": None}},
    ),
    "smells": Case(
        ["car", "truck", "bus"],
        concepts=[
            _concept("vehicle", "vehicle"),
            _concept("land", "land", ("vehicle",)),
            _concept("car", "car", ("land", "vehicle")),
            _concept("truck", "truck", ("land",)),
            _concept("bus", "bus", ("transport",)),
        ],
    ),
    "unseen": Case(
        ["car", "truck", "bus", "tank"], {"vehicle": {"car": None, "truck": None, "bus": None, "tank": None}}
    ),
}


def pipeline(name: str, *, legacy: bool) -> PipelineConfig:
    """Case `name` as a one-task pipeline: a `taxonomy` entry. Legacy scope's is refused."""
    if legacy:
        raise ValueError("scope no longer judges an ontology, so legacy's settings for this case do not build.")
    case = CASES[name]
    DatasetCache.clear_instances()
    ontology: dict[str, Any] | str = "vocab" if case.concepts else dict(case.ontology or {})
    entry: dict[str, Any] = {"name": "w", "type": "taxonomy", "ontology": ontology}
    if case.expected is not None:
        entry["representation"] = {"expected": case.expected}
    if case.label_pattern is not None:
        entry["ontology-validation"] = {"label_pattern": case.label_pattern}
    extra = {"ontologies": [{"name": "vocab", "concepts": case.concepts}]} if case.concepts else {}
    return chain_pipeline(
        workflows=[entry],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": Vehicles(case.names)},
        extra=extra,
    )
