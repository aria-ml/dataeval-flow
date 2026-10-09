"""Small in-memory datasets and pipeline builders for the audit, chain and evaluator tests.

Everything is synthetic and deterministic: no files, no network, no model downloads. The datasets satisfy the MAITE
image-classification protocol (``__getitem__`` returns ``(image, one_hot_target, metadata)``), so they go straight
into a ``maite`` dataset entry or ``dataeval_flow.run``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dataeval_flow.config import DatasetProtocolConfig, PipelineConfig, SourceConfig, TaskConfig
from dataeval_flow.config.extractors import FlattenExtractorConfig

# Flatten carries no batch size by default, and DataEval refuses to embed without one.
FLAT = FlattenExtractorConfig(name="flat", batch_size=8)


class Images:
    """A classification dataset of 3x16x16 images over ``n_classes`` classes, each with a ``site`` and an ``angle``.

    ``planted=True`` (the default) makes item 5 a byte-for-byte copy of item 0 and item 7 solid white, so a duplicates
    run finds one exact group and an outliers run one image. ``bright=True`` lifts every pixel by 100, out of the
    distribution of an unlifted set. Labels cycle through the classes; ``site`` cycles through ``north``, ``south``,
    ``east``.
    """

    def __init__(
        self,
        count: int = 12,
        seed: int = 0,
        *,
        n_classes: int = 2,
        planted: bool = True,
        bright: bool = False,
        dataset_id: str | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._images = [rng.integers(0, 255, (3, 16, 16), dtype=np.uint8) for _ in range(count)]
        if planted and count > 7:
            self._images[5] = self._images[0].copy()
            self._images[7] = np.full((3, 16, 16), 255, dtype=np.uint8)
        if bright:
            self._images = [np.clip(image.astype(np.int16) + 100, 0, 255).astype(np.uint8) for image in self._images]
        self._n_classes = n_classes
        names = "abcdefgh"
        self.metadata: Any = {
            "id": dataset_id or f"images-{seed}-{count}-{n_classes}{'p' if planted else ''}{'b' if bright else ''}",
            "index2label": {i: names[i] for i in range(n_classes)},
        }

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(self._n_classes, dtype=np.float32)
        target[index % self._n_classes] = 1.0
        site = ("north", "south", "east")[index % 3]
        return self._images[index], target, {"id": index, "site": site, "angle": float(index % 7)}


class Empty:
    """A dataset with no items."""

    metadata: Any = {"id": "empty", "index2label": {0: "a", 1: "b"}}

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> Any:
        raise IndexError(index)


def shifted_sources(count: int = 40) -> dict[str, Images]:
    """A reference, then test images brightened out of its distribution."""
    return {"reference": Images(count=count), "test": Images(count=count, seed=1, bright=True)}


def pipeline(
    datasets: Mapping[str, Any] | None = None,
    *,
    workflows: Sequence[Any] = (),
    evaluators: Sequence[Any] = (),
    tasks: Sequence[Mapping[str, Any] | TaskConfig] = (),
    extractor: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> PipelineConfig:
    """A pipeline with one in-memory dataset and one same-named source per entry of *datasets*.

    *workflows* and *evaluators* hold config instances or dicts; *tasks* are dicts as a config file writes them (or
    ``TaskConfig`` instances). *datasets* defaults to one source ``src`` over :class:`Images`.
    """
    datasets = datasets if datasets is not None else {"src": Images()}
    data: dict[str, Any] = {
        "datasets": [
            DatasetProtocolConfig(name=f"{name}_data", format="maite", dataset=ds) for name, ds in datasets.items()
        ],
        "sources": [SourceConfig(name=name, dataset=f"{name}_data") for name in datasets],
        "evaluators": list(evaluators) or None,
        "workflows": list(workflows) or None,
        "tasks": [t if isinstance(t, TaskConfig) else TaskConfig.model_validate(t) for t in tasks],
        **(extra or {}),
    }
    if extractor:
        data["extractors"] = [FLAT]
    return PipelineConfig.model_validate(data)


class Factors:
    """Images over three classes with two metadata factors: ``site`` follows the class and ``angle`` does not."""

    def __init__(self, count: int = 60) -> None:
        rng = np.random.default_rng(0)
        self._images = [rng.integers(0, 255, (3, 8, 8), dtype=np.uint8) for _ in range(count)]
        self.metadata: Any = {"id": f"factors-{count}", "index2label": {0: "cat", 1: "dog", 2: "bird"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(3, dtype=np.float32)
        target[index % 3] = 1.0
        return self._images[index], target, {"site": ("north", "south", "east")[index % 3], "angle": float(index % 7)}


def _shifted(count: int, *, validation: bool = False) -> dict[str, Images]:
    sources = {"reference": Images(count=count)}
    if validation:
        sources["validation"] = Images(count=count, seed=2)
    sources["test"] = Images(count=count, seed=1, bright=True)
    return sources


# For each built-in evaluator type: the data its task reads, and whether the task needs an extractor. A source
# mapping gives the sources in the order the evaluator reads them.
EVALUATOR_DATA: dict[str, tuple[Any, bool]] = {
    "balance": (Factors(), False),
    "completeness": (Images(40), True),
    "content-digest": (Images(40), False),
    "coverage": (Images(40), True),
    "divergence": (_shifted(40), True),
    "diversity": (Factors(), False),
    "drift-domain-classifier": (_shifted(40), True),
    "drift-kneighbors": (_shifted(40), True),
    "drift-mmd": (_shifted(40), True),
    "drift-univariate": (_shifted(40), True),
    "drift-wasserstein": (_shifted(40, validation=True), True),
    "duplicates": (Images(40), False),
    "factor-leakage": ({"a": Factors(), "b": Factors()}, False),
    "factor-summary": (Factors(), False),
    "factor-triage": (Factors(), False),
    "label-alignment": (Images(40), False),
    "label-health": (Images(40), False),
    "label-reconciliation": (Images(40), False),
    "ontology-validation": (Images(40), False),
    "ood-domain-classifier": (_shifted(40), True),
    "ood-kneighbors": (_shifted(40), True),
    "outliers": (Images(40), False),
    "parity": (Factors(), False),
    "prioritization": (Images(40), True),
    "profile": (Factors(), False),
    "representation": (Images(40), False),
}

# Settings a bare config cannot supply because the field has no default.
EVALUATOR_SETTINGS: dict[str, dict[str, Any]] = {
    "factor-leakage": {"factors": ["site"]},
    "label-alignment": {"ontology": {"a": None, "b": None}},
    "label-reconciliation": {"ontology": {"a": None, "b": None}},
    "ontology-validation": {"ontology": {"a": None, "b": None}},
    "representation": {"ontology": {"a": None, "b": None}},
}


class _Detection:
    """A duck-typed object-detection target (DataEval exports no constructible one)."""

    def __init__(self, boxes: np.ndarray, labels: np.ndarray) -> None:
        self.boxes = boxes
        self.labels = labels
        self.scores = np.ones(len(labels), dtype=np.float32)


class Detections:
    """An object-detection dataset of 3x16x16 images, each with one or two boxes.

    *labels* holds each item's box labels, *index2label* the class names. ``duplicate_of`` maps an item to an earlier
    item whose image it copies, making an exact duplicate.
    """

    def __init__(
        self,
        labels: Sequence[Sequence[int]],
        index2label: Mapping[int, str],
        *,
        duplicate_of: Mapping[int, int] | None = None,
        dataset_id: str = "detections",
    ) -> None:
        import zlib

        self._labels = [list(item) for item in labels]
        self._copy = dict(duplicate_of or {})
        self._seed = zlib.crc32(dataset_id.encode())  # two datasets never share an image by accident
        self.metadata: Any = {"id": dataset_id, "index2label": dict(index2label)}

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[np.ndarray, _Detection, dict[str, Any]]:
        rng = np.random.default_rng((self._seed, self._copy.get(index, index)))
        image = rng.integers(0, 60, size=(3, 16, 16), dtype=np.uint8)
        labels = np.asarray(self._labels[index], dtype=np.intp)
        boxes = np.asarray([[1 + 6 * i, 1, 6 + 6 * i, 9] for i in range(len(labels))], dtype=np.float32)
        return image, _Detection(boxes, labels), {"id": index, "site": ("north", "south", "east")[index % 3]}


def yaml_pipeline(
    text: str, datasets: Mapping[str, Any], *, extractor: bool = False, extra: Mapping[str, Any] | None = None
) -> PipelineConfig:
    """The pipeline *text* writes (``evaluators:``, ``workflows:``, ``tasks:`` and other keys), over in-memory
    datasets with a same-named source each."""
    import yaml

    data = yaml.safe_load(text)
    others = {k: v for k, v in data.items() if k not in ("workflows", "evaluators", "tasks")}
    return pipeline(
        datasets,
        workflows=data.get("workflows", ()),
        evaluators=data.get("evaluators", ()),
        tasks=data["tasks"],
        extractor=extractor,
        extra={**others, **(extra or {})},
    )
