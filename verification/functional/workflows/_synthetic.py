"""In-memory MAITE datasets and pipeline helpers shared by the dataset-quality, scope, shift, splits and matrix tests.

Everything is deterministic and small: no files, no network, no model.  The datasets follow MAITE's shape
(``__getitem__`` returns ``(image, target, datum_metadata)`` and ``metadata`` holds ``id`` and ``index2label``), so
they run through ``run_tasks`` and ``run`` exactly as a user's dataset would.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dataeval_flow.config import DatasetProtocolConfig, PipelineConfig, SourceConfig, TaskConfig
from dataeval_flow.config.extractors import FlattenExtractorConfig

FLAT = FlattenExtractorConfig(name="flat", batch_size=8)
"""The flatten extractor: pixels as the embedding, so no model is loaded."""

CLASSES = {0: "cat", 1: "dog", 2: "bird"}
ANIMALS = {**CLASSES, 3: "owl"}
"""``owl`` is declared where a test asks for it (``index2label=ANIMALS``) and never labelled."""


class Images:
    """Classification images over ``cat``, ``dog`` and ``bird`` with optional planted defects and metadata.

    Item ``i`` has class ``pattern[i % len(pattern)]`` (default 5 cat : 3 dog : 2 bird).

    - ``planted=True`` makes item 5 a byte-for-byte copy of item 0 (one exact duplicate group, ``[0, 5]``) and item 7
      solid white (one pixel outlier); ``near_duplicate=True`` makes item 9 a one-pixel edit of item 3.
    - ``tint=True`` gives a colour cast no untinted image has; ``bright`` lifts every pixel by 100, out of the
      distribution of an unlifted dataset of the same seed; ``value_range`` bounds the pixel noise.
    - ``tight`` names classes whose images are one picture plus a little noise.
    - ``factors=True`` gives each item a ``site`` that follows its class except for every tenth item (a shortcut) and
      an ``angle`` that does not.  ``extra`` adds metadata to every item: a fixed value, or a function of the item's
      index.
    """

    def __init__(
        self,
        count: int = 60,
        *,
        seed: int = 0,
        pattern: Sequence[int] = (0, 0, 0, 0, 0, 1, 1, 1, 2, 2),
        planted: bool = False,
        near_duplicate: bool = False,
        factors: bool = False,
        extra: Mapping[str, Any] | None = None,
        labeled: bool = True,
        bright: bool = False,
        tint: bool = False,
        tight: Sequence[int] = (),
        shape: tuple[int, int, int] = (3, 8, 8),
        index2label: Mapping[int, str] = CLASSES,
        value_range: tuple[int, int] = (0, 255),
        uid: str | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._images = [rng.integers(*value_range, shape, dtype=np.uint8) for _ in range(count)]
        if tight:  # these classes' images are one picture with a little noise: a class that varies in nothing
            base = rng.integers(*value_range, shape, dtype=np.uint8)
            for index in range(count):
                if int(pattern[index % len(pattern)]) in tight:
                    self._images[index] = np.clip(base + rng.integers(0, 3, shape), 0, 255).astype(np.uint8)
        if planted and count > 7:
            self._images[5] = self._images[0].copy()
            self._images[7] = np.full(shape, 255, dtype=np.uint8)
        if near_duplicate and count > 9:  # item 9 is item 3 with one pixel edited
            self._images[9] = self._images[3].copy()
            self._images[9][0, 0, 0] ^= 1
        if tint:  # first channel high, the others low: a colour cast that moves the direction of the pixel vector
            for image in self._images:
                image[0] = rng.integers(200, 255, image[0].shape, dtype=np.uint8)
                image[1:] = rng.integers(0, 55, image[1:].shape, dtype=np.uint8)
        if bright:
            self._images = [np.clip(image.astype(np.int16) + 100, 0, 255).astype(np.uint8) for image in self._images]
        self.labels = [int(pattern[index % len(pattern)]) for index in range(count)]
        self._factors, self._labeled, self._extra = factors, labeled, dict(extra or {})
        self.metadata: dict[str, Any] = {
            "id": uid or f"images-{seed}-{count}-{int(planted)}",
            "index2label": dict(index2label),
        }

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        label = self.labels[index]
        if self._labeled:
            target = np.zeros(len(self.metadata["index2label"]), dtype=np.float32)
            target[label] = 1.0
        else:
            target = np.zeros(0, dtype=np.float32)
        datum: dict[str, Any] = {"id": index}
        if self._factors:
            datum["site"] = f"site-{(label + 1) % 3}" if index % 10 == 9 else f"site-{label}"
            datum["angle"] = float(index % 7)
        datum.update({key: value(index) if callable(value) else value for key, value in self._extra.items()})
        return self._images[index], target, datum


class _Boxes:
    """An object-detection target: boxes (x0, y0, x1, y1), their labels, and one score row per box."""

    def __init__(self, boxes: Any, labels: Any) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.scores = np.ones((len(self.labels), 3), dtype=np.float32)


class Detections:
    """3x32x32 images; each holds two 8x8 boxes (``car`` and ``van``) and a 2x2 ``car`` box, every fourth image none.

    ``bus`` is declared and never labelled.  Of the 30 items holding boxes, a crop ``min_size`` of 4 drops the 30 tiny
    boxes and leaves 60 crops of one size, which the flatten extractor can embed.
    """

    def __init__(self, count: int = 40, *, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self._images = [rng.integers(0, 255, (3, 32, 32), dtype=np.uint8) for _ in range(count)]
        self.metadata: dict[str, Any] = {"id": f"detections-{count}", "index2label": {0: "car", 1: "van", 2: "bus"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        if index % 4 == 3:
            target = _Boxes(np.zeros((0, 4)), [])
        else:
            target = _Boxes([[1, 1, 9, 9], [12, 12, 20, 20], [24, 24, 26, 26]], [0, 1, 0])
        return self._images[index], target, {"id": index}


def pipeline(
    *,
    workflows: Sequence[Any] = (),
    evaluators: Sequence[Any] = (),
    tasks: Sequence[Mapping[str, Any]] = (),
    datasets: Mapping[str, Any],
    extractor: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> PipelineConfig:
    """A pipeline with one in-memory dataset and one same-named source per entry of *datasets*.

    *workflows* and *evaluators* hold dicts or config models; *tasks* are written as a config file writes them.
    """
    data: dict[str, Any] = {
        "datasets": [
            DatasetProtocolConfig(name=f"{name}_data", format="maite", dataset=ds) for name, ds in datasets.items()
        ],
        "sources": [SourceConfig(name=name, dataset=f"{name}_data") for name in datasets],
        "evaluators": list(evaluators),
        "workflows": list(workflows),
        "tasks": [TaskConfig.model_validate(task) for task in tasks],
        **(extra or {}),
    }
    if extractor:
        data["extractors"] = [FLAT, *data.get("extractors", [])]
    return PipelineConfig.model_validate(data)


def run_preset(
    entry: Mapping[str, Any],
    datasets: Any,
    *,
    extractor: bool = False,
    task: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    data_dir: Any = None,
    output_dir: Any = None,
) -> Any:
    """Run one workflow entry as the task ``t`` and return its result.

    *datasets* is one dataset (the source ``src``) or a mapping of source names to datasets, bound in order.
    *data_dir* is the root that paths in the config (an ontology file, a model) resolve against.
    """
    from dataeval_flow import run_tasks
    from dataeval_flow._cache import DatasetCache

    DatasetCache.clear_instances()
    data = dict(datasets) if isinstance(datasets, Mapping) else {"src": datasets}
    body: dict[str, Any] = {"name": "t", "workflow": "w", "sources": list(data)}
    if extractor:
        body["extractor"] = "flat"
    body.update(task or {})
    config = pipeline(
        workflows=[{"name": "w", **entry}],
        tasks=[body],
        datasets=data,
        extractor=extractor or "extractor" in body,
        extra={"seed": 0, **(extra or {})},  # a seed makes the detectors that draw random numbers repeatable
    )
    return run_tasks(config, data_dir=data_dir, output_dir=output_dir)["t"]


def by_title(result: Any) -> dict[str, Any]:
    """A result's findings keyed by title (the first one of a repeated title)."""
    found: dict[str, Any] = {}
    for finding in result.findings:
        found.setdefault(finding.title, finding)
    return found


class Concat:
    """Two datasets end to end, under one ``index2label`` (the first's)."""

    def __init__(self, first: Any, second: Any, uid: str = "concat") -> None:
        self._parts = (first, second)
        self.metadata: dict[str, Any] = {"id": uid, "index2label": dict(first.metadata["index2label"])}

    def __len__(self) -> int:
        return len(self._parts[0]) + len(self._parts[1])

    def __getitem__(self, index: int) -> Any:
        first = len(self._parts[0])
        image, target, datum = self._parts[0][index] if index < first else self._parts[1][index - first]
        return image, target, {**datum, "id": index}  # one id per item across both parts


def write_onnx_classifier(root: Any, *, size: int = 16, n_classes: int = 3) -> tuple[str, str]:
    """Write a tiny ONNX image classifier and DataEval's metadata for it under *root*; return their file names.

    The model is one linear layer from the flattened image to ``n_classes`` logits, with weights set so that class 0
    grows with the mean pixel and class 1 shrinks with it.  A dark image, a bright one and a mid-grey one therefore
    differ in how sure the model is, which is what the ``uncertainty`` extractor measures.
    """
    import json

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    pixels = 3 * size * size
    weights = np.zeros((pixels, n_classes), dtype=np.float32)
    weights[:, 0] = 8.0 / pixels
    weights[:, 1] = -8.0 / pixels
    bias = np.array([0.0, 4.0, 0.0], dtype=np.float32)[:n_classes]
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["image"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "w"], ["product"]),
            helper.make_node("Add", ["product", "b"], ["scores"]),
        ],
        "classifier",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["batch", n_classes])],
        [numpy_helper.from_array(weights, "w"), numpy_helper.from_array(bias, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(root / "model.onnx"))
    metadata = {
        "interface": {"name": "JATIC_ONNX", "version": "v1"},
        "io": {
            "batchSize": -1,
            "interface": "IMAGE_CLASSIFICATION",
            "input": {"channels": "RGB", "height": size, "width": size},
            "output": {"nClasses": n_classes},
        },
    }
    (root / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
    return "model.onnx", "model.json"


class Readings:
    """Classification items (alternating cat and dog) whose metadata is awkward to read.

    ``kind`` picks the awkwardness: ``"commas"`` makes ``weight`` a number except every tenth reading, written with a
    thousands comma (``"7,804"``); ``"markers"`` makes ``latitude`` a number except ``"N"`` where ``index % 7 == 3``
    and ``"S"`` at item 20; ``"continuous"`` gives a clean continuous ``altitude`` nobody pinned the bins of;
    ``"clean"`` gives a fixed three-valued ``weather`` and nothing else.
    """

    def __init__(self, kind: str, count: int = 60) -> None:
        self._kind, self._count = kind, count
        self._numbers = np.random.default_rng(0).integers(1000, 9000, count)
        self._altitude = np.random.default_rng(1).uniform(0.0, 1000.0, count)
        self.metadata: dict[str, Any] = {"id": f"readings-{kind}-{count}", "index2label": {0: "cat", 1: "dog"}}

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(2, dtype=np.float32)
        target[index % 2] = 1.0
        datum: dict[str, Any] = {"id": index}
        if self._kind == "commas":
            raw = int(self._numbers[index])
            datum["weight"] = f"{raw:,}" if index % 10 == 0 else raw
        elif self._kind == "markers":
            datum["latitude"] = "N" if index % 7 == 3 else "S" if index == 20 else float(index)
        elif self._kind == "continuous":
            datum["altitude"] = float(self._altitude[index])
        else:
            datum["weather"] = ("clear", "fog", "rain")[index % 3]
        return np.zeros((3, 8, 8), dtype=np.float32), target, datum
