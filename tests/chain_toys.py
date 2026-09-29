"""Toy transforms, toy detection datasets and pipeline builders shared by the chain tests."""

import zlib
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from dataeval.data import Indices, View
from dataeval.protocols import DatasetMetadata

from dataeval_flow import PipelineConfig
from dataeval_flow.config import DatasetProtocolConfig, SourceConfig, TaskConfig
from dataeval_flow.steps import DataType, Port, Transform, TransformConfig, TransformContext
from tests.evaluator_toys import FLAT, ToyImages


class KeepConfig(TransformConfig):
    input: str


class Keep(Transform[KeepConfig]):
    """Hands its input on unchanged."""

    name: ClassVar[str] = "toy-keep"
    description: ClassVar[str] = "Hands its input on unchanged."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    def run(self, config: KeepConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        return {"output": inputs["input"].value}


class FirstConfig(TransformConfig):
    input: str
    n: int = 3


class First(Transform[FirstConfig]):
    """The first `n` items of its input."""

    name: ClassVar[str] = "toy-first"
    description: ClassVar[str] = "The first n items."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    def run(self, config: FirstConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        return {"output": View(inputs["input"].value, Indices(list(range(config.n))))}


class ExplodeConfig(TransformConfig):
    input: str
    only: str | None = None


class Explode(Transform[ExplodeConfig]):
    """Raises, on every input or only on the node whose address is `only`."""

    name: ClassVar[str] = "toy-explode"
    description: ClassVar[str] = "Raises."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    def run(self, config: ExplodeConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        node = inputs["input"]
        if config.only is None or node.address == config.only:
            raise RuntimeError(f"boom on {node.address}")
        return {"output": node.value}


_TOYS = {
    "toy-keep": "tests.chain_toys:Keep",
    "toy-first": "tests.chain_toys:First",
    "toy-explode": "tests.chain_toys:Explode",
}


def register_toys(plugins: dict[str, list[tuple[str, str]]]) -> None:
    """Serve the toy transforms through the `plugins` fixture, before the first registry lookup."""
    plugins.setdefault("dataeval_flow.transforms", []).extend(_TOYS.items())


class _Detection:
    """A duck-typed object-detection target: DataEval exports no constructible one."""

    def __init__(self, boxes: np.ndarray, labels: np.ndarray) -> None:
        self.boxes = boxes
        self.labels = labels
        self.scores = np.ones(len(labels), dtype=np.float32)


class ToyDetections:
    """An in-memory object-detection dataset: 3x16x16 uint8 images, one or two boxes each.

    Parameters
    ----------
    labels : Sequence[Sequence[int]]
        Each item's box labels, one inner sequence per item.
    index2label : Mapping[int, str]
        The class names.
    duplicate_of : Mapping[int, int]
        Items whose image copies another item's, making exact duplicates.
    dataset_id : str
        The dataset's id.
    """

    def __init__(
        self,
        labels: Sequence[Sequence[int]],
        index2label: Mapping[int, str],
        *,
        duplicate_of: Mapping[int, int] | None = None,
        dataset_id: str = "toy-detections",
    ) -> None:
        self._labels = [list(item) for item in labels]
        self._copy = dict(duplicate_of or {})
        self._seed = zlib.crc32(dataset_id.encode())  # two corpora never share an image by accident
        self.metadata = DatasetMetadata(id=dataset_id, index2label=dict(index2label))

    def __len__(self) -> int:
        return len(self._labels)

    def _image(self, index: int) -> np.ndarray:
        source = self._copy.get(index, index)
        rng = np.random.default_rng((self._seed, source))
        return rng.integers(0, 255, size=(3, 16, 16), dtype=np.uint8)

    def __getitem__(self, index: int) -> tuple[np.ndarray, _Detection, dict[str, Any]]:
        labels = np.asarray(self._labels[index], dtype=np.intp)
        boxes = np.asarray([[1 + 6 * i, 1, 6 + 6 * i, 9] for i in range(len(labels))], dtype=np.float32)
        return self._image(index), _Detection(boxes, labels), {"id": index}


def chain_pipeline(
    *,
    workflows: Sequence[Mapping[str, Any]] = (),
    evaluators: Sequence[Any] = (),
    tasks: Sequence[Mapping[str, Any]] = (),
    datasets: Mapping[str, Any] | None = None,
    extractor: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> PipelineConfig:
    """A pipeline with one in-memory dataset and one same-named source per entry of `datasets`.

    `workflows` and `tasks` are written as a config file would write them. `datasets` defaults to one source
    `src` over :class:`ToyImages`.
    """
    datasets = datasets if datasets is not None else {"src": ToyImages()}
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
        data["extractors"] = [FLAT]
    return PipelineConfig.model_validate(data)
