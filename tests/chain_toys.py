"""Toy transforms, toy detection datasets and pipeline builders shared by the chain tests."""

import zlib
from collections.abc import Collection, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from dataeval.data import Indices, View
from dataeval.protocols import DatasetMetadata

from dataeval_flow import PipelineConfig, SourceCount
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


class HalvesConfig(TransformConfig):
    input: str


class Halves(Transform[HalvesConfig]):
    """Two outputs: the first half of its input, and the rest."""

    name: ClassVar[str] = "toy-halves"
    description: ClassVar[str] = "Splits in two."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("head", DataType.DATASET), Port("tail", DataType.DATASET))

    def run(self, config: HalvesConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        dataset = inputs["input"].value
        half = len(dataset) // 2
        return {
            "head": View(dataset, Indices(list(range(half)))),
            "tail": View(dataset, Indices(list(range(half, len(dataset))))),
        }


class GatherConfig(TransformConfig):
    input: str


class Gather(Transform[GatherConfig]):
    """Takes a whole list and hands on its first present element."""

    name: ClassVar[str] = "toy-gather"
    description: ClassVar[str] = "The first element of a list."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET, is_list=True),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    def run(self, config: GatherConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        present = [element for element in inputs["input"].elements.values() if hasattr(element, "value")]
        return {"output": present[0].value}


class PairConfig(TransformConfig):
    input: list[str]


class Pair(Transform[PairConfig]):
    """Takes exactly two Datasets, a count only its port declares, and hands on the first."""

    name: ClassVar[str] = "toy-pair"
    description: ClassVar[str] = "The first of two Datasets."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET, count=SourceCount.TWO),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    def run(self, config: PairConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        return {"output": inputs["input"][0].value}


class SpreadConfig(TransformConfig):
    input: str
    parts: int = 2


class Spread(Transform[SpreadConfig]):
    """A list output: its input split into `parts` interleaved parts, keyed "0".."parts-1"."""

    name: ClassVar[str] = "toy-spread"
    description: ClassVar[str] = "Interleaved parts, as a list."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("parts", DataType.DATASET, is_list=True),)

    @classmethod
    def output_keys(cls, config: SpreadConfig) -> Mapping[str, tuple[str, ...]]:
        return {"parts": tuple(str(i) for i in range(config.parts))}

    def run(self, config: SpreadConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        dataset = inputs["input"].value
        return {
            "parts": {
                str(i): View(dataset, Indices(list(range(i, len(dataset), config.parts)))) for i in range(config.parts)
            }
        }


class DetectionsOnlyConfig(TransformConfig):
    input: str


class DetectionsOnly(Transform[DetectionsOnlyConfig]):
    """Takes object-detection Datasets only, and hands them on."""

    name: ClassVar[str] = "toy-detections-only"
    description: ClassVar[str] = "Object detection only."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET, kinds=frozenset({"object_detection"})),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    def run(
        self, config: DetectionsOnlyConfig, inputs: Mapping[str, Any], context: TransformContext
    ) -> Mapping[str, Any]:
        return {"output": inputs["input"].value}


_TOYS = {
    "toy-keep": "tests.chain_toys:Keep",
    "toy-first": "tests.chain_toys:First",
    "toy-explode": "tests.chain_toys:Explode",
    "toy-halves": "tests.chain_toys:Halves",
    "toy-gather": "tests.chain_toys:Gather",
    "toy-pair": "tests.chain_toys:Pair",
    "toy-spread": "tests.chain_toys:Spread",
    "toy-detections-only": "tests.chain_toys:DetectionsOnly",
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

    Each datum's metadata holds its ``id`` and a ``site`` (north, south, east in turn), a factor for bias steps.

    Parameters
    ----------
    labels : Sequence[Sequence[int]]
        Each item's box labels, one inner sequence per item.
    index2label : Mapping[int, str]
        The class names.
    duplicate_of : Mapping[int, int]
        Items whose image copies another item's, making exact duplicates.
    bright : Collection[tuple[int, int]]
        ``(item, box)`` pairs whose box region is drawn white, standing out from the dim rest.
    dataset_id : str
        The dataset's id.
    """

    def __init__(
        self,
        labels: Sequence[Sequence[int]],
        index2label: Mapping[int, str],
        *,
        duplicate_of: Mapping[int, int] | None = None,
        bright: Collection[tuple[int, int]] = (),
        dataset_id: str = "toy-detections",
    ) -> None:
        self._labels = [list(item) for item in labels]
        self._copy = dict(duplicate_of or {})
        self._bright = set(bright)
        self._seed = zlib.crc32(dataset_id.encode())  # two corpora never share an image by accident
        self.metadata = DatasetMetadata(id=dataset_id, index2label=dict(index2label))

    def __len__(self) -> int:
        return len(self._labels)

    def _image(self, index: int) -> np.ndarray:
        source = self._copy.get(index, index)
        rng = np.random.default_rng((self._seed, source))
        image = rng.integers(0, 60, size=(3, 16, 16), dtype=np.uint8)
        for item, box in self._bright:
            if item == index:
                x0 = 1 + 6 * box
                image[:, 1:9, x0 : x0 + 5] = 255
        return image

    def __getitem__(self, index: int) -> tuple[np.ndarray, _Detection, dict[str, Any]]:
        labels = np.asarray(self._labels[index], dtype=np.intp)
        boxes = np.asarray([[1 + 6 * i, 1, 6 + 6 * i, 9] for i in range(len(labels))], dtype=np.float32)
        site = ("north", "south", "east")[index % 3]
        return self._image(index), _Detection(boxes, labels), {"id": index, "site": site}


def chain_pipeline(
    *,
    workflows: Sequence[Any] = (),
    evaluators: Sequence[Any] = (),
    tasks: Sequence[Mapping[str, Any]] = (),
    datasets: Mapping[str, Any] | None = None,
    extractor: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> PipelineConfig:
    """A pipeline with one in-memory dataset and one same-named source per entry of `datasets`.

    `workflows` holds dicts, model instances, or both mixed; `tasks` are written as a config file would write
    them. `datasets` defaults to one source `src` over :class:`ToyImages`.
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


def yaml_pipeline(text: str, datasets: Mapping[str, Any]) -> PipelineConfig:
    """The pipeline `text` writes, with one in-memory dataset and same-named source per entry of `datasets`."""
    import yaml

    data = yaml.safe_load(text)
    data["datasets"] = [
        DatasetProtocolConfig(name=f"{name}_data", format="maite", dataset=ds) for name, ds in datasets.items()
    ]
    data["sources"] = [SourceConfig(name=name, dataset=f"{name}_data") for name in datasets]
    return PipelineConfig.model_validate(data)


def run_chain_task(config: PipelineConfig, task: str = "t"):
    """Run one task through the orchestrator, as the CLI would, and return its result."""
    from dataeval_flow import run_tasks

    return run_tasks(config, task)[task]


def run_toy_chain(
    config: PipelineConfig, workflow: str, sources: Sequence[str], *, step_contexts: Mapping[str, Any] | None = None
):
    """Run custom workflow `workflow` on `sources` straight through the executor, without the orchestrator."""
    from dataeval_flow._cache import DatasetCache
    from dataeval_flow._chain._graph import build_graph
    from dataeval_flow._chain._run import RunSettings, bind_inputs, run_chain
    from dataeval_flow._sources import resolve_source
    from dataeval_flow.workflows._context import DatasetContext

    entry = next(item for item in config.workflows or () if item.name == workflow)
    graph = build_graph(entry, config)  # type: ignore[arg-type]
    contexts, resolved = {}, {}
    for name in sources:
        source = resolve_source(name, config)
        resolved[name] = source
        contexts[name] = DatasetContext(
            name=name,
            dataset=source.dataset,
            view_operations=source.view_config.operations if source.view_config else None,
            cache=DatasetCache.get_or_create(None, source.cache_name, source.cache_key),
        )
    inputs = bind_inputs(graph, list(sources), contexts, resolved)
    return run_chain(graph, inputs, RunSettings(task="t", pipeline=config, step_contexts=step_contexts or {}))
