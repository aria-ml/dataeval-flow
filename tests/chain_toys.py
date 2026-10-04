"""Toy transforms, toy detection datasets and pipeline builders shared by the chain tests."""

import zlib
from collections.abc import Collection, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from dataeval.data import Indices, View
from dataeval.protocols import DatasetMetadata
from dataeval.quality import DuplicatesOutput
from dataeval.shift import DriftOutput
from pydantic import BaseModel

from dataeval_flow import PipelineConfig, SourceCount
from dataeval_flow.config import DatasetProtocolConfig, SourceConfig, TaskConfig
from dataeval_flow.steps import (
    Check,
    CheckConfig,
    CheckContext,
    Combine,
    CombineConfig,
    CombineContext,
    DataType,
    Port,
    StepSkipped,
    Transform,
    TransformConfig,
    TransformContext,
)
from dataeval_flow.workflows import Finding
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


class YieldConfig(TransformConfig):
    input: str


class Yield(Transform[YieldConfig]):
    """Raises `StepSkipped`: a transform that declines to run."""

    name: ClassVar[str] = "toy-yield"
    description: ClassVar[str] = "Declines."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    def run(self, config: YieldConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        raise StepSkipped("it declines")


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


class OnePlaceConfig(TransformConfig):
    input: str


class OnePlace(Transform[OnePlaceConfig]):
    """Hands its input on, and cannot run once per element of a list."""

    name: ClassVar[str] = "toy-one-place"
    description: ClassVar[str] = "Runs once, never per element."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)
    broadcasts: ClassVar[bool] = False

    def run(self, config: OnePlaceConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        return {"output": inputs["input"].value}


class GroupCount(BaseModel):
    """How many duplicate groups an Output holds: a combine's output."""

    groups: int


class CountGroupsConfig(CombineConfig):
    input: str


class CountGroups(Combine[CountGroupsConfig]):
    """Counts a Duplicates Output's groups: one row of its table each."""

    name: ClassVar[str] = "toy-count-groups"
    description: ClassVar[str] = "Counts duplicate groups."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(DuplicatesOutput,)),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(GroupCount,)),)

    def run(self, config: CountGroupsConfig, inputs: Mapping[str, Any], context: CombineContext) -> Mapping[str, Any]:
        return {"output": GroupCount(groups=len(inputs["input"].value.data()))}


class GroupLimitConfig(CheckConfig):
    input: str
    most: float | None = 0.0


class GroupLimit(Check[GroupLimitConfig]):
    """Warns when a count of groups passes `most`; `None` judges nothing."""

    name: ClassVar[str] = "toy-at-most"
    description: ClassVar[str] = "Warns above a group count."
    title: ClassVar[str] = "Group count"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(GroupCount,)),)

    def run(self, config: GroupLimitConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:
        count = inputs["input"].value.groups
        severity = "info" if config.most is None else ("warning" if count > config.most else "ok")
        return [Finding(severity=severity, title=self.title, brief=f"{count} groups")]


class UnassessableConfig(CheckConfig):
    input: str
    only: str | None = None


class Unassessable(Check[UnassessableConfig]):
    """Raises `StepSkipped` to say it cannot assess, on every input or only on the node whose address is `only`."""

    name: ClassVar[str] = "toy-unassessable"
    description: ClassVar[str] = "Cannot assess."
    title: ClassVar[str] = "Unassessable"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(GroupCount,)),)

    def run(self, config: UnassessableConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:
        node = inputs["input"]
        if config.only is None or node.address == config.only:
            raise StepSkipped(f"nothing to judge in {node.address}")
        return [Finding(severity="ok", title=self.title, brief="judged")]


class WorstConfig(CheckConfig):
    input: str


class Worst(Check[WorstConfig]):
    """Judges a whole list of group counts at once: its largest."""

    name: ClassVar[str] = "toy-worst"
    description: ClassVar[str] = "The largest group count in a list."
    title: ClassVar[str] = "Worst group count"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(GroupCount,), is_list=True),)

    def run(self, config: WorstConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:
        counts = {key: node.value.groups for key, node in inputs["input"].present.items()}
        worst = max(counts, key=lambda key: counts[key])
        missing = sorted(set(inputs["input"].elements) - set(counts))
        brief = f"worst: {worst} ({counts[worst]} groups)" + (f", missing {', '.join(missing)}" if missing else "")
        return [Finding(severity="info", title=self.title, brief=brief)]


class WorstOfConfig(CheckConfig):
    input: list[str]


class WorstOf(Check[WorstOfConfig]):
    """Judges several whole lists of group counts at once: the largest across them."""

    name: ClassVar[str] = "toy-worst-of"
    description: ClassVar[str] = "The largest group count across lists."
    title: ClassVar[str] = "Worst group count of several"
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("input", DataType.OUTPUT, classes=(GroupCount,), is_list=True, count=SourceCount.ONE_OR_MORE),
    )

    def run(self, config: WorstOfConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:
        counts = [node.value.groups for listed in inputs["input"] for node in listed.present.values()]
        return [Finding(severity="info", title=self.title, brief=f"{len(counts)} counts, worst {max(counts)}")]


class AgainstConfig(CheckConfig):
    reference: str
    others: str


class Against(Check[AgainstConfig]):
    """Judges one group count against a list of others that may be empty."""

    name: ClassVar[str] = "toy-against"
    description: ClassVar[str] = "A group count against others."
    title: ClassVar[str] = "Group count against others"
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("reference", DataType.OUTPUT, classes=(GroupCount,)),
        Port("others", DataType.OUTPUT, classes=(GroupCount,), is_list=True, may_be_empty=True),
    )

    def run(self, config: AgainstConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:
        others = inputs["others"].present
        brief = f"{inputs['reference'].value.groups} groups against {len(others)} others"
        return [Finding(severity="info", title=self.title, brief=brief)]


class DriftedConfig(CheckConfig):
    input: str


class Drifted(Check[DriftedConfig]):
    """``toy-drifted``: a warning when a drift Output drifted, ok otherwise."""

    name: ClassVar[str] = "toy-drifted"
    description: ClassVar[str] = "Warns on drift."
    title: ClassVar[str] = "Drifted"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(DriftOutput,)),)

    def run(self, config: DriftedConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:
        drifted = inputs["input"].value.drifted
        return [Finding(severity="warning" if drifted else "ok", title=self.title)]


_TOYS = {
    "toy-keep": "tests.chain_toys:Keep",
    "toy-first": "tests.chain_toys:First",
    "toy-explode": "tests.chain_toys:Explode",
    "toy-halves": "tests.chain_toys:Halves",
    "toy-gather": "tests.chain_toys:Gather",
    "toy-pair": "tests.chain_toys:Pair",
    "toy-spread": "tests.chain_toys:Spread",
    "toy-detections-only": "tests.chain_toys:DetectionsOnly",
    "toy-one-place": "tests.chain_toys:OnePlace",
    "toy-yield": "tests.chain_toys:Yield",
}


_COMBINE_TOYS = {"toy-count-groups": "tests.chain_toys:CountGroups"}
_CHECK_TOYS = {
    "toy-at-most": "tests.chain_toys:GroupLimit",
    "toy-unassessable": "tests.chain_toys:Unassessable",
    "toy-worst": "tests.chain_toys:Worst",
    "toy-worst-of": "tests.chain_toys:WorstOf",
    "toy-against": "tests.chain_toys:Against",
    "toy-drifted": "tests.chain_toys:Drifted",
}


def register_toys(plugins: dict[str, list[tuple[str, str]]]) -> None:
    """Serve the toy transforms, combines and checks through the `plugins` fixture, before the first lookup."""
    plugins.setdefault("dataeval_flow.transforms", []).extend(_TOYS.items())
    plugins.setdefault("dataeval_flow.combines", []).extend(_COMBINE_TOYS.items())
    plugins.setdefault("dataeval_flow.checks", []).extend(_CHECK_TOYS.items())


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
        self._seed = zlib.crc32(dataset_id.encode())  # two datasets never share an image by accident
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
