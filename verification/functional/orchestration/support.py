"""Shared support for the configuration, dataset, extractor, preprocessing and orchestration tests.

Not a test module. It holds:

- ``plugins``, a fixture that serves entry points from a dict, as an installed plug-in package would, and resets
  every registry around the test;
- a small plug-in written against the public API alone (a workflow with its combine and check, an evaluator, an
  extractor and an image transform), which the tests register through ``plugins``;
- builders for a pipeline on disk and for a pipeline over in-memory datasets.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml
from dataeval.flags import ImageStats
from dataeval.quality import Outliers, OutliersOutput
from pydantic import BaseModel, Field

import dataeval_flow._registry as registry_module
from dataeval_flow import InputKind, InputSpec, SourceCount
from dataeval_flow.config import StatsConfigMixin
from dataeval_flow.config.extractors import Extractor, ExtractorConfig
from dataeval_flow.config.image_transforms import ImageTransform
from dataeval_flow.evaluators import Evaluator, EvaluatorConfig, EvaluatorInputs, EvaluatorResult
from dataeval_flow.steps import (
    ChainResult,
    Check,
    CheckConfig,
    CheckContext,
    Combine,
    CombineConfig,
    CombineContext,
    DataType,
    Finding,
    InputSlot,
    Port,
)
from dataeval_flow.workflows import Preset, PresetChain, Workflow, WorkflowConfig
from verification.fixtures import write_image_folder

MODULE = "verification.functional.orchestration.support"

# ---------------------------------------------------------------------------
# The plug-in: names carry the `example.` prefix so none can clash with a built-in.
# ---------------------------------------------------------------------------


class ItemCount(BaseModel):
    """How many items a Dataset holds."""

    items: int = Field(description="Items in the Dataset.")


class ItemsConfig(CombineConfig):
    """Settings for `example.items`."""

    input: str = Field(description="The Dataset to count.")
    explode: bool = Field(default=False, description="Raise instead of counting.")


class ItemsCombine(Combine[ItemsConfig]):
    """Counts a Dataset's items."""

    name: ClassVar[str] = "example.items"
    description: ClassVar[str] = "Counts a Dataset's items."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(ItemCount,)),)

    def run(self, config: ItemsConfig, inputs: Mapping[str, Any], context: CombineContext) -> Mapping[str, Any]:
        if config.explode:
            raise RuntimeError("example.items was told to explode")
        return {"output": ItemCount(items=len(inputs["input"].value))}


class AtLeastConfig(CheckConfig):
    """Settings for `example.at-least`."""

    input: str = Field(description="The count to judge.")
    minimum: int = Field(default=0, ge=0, description="Fewest items a Dataset may hold before it warns.")


class AtLeastCheck(Check[AtLeastConfig]):
    """Warns when a Dataset holds fewer than `minimum` items."""

    name: ClassVar[str] = "example.at-least"
    description: ClassVar[str] = "Warns below a count of items."
    title: ClassVar[str] = "Items"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(ItemCount,)),)

    def run(self, config: AtLeastConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:
        n = inputs["input"].value.items
        return [Finding(severity="warning" if n < config.minimum else "ok", title=self.title, brief=f"{n} items")]


class CountConfig(WorkflowConfig[ChainResult]):
    """Settings for `example.count`."""

    type: str = "example.count"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset(), sources=SourceCount.ONE)
    minimum: int = Field(default=0, ge=0, description="Fewest items the source may hold before it warns.")
    explode: bool = Field(default=False, description="Make the count raise, to test failure handling.")


class CountWorkflow(Preset, Workflow[CountConfig, ChainResult]):
    """Counts the source's items and warns when it holds fewer than `minimum`."""

    name: ClassVar[str] = "example.count"
    description: ClassVar[str] = "Counts the items in a source."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: CountConfig) -> PresetChain:
        return PresetChain(
            steps=[
                {"name": "items", "combine": "example.items", "input": "data", "explode": config.explode},
                {"name": "at-least", "check": "example.at-least", "input": "items", "minimum": config.minimum},
            ]
        )


class BrightnessResult(EvaluatorResult[OutliersOutput[Any]]):
    """The result of an `example.brightness` run."""


class BrightnessConfig(EvaluatorConfig[BrightnessResult], StatsConfigMixin):
    """Settings for `example.brightness`; `explode` makes the run raise, to test failure handling."""

    type: str = "example.brightness"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.STATS}), sources=SourceCount.ONE)
    explode: bool = Field(default=False, description="Raise instead of running.")

    def stats_request(self) -> dict[str, Any]:
        return {"outlier_flags": ImageStats.VISUAL_BRIGHTNESS}


class BrightnessEvaluator(Evaluator[BrightnessConfig, OutliersOutput[Any]]):
    """Flags images whose brightness is a z-score outlier."""

    name: ClassVar[str] = "example.brightness"
    description: ClassVar[str] = "Images whose brightness is an outlier."
    dataeval_class: ClassVar[type] = Outliers
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.STATS: "from_stats"}
    calls: ClassVar[int] = 0
    """How many runs have started: lets a test see how many tasks had run when a callback fired."""

    def run(self, config: BrightnessConfig, inputs: Sequence[EvaluatorInputs]) -> OutliersOutput[Any]:
        BrightnessEvaluator.calls += 1
        if config.explode:
            raise RuntimeError("example.brightness was told to explode")
        (source,) = inputs
        assert source.stats is not None
        return Outliers(flags=ImageStats.VISUAL_BRIGHTNESS, outlier_threshold="zscore").from_stats(source.stats)


class MeanConfig(ExtractorConfig):
    """Settings for `example.mean`."""

    model: str = "example.mean"


class Means:
    """Each image's per-channel mean, after the entry's preprocessing: a DataEval `FeatureExtractor`.

    ``batches`` records the size of every call, ``dtypes`` the dtype of each image after preprocessing and ``firsts``
    the first image of each call as it arrived, so a test can see how Flow fed it.
    """

    batches: ClassVar[list[int]] = []
    dtypes: ClassVar[list[str]] = []
    firsts: ClassVar[list[Any]] = []

    def __init__(self, transforms: Any) -> None:
        self.transforms = transforms

    def __call__(self, data: Any, /) -> Any:
        Means.batches.append(len(data))
        images = data if self.transforms is None else [self.transforms(image) for image in data]
        Means.dtypes.extend(str(np.asarray(image).dtype) for image in images)
        Means.firsts.append(np.asarray(images[0]))
        return np.stack([np.asarray(image, dtype=np.float32).mean(axis=(-2, -1)) for image in images])


class MeanExtractor(Extractor[MeanConfig]):
    """Embeds each image as its per-channel means."""

    name: ClassVar[str] = "example.mean"
    description: ClassVar[str] = "Per-channel mean of each image."

    def build(self, config: MeanConfig, transforms: Any) -> Any:
        return Means(transforms)


class Invert(ImageTransform):
    """Inverts an image whose values lie in [0, maximum]."""

    name: ClassVar[str] = "example.Invert"
    description: ClassVar[str] = "maximum - x for each value."

    def __init__(self, maximum: float = 1.0) -> None:
        self.maximum = maximum

    def __call__(self, data: Any, /) -> Any:
        return self.maximum - data

    def __repr__(self) -> str:
        return f"Invert(maximum={self.maximum!r})"


# Every group the plug-in registers under, as entry points would declare them.
EXAMPLE_ENTRY_POINTS: dict[str, list[tuple[str, str]]] = {
    "dataeval_flow.workflows": [("example.count", f"{MODULE}:CountWorkflow")],
    "dataeval_flow.combines": [("example.items", f"{MODULE}:ItemsCombine")],
    "dataeval_flow.checks": [("example.at-least", f"{MODULE}:AtLeastCheck")],
    "dataeval_flow.evaluators": [("example.brightness", f"{MODULE}:BrightnessEvaluator")],
    "dataeval_flow.extractors": [("example.mean", f"{MODULE}:MeanExtractor")],
    "dataeval_flow.image_transforms": [("example.Invert", f"{MODULE}:Invert")],
}


def _registries() -> list[Any]:
    from dataeval_flow.config.extractors._registry import EXTRACTORS
    from dataeval_flow.config.image_transforms._registry import IMAGE_TRANSFORMS
    from dataeval_flow.evaluators._registry import EVALUATORS
    from dataeval_flow.steps._registry import CHECKS, COMBINES, TRANSFORMS
    from dataeval_flow.workflows._registry import WORKFLOWS

    return [WORKFLOWS, EVALUATORS, EXTRACTORS, IMAGE_TRANSFORMS, TRANSFORMS, COMBINES, CHECKS]


def _serve(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, served: dict[str, list[tuple[str, str]]]
) -> dict[str, list[tuple[str, str]]]:
    """Serve `served` as the installed entry points, and reset every registry now and when the test ends."""

    def entry_points(*, group: str) -> list[EntryPoint]:
        return [EntryPoint(name=name, value=value, group=group) for name, value in served.get(group, [])]

    monkeypatch.setattr(registry_module, "entry_points", entry_points)
    for registry in _registries():
        registry._reset()
    request.addfinalizer(lambda: [registry._reset() for registry in _registries()])
    return served


@pytest.fixture
def plugins(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> dict[str, list[tuple[str, str]]]:
    """Serve entry points from ``{group: [(name, "module:attribute")]}``, resetting every registry around the test.

    Add entries before the first registry lookup; registries load on first use.
    """
    return _serve(monkeypatch, request, {})


@pytest.fixture
def fresh_caches() -> Iterator[None]:
    """Forget the in-memory dataset caches around a test, so one test's embeddings are never served to another."""
    from dataeval_flow._cache import DatasetCache

    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


@pytest.fixture
def unseeded() -> Iterator[None]:
    """Leave DataEval's process-wide seed unset after a test: a pipeline with a ``seed`` sets it and a pipeline without
    one leaves it alone, so a seed would otherwise carry into later tests."""
    from dataeval.config import set_seed

    set_seed(None)
    yield
    set_seed(None)


@pytest.fixture
def example_plugin(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> dict[str, list[tuple[str, str]]]:
    """The entry points served with the whole example plug-in registered; add to them before the first lookup."""
    return _serve(monkeypatch, request, {group: list(entries) for group, entries in EXAMPLE_ENTRY_POINTS.items()})


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


class InMemoryImages:
    """A MAITE image classification dataset held in memory: `n` 3x8x8 images with one-hot labels over two classes.

    Item 5 is a copy of item 0 once there are six or more, so a duplicates run has a group to find. Built here, not
    from ``verification.fixtures.make_synthetic_dataset``, whose integer targets DataEval no longer accepts.
    """

    def __init__(self, n: int = 12, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._images = [rng.integers(0, 255, (3, 8, 8), dtype=np.uint8) for _ in range(n)]
        if n > 5:
            self._images[5] = self._images[0].copy()
        self.metadata = {"id": f"in-memory-{n}-{seed}", "index2label": {0: "a", 1: "b"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(2, dtype=np.float32)
        target[index % 2] = 1.0
        return self._images[index], target, {"id": index}


QUALITY: dict[str, Any] = {
    "name": "q",
    "type": "quality",
    "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"},
}


def pipeline_dict(**sections: Any) -> dict[str, Any]:
    """A pipeline over the image folder ``imgs`` (labelled by sub-folder) as source ``main``, plus `sections`.

    The base has a dataset ``ds``, a source ``main``, a Flatten extractor ``flat`` and no tasks; `sections`
    replace or add top-level keys.
    """
    base: dict[str, Any] = {
        "datasets": [{"name": "ds", "format": "image_folder", "path": "imgs", "infer_labels": True}],
        "sources": [{"name": "main", "dataset": "ds"}],
        "extractors": [{"name": "flat", "model": "flatten", "batch_size": 8}],
    }
    return {**base, **sections}


def write_project(
    root: Path, *, n_per_class: int = 5, n_classes: int = 2, **sections: Any
) -> tuple[Path, dict[str, Any]]:
    """Write ``root/imgs`` and ``root/config.yaml`` for :func:`pipeline_dict` and return ``(config_path, dict)``."""
    write_image_folder(root / "imgs", n_per_class=n_per_class, n_classes=n_classes)
    config = pipeline_dict(**sections)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config
