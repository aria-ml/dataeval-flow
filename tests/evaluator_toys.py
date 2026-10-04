"""Tiny in-memory datasets and pipelines for the evaluator tests.

Twelve 3x16x16 images by default: item 5 is a byte-for-byte copy of item 0, and item 7 is
solid white. A duplicates run has one exact group to find, and an outliers run one image to
flag. ``ToyImages(near_duplicate=True)`` also makes item 9 a one-pixel edit of item 3, so a
duplicates run finds a near group as well.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from dataeval_flow import PipelineConfig
from dataeval_flow.config import DatasetProtocolConfig, SourceConfig
from dataeval_flow.config.extractors import FlattenExtractorConfig

if TYPE_CHECKING:
    from dataeval.protocols import DatasetMetadata

    from dataeval_flow import Result
    from dataeval_flow.config.extractors import ExtractorConfig
    from dataeval_flow.evaluators import EvaluatorResult

# Flatten carries no batch size, and DataEval refuses to embed without one.
FLAT = FlattenExtractorConfig(name="flat", batch_size=8)


class ToyImages:
    """A MAITE-shaped image classification dataset with one planted duplicate and one planted outlier.

    ``ToyImages(labeled=False)`` gives every item an empty target, as a dataset without labels has.
    ``ToyImages(bright=True)`` lifts every pixel by 100, out of the distribution of the rest.
    """

    def __init__(
        self,
        count: int = 12,
        seed: int = 0,
        *,
        near_duplicate: bool = False,
        labeled: bool = True,
        bright: bool = False,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._images = [rng.integers(0, 255, (3, 16, 16), dtype=np.uint8) for _ in range(count)]
        if count > 7:
            self._images[5] = self._images[0].copy()
            self._images[7] = np.full((3, 16, 16), 255, dtype=np.uint8)
        if near_duplicate and count > 9:
            self._images[9] = self._images[3].copy()
            self._images[9][0, 0, 0] ^= 1
        self._labeled = labeled
        self.metadata: DatasetMetadata = {"id": f"toy-{seed}-{count}", "index2label": {0: "a", 1: "b"}}
        if bright:
            self._images = [np.clip(image.astype(np.int16) + 100, 0, 255).astype(np.uint8) for image in self._images]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        if not self._labeled:
            return self._images[index], np.zeros(0, dtype=np.float32), {"id": index}
        target = np.zeros(2, dtype=np.float32)
        target[index % 2] = 1.0
        return self._images[index], target, {"id": index}


class ToyFactors:
    """Images across three classes, each with two metadata factors: `site` follows the class, and `angle` does not."""

    def __init__(self, count: int = 60) -> None:
        rng = np.random.default_rng(0)
        self._images = [rng.integers(0, 255, (3, 8, 8), dtype=np.uint8) for _ in range(count)]
        self.metadata: DatasetMetadata = {"id": f"factors-{count}", "index2label": {0: "cat", 1: "dog", 2: "bird"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(3, dtype=np.float32)
        target[index % 3] = 1.0
        return self._images[index], target, {"site": ["north", "south", "east"][index % 3], "angle": float(index % 7)}


class Items:
    """A dataset of exactly the items given, with the class names given (``{0: "a", 1: "b"}`` by default).

    ``Items(items, metadata=False)`` gives it no ``metadata`` attribute at all, as some datasets have none.
    """

    def __init__(
        self, items: Sequence[Any], index2label: Mapping[int, str] | None = None, *, metadata: bool = True
    ) -> None:
        self._items = list(items)
        if metadata:
            self.metadata = {"id": "items", "index2label": dict(index2label or {0: "a", 1: "b"})}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Any:
        return self._items[index]


def toy_pipeline(
    *,
    evaluators: Sequence[Any] = (),
    workflows: Sequence[Any] = (),
    tasks: Sequence[Any] = (),
    sources: Sequence[str] = ("src",),
    dataset: Any = None,
    datasets: "Mapping[str, Any] | None" = None,
    extractor: bool = False,
) -> PipelineConfig:
    """A pipeline whose every source reads one toy dataset, with a flatten extractor on request.

    `datasets` maps source names to their own datasets instead, for a task that compares sources.
    """
    if datasets is not None:
        dataset_configs = [
            DatasetProtocolConfig(name=f"{name}_data", format="maite", dataset=data) for name, data in datasets.items()
        ]
        source_configs = [SourceConfig(name=name, dataset=f"{name}_data") for name in datasets]
    else:
        dataset_configs = [
            DatasetProtocolConfig(name="toy", format="maite", dataset=dataset if dataset is not None else ToyImages())
        ]
        source_configs = [SourceConfig(name=name, dataset="toy") for name in sources]
    return PipelineConfig(
        datasets=dataset_configs,
        sources=source_configs,
        extractors=[FLAT] if extractor else None,
        evaluators=list(evaluators) or None,
        workflows=list(workflows) or None,
        tasks=list(tasks) or None,
    )


def shifted_sources(count: int = 40, *, validation: bool = False) -> dict[str, ToyImages]:
    """A reference, then test images brightened out of its distribution; between them, a validation set on request."""
    sources = {"reference": ToyImages(count=count)}
    if validation:
        sources["validation"] = ToyImages(count=count, seed=2)
    sources["test"] = ToyImages(count=count, seed=1, bright=True)
    return sources


def exact_groups(rows: Sequence[dict[str, Any]]) -> set[tuple[int, ...]]:
    """The item-level exact-duplicate groups in a ``duplicates`` table, as sorted index tuples."""
    return {tuple(sorted(row["item_indices"])) for row in rows if row["dup_type"] == "exact" and row["level"] == "item"}


def output_json(result: "Result[Any, Any]") -> dict[str, Any]:
    """An evaluator result's output as JSON, the form ``to_dict()`` and ``export()`` write."""
    return cast("dict[str, Any]", result.to_dict()["output"])


# Toy data each built-in evaluator can read, by type: the data `run` takes, and the extractor its task needs. Each
# factory takes the item count, so a test can ask for an empty source. A task adding a type adds its row.
_TOY_DATA: "dict[str, Callable[[int], tuple[Any, ExtractorConfig | None]]]" = {
    "duplicates": lambda count: (ToyImages(count=count), None),
    "label-health": lambda count: (ToyImages(count=count), None),
    "factor-triage": lambda count: (ToyFactors(count=count), None),
    "content-digest": lambda count: (ToyImages(count=count), None),
    "metadata-summary": lambda count: (ToyFactors(count=count), None),
    "outliers": lambda count: (ToyImages(count=count), None),
    "balance": lambda count: (ToyFactors(count=count), None),
    "diversity": lambda count: (ToyFactors(count=count), None),
    "parity": lambda count: (ToyFactors(count=count), None),
    "representation": lambda count: (ToyImages(count=count), None),
    "coverage": lambda count: (ToyImages(count=count), FLAT),
    "prioritize": lambda count: (ToyImages(count=count), FLAT),
    "completeness": lambda count: (ToyImages(count=count), FLAT),
    "drift-domain-classifier": lambda count: (shifted_sources(count), FLAT),
    "drift-kneighbors": lambda count: (shifted_sources(count), FLAT),
    "drift-mmd": lambda count: (shifted_sources(count), FLAT),
    "drift-univariate": lambda count: (shifted_sources(count), FLAT),
    "drift-wasserstein": lambda count: (shifted_sources(count, validation=True), FLAT),
    "ood-domain-classifier": lambda count: (shifted_sources(count), FLAT),
    "ood-kneighbors": lambda count: (shifted_sources(count), FLAT),
    "divergence": lambda count: (shifted_sources(count), FLAT),
    "label-alignment": lambda count: (ToyImages(count=count), None),
    "label-reconciliation": lambda count: (ToyImages(count=count), None),
    "ontology-validation": lambda count: (ToyImages(count=count), None),
}

# Config values a bare `config_type()` cannot supply, because the field has no default. `label-alignment`
# needs a target ontology; a flat one matching `ToyImages`'s own `index2label` aligns losslessly.
_EXTRA_CONFIG: "dict[str, dict[str, Any]]" = {
    "label-alignment": {"ontology": {"a": None, "b": None}},
    "label-reconciliation": {"ontology": {"a": None, "b": None}},
    "ontology-validation": {"ontology": {"a": None, "b": None}},
}


def toy_run(name: str, count: int = 40) -> "EvaluatorResult[Any]":
    """A run of the built-in evaluator `name`, with its config's defaults, on toy data it can read, through `run`."""
    from dataeval_flow import run
    from dataeval_flow.evaluators import get_evaluator

    assert name in _TOY_DATA, f"give {name} toy data in tests/evaluator_toys.py"
    data, extractor = _TOY_DATA[name](count)
    config = get_evaluator(name).config_type(**_EXTRA_CONFIG.get(name, {}))  # type: ignore[call-arg]
    return run(config, data, extractor=extractor)


def toy_task_run(name: str, count: int = 40) -> "Result[Any, Any]":
    """`toy_run`'s run as a task of a pipeline, through `run_tasks`, the way a config file runs it."""
    from dataeval_flow import run_tasks
    from dataeval_flow.config import TaskConfig
    from dataeval_flow.evaluators import get_evaluator

    data, extractor = _TOY_DATA[name](count)
    several = isinstance(data, Mapping)
    task = TaskConfig(
        name="t",
        workflow="e",
        sources=list(data) if several else ["src"],
        kind="evaluator",
        extractor="flat" if extractor is not None else None,
    )
    config = toy_pipeline(
        evaluators=[get_evaluator(name).config_type(name="e", **_EXTRA_CONFIG.get(name, {}))],  # type: ignore[call-arg]
        tasks=[task],
        dataset=None if several else data,
        datasets=data if several else None,
        extractor=extractor is not None,
    )
    return run_tasks(config)["t"]
