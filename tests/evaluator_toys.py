"""Tiny in-memory datasets and pipelines for the evaluator tests.

Twelve 3x16x16 images by default: item 5 is a byte-for-byte copy of item 0, and item 7 is
solid white. A duplicates run has one exact group to find, and an outliers run one image to
flag. ``ToyImages(near_duplicate=True)`` also makes item 9 a one-pixel edit of item 3, so a
duplicates run finds a near group as well.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from dataeval_flow import PipelineConfig
from dataeval_flow.config import DatasetProtocolConfig, SourceConfig
from dataeval_flow.config.extractors import FlattenExtractorConfig

if TYPE_CHECKING:
    from dataeval.protocols import DatasetMetadata

    from dataeval_flow import Result


class ToyImages:
    """A MAITE-shaped image classification dataset with one planted duplicate and one planted outlier."""

    def __init__(self, count: int = 12, seed: int = 0, *, near_duplicate: bool = False) -> None:
        rng = np.random.default_rng(seed)
        self._images = [rng.integers(0, 255, (3, 16, 16), dtype=np.uint8) for _ in range(count)]
        if count > 7:
            self._images[5] = self._images[0].copy()
            self._images[7] = np.full((3, 16, 16), 255, dtype=np.uint8)
        if near_duplicate and count > 9:
            self._images[9] = self._images[3].copy()
            self._images[9][0, 0, 0] ^= 1
        self.metadata: DatasetMetadata = {"id": f"toy-{seed}-{count}", "index2label": {0: "a", 1: "b"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(2, dtype=np.float32)
        target[index % 2] = 1.0
        return self._images[index], target, {"id": index}


def toy_pipeline(
    *,
    evaluators: Sequence[Any] = (),
    workflows: Sequence[Any] = (),
    tasks: Sequence[Any] = (),
    sources: Sequence[str] = ("src",),
    dataset: Any = None,
    extractor: bool = False,
) -> PipelineConfig:
    """A pipeline whose every source reads one toy dataset, with a flatten extractor on request."""
    return PipelineConfig(
        datasets=[
            DatasetProtocolConfig(name="toy", format="maite", dataset=dataset if dataset is not None else ToyImages())
        ],
        sources=[SourceConfig(name=name, dataset="toy") for name in sources],
        extractors=[FlattenExtractorConfig(name="flat")] if extractor else None,
        evaluators=list(evaluators) or None,
        workflows=list(workflows) or None,
        tasks=list(tasks) or None,
    )


def exact_groups(rows: Sequence[dict[str, Any]]) -> set[tuple[int, ...]]:
    """The item-level exact-duplicate groups in a ``quality.duplicates`` table, as sorted index tuples."""
    return {tuple(sorted(row["item_indices"])) for row in rows if row["dup_type"] == "exact" and row["level"] == "item"}


def output_json(result: "Result[Any, Any]") -> dict[str, Any]:
    """An evaluator result's output as JSON, the form ``to_dict()`` and ``export()`` write."""
    return cast("dict[str, Any]", result.to_dict()["output"])
