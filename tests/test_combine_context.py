"""What a combine may read and show: a Dataset's statistics under its stats policy, and its own report section with
thumbnails (ood-detection spec §6.2, §6.3)."""

from collections.abc import Mapping
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from dataeval_flow import run_task
from dataeval_flow._blocks import Column, ItemRef, Table
from dataeval_flow._cache import DatasetCache
from dataeval_flow._input_spec import InputKind
from dataeval_flow.config import StatsConfigMixin, TaskConfig
from dataeval_flow.steps import ChainResult, Combine, CombineConfig, CombineContext, DataType, Port
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages


class Measured(BaseModel):
    """The statistics a combine read: their names and the image count."""

    names: list[str]
    images: int


class MeasureConfig(CombineConfig, StatsConfigMixin):
    input: str


class Measure(Combine[MeasureConfig]):
    """Reads its Dataset's statistics, and pictures its first item."""

    name: ClassVar[str] = "toy-measure"
    description: ClassVar[str] = "Reads a Dataset's statistics."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET, derives=frozenset({InputKind.STATS})),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(Measured,)),)

    def run(self, config: MeasureConfig, inputs: Mapping[str, Any], context: CombineContext) -> Mapping[str, Any]:
        stats = context.derive_stats(inputs["input"])
        return {"output": Measured(names=sorted(stats["stats"]), images=stats["image_count"])}

    def section(self, record: Any) -> list[Any]:
        column = Column(key="image", kind="image")
        return [Table(columns=[column], rows=[{"image": ItemRef(source="a", index=0)}])]


@pytest.fixture(autouse=True)
def _fresh_cache():
    """The in-memory cache outlives a run: one run's every statistic would answer another's policy."""
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _run(stats: str | None = None) -> ChainResult:
    step = {"name": "m", "combine": "toy-measure", "input": "a", **({"stats": stats} if stats else {})}
    policy = {"name": "pix", "measure": [{"bands": None, "families": ["pixel"]}]}
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["a"], "steps": [step]}],
        datasets={"src": ToyImages(count=6)},
        extra={"stats": [policy]},
    )
    result = run_task(TaskConfig(name="t", workflow="w", sources="src"), config, report_images=True)
    assert isinstance(result, ChainResult)
    return result


def test_a_combine_reads_every_statistic_per_image_where_it_names_no_policy(plugins) -> None:
    plugins.setdefault("dataeval_flow.combines", []).append(("toy-measure", "tests.test_combine_context:Measure"))
    measured = _run().steps["m"].output
    assert measured.images == 6
    assert {"brightness", "mean"} <= set(measured.names)


def test_a_combine_reads_statistics_under_its_stats_policy(plugins) -> None:
    plugins.setdefault("dataeval_flow.combines", []).append(("toy-measure", "tests.test_combine_context:Measure"))
    measured = _run("pix").steps["m"].output
    assert "mean" in measured.names
    assert "brightness" not in measured.names


def test_a_combine_s_section_is_shown_with_its_thumbnails(plugins) -> None:
    plugins.setdefault("dataeval_flow.combines", []).append(("toy-measure", "tests.test_combine_context:Measure"))
    result = _run()
    assert [(asset.item.source, asset.item.index) for asset in result.assets] == [("a", 0)]
    assert '<details class="thumb">' in result.to_html()
