"""`load_source`: a configured source as a run reads it, for a training job to digest (follow-ons spec §4)."""

from typing import Any

import pytest

from dataeval_flow import dataset_digest, load_source, run_task
from dataeval_flow._cache import DatasetCache
from dataeval_flow._sources import _refuse_unseeded, resolve_source
from dataeval_flow.config import DatasetProtocolConfig, PipelineConfig, SourceConfig, TaskConfig
from dataeval_flow.evaluators.quality import ContentDigestConfig
from tests.evaluator_toys import ToyImages

_LIMIT = {"type": "Limit", "params": {"size": 5}}


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _viewed(*operations: dict[str, Any], merged: bool = False) -> PipelineConfig:
    """Source `train` over twelve toy images through view `v` of `operations`; `merged`, a merge of two operands each
    read through `v`, with no view of its own."""
    data: dict[str, Any] = {
        "datasets": [DatasetProtocolConfig(name="toy_data", format="maite", dataset=ToyImages())],
        "views": [{"name": "v", "operations": list(operations)}],
        "evaluators": [ContentDigestConfig(name="digest")],
        "tasks": [TaskConfig(name="t", workflow="digest", sources=["train"], kind="evaluator")],
    }
    if merged:
        data["sources"] = [
            SourceConfig(name="a", dataset="toy_data", view="v"),
            SourceConfig(name="b", dataset="toy_data", view="v"),
            {"name": "train", "merge": ["a", "b"]},
        ]
    else:
        data["sources"] = [SourceConfig(name="train", dataset="toy_data", view="v")]
    return PipelineConfig.model_validate(data)


def test_a_viewed_source_loads_the_items_a_run_reads() -> None:
    config = _viewed(_LIMIT)
    loaded = load_source(config, "train")
    assert len(loaded) == 5
    assert dataset_digest(loaded).content == run_task(config, "t").output.data()["content"]


def test_an_unseeded_shuffle_before_a_limit_is_refused() -> None:
    with pytest.raises(ValueError, match=r"view 'v' runs `Shuffle` with no `seed`"):
        load_source(_viewed({"type": "Shuffle"}, _LIMIT), "train")


def test_an_unseeded_shuffle_as_the_last_operation_only_reorders() -> None:
    reordered = load_source(_viewed(_LIMIT, {"type": "Shuffle"}), "train")
    assert dataset_digest(reordered) == dataset_digest(load_source(_viewed(_LIMIT), "train"))


def test_a_seeded_shuffle_before_a_limit_draws_the_same_items_each_load() -> None:
    config = _viewed({"type": "Shuffle", "params": {"seed": 3}}, _LIMIT)
    assert dataset_digest(load_source(config, "train")) == dataset_digest(load_source(config, "train"))


def test_a_merge_whose_operand_view_shuffles_unseeded_is_refused() -> None:
    with pytest.raises(ValueError, match="`Shuffle` with no `seed`"):
        load_source(_viewed({"type": "Shuffle"}, merged=True), "train")


def test_an_unknown_source_is_refused_naming_the_sources() -> None:
    with pytest.raises(ValueError, match=r"Unknown source: 'nope'. Available: \['train'\]"):
        load_source(_viewed(_LIMIT), "nope")


def test_a_stride_without_jitter_needs_no_seed() -> None:
    # Not loaded: DataEval's `Stride` lacks `requires`, so `View` can't hold it; the seed check is what's pinned.
    _refuse_unseeded(resolve_source("train", _viewed({"type": "Stride", "params": {"step": 2}}), None))


def test_a_jittered_stride_with_no_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="`Stride` with `jitter` and no `seed`"):
        load_source(_viewed({"type": "Stride", "params": {"step": 2, "jitter": True}}), "train")


def test_a_shuffle_with_a_null_seed_before_a_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="`Shuffle` with no `seed`"):
        load_source(_viewed({"type": "Shuffle", "params": {"seed": None}}, _LIMIT), "train")


def test_a_merge_sharing_a_view_with_its_operands_exempts_only_its_own_last_operation() -> None:
    config = _viewed(_LIMIT, {"type": "Shuffle"}, merged=True)
    assert config.sources is not None
    config.sources[2].view = "v"
    with pytest.raises(ValueError, match="`Shuffle` with no `seed`"):
        load_source(config, "train")
