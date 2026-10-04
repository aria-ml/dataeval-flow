"""The `content-digest` evaluator: `dataset_digest`'s values for a source, read from every item (audit spec §7.1)."""

from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import DatasetDigest, dataset_digest, run
from dataeval_flow._cache import DatasetCache, dataset_fingerprint
from dataeval_flow.evaluators.quality import ContentDigestConfig, ContentDigestResult
from dataeval_flow.steps import ChainResult
from tests.chain_toys import chain_pipeline, run_chain_task
from tests.evaluator_toys import Items, ToyImages


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _values(digest: DatasetDigest) -> dict[str, Any]:
    return {"content": digest.content, "metadata": digest.metadata, "items": digest.items, "scheme": digest.scheme}


def test_a_run_gives_dataset_digests_values() -> None:
    data = ToyImages()
    result = run(ContentDigestConfig(), data)
    assert isinstance(result, ContentDigestResult)
    assert result.success, result.errors
    assert result.output.data() == _values(dataset_digest(data))
    assert result.to_dict()["output"]["data"] == _values(dataset_digest(data))


def test_the_report_gives_both_digests_in_full() -> None:
    data = ToyImages()
    text = run(ContentDigestConfig(), data).report(width=200)
    digest = dataset_digest(data)
    assert digest.content in text
    assert digest.metadata in text


def test_a_chain_step_digests_the_node_it_reads() -> None:
    data = ToyImages()
    part = {"type": "Indices", "params": {"indices": [0, 2, 4]}}
    config = chain_pipeline(
        workflows=[
            {
                "name": "w",
                "inputs": ["data"],
                "steps": [
                    {"name": "whole", "evaluator": "digest", "input": "data"},
                    {"name": "part", "transform": "view", "input": "data", "operations": [part]},
                    {"name": "sub", "evaluator": "digest", "input": "part"},
                ],
            }
        ],
        evaluators=[{"name": "digest", "type": "content-digest"}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": data},
    )
    result = run_chain_task(config)
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert result.steps["whole"].output.data() == _values(dataset_digest(data))
    subset = Items([data[0], data[2], data[4]], data.metadata["index2label"])
    assert result.steps["sub"].output.data() == _values(dataset_digest(subset))


def test_an_edit_the_cache_fingerprint_skips_still_changes_the_digest(tmp_path: Path) -> None:
    """The cache keys a dataset by a sample of 64 of its items, so an edit between the samples keeps the key. The
    digest still sees it, because it reads every item, never a cached value."""
    toy = ToyImages(count=200)
    items = [toy[index] for index in range(200)]
    image, target, meta = items[1]  # item 1 lies between the samples at 0 and 3
    image = image.copy()
    image[0, 0, 0] ^= 1
    original, edited = Items(items), Items([items[0], (image, target, meta), *items[2:]])
    assert dataset_fingerprint(edited) == dataset_fingerprint(original)  # the cache can't tell them apart
    first = run(ContentDigestConfig(), original, cache_dir=tmp_path)
    second = run(ContentDigestConfig(), edited, cache_dir=tmp_path)
    assert second.output.data()["content"] != first.output.data()["content"]
    assert second.output.data() == _values(dataset_digest(edited))
