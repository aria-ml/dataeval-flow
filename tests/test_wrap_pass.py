"""`wrap` with `other_kinds: pass`: detection data is cropped, anything else handed on, sharing its node's key
(coverage spec §5.1, §17)."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import dataeval_flow._cache as cache
from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps.transforms import WrapConfig, WrapTransform
from tests.chain_toys import ToyDetections, chain_pipeline
from tests.evaluator_toys import ToyImages


def _run(datasets: dict, steps: list, *, extractor: bool = False, evaluators: list = ()) -> ChainResult:  # type: ignore[assignment]
    DatasetCache.clear_instances()
    task = {"name": "t", "workflow": "w", "sources": ["src"]} | ({"extractor": "flat"} if extractor else {})
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        evaluators=list(evaluators),
        tasks=[task],
        datasets=datasets,
        extractor=extractor,
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


_CROPS = {"name": "crops", "transform": "wrap", "input": "data", "wrapper": "DetectionCrops", "other_kinds": "pass"}


def test_it_refuses_another_kind_by_default() -> None:
    assert WrapConfig(input="data", wrapper="DetectionCrops").other_kinds == "refuse"


def test_detection_data_is_cropped_and_counted() -> None:
    boxes = ToyDetections([[0, 1], [1], [0, 0]], {0: "a", 1: "b"})
    record = _run({"src": boxes}, [_CROPS]).steps["crops"]
    assert len(record.output) == 5
    assert record.details == {"wrapped": True, "kind": "object_detection", "items": 5, "dropped": 0}


def test_classification_data_is_handed_on_unchanged() -> None:
    result = _run({"src": ToyImages(count=8)}, [_CROPS])
    record = result.steps["crops"]
    assert record.details == {"wrapped": False, "kind": "classification"}
    assert len(record.output) == 8
    names = [entry.name for entry in result.metadata.lineage]
    assert names.count("crops") == 1
    assert names.count("data") == 1


def test_a_handed_on_dataset_shares_its_input_s_embeddings() -> None:
    """Coverage on `crops` and on `data` extract once: the pass-through node reads the source's cache key."""
    real = cache._do_compute_embeddings
    calls: list[int] = []

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    steps = [
        _CROPS,
        {"name": "on-data", "evaluator": "cov", "input": "data"},
        {"name": "on-crops", "evaluator": "cov", "input": "crops"},
    ]
    evaluators = [{"name": "cov", "type": "coverage", "num_observations": 5}]
    with patch.object(cache, "_do_compute_embeddings", counting):
        _run({"src": ToyImages(count=30)}, steps, extractor=True, evaluators=evaluators)
    assert len(calls) == 1


def test_its_section_says_what_it_did() -> None:
    boxes = ToyDetections([[0, 1], [1]], {0: "a", 1: "b"})
    record = _run({"src": boxes}, [_CROPS]).steps["crops"]
    texts = [getattr(block, "text", "") for block in WrapTransform().section(record)]
    assert any(text.startswith("Cropped 3 detections into items") for text in texts)


def test_its_section_counts_the_boxes_it_dropped() -> None:
    boxes = ToyDetections([[0, 1], [1]], {0: "a", 1: "b"})
    crops = {**_CROPS, "params": {"min_size": 1000}}
    record = _run({"src": boxes}, [crops]).steps["crops"]
    assert record.details is not None
    assert record.details["dropped"] == 3
    texts = [getattr(block, "text", "") for block in WrapTransform().section(record)]
    assert any("3 were too small or degenerate to embed" in text for text in texts)


def test_its_section_says_when_the_input_passed_through_is_empty() -> None:
    record = SimpleNamespace(details={"wrapped": False, "kind": None})
    texts = [getattr(block, "text", "") for block in WrapTransform().section(record)]
    assert texts == ["Passed through: the input is empty."]
