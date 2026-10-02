"""A subset node reads its embeddings from rows the run already extracted for its parent, or an ancestor, through
views that change no pixels (data-splitting spec §5.3)."""

from typing import Any

import numpy as np
import pytest

import dataeval_flow._cache as cache_module
from dataeval_flow._cache import DatasetCache
from dataeval_flow._embeddings import node_embeddings
from dataeval_flow.config.extractors import FlattenExtractorConfig, UncertaintyExtractorConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows._context import DatasetContext, Subset
from tests.chain_toys import chain_pipeline, run_chain_task
from tests.evaluator_toys import FLAT, ToyImages

_ROWS = np.arange(12.0).reshape(6, 2)


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _context(parent: Subset | None = None, extractor: Any = FLAT) -> DatasetContext:
    return DatasetContext(
        name="n",
        dataset=None,  # type: ignore[arg-type]
        extractor=extractor,
        parent=parent,
        embedded={},
    )


def _never() -> Any:
    raise AssertionError("extracted rows a parent already had")


def test_a_subset_reads_its_parent_s_rows_at_its_indices_without_extracting() -> None:
    parent = _context()
    node_embeddings(parent, lambda: _ROWS)
    subset = _context(Subset(parent, (4, 1, 1)))
    assert node_embeddings(subset, _never).tolist() == _ROWS[[4, 1, 1]].tolist()


def test_a_subset_of_a_subset_composes_down_to_the_ancestor_with_rows() -> None:
    grand = _context()
    node_embeddings(grand, lambda: _ROWS)
    middle = _context(Subset(grand, (3, 5)))
    leaf = _context(Subset(middle, (1, 0)))
    assert node_embeddings(leaf, _never).tolist() == _ROWS[[5, 3]].tolist()


def test_a_subset_of_a_parent_not_yet_embedded_extracts_its_own() -> None:
    own = np.ones((2, 2))
    subset = _context(Subset(_context(), (0, 1)))
    assert node_embeddings(subset, lambda: own).tolist() == own.tolist()


def test_other_extractor_settings_are_not_sliced() -> None:
    parent = _context()
    node_embeddings(parent, lambda: _ROWS)
    other = FlattenExtractorConfig(name="other", batch_size=4)
    own = np.zeros((2, 2))
    assert node_embeddings(_context(Subset(parent, (0, 1)), extractor=other), lambda: own).tolist() == own.tolist()


def test_a_model_s_rows_are_neither_remembered_nor_sliced() -> None:
    model = UncertaintyExtractorConfig(
        name="unc", model_path="model.onnx", metadata_path="model.json", preds_type="logits"
    )
    parent = _context(extractor=model)
    node_embeddings(parent, lambda: _ROWS)
    assert parent.embedded == {}
    own = np.zeros((2, 2))
    assert node_embeddings(_context(Subset(parent, (0, 1)), extractor=model), lambda: own).tolist() == own.tolist()


def _extractions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The size of each Dataset embeddings are extracted for, in order."""
    sizes: list[int] = []
    original = cache_module.get_or_compute_embeddings

    def counting(dataset: Any, *args: Any, **kwargs: Any) -> Any:
        sizes.append(len(dataset))
        return original(dataset, *args, **kwargs)

    monkeypatch.setattr(cache_module, "get_or_compute_embeddings", counting)
    return sizes


def _chain(steps: list[dict[str, Any]], evaluators: list[dict[str, Any]]) -> ChainResult:
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        evaluators=evaluators,
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"], "extractor": "flat"}],
        datasets={"src": ToyImages(count=12)},
        extractor=True,
    )
    result = run_chain_task(config)
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


_COVERAGE = [{"name": "cov", "type": "coverage", "num_observations": 3}]
_PART = {
    "name": "part",
    "transform": "view",
    "input": "data",
    "operations": [{"type": "Indices", "params": {"indices": [0, 2, 4, 6, 8, 10, 1, 1]}}],
}
_WHOLE = {"name": "whole", "evaluator": "cov", "input": "data"}
_SUB = {"name": "sub", "evaluator": "cov", "input": "part"}


def test_a_view_of_an_embedded_node_extracts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes = _extractions(monkeypatch)
    _chain([_WHOLE, _PART, _SUB], _COVERAGE)
    assert sizes == [12]


def test_a_view_embedded_before_its_parent_extracts_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes = _extractions(monkeypatch)
    _chain([_PART, _SUB, _WHOLE], _COVERAGE)
    assert sizes == [8, 12]


def test_a_view_that_changes_pixels_extracts_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes = _extractions(monkeypatch)
    resized = {
        "name": "part",
        "transform": "view",
        "input": "data",
        "operations": [{"type": "Resize", "params": {"size": 8}}],
    }
    _chain([_WHOLE, resized, _SUB], _COVERAGE)
    assert sizes == [12, 12]


def test_clusters_slice_too(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes = _extractions(monkeypatch)
    whole = {"name": "whole", "evaluator": "clustered", "input": "data"}
    sub = {"name": "sub", "evaluator": "clustered", "input": "part"}
    _chain([whole, _PART, sub], [{"name": "clustered", "type": "outliers", "cluster_threshold": 2.0}])
    assert sizes == [12]


def test_a_sliced_node_reads_the_rows_it_would_extract() -> None:
    """A view over a view, so the indices into the parent differ from those into the root."""
    again = {
        "name": "again",
        "transform": "view",
        "input": "part",
        "operations": [{"type": "Indices", "params": {"indices": [5, 0, 3, 7, 2, 6]}}],
    }
    sub = {"name": "sub", "evaluator": "cov", "input": "again"}
    sliced = _chain([_WHOLE, _PART, again, sub], _COVERAGE).steps["sub"].output
    DatasetCache.clear_instances()
    extracted = _chain([_PART, again, sub], _COVERAGE).steps["sub"].output
    assert sliced.critical_value_radii.tolist() == extracted.critical_value_radii.tolist()
    assert sliced.uncovered_indices.tolist() == extracted.uncovered_indices.tolist()
