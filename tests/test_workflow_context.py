"""A workflow reads its inputs through public, cached, policy-aware methods on its context."""

import uuid

import pytest
from dataeval.flags import ImageStats

from dataeval_flow._cache import DatasetCache
from dataeval_flow._stats import ResolvedStatsPolicy
from dataeval_flow.config.extractors import FlattenExtractorConfig
from dataeval_flow.workflows import DatasetContext, WorkflowContext
from tests.evaluator_toys import ToyImages


def _context(*, extractor: bool = True, stats_policy: ResolvedStatsPolicy | None = None) -> WorkflowContext:
    toy = ToyImages()
    cache = DatasetCache.get_or_create(cache_dir=None, name="toy", cache_key=f"toy-context-{uuid.uuid4()}")
    # The installed DataEval requires a resolved batch_size before it will build Embeddings;
    # a flatten extractor carries none of its own, so this passes one directly (see
    # tests/test_quality_evaluators.py's `_context` for the same idiom).
    dc = DatasetContext(
        name="src",
        dataset=toy,
        cache=cache,
        extractor=FlattenExtractorConfig(name="flat") if extractor else None,
        batch_size=8,
    )
    return WorkflowContext(dataset_contexts={"src": dc}, stats_policy=stats_policy)


def test_sources_are_the_task_order() -> None:
    assert _context().sources == ["src"]


def test_stats_are_computed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataeval_flow._cache as cache_module

    calls: list[int] = []
    original = cache_module._do_compute_stats
    monkeypatch.setattr(cache_module, "_do_compute_stats", lambda *a, **k: calls.append(1) or original(*a, **k))
    context = _context()
    context.stats("src")
    context.stats("src")
    assert len(calls) == 1


def test_stats_reads_a_declared_policys_measure_without_flags() -> None:
    context = _context(stats_policy=ResolvedStatsPolicy.of_flags(ImageStats.DIMENSION))
    result = context.stats("src")
    assert "width" in result["stats"]
    assert "brightness" not in result["stats"]


def test_stats_checks_explicit_flags_against_a_declared_policy() -> None:
    context = _context(stats_policy=ResolvedStatsPolicy.of_flags(ImageStats.DIMENSION))
    with pytest.raises(ValueError, match="flags"):
        context.stats("src", flags=ImageStats.VISUAL_BRIGHTNESS)


def test_embeddings_have_one_row_per_item() -> None:
    assert _context().embeddings("src").shape[0] == len(ToyImages())


def test_embeddings_are_computed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataeval_flow._cache as cache_module

    calls: list[int] = []
    original = cache_module._do_compute_embeddings
    monkeypatch.setattr(cache_module, "_do_compute_embeddings", lambda *a, **k: calls.append(1) or original(*a, **k))
    context = _context()
    context.embeddings("src")
    context.embeddings("src")
    assert len(calls) == 1


def test_clusters_cover_every_item() -> None:
    assert len(_context().clusters("src")["clusters"]) == len(ToyImages())


def test_clusters_reuse_the_cached_embeddings_and_compute_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataeval_flow._cache as cache_module

    embedding_calls: list[int] = []
    cluster_calls: list[int] = []
    original_embeddings = cache_module._do_compute_embeddings
    original_clusters = cache_module._do_compute_clusters
    monkeypatch.setattr(
        cache_module,
        "_do_compute_embeddings",
        lambda *a, **k: embedding_calls.append(1) or original_embeddings(*a, **k),
    )
    monkeypatch.setattr(
        cache_module,
        "_do_compute_clusters",
        lambda *a, **k: cluster_calls.append(1) or original_clusters(*a, **k),
    )
    context = _context()
    context.clusters("src")
    context.clusters("src")
    assert len(embedding_calls) == 1
    assert len(cluster_calls) == 1


def test_labels_match_the_dataset() -> None:
    assert list(_context().labels("src")) == [i % 2 for i in range(len(ToyImages()))]


def test_embeddings_without_an_extractor_explain_themselves() -> None:
    with pytest.raises(ValueError, match="extractor"):
        _context(extractor=False).embeddings("src")


def test_an_unknown_source_lists_the_real_ones() -> None:
    with pytest.raises(KeyError, match="src"):
        _context().dataset("nope")
