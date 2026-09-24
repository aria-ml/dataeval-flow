"""`quality.duplicates` and `quality.outliers` against real DataEval, on a dataset with planted answers."""

import pytest

from dataeval_flow.config import FlattenExtractorConfig
from dataeval_flow.evaluator import get_evaluator, list_evaluators
from dataeval_flow.evaluator.inputs import Inputs
from dataeval_flow.evaluators.quality.evaluator import find_duplicates, find_outliers
from dataeval_flow.evaluators.quality.params import DuplicatesParameters, OutliersParameters
from dataeval_flow.workflow import DatasetContext, WorkflowContext
from tests.evaluator_toys import ToyImages, exact_groups


def _context(*names: str, count: int = 12, extractor: bool = False) -> WorkflowContext:
    toy = ToyImages(count=count)
    flat = FlattenExtractorConfig(name="flat") if extractor else None
    # The installed DataEval requires a resolved batch_size before it will build Embeddings;
    # a flatten extractor carries none of its own, so cluster-mode contexts pass one directly.
    return WorkflowContext(
        dataset_contexts={name: DatasetContext(name=name, dataset=toy, extractor=flat, batch_size=8) for name in names}
    )


class TestDuplicates:
    def test_the_planted_copy_is_an_exact_group(self):
        result = get_evaluator("quality.duplicates").execute(_context("src"), DuplicatesParameters())
        assert result.success, result.errors
        assert exact_groups(result.output["rows"]) == {(0, 5)}

    def test_across_sources_every_group_says_where_it_lives(self):
        result = get_evaluator("quality.duplicates").execute(_context("a", "b"), DuplicatesParameters())
        assert result.success, result.errors
        assert all("dataset_indices" in row for row in result.output["rows"])

    def test_per_target_reaches_dataeval(self):
        result = get_evaluator("quality.duplicates").execute(_context("src"), DuplicatesParameters(per_target=True))
        assert result.success, result.errors

    def test_cluster_mode_keeps_the_hash_groups(self):
        params = DuplicatesParameters(cluster_sensitivity=1.0, cluster_algorithm="kmeans", n_clusters=2)
        result = get_evaluator("quality.duplicates").execute(_context("src", count=40, extractor=True), params)
        assert result.success, result.errors
        assert (0, 5) in exact_groups(result.output["rows"])


class TestOutliers:
    def test_the_white_image_is_flagged(self):
        result = get_evaluator("quality.outliers").execute(_context("src"), OutliersParameters())
        assert result.success, result.errors
        assert 7 in {row["item_index"] for row in result.output["rows"]}

    def test_a_named_threshold_reaches_dataeval(self):
        params = OutliersParameters(flags=["visual"], outlier_threshold=("zscore", 3.0))
        result = get_evaluator("quality.outliers").execute(_context("src"), params)
        assert result.success, result.errors
        assert {row["metric_name"] for row in result.output["rows"]} <= {
            "brightness",
            "contrast",
            "darkness",
            "missing",
            "sharpness",
            "zeros",
        }

    def test_cluster_mode_runs(self):
        params = OutliersParameters(cluster_threshold=2.0, cluster_algorithm="kmeans", n_clusters=2)
        result = get_evaluator("quality.outliers").execute(_context("src", count=40, extractor=True), params)
        assert result.success, result.errors


def test_ensure_initialized_never_partially_fills_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A build that fails partway through must not leave `_EVALUATORS` half filled — that
    would look "initialized" (truthy) to another thread's `if not _EVALUATORS:` check."""
    import dataeval_flow.evaluator as evaluator_module
    from dataeval_flow.evaluators.quality.evaluator import DuplicatesEvaluator

    class _Boom:
        def __init__(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(evaluator_module, "_EVALUATORS", {})
    monkeypatch.setattr(evaluator_module, "_registered", lambda: [DuplicatesEvaluator, _Boom])

    with pytest.raises(RuntimeError, match="boom"):
        evaluator_module._ensure_initialized()

    assert evaluator_module._EVALUATORS == {}


def test_both_are_registered_and_listed():
    listed = {entry["name"]: entry for entry in list_evaluators()}
    assert set(listed) >= {"quality.duplicates", "quality.outliers"}
    assert listed["quality.duplicates"]["consumes"] == "stats, clusters (optional)"
    assert listed["quality.duplicates"]["sources"] == "1+"


@pytest.mark.parametrize("name", ["quality.duplicates", "quality.outliers"])
def test_results_record_dataeval(name: str):
    params = DuplicatesParameters() if name == "quality.duplicates" else OutliersParameters()
    result = get_evaluator(name).execute(_context("src"), params)
    assert result.metadata.evaluator == name
    assert result.metadata.dataeval.version


class TestMissingInputsAreNamed:
    """A producer that did not run names the source and the kind it skipped, not a stack trace."""

    def test_find_duplicates_names_the_source_missing_stats(self):
        with pytest.raises(ValueError, match="Source 'src' arrived without stats"):
            find_duplicates(DuplicatesParameters(), [Inputs(source="src")])

    def test_find_outliers_names_the_source_missing_its_stats_policy(self):
        with pytest.raises(ValueError, match="Source 'src' arrived without a stats policy"):
            find_outliers(OutliersParameters(), [Inputs(source="src")])
