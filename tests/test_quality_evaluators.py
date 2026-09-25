"""`quality.duplicates` and `quality.outliers` against real DataEval, on a dataset with planted answers."""

import pytest

from dataeval_flow.config.extractors import FlattenExtractorConfig
from dataeval_flow.evaluators import EvaluatorInputs, get_evaluator, list_evaluators
from dataeval_flow.evaluators._execute import execute
from dataeval_flow.evaluators.quality import DuplicatesConfig, OutliersConfig
from dataeval_flow.evaluators.quality._evaluator import find_duplicates, find_outliers
from dataeval_flow.workflows import DatasetContext, WorkflowContext
from tests.evaluator_toys import ToyImages, exact_groups, output_json


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
        result = execute(get_evaluator("quality.duplicates")(), _context("src"), DuplicatesConfig())
        assert result.success, result.errors
        assert exact_groups(output_json(result)["rows"]) == {(0, 5)}

    def test_across_sources_every_group_says_where_it_lives(self):
        result = execute(get_evaluator("quality.duplicates")(), _context("a", "b"), DuplicatesConfig())
        assert result.success, result.errors
        assert all("dataset_indices" in row for row in output_json(result)["rows"])

    def test_per_target_reaches_dataeval(self):
        result = execute(get_evaluator("quality.duplicates")(), _context("src"), DuplicatesConfig(per_target=True))
        assert result.success, result.errors

    def test_cluster_mode_keeps_the_hash_groups(self):
        params = DuplicatesConfig(cluster_sensitivity=1.0, cluster_algorithm="kmeans", n_clusters=2)
        result = execute(get_evaluator("quality.duplicates")(), _context("src", count=40, extractor=True), params)
        assert result.success, result.errors
        assert (0, 5) in exact_groups(output_json(result)["rows"])


class TestOutliers:
    def test_the_white_image_is_flagged(self):
        result = execute(get_evaluator("quality.outliers")(), _context("src"), OutliersConfig())
        assert result.success, result.errors
        assert 7 in {row["item_index"] for row in output_json(result)["rows"]}

    def test_a_named_threshold_reaches_dataeval(self):
        params = OutliersConfig(flags=["visual"], outlier_threshold=("zscore", 3.0))
        result = execute(get_evaluator("quality.outliers")(), _context("src"), params)
        assert result.success, result.errors
        assert {row["metric_name"] for row in output_json(result)["rows"]} <= {
            "brightness",
            "contrast",
            "darkness",
            "missing",
            "sharpness",
            "zeros",
        }

    def test_cluster_mode_runs(self):
        params = OutliersConfig(cluster_threshold=2.0, cluster_algorithm="kmeans", n_clusters=2)
        result = execute(get_evaluator("quality.outliers")(), _context("src", count=40, extractor=True), params)
        assert result.success, result.errors


def test_both_are_registered_and_listed():
    from dataeval_flow.__main__ import _evaluator_entry

    listed = {cls.name: _evaluator_entry(cls) for cls in list_evaluators()}
    assert set(listed) >= {"quality.duplicates", "quality.outliers"}
    assert listed["quality.duplicates"]["consumes"] == "stats, clusters (optional)"
    assert listed["quality.duplicates"]["sources"] == "1+"


@pytest.mark.parametrize("name", ["quality.duplicates", "quality.outliers"])
def test_results_record_dataeval(name: str):
    params = DuplicatesConfig() if name == "quality.duplicates" else OutliersConfig()
    result = execute(get_evaluator(name)(), _context("src"), params)
    assert result.metadata.evaluator == name
    assert result.metadata.dataeval.version


class TestMissingInputsAreNamed:
    """A producer that did not run names the source and the kind it skipped, not a stack trace."""

    def test_find_duplicates_names_the_source_missing_stats(self):
        with pytest.raises(ValueError, match="Source 'src' arrived without stats"):
            find_duplicates(DuplicatesConfig(), [EvaluatorInputs(source="src")])

    def test_find_outliers_names_the_source_missing_its_stats_policy(self):
        with pytest.raises(ValueError, match="Source 'src' arrived without a stats policy"):
            find_outliers(OutliersConfig(), [EvaluatorInputs(source="src")])
