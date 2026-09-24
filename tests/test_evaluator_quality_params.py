"""Quality evaluator parameters: DataEval's names, DataEval's defaults, and DataEval's validation."""

import pytest
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates, Outliers
from pydantic import ValidationError

from dataeval_flow.config import DuplicatesEvaluatorConfig, OutliersEvaluatorConfig, PipelineConfig
from dataeval_flow.evaluator import InputKind
from dataeval_flow.evaluators.quality.params import DuplicatesParameters, OutliersParameters


class TestDuplicatesParameters:
    def test_unset_fields_are_left_to_dataeval(self):
        assert DuplicatesParameters().constructor_kwargs() == {}
        assert DuplicatesParameters().call_kwargs() == {}

    def test_flags_map_to_image_stats(self):
        params = DuplicatesParameters(flags=["hash_basic", "hash_d4"])
        assert params.constructor_kwargs()["flags"] == ImageStats.HASH_DUPLICATES_BASIC | ImageStats.HASH_DUPLICATES_D4

    def test_unset_flags_request_what_dataeval_would_use(self):
        assert DuplicatesParameters().stats_flags() == Duplicates.Config().flags

    def test_stats_are_requested_as_the_duplicate_consumer(self):
        assert DuplicatesParameters().stats_request()["duplicate_flags"] == Duplicates.Config().flags

    def test_cluster_sensitivity_switches_on_clusters(self):
        assert DuplicatesParameters().wanted_kinds() == {InputKind.STATS}
        assert DuplicatesParameters(cluster_sensitivity=1.0).wanted_kinds() == {InputKind.STATS, InputKind.CLUSTERS}

    def test_cluster_mode_reads_one_source(self):
        params = DuplicatesParameters(cluster_sensitivity=1.0)
        assert params.source_problem(1) is None
        problem = params.source_problem(2)
        assert problem is not None
        assert "exactly one source in cluster mode" in problem

    def test_hash_mode_reads_any_number_of_sources(self):
        assert DuplicatesParameters().source_problem(3) is None

    def test_call_kwargs_carry_only_set_values(self):
        assert DuplicatesParameters(per_target=True).call_kwargs() == {"per_target": True}

    def test_empty_flags_are_refused(self):
        with pytest.raises(ValidationError):
            DuplicatesParameters(flags=[])

    def test_hash_radius_reaches_dataeval(self):
        params = DuplicatesParameters(hash_radius=5)
        assert params.constructor_kwargs()["hash_radius"] == 5

    def test_unset_hash_radius_is_left_to_dataeval(self):
        assert "hash_radius" not in DuplicatesParameters().constructor_kwargs()

    def test_a_negative_hash_radius_is_refused_at_load(self):
        with pytest.raises(ValidationError):
            DuplicatesParameters(hash_radius=-1)

    def test_a_misspelled_argument_fails_the_load(self):
        """Review Focus 1: a typo must not silently run DataEval's default."""
        with pytest.raises(ValidationError, match="merge_near_duplicate"):
            DuplicatesEvaluatorConfig.model_validate(
                {"name": "d", "type": "quality.duplicates", "merge_near_duplicate": True}
            )


class TestOutliersParameters:
    @pytest.mark.parametrize(
        "threshold",
        ["zscore", ["zscore", 3.0], ["iqr", [1.0, 3.0]], {"brightness": ["modzscore", 3.5]}],
    )
    def test_dataeval_threshold_spellings_load(self, threshold: object):
        params = OutliersParameters.model_validate({"outlier_threshold": threshold})
        assert "outlier_threshold" in params.constructor_kwargs()

    def test_dataeval_refuses_an_unknown_method_at_load(self):
        with pytest.raises(ValidationError, match="DataEval rejected these parameters"):
            OutliersParameters(outlier_threshold="bogus")

    def test_unset_flags_request_what_dataeval_would_use(self):
        assert OutliersParameters().stats_flags() == Outliers.Config().flags

    def test_flags_map_to_image_stats(self):
        assert OutliersParameters(flags=["pixel"]).stats_flags() == ImageStats.PIXEL

    def test_stats_are_requested_as_the_outlier_consumer(self):
        assert set(OutliersParameters().stats_request()) == {"outlier_flags"}

    def test_cluster_threshold_switches_on_clusters(self):
        assert InputKind.CLUSTERS in OutliersParameters(cluster_threshold=2.0).wanted_kinds()


class TestPipelineConfigEvaluators:
    def test_entries_parse_by_type(self):
        config = PipelineConfig.model_validate(
            {
                "evaluators": [
                    {"name": "d", "type": "quality.duplicates"},
                    {"name": "o", "type": "quality.outliers", "outlier_threshold": "zscore"},
                ]
            }
        )
        assert config.evaluators is not None
        assert [type(e) for e in config.evaluators] == [DuplicatesEvaluatorConfig, OutliersEvaluatorConfig]

    def test_an_unknown_type_is_refused(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"evaluators": [{"name": "x", "type": "quality.nope"}]})

    def test_names_are_unique(self):
        with pytest.raises(ValidationError, match="Duplicate name 'd' in evaluators"):
            PipelineConfig.model_validate(
                {"evaluators": [{"name": "d", "type": "quality.duplicates"}, {"name": "d", "type": "quality.outliers"}]}
            )
