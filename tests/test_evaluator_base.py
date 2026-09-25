"""The evaluator input vocabulary and the rules a task must meet to run an evaluator."""

from typing import ClassVar

import pytest
from pydantic import ValidationError

from dataeval_flow import InputKind, InputSpec, SourceCount
from dataeval_flow._kind import input_problem
from dataeval_flow.evaluators import EvaluatorConfig


class _StatsParams(EvaluatorConfig):
    type: str = "test.stats"
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.STATS}),
        optional=frozenset({InputKind.CLUSTERS}),
        sources=SourceCount.ONE_OR_MORE,
    )
    cluster: bool = False

    def wanted_kinds(self) -> frozenset[InputKind]:
        return self.inputs.required | (frozenset({InputKind.CLUSTERS}) if self.cluster else frozenset())


class _MetadataParams(EvaluatorConfig):
    type: str = "test.metadata"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)


class TestSourceCount:
    @pytest.mark.parametrize(
        ("rule", "count", "allowed"),
        [
            (SourceCount.ONE, 1, True),
            (SourceCount.ONE, 0, False),
            (SourceCount.ONE, 2, False),
            (SourceCount.ONE_OR_MORE, 1, True),
            (SourceCount.ONE_OR_MORE, 5, True),
            (SourceCount.ONE_OR_MORE, 0, False),
            (SourceCount.ONE_OR_TWO, 2, True),
            (SourceCount.ONE_OR_TWO, 3, False),
            (SourceCount.TWO, 2, True),
            (SourceCount.TWO, 1, False),
        ],
    )
    def test_allows(self, rule: SourceCount, count: int, allowed: bool):
        assert rule.allows(count) is allowed

    def test_every_rule_reads_as_a_phrase(self):
        assert SourceCount.ONE.phrase == "exactly one source"
        assert all(rule.phrase for rule in SourceCount)


class TestInputKind:
    def test_only_embedding_kinds_need_an_extractor(self):
        assert {kind for kind in InputKind if kind.needs_extractor} == {InputKind.CLUSTERS, InputKind.EMBEDDINGS}

    def test_values_are_the_names_proposed_to_dataeval(self):
        assert [kind.value for kind in InputKind] == ["stats", "clusters", "metadata", "labels", "embeddings"]


class TestInputSpec:
    def test_an_optional_embedding_kind_admits_an_extractor(self):
        assert _StatsParams.inputs.accepts_extractor

    def test_no_embedding_kind_refuses_one(self):
        assert not _MetadataParams.inputs.accepts_extractor

    def test_kinds_are_required_and_optional_together(self):
        assert _StatsParams.inputs.kinds == {InputKind.STATS, InputKind.CLUSTERS}


class TestConfigBase:
    def test_unknown_keys_are_rejected(self):
        with pytest.raises(ValidationError, match="clustr"):
            _StatsParams.model_validate({"clustr": True})

    def test_optional_kinds_follow_the_values(self):
        assert _StatsParams().wanted_kinds() == {InputKind.STATS}
        assert _StatsParams(cluster=True).wanted_kinds() == {InputKind.STATS, InputKind.CLUSTERS}

    def test_an_extractor_is_needed_only_for_a_wanted_embedding_kind(self):
        assert _StatsParams(cluster=True).requires_extractor()
        assert not _StatsParams().requires_extractor()

    def test_no_source_rule_beyond_the_spec_by_default(self):
        assert _StatsParams().check_inputs(3) is None


class TestInputProblem:
    def test_a_runnable_task_has_none(self):
        assert input_problem(_StatsParams(), source_count=2, has_extractor=False) is None

    def test_the_source_count_is_checked(self):
        problem = input_problem(_MetadataParams(), source_count=2, has_extractor=False)
        assert problem == "takes exactly one source, but the task names 2."

    def test_a_missing_extractor_names_the_kind_that_needs_it(self):
        problem = input_problem(_StatsParams(cluster=True), source_count=1, has_extractor=False)
        assert problem == "needs an extractor to produce clusters; name one with `extractor:`."

    def test_an_extractor_nothing_uses_is_refused(self):
        problem = input_problem(_MetadataParams(), source_count=1, has_extractor=True)
        assert problem == "does not use an extractor; remove `extractor:` from the task."

    def test_an_unused_optional_kind_still_admits_an_extractor(self):
        assert input_problem(_StatsParams(), source_count=1, has_extractor=True) is None

    def test_the_params_can_add_a_source_rule(self):
        class _OneWhenClustering(_StatsParams):
            def check_inputs(self, count: int) -> str | None:
                return "reads exactly one source in cluster mode." if self.cluster and count > 1 else None

        problem = input_problem(_OneWhenClustering(cluster=True), source_count=2, has_extractor=True)
        assert problem == "reads exactly one source in cluster mode."
