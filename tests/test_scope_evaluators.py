"""The scope evaluators against real DataEval, on toy images whose labels and embeddings are known."""

import logging
from typing import Any, TypedDict

import numpy as np
import pytest

from dataeval_flow import run
from dataeval_flow.evaluators import EvaluatorInputs
from dataeval_flow.evaluators.scope import (
    CoverageConfig,
    CoverageResult,
    PrioritizeConfig,
    PrioritizeResult,
    RepresentationConfig,
    RepresentationResult,
)
from dataeval_flow.evaluators.scope._evaluator import PrioritizeEvaluator, usable_labels
from tests.evaluator_toys import FLAT, ToyImages, output_json


def _declaring(*names: str, **kwargs: Any) -> ToyImages:
    """Toy images whose `index2label` declares `names`; only the first two occur."""
    toy = ToyImages(**kwargs)
    toy.metadata["index2label"] = dict(enumerate(names))  # type: ignore[reportTypedDictNotRequiredAccess]
    return toy


class TestRepresentation:
    def test_a_declared_class_with_no_samples_is_on_the_worklist(self):
        result = run(RepresentationConfig(), _declaring("a", "b", "c"))
        assert isinstance(result, RepresentationResult)
        assert result.success, result.errors
        worklist = {row["concept"]: row for row in output_json(result)["rows"]}
        assert worklist["c"]["count"] == 0
        assert worklist["c"]["deficit"] > 0
        assert output_json(result)["extras"]["leaf_coverage"] == pytest.approx(2 / 3)

    def test_an_inline_ontology_is_counted_against(self):
        result = run(RepresentationConfig(ontology={"animal": ["a", "b", "c", "d"]}), _declaring("a", "b"))
        assert result.success, result.errors
        assert output_json(result)["extras"]["leaf_coverage"] == pytest.approx(0.5)

    def test_expected_shares_reach_dataeval(self):
        result = run(RepresentationConfig(expected={"a": 0.9}), _declaring("a", "b"))
        assert result.success, result.errors
        violations = output_json(result)["extras"]["violations"]["rows"]
        assert [row["concept"] for row in violations] == ["a"]

    def test_an_ontology_that_does_not_resolve_fails_the_run(self):
        result = run(RepresentationConfig(ontology="missing.ttl"), _declaring("a", "b"))
        assert not result.success
        assert "could not be resolved" in result.errors[0]

    def test_a_dataset_without_labels_fails_naming_its_source(self):
        result = run(RepresentationConfig(), ToyImages(labeled=False))
        assert not result.success
        assert "Source 'dataset' has no labels" in result.errors[0]


class _SmallCoverage(TypedDict):
    num_observations: int
    min_class_samples: int


_SMALL: _SmallCoverage = {"num_observations": 5, "min_class_samples": 5}


class TestCoverage:
    def test_each_class_gets_a_row_and_the_source_its_radius(self):
        result = run(CoverageConfig(**_SMALL), ToyImages(count=40), extractor=FLAT)
        assert isinstance(result, CoverageResult)
        assert result.success, result.errors
        assert {row["class"] for row in output_json(result)["rows"]} == {"a", "b"}
        extras = output_json(result)["extras"]
        assert set(extras) == {"uncovered_indices", "coverage_radius", "critical_value_radii", "uncovered_classes"}
        assert len(extras["critical_value_radii"]) == 40

    def test_coverage_names_each_uncovered_item_s_class(self):
        output = run(CoverageConfig(**_SMALL), ToyImages(count=40), extractor=FLAT).output
        assert len(output.uncovered_indices) > 0
        assert len(output.uncovered_classes) == len(output.uncovered_indices)
        assert set(output.uncovered_classes) <= {"a", "b"}

    def test_coverage_without_a_class_breakdown_names_none(self):
        output = run(CoverageConfig(**_SMALL), ToyImages(count=40, labeled=False), extractor=FLAT).output
        assert len(output.uncovered_classes) == len(output.uncovered_indices) > 0
        assert all(name is None for name in output.uncovered_classes)

    def test_without_labels_it_runs_as_one_class_and_warns(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            result = run(CoverageConfig(**_SMALL), ToyImages(count=40, labeled=False), extractor=FLAT)
        assert result.success, result.errors
        assert [(row["class"], row["count"]) for row in output_json(result)["rows"]] == [("0", 40)]
        assert "has no labels, so it runs without a class breakdown" in caplog.text

    def test_a_setting_reaches_dataeval(self):
        result = run(CoverageConfig(num_observations=5, min_class_samples=30), ToyImages(count=40), extractor=FLAT)
        assert result.success, result.errors
        assert not any(row["assessable"] for row in output_json(result)["rows"])


class TestUsableLabels:
    def test_one_label_per_item_is_used(self):
        source = EvaluatorInputs(source="s", labels=np.array([0, 1]), index2label={0: "a", 1: "b"})
        labels = usable_labels(source, 2, "coverage")
        assert labels is not None
        assert labels.tolist() == [0, 1]

    def test_labels_per_target_are_not_used(self, caplog: pytest.LogCaptureFixture):
        """A detection dataset's labels count targets, so they do not line up with one embedding per item."""
        source = EvaluatorInputs(source="s", labels=np.array([0, 1, 1]), index2label={})
        with caplog.at_level(logging.WARNING):
            assert usable_labels(source, 2, "coverage") is None
        assert "has 3 labels for 2 items, one per target rather than one per item" in caplog.text

    def test_a_source_without_labels_says_so(self, caplog: pytest.LogCaptureFixture):
        source = EvaluatorInputs(source="s", labels=np.array([], dtype=np.intp), index2label={})
        with caplog.at_level(logging.WARNING):
            assert usable_labels(source, 2, "prioritize") is None
        assert "prioritize: source 's' has no labels" in caplog.text


class TestPrioritize:
    def test_every_item_is_ranked_with_its_score(self):
        result = run(PrioritizeConfig(), ToyImages(count=40), extractor=FLAT)
        assert isinstance(result, PrioritizeResult)
        assert result.success, result.errors
        assert sorted(output_json(result)["data"]) == list(range(40))
        assert len(output_json(result)["extras"]["scores"]) == 40

    def test_a_second_source_is_the_reference(self):
        data = ToyImages(count=40, seed=1)
        alone = run(PrioritizeConfig(), data, extractor=FLAT)
        relative = run(PrioritizeConfig(), {"data": data, "reference": ToyImages(count=40)}, extractor=FLAT)
        assert relative.success, relative.errors
        assert relative.sources is not None
        assert list(relative.sources) == ["data", "reference"]
        assert sorted(output_json(relative)["data"]) == list(range(40))
        assert output_json(relative)["data"] != output_json(alone)["data"]

    def test_the_order_reaches_dataeval(self):
        easy = output_json(run(PrioritizeConfig(order="easy_first"), ToyImages(count=40), extractor=FLAT))["data"]
        hard = output_json(run(PrioritizeConfig(order="hard_first"), ToyImages(count=40), extractor=FLAT))["data"]
        assert hard == easy[::-1]


class TestPrioritizeEmptySources:
    def test_an_empty_dataset_ranks_to_an_empty_ranking(self):
        empty = EvaluatorInputs(source="pool", embeddings=np.empty((0, 4), dtype=np.float32))
        reference = EvaluatorInputs(source="ref", embeddings=np.random.default_rng(0).random((10, 4)))
        output = PrioritizeEvaluator().run(PrioritizeConfig(), [empty, reference])
        assert output.data().tolist() == []
        assert output.scores is not None
        assert output.scores.tolist() == []

    def test_an_empty_reference_is_refused_by_name(self):
        pool = EvaluatorInputs(source="pool", embeddings=np.random.default_rng(0).random((10, 4)), labels=None)
        empty = EvaluatorInputs(source="ref", embeddings=np.empty((0, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="`ref` has no items to rank against."):
            PrioritizeEvaluator().run(PrioritizeConfig(), [pool, empty])
