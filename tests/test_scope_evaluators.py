"""`scope.*` against real DataEval, on toy images whose labels and embeddings are known."""

from typing import Any

import pytest

from dataeval_flow import run
from dataeval_flow.evaluators.scope import RepresentationConfig, RepresentationResult
from tests.evaluator_toys import ToyImages, output_json


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
