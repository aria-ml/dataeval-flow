"""`label-reconciliation`: which class names resolve to one ontology concept (coverage spec §3.3)."""

from dataeval_flow import run
from dataeval_flow.evaluators.scope import LabelReconciliationConfig, LabelReconciliationOutput
from tests.evaluator_toys import ToyImages


def test_names_the_ontology_knows_conform() -> None:
    result = run(LabelReconciliationConfig(ontology={"x": {"a": None, "b": None}}), ToyImages(count=10))
    assert result.success, result.errors
    assert isinstance(result.output, LabelReconciliationOutput)
    data = result.output.data()
    assert data["conforms"] is True
    assert sorted(data["matched"]) == ["a", "b"]
    assert (data["unmatched"], data["ambiguous"]) == ([], {})


def test_a_name_the_ontology_lacks_is_unmatched() -> None:
    data = run(LabelReconciliationConfig(ontology={"x": {"a": None}}), ToyImages(count=10)).output.data()
    assert data["conforms"] is False
    assert data["unmatched"] == ["b"]


def test_a_name_two_concepts_share_is_ambiguous_with_their_ids() -> None:
    from dataeval_flow import run_tasks
    from tests.chain_toys import chain_pipeline

    concepts = [
        {"id": "x", "label": "x"},
        {"id": "a1", "label": "a", "parents": ["x"]},
        {"id": "a2", "label": "a", "parents": ["x"]},
        {"id": "b", "label": "b", "parents": ["x"]},
    ]
    config = chain_pipeline(
        evaluators=[{"name": "rec", "type": "label-reconciliation", "ontology": "vocab"}],
        tasks=[{"name": "t", "workflow": "rec", "kind": "evaluator", "sources": ["src"]}],
        extra={"ontologies": [{"name": "vocab", "concepts": concepts}]},
    )
    result = run_tasks(config)["t"]
    assert result.success, result.errors
    assert sorted(result.output.data()["ambiguous"]["a"]) == ["a1", "a2"]


def test_an_ontology_that_does_not_load_fails_the_run() -> None:
    result = run(LabelReconciliationConfig(ontology="missing/vocab.ttl"), ToyImages(count=10))
    assert not result.success
    assert any("could not read ontology file 'missing/vocab.ttl'" in error for error in result.errors), result.errors


def test_its_reads_note_no_factor() -> None:
    from dataeval_flow.evaluators import get_evaluator

    assert get_evaluator("label-reconciliation").reads_factors is False


def test_its_section_counts_the_names() -> None:
    from tests.evaluator_toys import output_json

    result = run(LabelReconciliationConfig(ontology={"x": {"a": None}}), ToyImages(count=10))
    section = result._section(output_json(result), ["src"], detailed=False)
    assert section is not None
    (fields,) = section
    assert dict(fields.items) == {"Matched": 1, "Unmatched": 1, "Ambiguous": 0}  # type: ignore[attr-defined]
