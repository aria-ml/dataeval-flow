"""`ontology-validation`: an ontology's structural and naming facts (coverage spec §3.3)."""

from dataeval_flow import run, run_tasks
from dataeval_flow._blocks import Fields
from dataeval_flow.evaluators.scope import OntologyValidationConfig, OntologyValidationOutput, OntologyValidationResult
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages, output_json


def test_it_counts_concepts_leaves_and_depth() -> None:
    result = run(OntologyValidationConfig(ontology={"x": {"a": None, "b": None}}), ToyImages(count=10))
    assert result.success, result.errors
    assert isinstance(result.output, OntologyValidationOutput)
    data = result.output.data()
    assert (data["concept_count"], data["leaf_count"], data["max_depth"]) == (3, 2, 1)
    assert data["label_collisions"] == {}


def test_a_label_two_concepts_share_is_a_collision() -> None:
    concepts = [
        {"id": "x", "label": "x"},
        {"id": "a1", "label": "a", "parents": ["x"]},
        {"id": "a2", "label": "a", "parents": ["x"]},
    ]
    config = chain_pipeline(
        evaluators=[{"name": "lint", "type": "ontology-validation", "ontology": "vocab"}],
        tasks=[{"name": "t", "workflow": "lint", "kind": "evaluator", "sources": ["src"]}],
        extra={"ontologies": [{"name": "vocab", "concepts": concepts}]},
    )
    result = run_tasks(config)["t"]
    assert result.success, result.errors
    assert sorted(result.output.data()["label_collisions"]["a"]) == ["a1", "a2"]


def test_a_label_pattern_names_the_labels_that_break_it() -> None:
    config = OntologyValidationConfig(ontology={"x": {"a": None, "B": None}}, label_pattern="^[a-z]+$")
    data = run(config, ToyImages(count=10)).output.data()
    assert data["nonconforming_labels"] == {"B": "B"}  # concept id to its label


def test_its_json_has_no_tuples() -> None:
    concepts = [
        {"id": "vehicle", "label": "vehicle"},
        {"id": "land", "label": "land", "parents": ["vehicle"]},
        {"id": "car", "label": "car", "parents": ["land", "vehicle"]},
    ]
    config = chain_pipeline(
        evaluators=[{"name": "lint", "type": "ontology-validation", "ontology": "vocab"}],
        tasks=[{"name": "t", "workflow": "lint", "kind": "evaluator", "sources": ["src"]}],
        extra={"ontologies": [{"name": "vocab", "concepts": concepts}]},
    )
    result = run_tasks(config)["t"]
    assert result.success, result.errors
    assert output_json(result)["data"]["redundant_edges"] == [["vehicle", "car"]]


def test_its_section_summarizes_the_ontology() -> None:
    result = run(OntologyValidationConfig(ontology={"x": {"a": None, "b": None}}), ToyImages(count=10))
    assert isinstance(result, OntologyValidationResult)
    (block,) = result._section(output_json(result), ["src"], detailed=False) or []
    assert isinstance(block, Fields)
    assert dict(block.items) == {
        "Concepts": 3,
        "Leaves": 2,
        "Max Depth": 1,
        "Roots": 1,
        "Label Collisions": 0,
    }
