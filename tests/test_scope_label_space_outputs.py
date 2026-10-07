"""What `representation` and `label-alignment` record for `taxonomy`'s checks (coverage spec §3.3)."""

from typing import Any

from dataeval_flow import run
from dataeval_flow._blocks import Fields
from dataeval_flow.evaluators.scope import LabelAlignmentConfig, RepresentationConfig
from tests.evaluator_toys import ToyImages, output_json


def _fields(blocks: Any) -> dict[str, Any]:
    (fields,) = [block for block in blocks if isinstance(block, Fields)]
    return dict(fields.items)


def test_representation_records_the_expected_names_it_ignored() -> None:
    config = RepresentationConfig(ontology={"x": {"a": None, "b": None}}, expected={"a": 0.5, "nope": 0.1})
    result = run(config, ToyImages(count=20))
    assert result.success, result.errors
    assert result.output.ignored_expected == ["nope"]
    assert output_json(result)["extras"]["ignored_expected"] == ["nope"]


def test_representation_says_how_its_ontology_was_named() -> None:
    declared = run(RepresentationConfig(ontology={"x": {"a": None, "b": None}}), ToyImages(count=20))
    synthesized = run(RepresentationConfig(), ToyImages(count=20))
    assert declared.output.ontology_source == "inline"
    assert synthesized.output.ontology_source == "index2label"
    assert "ontology_source" not in output_json(declared).get("extras", {})


def test_representation_s_section_summarizes_rather_than_listing_the_worklist() -> None:
    result = run(RepresentationConfig(ontology={"x": {"a": None, "b": None, "c": None}}), ToyImages(count=20))
    section = result._section(output_json(result), ["src"], detailed=False)
    assert section is not None
    fields = _fields(section)
    assert len(result.output.data()) == 1
    assert fields == {"Leaf coverage": "66.7%", "Total deficit": 7, "Concepts short": 1}


def test_label_alignment_s_section_summarizes_rather_than_listing_correspondences() -> None:
    result = run(LabelAlignmentConfig(ontology={"a": None, "b": None}), ToyImages(count=20))
    section = result._section(output_json(result), ["src"], detailed=False)
    assert section is not None
    fields = _fields(section)
    assert fields == {"Mergeability": "lossless", "Correspondences": 2, "Dropped": 0, "Not covered": 0}


def test_a_detailed_representation_section_adds_the_output_blocks() -> None:
    result = run(RepresentationConfig(ontology={"x": {"a": None, "b": None, "c": None}}), ToyImages(count=20))
    short = result._section(output_json(result), ["src"], detailed=False)
    detailed = result._section(output_json(result), ["src"], detailed=True)
    assert short is not None
    assert detailed is not None
    assert detailed[: len(short)] == short
    assert any(getattr(block, "title", None) == "Output" for block in detailed[len(short) :])


def test_a_detailed_label_alignment_section_adds_the_output_blocks() -> None:
    result = run(LabelAlignmentConfig(ontology={"a": None, "b": None}), ToyImages(count=20))
    short = result._section(output_json(result), ["src"], detailed=False)
    detailed = result._section(output_json(result), ["src"], detailed=True)
    assert short is not None
    assert detailed is not None
    assert detailed[: len(short)] == short
    assert any(getattr(block, "title", None) == "Output" for block in detailed[len(short) :])
