"""Every field naming another pool's entry carries `x-flow-ref`, so an editor finds each reference to rename (builder
spec §4.3)."""

from collections.abc import Iterator
from typing import Any

from dataeval_flow.config import PipelineConfig

_POOL_NAMED = {
    "dataset", "merge", "view", "preprocessor", "workflow", "evaluator", "sources", "extractor", "metadata", "stats",
    "ontology", "source",
}  # fmt: skip
# Text fields named like a pool that hold something else, with what they hold.
_NOT_REFERENCES = {
    ("DemoDatasetConfig", "dataset"): "a built-in tutorial dataset's name",
    ("AggregatorConfig", "source"): "a factor level whose rows are rolled up",
    ("OntologyConfig", "source"): "a path under the data root to a serialized RDF artifact",
}


def _holds_text(body: dict[str, Any]) -> bool:
    if body.get("type") == "string":
        return True
    if body.get("type") == "array":
        return _holds_text(body.get("items", {}))
    return any(_holds_text(option) for option in body.get("anyOf", []))


def _fields(schema: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for model, definition in {**schema.get("$defs", {}), "PipelineConfig": schema}.items():
        for name, body in definition.get("properties", {}).items():
            yield model, name, body


def test_every_text_field_named_for_a_pool_marks_its_references() -> None:
    unmarked = sorted(
        (model, name)
        for model, name, body in _fields(PipelineConfig.model_json_schema())
        if name in _POOL_NAMED
        and _holds_text(body)
        and "x-flow-ref" not in body
        and (model, name) not in _NOT_REFERENCES
    )
    assert unmarked == []


def test_each_reference_names_a_pool_of_the_pipeline() -> None:
    pools = set(PipelineConfig.model_fields)
    refs = [body["x-flow-ref"] for _, _, body in _fields(PipelineConfig.model_json_schema()) if "x-flow-ref" in body]
    assert len(refs) >= 60
    for ref in refs:
        assert ref["pool"] in pools, ref


def test_a_reference_keeps_its_mark_through_the_step_union() -> None:
    schema = PipelineConfig.model_json_schema()["$defs"]
    assert schema["EvaluatorStep"]["properties"]["evaluator"]["x-flow-ref"] == {"pool": "evaluators"}
    assert schema["EvaluatorStep"]["properties"]["extractor"]["x-flow-ref"] == {"pool": "extractors"}
    assert schema["WorkflowStep"]["properties"]["workflow"]["x-flow-ref"] == {"pool": "workflows"}
    assert schema["TaskConfig"]["properties"]["workflow"]["x-flow-ref"] == {"pool": "workflows"}
    assert schema["TaskConfig"]["properties"]["evaluator"]["x-flow-ref"] == {"pool": "evaluators"}
    assert schema["ExportConfig"]["properties"]["ontology"]["x-flow-ref"] == {"pool": "ontologies", "or_path": True}
