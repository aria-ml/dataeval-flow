"""`label-space`: a dataset's labels judged against a declared ontology, as chained steps (coverage spec §3)."""

import pytest
from pydantic import ValidationError

from dataeval_flow import MatrixResult, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.label_space import LabelSpaceConfig, LabelSpaceWorkflow
from tests.chain_toys import ToyDetections, chain_pipeline
from tests.evaluator_toys import ToyImages

_ONTOLOGY = {"x": {"a": None, "b": None, "c": None}}
_TITLES = ["Leaf Coverage", "Label Conformance", "Label Mergeability", "Ontology Structure"]


def _run(entry: dict, datasets: dict | None = None, **task: object) -> ChainResult:
    DatasetCache.clear_instances()
    config = chain_pipeline(
        workflows=[{"name": "w", "type": "label-space", **entry}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"], **task}],
        datasets=datasets or {"src": ToyImages(count=20)},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


def test_its_chain_is_four_evaluators_each_judged() -> None:
    chain = LabelSpaceWorkflow.chain(LabelSpaceConfig(name="w", ontology=_ONTOLOGY))
    names = [step["name"] for step in chain.steps]  # type: ignore[index]
    assert names == [
        "representation",
        "leaf-coverage",
        "label-reconciliation",
        "label-conformance",
        "label-alignment",
        "label-mergeability",
        "ontology-validation",
        "ontology-structure",
    ]
    assert all(entry.ontology == _ONTOLOGY for entry in chain.evaluators)  # type: ignore[attr-defined]


def test_it_makes_legacy_s_four_findings_in_order_and_stamps_the_digest() -> None:
    result = _run({"ontology": _ONTOLOGY})
    assert result.success, result.errors
    assert [finding.title for finding in result.findings] == _TITLES
    assert result.metadata.label_space_digest == result.steps["label-alignment"].output.alignment.label_space_digest
    assert result.metadata.metadata_binning is None


def test_without_an_ontology_it_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match="judges labels against an ontology: name one with `ontology:`"):
        LabelSpaceConfig(name="w")


def test_an_ontology_that_does_not_load_fails_the_task_with_the_loader_s_message() -> None:
    result = _run({"ontology": "missing/vocab.ttl"})
    assert not result.success
    assert any("missing/vocab.ttl" in error for error in result.errors), result.errors


def test_it_runs_on_detection_data_counting_box_labels() -> None:
    boxes = ToyDetections([[0], [1, 0], [0], [1], [0, 1], [1]], {0: "car", 1: "van", 2: "bus"})
    result = _run({"ontology": {"x": {"car": None, "van": None, "bus": None}}}, {"src": boxes})
    assert result.success, result.errors
    assert [finding.title for finding in result.findings] == _TITLES
    # Eight boxes over three leaves: each wants three, and `bus` has none.
    (row,) = result.steps["representation"].output.data().to_dicts()
    assert (row["concept"], row["count"], row["target"], row["deficit"]) == ("bus", 0, 3, 3)


def test_thresholds_are_keyed_by_check_type_and_reach_the_checks() -> None:
    result = _run({"ontology": _ONTOLOGY, "checks": {"leaf-coverage": {"coverage": None, "empty_branches": None}}})
    leaf = next(finding for finding in result.findings if finding.title == "Leaf Coverage")
    assert leaf.severity == "info"  # `c` has no examples, but neither criterion judges


def test_a_matrix_varies_a_hyphenated_threshold() -> None:
    DatasetCache.clear_instances()
    config = chain_pipeline(
        workflows=[
            {
                "name": "w",
                "type": "label-space",
                "ontology": _ONTOLOGY,
                "checks": {"leaf-coverage": {"empty_branches": None}},
            }
        ],
        tasks=[
            {
                "name": "t",
                "workflow": "w",
                "sources": ["src"],
                "matrix": {"checks.leaf-coverage.coverage": [0.1, 0.9]},
            }
        ],
        datasets={"src": ToyImages(count=20)},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, MatrixResult)
    severities = [run.result.findings[0].severity for run in result.runs]  # type: ignore[union-attr]
    assert severities == ["info", "warning"]  # `c` has no examples: 2 of 3 leaves is over 0.1 and under 0.9
