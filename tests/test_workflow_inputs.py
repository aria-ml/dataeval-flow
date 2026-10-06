"""Every workflow declares its inputs, and every task is checked against them when the config loads."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import InputKind, SourceCount
from dataeval_flow.config import PipelineConfig
from dataeval_flow.workflows import list_workflows

SPECS = {
    "audit": (SourceCount.ONE_OR_MORE, {InputKind.METADATA, InputKind.STATS}, {InputKind.EMBEDDINGS}),
    "data-bias": (SourceCount.ONE, {InputKind.METADATA}, set()),
    "data-cleaning": (SourceCount.ONE, {InputKind.STATS, InputKind.METADATA}, {InputKind.CLUSTERS}),
    "data-coverage": (SourceCount.ONE, {InputKind.LABELS}, {InputKind.EMBEDDINGS}),
    "data-prioritization": (SourceCount.TWO_OR_MORE, {InputKind.EMBEDDINGS}, set()),
    "data-splitting": (SourceCount.ONE, {InputKind.METADATA}, set()),
    "drift-monitoring": (SourceCount.TWO_OR_MORE, {InputKind.EMBEDDINGS}, {InputKind.LABELS}),
    "label-space": (SourceCount.ONE, {InputKind.LABELS}, set()),
    "metadata-triage": (SourceCount.ONE, {InputKind.METADATA}, set()),
    "ood-detection": (SourceCount.TWO_OR_MORE, {InputKind.EMBEDDINGS}, {InputKind.METADATA, InputKind.STATS}),
}


@pytest.mark.parametrize("workflow", list_workflows(), ids=lambda cls: cls.name)
def test_each_builtin_declares_its_inputs(workflow: Any) -> None:
    sources, required, optional = SPECS[workflow.name]
    spec = workflow.config_type.inputs
    assert (spec.sources, set(spec.required), set(spec.optional)) == (sources, required, optional)


def _pipeline(workflow: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    return {
        "datasets": [{"name": "ds", "format": "huggingface", "path": "./d", "task": "image_classification"}],
        "sources": [{"name": "a", "dataset": "ds"}, {"name": "b", "dataset": "ds"}],
        "extractors": [{"name": "flat", "model": "flatten"}],
        "workflows": [{"name": "w", **workflow}],
        "tasks": [{"name": "t", "workflow": "w", **task}],
    }


def test_drift_with_one_source_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match="two or more sources"):
        PipelineConfig.model_validate(
            _pipeline(
                {"type": "drift-monitoring", "detectors": [{"type": "drift-mmd"}]},
                {"sources": "a", "extractor": "flat"},
            )
        )


def test_triage_with_an_extractor_is_refused() -> None:
    with pytest.raises(ValidationError, match="does not use an extractor"):
        PipelineConfig.model_validate(_pipeline({"type": "metadata-triage"}, {"sources": "a", "extractor": "flat"}))


def test_cluster_cleaning_without_an_extractor_is_refused() -> None:
    cleaning = {
        "type": "data-cleaning",
        "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore", "cluster_threshold": 2.0},
    }
    with pytest.raises(ValidationError, match="needs an extractor"):
        PipelineConfig.model_validate(_pipeline(cleaning, {"sources": "a"}))


def test_a_task_naming_an_undefined_workflow_is_refused_at_load() -> None:
    config = _pipeline({"type": "metadata-triage"}, {"sources": "a"})
    config["tasks"][0]["workflow"] = "nope"
    with pytest.raises(ValidationError, match="`workflows:` does not define"):
        PipelineConfig.model_validate(config)
