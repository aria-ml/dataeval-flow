"""Where a custom workflow's steps came from: historical, never a reference (preset decomposition spec §4)."""

import yaml

from dataeval_flow import run_tasks
from dataeval_flow.config import PipelineConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult, CustomWorkflowConfig
from tests.chain_toys import chain_pipeline

_ORIGIN = {
    "type": "quality",
    "entry": "gone",
    "settings": {"name": "gone", "type": "quality", "outliers": {"flags": ["pixel"]}},
    "flow_version": "0.0.0",
}
_WORKFLOW = {
    "name": "w",
    "inputs": ["data"],
    "steps": [
        {"name": "dupes", "evaluator": "dupes", "input": "data"},
        {"name": "judge", "check": "image-duplicates", "input": "dupes"},
    ],
}


def test_provenance_round_trips_through_yaml() -> None:
    workflow = CustomWorkflowConfig.model_validate({**_WORKFLOW, "derived_from": [_ORIGIN]})
    written = yaml.safe_load(workflow.to_yaml())["workflows"][0]
    assert written["derived_from"] == [_ORIGIN]
    assert CustomWorkflowConfig.model_validate(written) == workflow


def test_a_workflow_with_no_provenance_writes_none() -> None:
    assert (
        "derived_from" not in yaml.safe_load(CustomWorkflowConfig.model_validate(_WORKFLOW).to_yaml())["workflows"][0]
    )


def test_provenance_naming_an_entry_that_is_gone_still_loads_and_runs() -> None:
    config = chain_pipeline(
        workflows=[{**_WORKFLOW, "derived_from": [_ORIGIN]}],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors


def test_provenance_is_no_reference_in_the_schema() -> None:
    derivation = PipelineConfig.model_json_schema()["$defs"]["DerivationConfig"]
    assert "x-flow-ref" not in yaml.safe_dump(derivation)


def test_inlined_provenance_names_its_steps() -> None:
    origin = {**_ORIGIN, "steps": ["dupes"]}
    workflow = CustomWorkflowConfig.model_validate({**_WORKFLOW, "derived_from": [origin, {**_ORIGIN, "entry": "b"}]})
    assert [d.steps for d in workflow.derived_from] == [["dupes"], None]
