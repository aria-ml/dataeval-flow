"""TC-6-1 — discovering workflows and evaluators."""

from __future__ import annotations

import json
from typing import Any

import pytest

from dataeval_flow import InputSpec, Result, SourceCount
from dataeval_flow.config import PipelineConfig
from dataeval_flow.evaluators import Evaluator, EvaluatorConfig, get_evaluator, list_evaluators
from dataeval_flow.workflows import Workflow, WorkflowConfig, get_workflow, list_workflows
from verification.helpers import run_cli

pytestmark = pytest.mark.required

WORKFLOWS = [
    "audit",
    "bias",
    "prioritization",
    "quality",
    "scope",
    "shift",
    "splits",
    "taxonomy",
    "triage",
]
EVALUATORS = [
    "balance",
    "completeness",
    "content-digest",
    "coverage",
    "divergence",
    "diversity",
    "drift-domain-classifier",
    "drift-kneighbors",
    "drift-mmd",
    "drift-univariate",
    "drift-wasserstein",
    "duplicates",
    "factor-leakage",
    "factor-summary",
    "factor-triage",
    "label-alignment",
    "label-health",
    "label-reconciliation",
    "ontology-validation",
    "ood-domain-classifier",
    "ood-kneighbors",
    "outliers",
    "parity",
    "prioritization",
    "profile",
    "representation",
]


class TestWorkflowDiscovery:
    def test_list_workflows_returns_the_nine_workflows_by_name(self) -> None:
        listed = list_workflows()
        assert [cls.name for cls in listed] == WORKFLOWS
        assert all(cls.description for cls in listed)

    def test_get_workflow_returns_the_class_registered_under_the_name(self) -> None:
        cls = get_workflow("quality")
        assert issubclass(cls, Workflow)
        assert cls.name == "quality"
        assert issubclass(cls.config_type, WorkflowConfig)
        assert cls.config_type.model_fields["type"].default == "quality"

    def test_get_workflow_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown workflow: 'does-not-exist'. Installed: \['audit', 'bias'"):
            get_workflow("does-not-exist")

    def test_the_old_workflow_names_are_gone(self) -> None:
        for retired in ("data-cleaning", "data-analysis", "drift-monitoring", "parameter-sweep"):
            with pytest.raises(ValueError, match="Unknown workflow"):
                get_workflow(retired)

    @pytest.mark.parametrize("name", WORKFLOWS)
    def test_a_workflow_declares_its_inputs_and_its_result_class(self, name: str) -> None:
        config_type = get_workflow(name).config_type
        assert isinstance(config_type.inputs, InputSpec)
        assert isinstance(config_type.inputs.sources, SourceCount)
        assert issubclass(config_type.result_type, Result)


class TestEvaluatorDiscovery:
    def test_list_evaluators_returns_every_evaluator_by_name(self) -> None:
        listed = list_evaluators()
        assert [cls.name for cls in listed] == EVALUATORS
        assert all(cls.description for cls in listed)

    def test_get_evaluator_returns_the_class_registered_under_the_name(self) -> None:
        cls = get_evaluator("duplicates")
        assert issubclass(cls, Evaluator)
        assert issubclass(cls.config_type, EvaluatorConfig)
        assert cls.config_type.model_fields["type"].default == "duplicates"

    def test_get_evaluator_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown evaluator: 'nope'. Installed: \['balance', 'completeness'"):
            get_evaluator("nope")

    @pytest.mark.parametrize("name", EVALUATORS)
    def test_an_evaluator_declares_its_inputs_and_its_result_class(self, name: str) -> None:
        config_type = get_evaluator(name).config_type
        assert isinstance(config_type.inputs, InputSpec)
        assert config_type.inputs.required
        assert issubclass(config_type.result_type, Result)

    def test_a_workflow_and_an_evaluator_may_share_a_name(self) -> None:
        """`prioritization` is both; a task says which it means with `workflow:` or `evaluator:`."""
        assert get_workflow("prioritization") is not get_evaluator("prioritization")


class TestTypeIdsInConfigFiles:
    @pytest.mark.parametrize("name", ["bias", "splits", "triage", "scope", "shift", "prioritization"])
    def test_a_workflow_entry_is_validated_with_the_config_its_type_names(self, name: str) -> None:
        entry: dict[str, Any] = {"type": name}
        if name == "shift":
            entry["detectors"] = [{"type": "drift-mmd"}]
        config = PipelineConfig.model_validate({"workflows": [entry]})
        assert config.workflows is not None
        assert type(config.workflows[0]) is get_workflow(name).config_type
        assert config.workflows[0].name == name  # unnamed entries are named after their type

    @pytest.mark.parametrize("name", ["duplicates", "outliers", "label-health", "profile", "drift-mmd"])
    def test_an_evaluator_entry_is_validated_with_the_config_its_type_names(self, name: str) -> None:
        config = PipelineConfig.model_validate({"evaluators": [{"type": name}]})
        assert config.evaluators is not None
        assert type(config.evaluators[0]) is get_evaluator(name).config_type

    def test_an_entry_of_the_wrong_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown evaluator: 'quality'"):
            PipelineConfig.model_validate({"evaluators": [{"type": "quality"}]})
        with pytest.raises(ValueError, match="Unknown workflow: 'duplicates'"):
            PipelineConfig.model_validate({"workflows": [{"type": "duplicates"}]})

    def test_a_workflow_entry_needs_a_type_or_steps(self) -> None:
        with pytest.raises(ValueError, match="needs a `type:`, or `steps:`"):
            PipelineConfig.model_validate({"workflows": [{"name": "w"}]})
        with pytest.raises(ValueError, match="needs a `type:`"):
            PipelineConfig.model_validate({"evaluators": [{"name": "e"}]})


class TestCommandLineListings:
    def test_workflows_lists_what_the_library_lists(self) -> None:
        result = run_cli("workflows", "--json")
        assert result.returncode == 0, result.stderr
        assert [entry["name"] for entry in json.loads(result.stdout)] == WORKFLOWS
        assert {entry["name"]: entry["description"] for entry in json.loads(result.stdout)} == {
            cls.name: cls.description for cls in list_workflows()
        }

    def test_evaluators_lists_what_the_library_lists_with_their_inputs(self) -> None:
        result = run_cli("evaluators", "--json")
        assert result.returncode == 0, result.stderr
        listing = {entry["name"]: entry for entry in json.loads(result.stdout)}
        assert list(listing) == EVALUATORS
        assert listing["drift-mmd"]["sources"] == "2"
        assert listing["duplicates"]["consumes"] == "stats, clusters (optional)"

    def test_a_workflow_name_prints_the_schema_of_its_settings(self) -> None:
        result = run_cli("workflows", "quality")
        assert result.returncode == 0, result.stderr
        schema = json.loads(result.stdout)
        assert schema["title"] == "QualityConfig"
        assert schema["properties"]["type"]["const"] == "quality"

    def test_an_unknown_workflow_name_is_an_error(self) -> None:
        result = run_cli("workflows", "nope")
        assert result.returncode != 0
        assert "Unknown workflow: 'nope'" in result.stderr
