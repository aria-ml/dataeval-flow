"""Evaluator tasks: told apart from workflow tasks by their key, and checked against their evaluator at load."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow.config import (
    DataCleaningTaskConfig,
    EvaluatorTaskConfig,
    PipelineConfig,
    TaskConfig,
    load_config_folder,
)


def _config(tasks: list[Any]) -> dict[str, Any]:
    return {
        "datasets": [{"name": "ds", "format": "huggingface", "path": "./d", "task": "image_classification"}],
        "sources": [{"name": "a", "dataset": "ds"}, {"name": "b", "dataset": "ds"}],
        "extractors": [{"name": "flat", "model": "flatten"}],
        "workflows": [
            {"name": "clean", "type": "data-cleaning", "outlier_method": "zscore", "outlier_flags": ["pixel"]}
        ],
        "evaluators": [
            {"name": "dupes", "type": "quality.duplicates"},
            {"name": "dupes_cluster", "type": "quality.duplicates", "cluster_sensitivity": 1.0},
        ],
        "tasks": tasks,
    }


class TestTaskKinds:
    def test_a_task_naming_an_evaluator_is_an_evaluator_task(self):
        config = PipelineConfig.model_validate(_config([{"name": "t", "evaluator": "dupes", "sources": "a"}]))
        assert config.tasks is not None
        task = config.tasks[0]
        assert type(task) is TaskConfig
        assert (task.kind, task.workflow) == ("evaluator", "dupes")

    def test_a_task_naming_a_workflow_is_unchanged(self):
        config = PipelineConfig.model_validate(_config([{"name": "t", "workflow": "clean", "sources": "a"}]))
        assert config.tasks is not None
        task = config.tasks[0]
        assert type(task) is TaskConfig
        assert (task.kind, task.workflow) == ("workflow", "clean")

    def test_an_empty_evaluator_key_counts_as_absent(self):
        """A TUI form's unset field, or `evaluator: null` in YAML, leaves a workflow task."""
        task = TaskConfig.model_validate({"name": "t", "workflow": "clean", "evaluator": None, "sources": "a"})
        assert (task.kind, task.workflow) == ("workflow", "clean")

    def test_naming_both_is_refused(self):
        with pytest.raises(ValidationError, match="Task 't' names both a workflow and an evaluator"):
            PipelineConfig.model_validate(
                _config([{"name": "t", "workflow": "clean", "evaluator": "dupes", "sources": "a"}])
            )

    def test_naming_neither_is_refused(self):
        with pytest.raises(ValidationError, match="Task 't' names neither a workflow nor an evaluator"):
            PipelineConfig.model_validate(_config([{"name": "t", "sources": "a"}]))

    def test_a_task_built_in_python_is_checked_too(self):
        with pytest.raises(ValidationError, match="Task 't' names both a workflow and an evaluator"):
            TaskConfig.model_validate({"name": "t", "workflow": "clean", "evaluator": "dupes", "sources": "a"})

    def test_an_evaluator_key_cannot_carry_a_workflow_kind(self):
        with pytest.raises(ValidationError, match="names an evaluator, so its kind cannot be 'workflow'"):
            TaskConfig.model_validate({"name": "t", "evaluator": "dupes", "kind": "workflow", "sources": "a"})

    def test_typed_workflow_tasks_keep_their_type(self):
        task = DataCleaningTaskConfig(name="t", workflow="clean", sources="a")
        config = PipelineConfig.model_validate(_config([task]))
        assert config.tasks is not None
        assert isinstance(config.tasks[0], DataCleaningTaskConfig)

    def test_source_names_are_always_a_list(self):
        assert EvaluatorTaskConfig(name="t", workflow="e", sources="a").source_names == ["a"]
        assert EvaluatorTaskConfig(name="t", workflow="e", sources=["a", "b"]).source_names == ["a", "b"]


class TestEvaluatorTaskConfig:
    def test_its_kind_is_always_evaluator(self):
        assert EvaluatorTaskConfig(name="t", workflow="dupes", sources="a").kind == "evaluator"

    def test_the_file_key_works_too(self):
        task = EvaluatorTaskConfig.model_validate({"name": "t", "evaluator": "dupes", "sources": "a"})
        assert (task.kind, task.workflow) == ("evaluator", "dupes")

    def test_a_workflow_kind_is_refused(self):
        with pytest.raises(ValidationError, match="its kind cannot be 'workflow'"):
            EvaluatorTaskConfig(name="t", workflow="dupes", kind="workflow", sources="a")


class TestTheFileShape:
    """A task is saved and described the way a config file names it: `workflow:` or `evaluator:`, no `kind`."""

    def test_an_evaluator_task_dumps_under_its_key(self):
        task = TaskConfig.model_validate({"name": "t", "evaluator": "dupes", "sources": "a"})
        dumped = task.model_dump()
        assert dumped["evaluator"] == "dupes"
        assert "workflow" not in dumped
        assert "kind" not in dumped
        assert TaskConfig.model_validate(dumped) == task

    def test_a_workflow_task_dumps_as_it_always_has(self):
        task = TaskConfig(name="t", workflow="clean", sources="a")
        assert task.model_dump() == {
            "name": "t",
            "workflow": "clean",
            "enabled": True,
            "sources": "a",
            "extractor": None,
        }

    def test_a_pipeline_round_trips_its_evaluator_tasks(self):
        config = PipelineConfig.model_validate(_config([{"name": "t", "evaluator": "dupes", "sources": "a"}]))
        reloaded = PipelineConfig.model_validate(json.loads(config.model_dump_json()))
        assert reloaded.tasks is not None
        assert (reloaded.tasks[0].kind, reloaded.tasks[0].workflow) == ("evaluator", "dupes")

    def test_the_schema_offers_both_keys_and_no_kind(self):
        schema = PipelineConfig.model_json_schema()["$defs"]["TaskConfig"]
        assert {"workflow", "evaluator"} <= set(schema["properties"])
        assert "kind" not in schema["properties"]
        assert "workflow" not in schema["required"]
        assert schema["oneOf"] == [{"required": ["workflow"]}, {"required": ["evaluator"]}]


class TestEvaluatorTaskValidation:
    def test_an_undefined_evaluator_is_refused(self):
        with pytest.raises(ValidationError, match="names evaluator 'nope', which `evaluators:` does not define"):
            PipelineConfig.model_validate(_config([{"name": "t", "evaluator": "nope", "sources": "a"}]))

    def test_hash_mode_reads_several_sources(self):
        config = PipelineConfig.model_validate(_config([{"name": "t", "evaluator": "dupes", "sources": ["a", "b"]}]))
        assert config.tasks is not None
        assert config.tasks[0].kind == "evaluator"

    def test_cluster_mode_needs_an_extractor(self):
        match = r"runs evaluator 'dupes_cluster' \(quality\.duplicates\), which needs an extractor to produce clusters"
        with pytest.raises(ValidationError, match=match):
            PipelineConfig.model_validate(_config([{"name": "t", "evaluator": "dupes_cluster", "sources": "a"}]))

    def test_cluster_mode_reads_one_source(self):
        task = {"name": "t", "evaluator": "dupes_cluster", "sources": ["a", "b"], "extractor": "flat"}
        with pytest.raises(ValidationError, match="reads exactly one source in cluster mode"):
            PipelineConfig.model_validate(_config([task]))

    def test_hash_mode_admits_an_extractor(self):
        task = {"name": "t", "evaluator": "dupes", "sources": "a", "extractor": "flat"}
        config = PipelineConfig.model_validate(_config([task]))
        assert config.tasks is not None
        assert config.tasks[0].kind == "evaluator"

    def test_a_config_split_across_files_validates_once_merged(self, tmp_path: Path):
        """Review Focus 2: the evaluator and the task that runs it may live in different files."""
        (tmp_path / "00-data.yaml").write_text(
            "datasets:\n"
            "  - name: ds\n"
            "    format: huggingface\n"
            "    path: ./d\n"
            "    task: image_classification\n"
            "sources:\n"
            "  - name: a\n"
            "    dataset: ds\n"
            "tasks:\n"
            "  - name: t\n"
            "    evaluator: dupes\n"
            "    sources: a\n"
        )
        (tmp_path / "01-evaluators.yaml").write_text("evaluators:\n  - name: dupes\n    type: quality.duplicates\n")
        config = load_config_folder(tmp_path)
        assert config.tasks is not None
        assert config.tasks[0].kind == "evaluator"
