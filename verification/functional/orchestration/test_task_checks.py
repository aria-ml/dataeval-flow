"""TC-6-2 — tasks: workflow and evaluator kinds, and the checks made when the configuration loads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import load_config, run, run_task
from dataeval_flow.config import PipelineConfig, TaskConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.evaluators.scope import CompletenessConfig
from dataeval_flow.workflows.quality import QualityConfig
from verification.fixtures import write_image_folder
from verification.functional.orchestration.support import InMemoryImages

pytestmark = pytest.mark.required

QUALITY = {"name": "q", "type": "quality", "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}
CLUSTERED = {"name": "qc", "type": "quality", "outliers": {**QUALITY["outliers"], "cluster_threshold": 2.0}}
SHIFT = {"name": "sh", "type": "shift", "detectors": [{"type": "drift-mmd"}]}


def _config(**task: Any) -> dict[str, Any]:
    """A pipeline with two sources, one extractor, four workflows and four evaluators, and one task `t` of `task`."""
    return {
        "datasets": [{"name": "ds", "format": "image_folder", "path": "imgs", "infer_labels": True}],
        "sources": [{"name": "a", "dataset": "ds"}, {"name": "b", "dataset": "ds"}],
        "extractors": [{"name": "flat", "model": "flatten", "batch_size": 4}],
        "workflows": [QUALITY, CLUSTERED, SHIFT, {"name": "tri", "type": "triage"}],
        "evaluators": [
            {"name": "dup", "type": "duplicates"},
            {"name": "comp", "type": "completeness"},
            {"name": "mmd", "type": "drift-mmd"},
            {"name": "digest", "type": "content-digest"},
        ],
        "tasks": [{"name": "t", **task}],
    }


class TestTaskKinds:
    def test_a_task_naming_a_workflow_is_a_workflow_task(self) -> None:
        task = TaskConfig.model_validate({"name": "t", "workflow": "q", "sources": "a"})
        assert (task.kind, task.workflow) == ("workflow", "q")

    def test_a_task_naming_an_evaluator_is_an_evaluator_task(self) -> None:
        task = TaskConfig.model_validate({"name": "t", "evaluator": "dup", "sources": "a"})
        assert (task.kind, task.workflow) == ("evaluator", "dup")

    def test_a_task_names_exactly_one_of_the_two(self) -> None:
        with pytest.raises(ValidationError, match="names both a workflow and an evaluator"):
            TaskConfig.model_validate({"name": "t", "workflow": "q", "evaluator": "dup", "sources": "a"})
        with pytest.raises(ValidationError, match="names neither a workflow nor an evaluator"):
            TaskConfig.model_validate({"name": "t", "sources": "a"})

    def test_an_empty_key_counts_as_absent(self) -> None:
        task = TaskConfig.model_validate({"name": "t", "workflow": None, "evaluator": "dup", "sources": "a"})
        assert task.kind == "evaluator"

    def test_a_task_is_written_back_under_the_key_it_was_read_from(self) -> None:
        workflow = TaskConfig.model_validate({"name": "w", "workflow": "q", "sources": "a"}).model_dump()
        evaluator = TaskConfig.model_validate({"name": "e", "evaluator": "dup", "sources": "a"}).model_dump()
        assert workflow["workflow"] == "q"
        assert "evaluator" not in workflow
        assert evaluator["evaluator"] == "dup"
        assert "workflow" not in evaluator
        assert "kind" not in workflow
        assert "kind" not in evaluator

    def test_a_pipeline_round_trips_its_workflow_and_evaluator_tasks(self) -> None:
        data = _config()
        data["tasks"] = [
            {"name": "w", "workflow": "q", "sources": "a"},
            {"name": "e", "evaluator": "dup", "sources": ["a", "b"]},
        ]
        config = PipelineConfig.model_validate(data)
        reloaded = PipelineConfig.model_validate(json.loads(config.model_dump_json()))
        assert reloaded == config
        assert reloaded.tasks is not None
        assert [(task.kind, task.workflow) for task in reloaded.tasks] == [("workflow", "q"), ("evaluator", "dup")]

    def test_sources_are_one_name_or_a_list_and_always_read_as_a_list(self) -> None:
        assert TaskConfig.model_validate({"name": "t", "workflow": "q", "sources": "a"}).source_names == ["a"]
        task = TaskConfig.model_validate({"name": "t", "workflow": "q", "sources": ["a", "b"]})
        assert task.source_names == ["a", "b"]

    def test_a_source_named_twice_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="names source 'a' more than once"):
            TaskConfig.model_validate({"name": "t", "evaluator": "dup", "sources": ["a", "a"]})

    def test_a_task_is_enabled_unless_it_says_otherwise(self) -> None:
        assert TaskConfig.model_validate({"name": "t", "workflow": "q", "sources": "a"}).enabled is True
        assert (
            TaskConfig.model_validate({"name": "t", "workflow": "q", "sources": "a", "enabled": False}).enabled is False
        )

    def test_a_task_name_is_used_once(self) -> None:
        data = _config()
        data["tasks"] = [
            {"name": "t", "workflow": "q", "sources": "a"},
            {"name": "t", "workflow": "tri", "sources": "b"},
        ]
        with pytest.raises(ValidationError, match="Duplicate name 't' in tasks"):
            PipelineConfig.model_validate(data)

    def test_a_task_in_a_file_names_a_workflow_that_another_file_defines(self, tmp_path: Path) -> None:
        (tmp_path / "00-tasks.yaml").write_text(
            yaml.safe_dump(
                {
                    "datasets": [{"name": "ds", "format": "image_folder", "path": "imgs"}],
                    "sources": [{"name": "a", "dataset": "ds"}],
                    "tasks": [{"name": "t", "evaluator": "dup", "sources": "a"}],
                }
            )
        )
        (tmp_path / "01-evaluators.yaml").write_text(
            yaml.safe_dump({"evaluators": [{"name": "dup", "type": "duplicates"}]})
        )
        config = load_config(tmp_path)
        assert config.tasks is not None
        assert config.tasks[0].kind == "evaluator"


class TestLoadTimeChecks:
    """A task its workflow or evaluator cannot run costs a configuration error, before any data is read."""

    def test_a_valid_task_of_each_kind_loads(self) -> None:
        for task in (
            {"workflow": "q", "sources": "a"},
            {"workflow": "qc", "sources": "a", "extractor": "flat"},
            {"workflow": "sh", "sources": ["a", "b"], "extractor": "flat"},
            {"evaluator": "dup", "sources": ["a", "b"]},
            {"evaluator": "dup", "sources": "a", "extractor": "flat"},
            {"evaluator": "comp", "sources": "a", "extractor": "flat"},
            {"evaluator": "mmd", "sources": ["a", "b"], "extractor": "flat"},
        ):
            assert PipelineConfig.model_validate(_config(**task)).tasks, task

    @pytest.mark.parametrize(
        ("task", "message"),
        [
            ({"workflow": "nope", "sources": "a"}, "names workflow 'nope', which `workflows:` does not define"),
            ({"evaluator": "nope", "sources": "a"}, "names evaluator 'nope', which `evaluators:` does not define"),
            ({"workflow": "dup", "sources": "a"}, "names workflow 'dup', which `workflows:` does not define"),
            ({"evaluator": "q", "sources": "a"}, "names evaluator 'q', which `evaluators:` does not define"),
        ],
    )
    def test_a_task_naming_what_is_not_defined_is_refused(self, task: dict[str, Any], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            PipelineConfig.model_validate(_config(**task))

    def test_the_error_lists_what_is_defined(self) -> None:
        with pytest.raises(ValidationError, match=r"Defined: \['q', 'qc', 'sh', 'tri'\]"):
            PipelineConfig.model_validate(_config(workflow="nope", sources="a"))

    @pytest.mark.parametrize(
        ("task", "message"),
        [
            (
                {"workflow": "q", "sources": ["a", "b"]},
                r"quality\), which takes exactly one source, but the task names 2",
            ),
            (
                {"workflow": "sh", "sources": "a", "extractor": "flat"},
                r"shift\), which takes two or more sources, but the task names 1",
            ),
            (
                {"evaluator": "mmd", "sources": "a", "extractor": "flat"},
                r"drift-mmd\), which takes exactly two sources, but the task names 1",
            ),
        ],
    )
    def test_a_task_naming_the_wrong_number_of_sources_is_refused(self, task: dict[str, Any], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            PipelineConfig.model_validate(_config(**task))

    @pytest.mark.parametrize(
        ("task", "message"),
        [
            ({"workflow": "sh", "sources": ["a", "b"]}, r"shift\), which needs an extractor to produce embeddings"),
            ({"workflow": "qc", "sources": "a"}, r"quality\), which needs an extractor to produce clusters"),
            ({"evaluator": "comp", "sources": "a"}, r"completeness\), which needs an extractor to produce embeddings"),
        ],
    )
    def test_a_task_that_needs_an_extractor_and_names_none_is_refused(self, task: dict[str, Any], message: str) -> None:
        with pytest.raises(ValidationError, match=message + r"; name one with `extractor:`"):
            PipelineConfig.model_validate(_config(**task))

    @pytest.mark.parametrize(
        "task",
        [
            {"workflow": "tri", "sources": "a", "extractor": "flat"},
            {"evaluator": "digest", "sources": "a", "extractor": "flat"},
        ],
    )
    def test_a_task_that_uses_no_extractor_but_names_one_is_refused(self, task: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="does not use an extractor; remove `extractor:` from the task"):
            PipelineConfig.model_validate(_config(**task))

    def test_a_setting_that_limits_the_source_count_is_checked_too(self) -> None:
        """Cluster mode reads one source, though the evaluator otherwise takes several."""
        data = _config(evaluator="dup", sources=["a", "b"], extractor="flat")
        data["evaluators"][0] = {"name": "dup", "type": "duplicates", "cluster_sensitivity": 1.0}
        with pytest.raises(ValidationError, match="reads exactly one source in cluster mode"):
            PipelineConfig.model_validate(data)

    def test_every_problem_is_found_by_loading_the_file_not_by_running_it(self, tmp_path: Path) -> None:
        path = tmp_path / "params.yaml"
        path.write_text(yaml.safe_dump(_config(workflow="sh", sources="a", extractor="flat")))
        with pytest.raises(ValidationError, match="takes two or more sources"):
            load_config(path)

    def test_a_task_run_directly_is_checked_and_fails_with_the_same_message(
        self,
        tmp_path: Path,
        fresh_caches: None,
    ) -> None:
        """`run_task` takes a task that is not in `config.tasks`, so load never saw it: the run refuses it instead."""
        write_image_folder(tmp_path / "imgs", n_per_class=3, n_classes=2)
        config = PipelineConfig.model_validate(_config(workflow="q", sources="a"))
        task = TaskConfig.model_validate({"name": "extra", "workflow": "q", "sources": ["a", "b"]})
        result = run_task(config, task, data_dir=tmp_path)
        assert not result.success
        assert (
            "Task 'extra' runs workflow 'q' (quality), which takes exactly one source, but the task names 2"
            in (result.errors[0])
        )

    def test_run_checks_the_data_against_the_config_before_anything_runs(self, fresh_caches: None) -> None:
        data = {"a": InMemoryImages(12), "b": InMemoryImages(12, seed=1)}
        with pytest.raises(ValidationError, match="takes exactly one source, but the task names 2"):
            run(QualityConfig(outliers={"flags": ["pixel"], "outlier_threshold": "zscore"}), data)  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="needs an extractor to produce embeddings"):
            run(CompletenessConfig(), InMemoryImages(12))
        assert run(DuplicatesConfig(), data).success  # `duplicates` reads one source or more
