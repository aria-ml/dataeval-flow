"""TC-19-7 — defining a custom workflow: the checks made when a config loads, and `CustomWorkflowConfig` in Python."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import load_config, run, run_tasks
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult, CustomWorkflowConfig, StepEntry
from verification.fixtures import write_image_folder
from verification.functional.chains._toys import Images, pipeline

pytestmark = pytest.mark.required


def _workflow(*steps: dict, inputs: list | None = None) -> dict:
    return {"name": "w", "inputs": inputs or ["a"], "steps": list(steps)}


def _load(workflow: dict, **kwargs):
    return pipeline(
        {"a": Images()},
        workflows=[workflow],
        evaluators=[{"name": "d", "type": "duplicates"}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["a"]}],
        **kwargs,
    )


class TestChecksAtLoad:
    def test_a_workflow_with_inputs_and_steps_loads(self) -> None:
        config = _load(_workflow({"name": "s", "evaluator": "d", "input": "a"}))
        (workflow,) = config.workflows or ()
        assert isinstance(workflow, CustomWorkflowConfig)
        assert [step.name for step in workflow.steps] == ["s"]

    def test_a_step_naming_an_evaluator_the_config_does_not_define_is_refused(self) -> None:
        with pytest.raises(
            ValidationError, match="Step 's' names evaluator 'zzz', which `evaluators:` does not define"
        ):
            _load(_workflow({"name": "s", "evaluator": "zzz", "input": "a"}))

    def test_a_step_reading_an_address_that_is_no_input_or_earlier_step_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="Step 's' reads `nope`, which is no input or earlier step"):
            _load(_workflow({"name": "s", "evaluator": "d", "input": "nope"}))

    def test_a_step_reading_a_step_written_below_it_is_refused(self) -> None:
        steps = ({"name": "s", "evaluator": "d", "input": "later"}, {"name": "later", "evaluator": "d", "input": "a"})
        with pytest.raises(ValidationError, match="reads `later`, but step 'later' runs later"):
            _load(_workflow(*steps))

    def test_a_setting_the_transform_does_not_take_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=r"steps\.0\.bogus"):
            _load(_workflow({"name": "s", "transform": "split", "input": "a", "bogus": 1}))

    def test_a_step_must_name_exactly_one_kind(self) -> None:
        with pytest.raises(ValidationError, match="names 2 of evaluator, workflow, transform, combine, check"):
            _load(_workflow({"name": "s", "evaluator": "d", "transform": "split", "input": "a"}))
        with pytest.raises(ValidationError, match="names none of"):
            _load(_workflow({"name": "s", "input": "a"}))

    def test_settings_of_an_evaluator_step_belong_in_its_evaluators_entry(self) -> None:
        with pytest.raises(ValidationError, match="whose settings live in its `evaluators:` entry"):
            _load(_workflow({"name": "s", "evaluator": "d", "input": "a", "flags": ["hash_basic"]}))

    def test_two_steps_may_not_share_a_name_nor_a_step_take_an_input_s_name(self) -> None:
        twice = ({"name": "s", "evaluator": "d", "input": "a"}, {"name": "s", "evaluator": "d", "input": "a"})
        with pytest.raises(ValidationError, match="more than one step named 's'"):
            _load(_workflow(*twice))
        with pytest.raises(ValidationError, match="step 'a' shares its name with an input"):
            _load(_workflow({"name": "a", "evaluator": "d", "input": "a"}))

    def test_a_step_may_not_read_the_output_of_a_step_computed_on_another_dataset(self) -> None:
        workflow = _workflow(
            {"name": "on_a", "evaluator": "d", "input": "a"},
            {"name": "clean_b", "transform": "remove", "input": "b", "plans": {"on_a": {}}},
            inputs=["a", "b"],
        )
        with pytest.raises(ValidationError, match="computed on `a`, not on `b`"):
            pipeline(
                {"a": Images(), "b": Images()},
                workflows=[workflow],
                evaluators=[{"name": "d", "type": "duplicates"}],
                tasks=[{"name": "t", "workflow": "w", "sources": ["a", "b"]}],
            )


class TestCustomWorkflowConfig:
    @staticmethod
    def _workflow() -> CustomWorkflowConfig:
        return CustomWorkflowConfig(
            name="clean_and_check",
            inputs=["data"],
            steps=[
                StepEntry(name="dupes", evaluator="dupes", input="data"),
                StepEntry(name="clean", transform="remove", input="data", plans={"dupes": {"keep": "first"}}),
                StepEntry(name="found", check="image-duplicates", input="dupes"),
            ],
        )

    def test_run_executes_a_workflow_built_in_python_on_datasets_in_memory(self) -> None:
        result = run(self._workflow(), Images(12), definitions=[DuplicatesConfig(name="dupes")])
        assert isinstance(result, ChainResult)
        assert result.success, result.errors
        assert [(name, record.status) for name, record in result.steps.items()] == [
            ("dupes", "ok"),
            ("clean", "ok"),
            ("found", "ok"),
        ]
        assert len(result.steps["clean"].output) == 11
        assert [f.severity for f in result.findings] == ["warning"]

    def test_run_binds_the_datasets_of_a_mapping_to_the_inputs_in_order(self) -> None:
        workflow = CustomWorkflowConfig(
            name="two",
            inputs=["first", "second"],
            steps=[StepEntry(name="merged", transform="merge", input=["first", "second"])],
        )
        result = run(workflow, {"x": Images(5, planted=False), "y": Images(7, seed=1, planted=False)})
        assert len(result.steps["merged"].output) == 12

    def test_to_yaml_writes_the_workflow_as_a_config_fragment(self) -> None:
        fragment = yaml.safe_load(self._workflow().to_yaml([DuplicatesConfig(name="dupes")]))
        assert fragment["evaluators"] == [{"name": "dupes", "type": "duplicates"}]
        (entry,) = fragment["workflows"]
        assert entry["inputs"] == ["data"]
        assert entry["steps"][1] == {
            "name": "clean",
            "transform": "remove",
            "input": "data",
            "plans": {"dupes": {"keep": "first"}},
        }
        assert entry["steps"][2] == {"name": "found", "check": "image-duplicates", "input": "dupes"}

    def test_save_adds_the_workflow_and_replaces_the_entry_of_the_same_name(self, tmp_path: Path) -> None:
        target = tmp_path / "workflows.yaml"
        self._workflow().save(target)
        again = CustomWorkflowConfig(
            name="clean_and_check", inputs=["data"], steps=[StepEntry(name="d", evaluator="dupes", input="data")]
        )
        again.save(target)
        saved = yaml.safe_load(target.read_text())
        assert [w["name"] for w in saved["workflows"]] == ["clean_and_check"]
        assert [s["name"] for s in saved["workflows"][0]["steps"]] == ["d"]

    def test_save_keeps_the_file_s_other_keys_and_workflows(self, tmp_path: Path) -> None:
        target = tmp_path / "workflows.yaml"
        target.write_text(yaml.safe_dump({"seed": 3, "workflows": [{"name": "keep", "type": "quality"}]}))
        self._workflow().save(target)
        saved = yaml.safe_load(target.read_text())
        assert saved["seed"] == 3
        assert [w["name"] for w in saved["workflows"]] == ["keep", "clean_and_check"]

    def test_save_refuses_a_file_that_is_not_a_pipeline_config_and_a_missing_directory(self, tmp_path: Path) -> None:
        other = tmp_path / "other.yaml"
        other.write_text("foo: 1\n")
        with pytest.raises(ValueError, match="is not a pipeline config fragment; refusing to rewrite it"):
            self._workflow().save(other)
        assert other.read_text() == "foo: 1\n"
        with pytest.raises(FileNotFoundError):
            self._workflow().save(tmp_path / "missing" / "workflows.yaml")

    def test_a_saved_workflow_loads_beside_the_pipeline_and_runs_on_new_data(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        write_image_folder(tmp_path / "imgs", n_per_class=4, n_classes=2)
        (config_dir / "pipeline.yaml").write_text(
            yaml.safe_dump(
                {
                    "datasets": [{"name": "ds", "format": "image_folder", "path": "imgs", "infer_labels": True}],
                    "sources": [{"name": "main", "dataset": "ds"}],
                    "tasks": [{"name": "t", "workflow": "clean_and_check", "sources": ["main"]}],
                }
            )
        )
        self._workflow().save(config_dir / "workflows.yaml", definitions=[DuplicatesConfig(name="dupes")])
        config = load_config(config_dir)
        result = run_tasks(config, data_dir=tmp_path)["t"]
        assert isinstance(result, ChainResult)
        assert result.success, result.errors
        assert result.steps["dupes"].status == "ok"

    def test_building_the_workflow_checks_its_slots_and_step_names(self) -> None:
        with pytest.raises(ValidationError, match="names input 'a' more than once"):
            CustomWorkflowConfig(name="w", inputs=["a", "a"], steps=[StepEntry(name="s", evaluator="d", input="a")])
        with pytest.raises(ValidationError, match="Only the last input may be a list"):
            CustomWorkflowConfig.model_validate(
                {"name": "w", "inputs": [{"name": "a", "list": True}, "b"], "steps": [{"name": "s", "evaluator": "d"}]}
            )
