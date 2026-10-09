"""TC-6-4 — workflows and evaluators from installed plug-in packages."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import InputKind, InputSpec, SourceCount, load_config, run, run_tasks
from dataeval_flow.config import PipelineConfig
from dataeval_flow.config.extractors import list_extractors
from dataeval_flow.config.image_transforms import list_image_transforms
from dataeval_flow.evaluators import get_evaluator, list_evaluators
from dataeval_flow.steps import ChainResult, InputSlot
from dataeval_flow.workflows import Preset, PresetChain, Workflow, WorkflowConfig, get_workflow, list_workflows
from verification.functional.orchestration.support import (
    EXAMPLE_ENTRY_POINTS,
    MODULE,
    BrightnessConfig,
    BrightnessResult,
    CountConfig,
    InMemoryImages,
    MeanConfig,
    _registries,
    write_project,
)

pytestmark = [pytest.mark.required, pytest.mark.usefixtures("fresh_caches")]

BUILT_IN_WORKFLOWS = ["audit", "bias", "prioritization", "quality", "scope", "shift", "splits", "taxonomy", "triage"]


def _workflow_names() -> list[str]:
    return [cls.name for cls in list_workflows()]


class _NoInputsConfig(WorkflowConfig[ChainResult]):
    """Its `type` matches the name it is registered under, but it declares no `inputs`."""

    type: str = "example.noinputs"


class NoInputsWorkflow(Preset, Workflow[_NoInputsConfig, ChainResult]):
    """A workflow whose config declares no `inputs`."""

    name: ClassVar[str] = "example.noinputs"
    description: ClassVar[str] = "Its config declares no inputs."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: _NoInputsConfig) -> PresetChain:
        raise NotImplementedError


class _OtherConfig(WorkflowConfig[ChainResult]):
    """Configures `example.other`, though the class below is registered as `example.mismatched`."""

    type: str = "example.other"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.LABELS}), sources=SourceCount.ONE)


class MismatchedWorkflow(Preset, Workflow[_OtherConfig, ChainResult]):
    """A workflow registered under a name its config does not configure."""

    name: ClassVar[str] = "example.mismatched"
    description: ClassVar[str] = "Its config configures another type."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: _OtherConfig) -> PresetChain:
        raise NotImplementedError


class NotAWorkflow:
    """Registered as a workflow, though it is not one."""

    name = "example.plain"


class TestInstalledPlugins:
    def test_every_installed_kind_is_listed_beside_the_built_ins(self, example_plugin: dict[str, Any]) -> None:
        assert _workflow_names() == sorted([*BUILT_IN_WORKFLOWS, "example.count"])
        assert "example.brightness" in [cls.name for cls in list_evaluators()]
        assert "example.mean" in [cls.name for cls in list_extractors()]
        assert "example.Invert" in [cls.name for cls in list_image_transforms()]

    def test_get_workflow_and_get_evaluator_find_an_installed_type(self, example_plugin: dict[str, Any]) -> None:
        assert get_workflow("example.count").__name__ == "CountWorkflow"
        assert get_evaluator("example.brightness").__name__ == "BrightnessEvaluator"

    def test_an_installed_type_validates_from_a_config_file(
        self, example_plugin: dict[str, Any], tmp_path: Path
    ) -> None:
        text = "workflows:\n  - type: example.count\n    minimum: 5\nevaluators:\n  - type: example.brightness\n"
        path = tmp_path / "params.yaml"
        path.write_text(text)
        config = load_config(path)
        assert config.workflows is not None
        assert isinstance(config.workflows[0], CountConfig)
        assert (config.workflows[0].name, config.workflows[0].minimum) == ("example.count", 5)
        assert config.evaluators is not None
        assert isinstance(config.evaluators[0], BrightnessConfig)

    def test_an_installed_types_own_settings_are_validated_and_kept_in_a_dump(
        self,
        example_plugin: dict[str, Any],
    ) -> None:
        with pytest.raises(ValidationError, match="minimum"):
            PipelineConfig.model_validate({"workflows": [{"type": "example.count", "minimum": -1}]})
        with pytest.raises(ValidationError, match="Extra inputs"):
            PipelineConfig.model_validate({"workflows": [{"type": "example.count", "bogus": 1}]})
        config = PipelineConfig.model_validate({"workflows": [{"type": "example.count", "minimum": 3}]})
        reloaded = PipelineConfig.model_validate_json(config.model_dump_json())
        assert reloaded == config
        assert reloaded.workflows is not None
        assert reloaded.workflows[0].minimum == 3  # type: ignore[attr-defined]

    def test_an_installed_workflow_runs_as_a_task_like_a_built_in(
        self,
        example_plugin: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        path, _ = write_project(
            tmp_path,
            workflows=[
                {"name": "small", "type": "example.count", "minimum": 100},
                {"name": "big", "type": "example.count"},
            ],
            tasks=[
                {"name": "warns", "workflow": "small", "sources": "main"},
                {"name": "passes", "workflow": "big", "sources": "main"},
            ],
        )
        results = run_tasks(load_config(path), data_dir=tmp_path)
        warns, passes = results["warns"], results["passes"]
        assert isinstance(warns, ChainResult)
        assert warns.success, warns.errors
        assert warns.type == "example.count"
        assert warns.steps["items"].output.items == 10
        assert warns.warning_count == 1  # 10 items are fewer than the 100 it asks for
        assert passes.warning_count == 0

    def test_an_installed_evaluator_runs_as_a_task_and_in_memory(
        self,
        example_plugin: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        path, _ = write_project(
            tmp_path,
            evaluators=[{"name": "bright", "type": "example.brightness"}],
            tasks=[{"name": "t", "evaluator": "bright", "sources": "main"}],
        )
        piped = run_tasks(load_config(path), data_dir=tmp_path)["t"]
        direct = run(BrightnessConfig(), InMemoryImages(12))
        for result in (piped, direct):
            assert isinstance(result, BrightnessResult)
            assert result.success, result.errors
            assert (result.kind, result.type) == ("evaluator", "example.brightness")

    def test_a_task_naming_an_installed_type_is_checked_against_its_inputs(
        self,
        example_plugin: dict[str, Any],
    ) -> None:
        data = {
            "datasets": [{"name": "ds", "format": "image_folder", "path": "imgs"}],
            "sources": [{"name": "a", "dataset": "ds"}, {"name": "b", "dataset": "ds"}],
            "workflows": [{"name": "count", "type": "example.count"}],
            "tasks": [{"name": "t", "workflow": "count", "sources": ["a", "b"]}],
        }
        with pytest.raises(
            ValidationError, match=r"example\.count\), which takes exactly one source, but the task names 2"
        ):
            PipelineConfig.model_validate(data)

    def test_an_installed_type_is_found_through_real_package_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing is patched: `importlib.metadata` finds a `.dist-info` folder on `sys.path` as it would an install."""
        dist_info = tmp_path / "flow_example_plugin-0.1.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: flow-example-plugin\nVersion: 0.1\n")
        (dist_info / "entry_points.txt").write_text(
            "".join(
                f"[{group}]\n" + "".join(f"{name} = {target}\n" for name, target in entries) + "\n"
                for group, entries in EXAMPLE_ENTRY_POINTS.items()
            )
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        for registry in _registries():
            registry._reset()
        try:
            assert "example.count" in _workflow_names()
            assert "example.brightness" in [cls.name for cls in list_evaluators()]
            config = PipelineConfig.model_validate(
                yaml.safe_load("workflows:\n  - type: example.count\nextractors:\n  - model: example.mean\n")
            )
            assert config.workflows is not None
            assert isinstance(config.workflows[0], CountConfig)
            assert config.extractors is not None
            assert isinstance(config.extractors[0], MeanConfig)
            assert run(CountConfig(), InMemoryImages(12)).success
        finally:
            for registry in _registries():
                registry._reset()

    def test_a_class_that_is_defined_but_not_registered_is_not_found(self) -> None:
        with pytest.raises(
            ValueError, match=r"Unknown evaluator: 'example.brightness'.*entry point.*'dataeval_flow.evaluators'"
        ):
            run(BrightnessConfig(), InMemoryImages(12))
        with pytest.raises(ValueError, match="Unknown workflow: 'example.count'"):
            PipelineConfig.model_validate({"workflows": [{"type": "example.count"}]})


class TestRefusedPlugins:
    def test_a_plugin_that_fails_to_load_is_left_out_and_the_rest_keep_working(
        self,
        example_plugin: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        example_plugin["dataeval_flow.workflows"].append(("example.gone", "verification.no_such_module:Nope"))
        with caplog.at_level(logging.WARNING):
            names = _workflow_names()
        assert "example.gone" not in names
        assert {"quality", "example.count"} <= set(names)
        assert "example.gone" in caplog.text
        with pytest.raises(ValueError, match="no_such_module"):
            get_workflow("example.gone")
        assert run(CountConfig(), InMemoryImages(12)).success

    def test_a_configuration_naming_a_plugin_that_failed_to_load_says_why(
        self,
        plugins: dict[str, Any],
    ) -> None:
        plugins["dataeval_flow.workflows"] = [("example.gone", "verification.no_such_module:Nope")]
        with pytest.raises(ValidationError, match="failed to load verification.no_such_module:Nope"):
            PipelineConfig.model_validate({"workflows": [{"type": "example.gone"}]})

    @pytest.mark.parametrize(
        ("group", "name", "target"),
        [
            ("dataeval_flow.workflows", "quality", f"{MODULE}:CountWorkflow"),
            ("dataeval_flow.evaluators", "duplicates", f"{MODULE}:BrightnessEvaluator"),
        ],
    )
    def test_a_plugin_cannot_take_a_built_ins_name(
        self,
        plugins: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
        group: str,
        name: str,
        target: str,
    ) -> None:
        plugins[group] = [(name, target)]
        with caplog.at_level(logging.WARNING):
            registered = get_workflow(name) if group.endswith("workflows") else get_evaluator(name)
        assert registered.__module__.startswith("dataeval_flow.")  # the built-in is still the one registered
        assert f"'{name}' from an unknown package clashes with the one from dataeval-flow" in caplog.text

    def test_two_plugins_claiming_one_name_are_both_refused(
        self,
        plugins: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        plugins["dataeval_flow.workflows"] = [
            ("example.count", f"{MODULE}:CountWorkflow"),
            ("example.count", f"{__name__}:MismatchedWorkflow"),
        ]
        with caplog.at_level(logging.WARNING):
            assert "example.count" not in _workflow_names()
        with pytest.raises(ValueError, match="is claimed by .* and .*; with no way to choose, none is registered"):
            get_workflow("example.count")

    def test_a_plugin_registered_under_another_name_than_its_own_is_refused(
        self,
        plugins: dict[str, Any],
    ) -> None:
        plugins["dataeval_flow.workflows"] = [("example.other", f"{MODULE}:CountWorkflow")]
        with pytest.raises(ValueError, match="registered as 'example.other' but names itself 'example.count'"):
            get_workflow("example.other")

    def test_a_plugin_that_is_not_a_workflow_is_refused(self, plugins: dict[str, Any]) -> None:
        plugins["dataeval_flow.workflows"] = [("example.plain", f"{__name__}:NotAWorkflow")]
        with pytest.raises(ValueError, match="is not a subclass of Workflow"):
            get_workflow("example.plain")

    def test_a_plugin_whose_config_configures_another_type_is_refused(self, plugins: dict[str, Any]) -> None:
        plugins["dataeval_flow.workflows"] = [("example.mismatched", f"{__name__}:MismatchedWorkflow")]
        with pytest.raises(
            ValueError, match="its config's `type` defaults to 'example.other', not 'example.mismatched'"
        ):
            get_workflow("example.mismatched")

    def test_a_plugin_whose_config_declares_no_inputs_is_refused(self, plugins: dict[str, Any]) -> None:
        plugins["dataeval_flow.workflows"] = [("example.noinputs", f"{__name__}:NoInputsWorkflow")]
        with pytest.raises(ValueError, match="its config declares no `inputs`"):
            get_workflow("example.noinputs")
