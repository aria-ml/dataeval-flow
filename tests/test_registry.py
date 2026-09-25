"""The registry: built-ins from the table, plugins from entry points, one set of checks."""

import importlib.util
import json
import logging
import re
import threading
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from pydantic import ValidationError

import dataeval_flow._registry as registry_module
from dataeval_flow import PipelineConfig
from dataeval_flow.evaluators import list_evaluators
from dataeval_flow.workflows import Workflow, WorkflowConfig, WorkflowContext, get_workflow, list_workflows
from tests.example_plugin import BrightnessConfig, CountConfig, CountResult, CountWorkflow

BUILTIN_WORKFLOWS = [
    "data-analysis",
    "data-cleaning",
    "data-coverage",
    "data-prioritization",
    "data-splitting",
    "drift-monitoring",
    "metadata-triage",
    "ood-detection",
    "parameter-sweep",
]


def test_every_builtin_workflow_loads() -> None:
    assert [cls.name for cls in list_workflows()] == BUILTIN_WORKFLOWS


def test_a_plugin_is_listed_and_validates_from_yaml(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [("example.count", "tests.example_plugin:CountWorkflow")]
    assert "example.count" in [cls.name for cls in list_workflows()]
    config = PipelineConfig.model_validate(yaml.safe_load("workflows:\n  - type: example.count\n    minimum: 3\n"))
    assert config.workflows is not None
    assert config.workflows[0].minimum == 3  # type: ignore[attr-defined]


def test_dumping_keeps_a_plugins_own_fields(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [("example.count", "tests.example_plugin:CountWorkflow")]
    config = PipelineConfig.model_validate({"workflows": [{"type": "example.count", "minimum": 3}]})
    dumped = json.loads(config.model_dump_json())
    assert dumped["workflows"][0]["minimum"] == 3
    assert PipelineConfig.model_validate(dumped).workflows[0].minimum == 3  # type: ignore[index, union-attr]


def test_the_schema_has_a_branch_per_type_with_a_const_type(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [("example.count", "tests.example_plugin:CountWorkflow")]
    schema = PipelineConfig.model_json_schema()
    assert schema["$defs"]["CountConfig"]["properties"]["type"]["const"] == "example.count"
    assert schema["$defs"]["DataCleaningConfig"]["properties"]["type"]["const"] == "data-cleaning"


def test_an_unknown_type_lists_the_installed_ones() -> None:
    with pytest.raises(ValidationError, match="data-cleaning"):
        PipelineConfig.model_validate({"workflows": [{"type": "no-such-thing"}]})


def test_an_invalid_entry_is_located_by_its_index() -> None:
    entries = [{"type": "data-splitting"}, {"type": "data-cleaning", "outlier_method": "bogus", "outlier_flags": []}]
    with pytest.raises(ValidationError) as caught:
        PipelineConfig.model_validate({"workflows": entries})
    assert caught.value.errors()[0]["loc"] == ("workflows", 1, "outlier_method")


def test_a_broken_plugin_is_left_out_and_explains_itself(plugins, caplog) -> None:
    plugins["dataeval_flow.workflows"] = [("example.gone", "tests.example_plugin_missing:Nope")]
    with caplog.at_level(logging.WARNING):
        names = [cls.name for cls in list_workflows()]
    assert "example.gone" not in names
    assert "data-cleaning" in names
    assert "example.gone" in caplog.text
    with pytest.raises(ValueError, match="example_plugin_missing"):
        get_workflow("example.gone")


def test_a_plugin_cannot_shadow_a_builtin(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [("data-cleaning", "tests.example_plugin:CountWorkflow")]
    assert get_workflow("data-cleaning").__name__ == "DataCleaningWorkflow"


def test_a_plugin_whose_name_disagrees_with_its_class_is_refused(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [("example.other", "tests.example_plugin:CountWorkflow")]
    with pytest.raises(ValueError, match="example.count"):
        get_workflow("example.other")


def test_an_evaluator_plugin_is_listed(plugins) -> None:
    plugins["dataeval_flow.evaluators"] = [("example.brightness", "tests.example_plugin:BrightnessEvaluator")]
    assert "example.brightness" in [cls.name for cls in list_evaluators()]


def test_a_clash_names_both_sides(plugins, caplog) -> None:
    plugins["dataeval_flow.workflows"] = [("data-cleaning", "tests.example_plugin:CountWorkflow")]
    with caplog.at_level(logging.WARNING):
        list_workflows()
    assert "'data-cleaning' from an unknown package clashes with the one from dataeval-flow" in caplog.text


@pytest.mark.parametrize(
    ("name", "target", "reason"),
    [
        ("x.gone", "tests.example_plugin_missing:Nope", "example_plugin_missing"),
        ("data-cleaning", "dataeval_flow.workflows.data_cleaning._workflow:DataCleaningWorkflow", "not a subclass"),
    ],
)
def test_a_broken_builtin_raises_and_leaves_nothing_half_loaded(name: str, target: str, reason: str) -> None:
    """A built-in failing its checks is a bug: it raises, and no thread is left a half-filled table."""
    from dataeval_flow.evaluators import Evaluator

    registry = registry_module.Registry(
        kind="evaluator",
        group="dataeval_flow.tests.no-such-group",
        base=lambda: Evaluator,
        builtins={
            "quality.duplicates": "dataeval_flow.evaluators.quality._evaluator:DuplicatesEvaluator",
            name: target,
        },
    )
    with pytest.raises(RuntimeError, match=reason):
        registry.list()
    assert registry._loaded is None


class _Nested:
    """Holds a workflow one attribute down, as a dotted entry-point reference names it."""

    CountWorkflow = CountWorkflow


class _Mismatched(Workflow[CountConfig, CountResult]):
    """Registered as `example.mismatched`, but its config configures `example.count`."""

    name: ClassVar[str] = "example.mismatched"
    description: ClassVar[str] = "Its config configures another type."

    def run(self, config: CountConfig, context: WorkflowContext) -> CountResult:
        raise NotImplementedError


class _Abstract(Workflow):  # type: ignore[type-arg]
    """Abstract and unparameterized, so it has no `config_type` for the registry's check to read."""

    name: ClassVar[str] = "example.abstract"
    description: ClassVar[str] = "Never bound to a config."


class _NoInputsConfig(WorkflowConfig[CountResult]):
    """Its `type` matches what it is registered under, but it declares no `inputs`."""

    type: str = "example.noinputs"


class _NoInputs(Workflow[_NoInputsConfig, CountResult]):
    """Registered as `example.noinputs`; its config's `type` matches, but it declares no `inputs`."""

    name: ClassVar[str] = "example.noinputs"
    description: ClassVar[str] = "Its config declares no `inputs`."

    def run(self, config: _NoInputsConfig, context: WorkflowContext) -> CountResult:
        raise NotImplementedError


def _names() -> list[str]:
    return [cls.name for cls in list_workflows()]


def test_a_plugin_looking_up_its_own_registry_while_it_loads_is_refused(plugins, monkeypatch) -> None:
    plugins["dataeval_flow.workflows"] = [
        ("example.reentrant", "tests.reentrant_plugin:DataCleaning"),
        ("example.count", "tests.example_plugin:CountWorkflow"),
    ]
    listed: list[str] = []
    probe = threading.Thread(target=lambda: listed.extend(_names()), daemon=True)
    probe.start()
    probe.join(timeout=60)
    if probe.is_alive():
        # The stuck probe holds the load lock: give teardown's `_reset` a fresh one, so the suite fails, not hangs.
        monkeypatch.setattr(registry_module, "_LOAD_LOCK", threading.RLock())
        pytest.fail("Listing workflows hung on a plugin that looked up the workflow registry while it loaded.")
    assert "example.reentrant" not in listed
    assert {"example.count", "data-cleaning"} <= set(listed)
    with pytest.raises(ValueError, match="looked up the workflow registry while it was loading"):
        get_workflow("example.reentrant")


def test_a_plugin_may_list_another_kind_while_it_loads(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [("example.count", "tests.cross_kind_plugin:CountWorkflow")]
    assert "example.count" in _names()
    from tests.cross_kind_plugin import EVALUATORS_SEEN

    assert "quality.duplicates" in EVALUATORS_SEEN


def test_two_plugins_claiming_one_name_are_both_refused(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [
        ("example.count", "tests.example_plugin:CountWorkflow"),
        ("example.count", f"{__name__}:_Nested.CountWorkflow"),
    ]
    assert "example.count" not in _names()
    both = re.escape(f"claimed by tests.example_plugin:CountWorkflow from an unknown package and {__name__}:_Nested")
    with pytest.raises(ValueError, match=both):
        get_workflow("example.count")


def test_a_dotted_attribute_resolves(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [("example.count", f"{__name__}:_Nested.CountWorkflow")]
    assert get_workflow("example.count") is CountWorkflow


def test_a_plugin_that_is_not_a_workflow_is_refused(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [("example.count", "tests.example_plugin:CountConfig")]
    assert "example.count" not in _names()
    with pytest.raises(ValueError, match="tests.example_plugin:CountConfig is not a subclass of Workflow"):
        get_workflow("example.count")


def test_a_plugin_whose_config_configures_another_type_is_refused(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [("example.mismatched", f"{__name__}:_Mismatched")]
    assert "example.mismatched" not in _names()
    with pytest.raises(ValueError, match="`type` defaults to 'example.count', not 'example.mismatched'"):
        get_workflow("example.mismatched")


def test_a_plugin_whose_check_raises_is_refused_not_raised(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [("example.abstract", f"{__name__}:_Abstract")]
    assert "example.abstract" not in _names()
    with pytest.raises(ValueError, match="failed to load .*config_type"):
        get_workflow("example.abstract")


def test_a_plugin_whose_config_declares_no_inputs_is_refused(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [("example.noinputs", f"{__name__}:_NoInputs")]
    assert "example.noinputs" not in _names()
    with pytest.raises(ValueError, match="its config declares no `inputs`"):
        get_workflow("example.noinputs")


def test_an_evaluator_plugin_validates_dumps_and_round_trips(plugins) -> None:
    plugins["dataeval_flow.evaluators"] = [("example.brightness", "tests.example_plugin:BrightnessEvaluator")]
    config = PipelineConfig.model_validate({"evaluators": [{"type": "example.brightness", "stats": "dim"}]})
    dumped = json.loads(config.model_dump_json())
    assert dumped["evaluators"] == [{"name": "example.brightness", "type": "example.brightness", "stats": "dim"}]
    again = PipelineConfig.model_validate(dumped).evaluators
    assert again is not None
    assert isinstance(again[0], BrightnessConfig)
    assert again[0].stats == "dim"


def test_an_entry_without_a_type_says_so() -> None:
    with pytest.raises(ValidationError, match="Each `workflows:` entry needs a `type:`"):
        PipelineConfig.model_validate({"workflows": [{"name": "x"}]})


def test_the_checked_in_schema_holds_the_builtins_alone(plugins) -> None:
    plugins["dataeval_flow.workflows"] = [("example.count", "tests.example_plugin:CountWorkflow")]
    plugins["dataeval_flow.extractors"] = [("example.mean", "tests.example_plugin:MeanExtractor")]
    script = Path(__file__).resolve().parents[1] / "config" / "sync_schema.py"
    spec = importlib.util.spec_from_file_location("sync_schema", script)
    assert spec is not None
    assert spec.loader is not None
    sync_schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync_schema)
    rendered = sync_schema.render()
    assert "example.count" not in rendered
    assert "example.mean" not in rendered
    assert '"const": "data-cleaning"' in rendered
    assert '"const": "bovw"' in rendered
    installed = PipelineConfig.model_json_schema()["$defs"]
    assert "CountConfig" in installed
    assert "MeanConfig" in installed
