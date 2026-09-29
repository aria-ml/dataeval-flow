"""A custom workflow's own shape, and its round-trip through a config file (spec §3)."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import PipelineConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import CustomWorkflowConfig, InputSlot, StepEntry
from tests.chain_toys import chain_pipeline, register_toys
from tests.evaluator_toys import ToyImages

_WRITTEN = {
    "name": "audit",
    "inputs": ["a", {"name": "rest", "list": True}],
    "steps": [
        {"name": "kept", "transform": "toy-keep", "input": "a"},
        {"name": "few", "transform": "toy-first", "input": "kept", "n": 2},
        {"name": "dupes", "evaluator": "dupes", "input": ["few", "rest"], "extractor": "flat", "optional": True},
    ],
}


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    return plugins


@pytest.mark.usefixtures("toys")
def test_a_workflow_with_steps_is_a_custom_workflow() -> None:
    config = chain_pipeline(workflows=[_WRITTEN], evaluators=[DuplicatesConfig(name="dupes")], extractor=True)
    workflow = (config.workflows or [])[0]
    assert isinstance(workflow, CustomWorkflowConfig)
    assert [slot.name for slot in workflow.single_slots] == ["a"]
    assert workflow.list_slot == InputSlot.model_validate({"name": "rest", "list": True})
    assert [step.kind for step in workflow.steps] == ["transform", "transform", "evaluator"]
    assert workflow.steps[1].config.n == 2  # type: ignore[union-attr]


@pytest.mark.usefixtures("toys")
def test_a_custom_workflow_dumps_as_it_was_written() -> None:
    workflow = CustomWorkflowConfig.model_validate(_WRITTEN)
    assert workflow.model_dump(mode="json") == _WRITTEN


@pytest.mark.usefixtures("toys")
def test_a_pipeline_holding_a_custom_workflow_survives_load_and_save() -> None:
    # No sources are read here (no task runs), so `datasets={}` keeps `DatasetProtocolConfig`'s in-memory
    # dataset — "not serializable" by its own docstring — out of the JSON round-trip this test is about.
    config = chain_pipeline(
        workflows=[_WRITTEN], evaluators=[DuplicatesConfig(name="dupes")], extractor=True, datasets={}
    )
    assert PipelineConfig.model_validate(config.model_dump(mode="json")) == config


@pytest.mark.usefixtures("toys")
def test_an_entry_naming_type_and_steps_is_refused() -> None:
    with pytest.raises(ValidationError, match="names both `type` and `steps`"):
        chain_pipeline(workflows=[{**_WRITTEN, "type": "data-cleaning"}])


def test_an_entry_naming_neither_says_both_forms() -> None:
    with pytest.raises(ValidationError, match=r"needs a `type:`, or `steps:`"):
        PipelineConfig.model_validate({"workflows": [{"name": "x"}]})


@pytest.mark.usefixtures("toys")
@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (["a", "a"], "input 'a' more than once"),
        ([{"name": "r", "list": True}, "a"], "Only the last input may be a list"),
        (["a.b"], "String should match pattern"),
        ([], "at least 1 item"),
    ],
)
def test_malformed_slots_are_refused(inputs: list, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        CustomWorkflowConfig.model_validate({**_WRITTEN, "inputs": inputs})


@pytest.mark.usefixtures("toys")
def test_a_step_named_like_an_input_is_refused() -> None:
    steps = [{"name": "a", "transform": "toy-keep", "input": "a"}]
    with pytest.raises(ValidationError, match="step 'a' shares its name with an input"):
        CustomWorkflowConfig.model_validate({"name": "w", "inputs": ["a"], "steps": steps})


@pytest.mark.usefixtures("toys")
def test_two_steps_of_one_name_are_refused() -> None:
    steps = [{"name": "k", "transform": "toy-keep", "input": "a"}, {"name": "k", "transform": "toy-keep", "input": "k"}]
    with pytest.raises(ValidationError, match="more than one step named 'k'"):
        CustomWorkflowConfig.model_validate({"name": "w", "inputs": ["a"], "steps": steps})


@pytest.mark.usefixtures("toys")
@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"name": "s", "input": "a"}, "names none of"),
        ({"name": "s", "transform": "toy-keep", "evaluator": "dupes", "input": "a"}, "names 2 of"),
        ({"name": "s", "evaluator": "dupes", "input": "a", "keep": "first"}, "move keep to it"),
        ({"name": "s", "transform": "no-such", "input": "a"}, "Unknown transform: 'no-such'"),
        ({"name": "s", "transform": "toy-first", "input": "a", "m": 2}, "Extra inputs are not permitted"),
        ({"name": "s", "check": "rate", "input": "a"}, "No check types are installed"),
    ],
)
def test_a_malformed_step_is_refused(step: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        StepEntry.model_validate(step)


@pytest.mark.usefixtures("toys")
@pytest.mark.parametrize(("sources", "ok"), [(["x"], False), (["x", "y"], True), (["x", "y", "z"], True)])
def test_a_task_must_bind_every_single_slot_and_one_source_to_a_list_slot(sources: list, ok: bool) -> None:
    datasets = {name: ToyImages() for name in sources}
    task = {"name": "t", "workflow": "audit", "sources": sources, "extractor": "flat"}
    build = lambda: chain_pipeline(  # noqa: E731
        workflows=[_WRITTEN],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[task],
        datasets=datasets,
        extractor=True,
    )
    if ok:
        build()
    else:
        with pytest.raises(ValidationError, match="takes one source for 'a' and at least one for 'rest'"):
            build()


@pytest.mark.usefixtures("toys")
def test_to_yaml_writes_a_workflows_fragment() -> None:
    workflow = CustomWorkflowConfig.model_validate(_WRITTEN)
    assert yaml.safe_load(workflow.to_yaml()) == {"workflows": [_WRITTEN]}


@pytest.mark.usefixtures("toys")
def test_save_replaces_the_entry_of_its_name_and_keeps_the_rest(tmp_path: Path) -> None:
    path = tmp_path / "workflows.yaml"
    other = {"name": "other", "type": "data-cleaning", "outlier_method": "zscore", "outlier_flags": ["pixel"]}
    stale = {"name": "audit", "inputs": ["a"], "steps": [{"name": "k", "transform": "toy-keep", "input": "a"}]}
    path.write_text(yaml.safe_dump({"logging": {"app_level": "INFO"}, "workflows": [stale, other]}, sort_keys=False))

    CustomWorkflowConfig.model_validate(_WRITTEN).save(path)

    saved = yaml.safe_load(path.read_text())
    assert saved == {"logging": {"app_level": "INFO"}, "workflows": [_WRITTEN, other]}


@pytest.mark.usefixtures("toys")
def test_save_refuses_a_file_that_is_no_config(tmp_path: Path) -> None:
    path = tmp_path / "compose.yaml"
    path.write_text("services:\n  flow: {}\n")
    with pytest.raises(ValueError, match="not a pipeline config"):
        CustomWorkflowConfig.model_validate(_WRITTEN).save(path)


def test_the_schema_offers_a_custom_workflow_branch() -> None:
    schema = PipelineConfig.model_json_schema()
    assert "CustomWorkflowConfig" in schema["$defs"]
    assert "steps" in schema["$defs"]["CustomWorkflowConfig"]["properties"]
