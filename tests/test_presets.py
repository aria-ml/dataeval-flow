"""Presets: workflow types whose settings expand to a chain of steps (spec §10.1).

The literals were measured on the toy datasets: 12 toy images hold one exact-duplicate pair, so 2 of 12 images
(16.7%) sit in exact-duplicate groups, and removing all but the first of each group keeps 11.
"""

from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from dataeval_flow import run, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult, list_steps
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.workflows import Workflow, WorkflowConfig
from dataeval_flow.workflows._preset import Preset, PresetChain, preset_of
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages
from tests.preset_toys import ToyPreset, ToyPresetConfig, register_presets


@pytest.fixture(autouse=True)
def presets(plugins):
    register_presets(plugins)
    DatasetCache.clear_instances()
    yield plugins
    DatasetCache.clear_instances()


def _task(dataset: Any, *, evaluators: list[dict[str, Any]] | None = None, **settings: Any) -> ChainResult:
    config = chain_pipeline(
        workflows=[{"name": "toy", "type": "toy-preset", **settings}],
        evaluators=evaluators or [],
        tasks=[{"name": "t", "workflow": "toy", "sources": ["src"]}],
        datasets={"src": dataset},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def test_a_preset_task_returns_a_chain_result_typed_by_its_preset() -> None:
    result = _task(ToyImages(count=12))
    assert result.type == "toy-preset"
    assert result.metadata.workflow == "toy"
    assert list(result.steps) == ["dupes", "rate", "kept"]
    assert [(f.severity, f.title, f.brief, f.step) for f in result.findings] == [
        ("warning", "Duplicates", "2 exact (16.7%), 0 near (0.0%)", "rate")
    ]
    assert len(result.steps["kept"].output) == 11


def test_a_preset_s_settings_reach_its_steps() -> None:
    result = _task(ToyImages(count=12), exact=None)
    assert [(f.severity, f.brief) for f in result.findings] == [("info", "2 exact (16.7%), 0 near (0.0%)")]


def test_a_preset_s_envelope_records_its_entry() -> None:
    workflow = _task(ToyImages(count=12), exact=None).metadata.resolved_config["workflow"]
    assert (workflow["name"], workflow["type"], workflow["exact"]) == ("toy", "toy-preset", None)


def test_a_preset_s_own_evaluators_are_found_before_the_pipeline_s() -> None:
    result = _task(ToyImages(count=12), evaluators=[{"name": "dupes", "type": "quality.outliers", "flags": ["pixel"]}])
    assert result.steps["dupes"].type == "quality.duplicates"


def test_run_takes_a_preset_config() -> None:
    result = run(ToyPresetConfig(), ToyImages(count=12))
    assert isinstance(result, ChainResult)
    assert result.type == "toy-preset"
    assert len(result.steps["kept"].output) == 11


def test_a_preset_is_refused_the_sources_its_type_refuses() -> None:
    with pytest.raises(ValidationError, match=r"Task 't' runs workflow 'toy' \(toy-preset\)"):
        chain_pipeline(
            workflows=[{"name": "toy", "type": "toy-preset"}],
            tasks=[{"name": "t", "workflow": "toy", "sources": ["a", "b"]}],
            datasets={"a": ToyImages(count=4), "b": ToyImages(count=4)},
        )


def test_a_preset_s_chain_is_checked_when_the_config_loads() -> None:
    with pytest.raises(ValidationError, match="reads `nowhere`"):
        chain_pipeline(workflows=[{"name": "broken", "type": "toy-broken-preset"}])


def test_a_preset_s_own_run_is_refused() -> None:
    with pytest.raises(TypeError, match="ToyPreset is a preset: Flow runs its chain of steps"):
        ToyPreset().run(ToyPresetConfig(), None)  # type: ignore[arg-type]


def test_the_catalog_lists_a_preset_s_declared_outputs() -> None:
    entry = next(e for e in list_steps().steps if e.type == "toy-preset")
    assert [(port.port, port.type) for port in entry.outputs] == [("kept", DataType.DATASET)]


def test_a_preset_must_declare_its_slots() -> None:
    with pytest.raises(TypeError, match="must declare `slots`"):

        class _NoSlots(Preset, Workflow[ToyPresetConfig, ChainResult]):
            name: ClassVar[str] = "toy-preset"
            description: ClassVar[str] = "No slots."

            @classmethod
            def chain(cls, config: Any) -> PresetChain:  # noqa: ARG003
                return PresetChain(steps=[])


def test_a_preset_s_outputs_are_datasets() -> None:
    with pytest.raises(TypeError, match="declares outputs rate, which are not Datasets"):

        class _Findings(Preset, Workflow[ToyPresetConfig, ChainResult]):
            name: ClassVar[str] = "toy-preset"
            description: ClassVar[str] = "Declares findings."
            slots: ClassVar[tuple[str, ...]] = ("data",)
            outputs: ClassVar[tuple[Port, ...]] = (Port("rate", DataType.FINDINGS),)

            @classmethod
            def chain(cls, config: Any) -> PresetChain:  # noqa: ARG003
                return PresetChain(steps=[])


class _UnregisteredConfig(WorkflowConfig[ChainResult]):
    type: str = "toy-unregistered"


def test_a_workflow_type_no_plugin_registers_is_no_preset() -> None:
    assert preset_of(_UnregisteredConfig(name="x")) is None
