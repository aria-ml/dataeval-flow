"""Presets: workflow types whose settings expand to a chain of steps (spec §10.1).

The literals were measured on the toy datasets: 12 toy images hold one exact-duplicate pair, so 2 of 12 images
(16.7%) sit in exact-duplicate groups, and removing all but the first of each group keeps 11.
"""

import re
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from dataeval_flow import PipelineConfig, run, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._graph import ChainGraph, build_graph
from dataeval_flow._chain._preflight import _slots_reached
from dataeval_flow.steps import ChainResult, CustomWorkflowConfig, list_steps
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._workflow import InputSlot
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
    result = _task(ToyImages(count=12), evaluators=[{"name": "dupes", "type": "outliers", "flags": ["pixel"]}])
    assert result.steps["dupes"].type == "duplicates"


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
            slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)
            outputs: ClassVar[tuple[Port, ...]] = (Port("rate", DataType.FINDINGS),)

            @classmethod
            def chain(cls, config: Any) -> PresetChain:  # noqa: ARG003
                return PresetChain(steps=[])


class _UnregisteredConfig(WorkflowConfig[ChainResult]):
    type: str = "toy-unregistered"


def test_a_workflow_type_no_plugin_registers_is_no_preset() -> None:
    assert preset_of(_UnregisteredConfig(name="x")) is None


def _steps(source: str) -> list[dict[str, Any]]:
    """The toy preset over `source`, then duplicates found again on what it kept, and removed by that finding."""
    return [
        {"name": "cleaning", "workflow": "toy", "input": source},
        {"name": "again", "evaluator": "dupes", "input": "cleaning.kept"},
        {"name": "final", "transform": "remove", "input": "cleaning.kept", "plans": {"again": {}}},
    ]


def _outer(
    steps: list[dict[str, Any]],
    datasets: dict[str, Any],
    *,
    inputs: list[Any] | None = None,
    extractor: bool = False,
) -> PipelineConfig:
    return chain_pipeline(
        workflows=[
            {"name": "toy", "type": "toy-preset"},
            {"name": "outer", "inputs": inputs or ["data"], "steps": steps},
        ],
        evaluators=[{"name": "dupes", "type": "duplicates"}],
        tasks=[{"name": "t", "workflow": "outer", "sources": list(datasets)}],
        datasets=datasets,
        extractor=extractor,
    )


def _graph(config: PipelineConfig) -> ChainGraph:
    workflow = next(w for w in config.workflows or () if isinstance(w, CustomWorkflowConfig) and w.name == "outer")
    return build_graph(workflow, config)


def test_a_preset_step_runs_its_chain_inside_the_custom_workflow() -> None:
    result = run_tasks(_outer(_steps("data"), {"src": ToyImages(count=12)}))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert list(result.steps) == ["cleaning/dupes", "cleaning/rate", "cleaning/kept", "again", "final"]
    assert result.steps["cleaning/kept"].inputs == ["data", "cleaning/dupes"]
    assert [(f.severity, f.brief, f.step) for f in result.findings] == [
        ("warning", "2 exact (16.7%), 0 near (0.0%)", "cleaning/rate")
    ]
    assert result.steps["again"].inputs == ["cleaning.kept"]
    assert len(result.steps["final"].output) == 11
    assert result.steps["final"].details == {
        "removed": {"items": 0, "detections": 0, "tracks": 0, "frames": 0},
        "by_plan": {"again": {}},
    }


def test_a_preset_step_runs_its_whole_chain_once_per_element_of_a_list() -> None:
    config = _outer(
        _steps("splits"),
        {"s1": ToyImages(count=12), "s2": ToyImages(count=24)},
        inputs=[{"name": "splits", "list": True}],
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert [(f.brief, f.step) for f in result.findings] == [
        ("2 exact (16.7%), 0 near (0.0%)", "cleaning/rate[s1]"),
        ("2 exact (8.3%), 0 near (0.0%)", "cleaning/rate[s2]"),
    ]
    kept = result.steps["cleaning/kept"].elements or {}
    assert {key: len(element.output) for key, element in kept.items()} == {"s1": 11, "s2": 23}
    final = result.steps["final"].elements or {}
    assert {key: len(element.output) for key, element in final.items()} == {"s1": 11, "s2": 23}


@pytest.mark.parametrize(
    ("address", "message"),
    [
        ("cleaning.dupes", "Step 'again' reads `cleaning.dupes`, but `cleaning` has outputs kept."),
        (
            "cleaning",
            "Step 'again' reads `cleaning`, but step 'cleaning' has outputs kept: name one, such as `cleaning.kept`.",
        ),
        ("cleaning/dupes", "'cleaning/dupes' is not an address"),
    ],
)
def test_only_a_preset_step_s_declared_outputs_can_be_read(address: str, message: str) -> None:
    steps = [
        {"name": "cleaning", "workflow": "toy", "input": "data"},
        {"name": "again", "evaluator": "dupes", "input": address},
    ]
    with pytest.raises(ValidationError, match=re.escape(message)):
        _outer(steps, {"src": ToyImages(count=4)})


def test_a_preset_step_reads_one_dataset_per_slot() -> None:
    steps = [{"name": "cleaning", "workflow": "toy", "input": ["data", "data"]}]
    message = "Step 'cleaning' runs workflow 'toy' (toy-preset), whose inputs are `data`, but the step names 2."
    with pytest.raises(ValidationError, match=re.escape(message)):
        _outer(steps, {"src": ToyImages(count=4)})


def test_a_preset_step_reads_datasets() -> None:
    steps = [
        {"name": "d", "evaluator": "dupes", "input": "data"},
        {"name": "cleaning", "workflow": "toy", "input": "d"},
    ]
    message = "Step 'cleaning' reads `d`, which is an Output, but workflow 'toy' (toy-preset) reads Datasets."
    with pytest.raises(ValidationError, match=re.escape(message)):
        _outer(steps, {"src": ToyImages(count=4)})


def test_a_preset_step_whose_declared_output_no_step_makes_is_refused() -> None:
    message = "declares output `gone`, but no step of its chain named `gone` makes one Dataset"
    with pytest.raises(ValidationError, match=re.escape(message)):
        chain_pipeline(
            workflows=[
                {"name": "gone", "type": "toy-gone-preset"},
                {"name": "outer", "inputs": ["data"], "steps": [{"name": "g", "workflow": "gone", "input": "data"}]},
            ]
        )


def test_an_optional_preset_step_makes_each_step_of_its_chain_optional() -> None:
    config = _outer([{"name": "cleaning", "workflow": "toy", "input": "data", "optional": True}], {"src": ToyImages()})
    assert [(spec.name, spec.optional) for spec in _graph(config).steps] == [
        ("cleaning/dupes", True),
        ("cleaning/rate", True),
        ("cleaning/kept", True),
    ]


def test_a_preset_step_s_extractor_reaches_each_step_of_its_chain_that_embeds() -> None:
    steps = [{"name": "cleaning", "workflow": "toy", "input": "data", "extractor": "flat"}]
    config = _outer(steps, {"src": ToyImages()}, extractor=True)
    assert [(spec.name, spec.extractor) for spec in _graph(config).steps] == [
        ("cleaning/dupes", "flat"),
        ("cleaning/rate", None),
        ("cleaning/kept", None),
    ]


def test_a_step_reading_a_preset_output_descends_from_the_slot_the_preset_read() -> None:
    graph = _graph(_outer(_steps("data"), {"src": ToyImages()}))
    assert _slots_reached(graph)["again"] == ["data"]
    assert graph.aliases == {"cleaning.kept": "cleaning/kept"}


_POOLED = {"name": "pooled", "type": "toy-pool-preset"}


def _pooled_sources() -> dict[str, Any]:
    return {"ref": ToyImages(count=12), "p1": ToyImages(count=12, seed=1), "p2": ToyImages(count=24, seed=2)}


def test_a_preset_s_list_slot_takes_every_source_after_its_single_slots() -> None:
    sources = _pooled_sources()
    config = chain_pipeline(
        workflows=[_POOLED], tasks=[{"name": "t", "workflow": "pooled", "sources": list(sources)}], datasets=sources
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert result.steps["reference-dupes"].elements is None
    assert list(result.steps["dupes"].elements or {}) == ["p1", "p2"]
    kept = result.steps["kept"].elements or {}
    assert {key: len(element.output) for key, element in kept.items()} == {"p1": 11, "p2": 23}


def test_a_preset_with_a_list_slot_is_refused_a_task_without_a_source_for_it() -> None:
    message = (
        "Task 't' runs workflow 'pooled' (toy-pool-preset), which takes two or more sources, but the task names 1."
    )
    with pytest.raises(ValidationError, match=re.escape(message)):
        chain_pipeline(
            workflows=[_POOLED],
            tasks=[{"name": "t", "workflow": "pooled", "sources": ["ref"]}],
            datasets={"ref": ToyImages(count=4)},
        )


def test_a_preset_step_s_list_slot_reads_a_list() -> None:
    config = chain_pipeline(
        workflows=[
            _POOLED,
            {
                "name": "outer",
                "inputs": ["ref", {"name": "pools", "list": True}],
                "steps": [
                    {"name": "cleaning", "workflow": "pooled", "input": ["ref", "pools"]},
                    {"name": "again", "evaluator": "dupes", "input": "cleaning.kept"},
                ],
            },
        ],
        evaluators=[{"name": "dupes", "type": "duplicates"}],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["ref", "p1", "p2"]}],
        datasets=_pooled_sources(),
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert list(result.steps) == ["cleaning/reference-dupes", "cleaning/dupes", "cleaning/kept", "again"]
    kept = result.steps["cleaning/kept"].elements or {}
    assert {key: len(element.output) for key, element in kept.items()} == {"p1": 11, "p2": 23}
    assert list(result.steps["again"].elements or {}) == ["p1", "p2"]


def test_a_preset_step_s_list_slot_is_refused_one_dataset() -> None:
    steps = [{"name": "cleaning", "workflow": "pooled", "input": ["ref", "ref"]}]
    message = "Step 'cleaning' binds `ref` to `pools`, which takes a list of Datasets, but `ref` is one Dataset."
    with pytest.raises(ValidationError, match=re.escape(message)):
        chain_pipeline(
            workflows=[_POOLED, {"name": "outer", "inputs": ["ref"], "steps": steps}],
            tasks=[{"name": "t", "workflow": "outer", "sources": ["ref"]}],
            datasets={"ref": ToyImages(count=4)},
        )


def test_only_a_preset_s_last_slot_may_be_a_list() -> None:
    with pytest.raises(TypeError, match=re.escape("only a preset's last slot may take a list of sources")):

        class _ListFirst(Preset, Workflow[ToyPresetConfig, ChainResult]):
            name: ClassVar[str] = "toy-list-first"
            description: ClassVar[str] = "A list slot ahead of a single one."
            slots: ClassVar[tuple[str | InputSlot, ...]] = (
                InputSlot.model_validate({"name": "pools", "list": True}),
                "reference",
            )

            @classmethod
            def chain(cls, config: Any) -> PresetChain:  # noqa: ARG003
                return PresetChain(steps=[])


def test_a_preset_step_names_a_list_slot_as_a_list_when_its_count_is_off() -> None:
    outer = {
        "name": "outer",
        "inputs": ["ref", {"name": "pools", "list": True}],
        "steps": [{"name": "cleaning", "workflow": "pooled", "input": ["ref", "pools", "pools"]}],
    }
    message = (
        "Step 'cleaning' runs workflow 'pooled' (toy-pool-preset), whose inputs are `reference`, `pools` (a list), "
        "but the step names 3."
    )
    with pytest.raises(ValidationError, match=re.escape(message)):
        chain_pipeline(
            workflows=[_POOLED, outer],
            tasks=[{"name": "t", "workflow": "outer", "sources": ["ref", "p1"]}],
            datasets=_pooled_sources(),
        )


def _reading(preset: str, address: str, **entry: Any) -> PipelineConfig:
    """A custom workflow running `preset` as step `s` over 12 toy images, then finding duplicates on `address`."""
    return chain_pipeline(
        workflows=[
            {"name": "p", "type": preset, **entry},
            {
                "name": "outer",
                "inputs": ["data"],
                "steps": [
                    {"name": "s", "workflow": "p", "input": "data"},
                    {"name": "again", "evaluator": "dupes", "input": address},
                ],
            },
        ],
        evaluators=[{"name": "dupes", "type": "duplicates"}],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["src"]}],
        datasets={"src": ToyImages(count=12)},
    )


def test_a_preset_output_reads_the_output_of_a_step_with_several() -> None:
    config = _reading("toy-split-preset", "s.train")
    assert _graph(config).aliases == {"s.train": "s/parts.train", "s.val": "s/parts.val", "s.test": "s/parts.test"}
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert result.steps["again"].inputs == ["s.train"]
    assert len(result.steps["s/parts"].output["train"]) == 9


def test_a_preset_output_its_settings_leave_empty_is_refused() -> None:
    message = "Step 'again' reads `s.val`, which step 's' leaves empty with its settings."
    with pytest.raises(ValidationError, match=re.escape(message)):
        _reading("toy-split-preset", "s.val")


def test_a_preset_output_read_from_a_list_is_a_list_with_its_keys() -> None:
    result = run_tasks(_reading("toy-fold-preset", "s.train"))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert list(result.steps["again"].elements or {}) == ["0", "1"]


def test_a_preset_output_mapped_to_no_dataset_is_refused() -> None:
    message = "declares output `train` at `labels`, but `labels` is no Dataset its chain makes"
    with pytest.raises(ValidationError, match=re.escape(message)):
        _reading("toy-fold-preset", "s.train", to_labels=True)


def test_a_preset_with_list_outputs_over_a_list_is_refused() -> None:
    with pytest.raises(ValidationError, match="lists do not nest"):
        chain_pipeline(
            workflows=[
                {"name": "p", "type": "toy-fold-preset"},
                {
                    "name": "outer",
                    "inputs": [{"name": "all", "list": True}],
                    "steps": [{"name": "s", "workflow": "p", "input": "all"}],
                },
            ],
            tasks=[{"name": "t", "workflow": "outer", "sources": ["a", "b"]}],
            datasets={"a": ToyImages(count=12), "b": ToyImages(count=12)},
        )
