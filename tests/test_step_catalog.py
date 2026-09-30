"""The step catalog: every step described as data, for Flow's CLI and Studio's palette (spec §8)."""

import importlib
import json
import subprocess
import sys
from typing import ClassVar

import dataeval.data
import pytest
from dataeval.data._view import Operation

from dataeval_flow.steps import StepCatalog, list_steps
from dataeval_flow.steps._registry import _BUILTINS as TRANSFORM_BUILTINS
from tests.chain_toys import Keep, register_toys


class _NamedLikeAWorkflow(Keep):
    """A plugin transform that shares the built-in `data-cleaning` workflow's name."""

    name: ClassVar[str] = "data-cleaning"


def test_the_catalog_lists_every_built_in_step_of_every_kind() -> None:
    catalog = list_steps(plugins=False)
    kinds = {(entry.kind, entry.type) for entry in catalog.steps}
    assert {("transform", name) for name in TRANSFORM_BUILTINS} <= kinds
    assert ("evaluator", "duplicates") in kinds
    assert ("evaluator", "label-alignment") in kinds
    assert ("workflow", "data-cleaning") in kinds
    assert catalog.format == 1


def test_the_catalog_orders_steps_by_kind_then_name(plugins) -> None:
    register_toys(plugins)
    steps = [(entry.kind, entry.type) for entry in list_steps().steps]
    assert list(dict.fromkeys(kind for kind, _ in steps)) == ["evaluator", "transform", "combine", "check", "workflow"]
    assert steps.index(("transform", "split")) < steps.index(("transform", "toy-keep"))
    assert steps.index(("transform", "toy-keep")) < steps.index(("transform", "view"))


def test_an_entry_describes_its_ports_and_config() -> None:
    entry = next(e for e in list_steps(plugins=False).steps if e.type == "select")
    assert [(p.port, p.type) for p in entry.inputs] == [("input", "dataset"), ("ranking", "output")]
    assert entry.inputs[1].classes == ["dataeval.scope.PrioritizeOutput"]
    assert entry.outputs[0].type == "dataset"
    assert "ranking" in entry.config_schema["properties"]
    duplicates = next(e for e in list_steps(plugins=False).steps if e.type == "duplicates")
    assert duplicates.inputs[0].count == "1+"
    assert "stats" in duplicates.inputs[0].derives
    assert duplicates.origin == "dataeval-flow"


def test_merge_takes_two_or_more_datasets() -> None:
    merge = next(e for e in list_steps(plugins=False).steps if e.type == "merge")
    assert merge.inputs[0].count == "2+"


def test_a_dataset_port_of_any_kind_says_any_and_other_ports_name_no_kinds() -> None:
    steps = {e.type: e for e in list_steps(plugins=False).steps}
    assert steps["select"].inputs[0].kinds == ["any"]
    assert steps["select"].inputs[1].kinds is None
    assert [port.kinds for port in steps["split"].outputs] == [["any"], ["any"], ["any"]]
    assert steps["export"].inputs[0].kinds == ["object_detection"]
    assert steps["data-analysis"].outputs[0].kinds is None
    assert steps["data-cleaning"].outputs[0].kinds == ["any"]


def test_a_port_writes_the_spec_keys() -> None:
    entry = next(e for e in list_steps(plugins=False).steps if e.type == "kfold")
    assert set(entry.outputs[0].model_dump(mode="json")) == {
        "port",
        "type",
        "kinds",
        "list",
        "count",
        "derives",
        "classes",
    }
    assert entry.outputs[0].model_dump(mode="json")["list"] is True


def test_every_class_the_catalog_names_imports_from_a_public_path() -> None:
    for entry in list_steps(plugins=False).steps:
        for port in (*entry.inputs, *entry.outputs):
            for path in port.classes:
                assert not any(part.startswith("_") for part in path.split(".")), path
                module, _, name = path.rpartition(".")
                assert isinstance(getattr(importlib.import_module(module), name, None), type), path


def test_label_alignment_is_named_by_its_public_home() -> None:
    steps = {e.type: e for e in list_steps(plugins=False).steps}
    expected = ["dataeval_flow.evaluators.scope.LabelAlignmentOutput"]
    assert steps["label-alignment"].outputs[0].classes == expected
    assert steps["conform"].inputs[1].classes == expected


def test_the_catalog_round_trips_as_json() -> None:
    catalog = list_steps(plugins=False)
    assert StepCatalog.model_validate_json(catalog.model_dump_json()) == catalog


def test_a_plugin_step_appears_with_its_origin(plugins) -> None:
    register_toys(plugins)
    entry = next(e for e in list_steps().steps if e.type == "toy-keep")
    assert entry.kind == "transform"
    assert entry.origin == "an unknown package"  # the fixture serves entry points without a distribution


def test_the_cli_prints_the_catalog_as_json() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "dataeval_flow", "steps", "--json"], capture_output=True, text=True, check=True
    )
    catalog = json.loads(out.stdout)
    assert catalog["format"] == 1
    assert "list" in catalog["steps"][0]["inputs"][0]


def test_the_cli_describes_one_step_by_kind_and_name() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "dataeval_flow", "steps", "transform:remove"], capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout)["type"] == "remove"


def test_the_table_lists_each_step_with_its_ports(capsys: pytest.CaptureFixture[str]) -> None:
    from dataeval_flow.__main__ import _list_steps

    assert _list_steps(None, as_json=False) == 0
    lines = capsys.readouterr().out.splitlines()
    select = next(line for line in lines if " select " in line)
    assert select.split()[:2] == ["transform", "select"]
    assert "input, ranking -> output" in select
    merge = next(line for line in lines if " merge " in line)
    assert merge.index("Concatenates") == select.index("Keeps")  # descriptions line up past the ports column


def test_a_name_no_step_of_that_kind_has_exits_one(capsys: pytest.CaptureFixture[str]) -> None:
    from dataeval_flow.__main__ import _list_steps

    assert _list_steps("evaluator:remove", as_json=False) == 1
    assert "'evaluator:remove' names no step" in capsys.readouterr().err


def test_a_name_two_kinds_share_needs_its_kind(plugins, capsys: pytest.CaptureFixture[str]) -> None:
    from dataeval_flow.__main__ import _list_steps

    plugins["dataeval_flow.transforms"] = [("data-cleaning", f"{__name__}:_NamedLikeAWorkflow")]
    assert _list_steps("data-cleaning", as_json=False) == 1
    assert "'data-cleaning' names 2 steps (transform, workflow); name one as KIND:NAME" in capsys.readouterr().err
    assert _list_steps("workflow:data-cleaning", as_json=False) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "workflow"


# Every public dataeval.data name: an Operation (reached through `view`), reached by a named step, or left out.
_REACHED = {
    "View": "view",
    "merge_datasets": "merge",
    "split_dataset": "split, kfold",
    "DatasetSplits": "split, kfold",
    "TrainValSplit": "split, kfold",
    "DetectionCrops": "wrap",
}
_FRAMES = "frame selection for SequenceFrames: waits on a tracking loader, and Keyframes on DataEval 1b"
_VIDEO = "video and tracking: Flow has no tracking loader yet"
_LEFT_OUT = {
    "Operation": "the abstract base of every operation; `view` takes its concrete subclasses, never the base",
    **dict.fromkeys(
        [
            "AllFrames",
            "EvenlySpaced",
            "FrameCandidate",
            "FrameIndices",
            "FrameInput",
            "FrameRate",
            "FrameSelector",
            "FrameVerdict",
            "Redundancy",
            "Representative",
            "SequenceInfo",
            "Stride",
        ],
        _FRAMES,
    ),
    **dict.fromkeys(
        ["SequenceFrames", "VideoSegments", "VideoStitch", "SegmentPlanner", "Cuts", "Window", "build_tracks"],
        _VIDEO,
    ),
    **dict.fromkeys(
        ["SourceItem", "SourceLocator"], "a lookup from an evaluator's address back to its datum, not a transform"
    ),
    "unzip_dataset": "a helper that splits datum tuples; nothing to chain",
}


def test_every_dataeval_data_name_is_reached_or_left_out_on_purpose() -> None:
    unclassified = []
    for name in dataeval.data.__all__:
        value = getattr(dataeval.data, name)
        is_operation = isinstance(value, type) and issubclass(value, Operation) and value is not Operation
        if not (is_operation or name in _REACHED or name in _LEFT_OUT):
            unclassified.append(name)
    assert not unclassified, f"classify in _REACHED (naming the step) or _LEFT_OUT (with a reason): {unclassified}"


def test_every_classified_name_is_still_in_dataeval_data() -> None:
    stale = sorted((set(_REACHED) | set(_LEFT_OUT)) - set(dataeval.data.__all__))
    assert stale == [], f"no longer in dataeval.data; drop them: {stale}"
    assert set(_REACHED).isdisjoint(_LEFT_OUT)


def test_every_step_the_parity_table_names_is_a_built_in_transform() -> None:
    transforms = {e.type for e in list_steps(plugins=False).steps if e.kind == "transform"}
    named = {step for steps in _REACHED.values() for step in steps.split(", ")}
    assert named == {"view", "merge", "split", "kfold", "wrap"}
    assert named <= transforms
