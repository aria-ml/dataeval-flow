"""TC-19-2 — the step catalog: every step a workflow can chain, listed from the command line and from Python."""

from __future__ import annotations

import json

import pytest

from dataeval_flow.evaluators import list_evaluators
from dataeval_flow.steps import StepCatalog, list_checks, list_combines, list_steps, list_transforms
from dataeval_flow.workflows import list_workflows
from verification.functional.chains._cli import invoke
from verification.helpers import run_cli

pytestmark = pytest.mark.required

_TRANSFORMS = {"collect", "conform", "export", "kfold", "merge", "remove", "select", "split", "view", "wrap"}
_COMBINES = {"factor-deviation", "factor-gaps", "factor-predictors", "ood-union", "outliers-by-class"}


def _output(proc) -> str:
    return proc.stdout + proc.stderr


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """``cli(*args)``: the command in this process, with its exit code and output."""
    return lambda *args: invoke(monkeypatch, capsys, *args)


class TestStepCatalogFromPython:
    def test_list_steps_describes_every_registered_step_of_every_kind(self) -> None:
        catalog = list_steps()
        assert isinstance(catalog, StepCatalog)
        by_kind: dict[str, set[str]] = {}
        for entry in catalog.steps:
            by_kind.setdefault(entry.kind, set()).add(entry.type)
        assert by_kind["evaluator"] == {cls.name for cls in list_evaluators()}
        assert by_kind["workflow"] == {cls.name for cls in list_workflows()}
        assert by_kind["transform"] == {cls.name for cls in list_transforms()} == _TRANSFORMS
        assert by_kind["combine"] == {cls.name for cls in list_combines()} == _COMBINES
        assert by_kind["check"] == {cls.name for cls in list_checks()}
        assert len(by_kind["check"]) >= 20

    def test_the_catalog_is_ordered_by_kind_then_name(self) -> None:
        entries = list_steps().steps
        kinds = [entry.kind for entry in entries]
        order = ["evaluator", "transform", "combine", "check", "workflow"]
        assert kinds == sorted(kinds, key=order.index)
        for kind in order:
            names = [entry.type for entry in entries if entry.kind == kind]
            assert names == sorted(names)

    def test_each_entry_gives_its_ports_description_origin_and_settings_schema(self) -> None:
        for entry in list_steps().steps:
            assert entry.description.strip()
            assert entry.origin == "dataeval-flow"
            assert entry.inputs, entry.type
            assert entry.config_schema["type"] == "object"

    def test_a_transform_s_ports_say_what_it_reads_and_makes(self) -> None:
        by_type = {entry.type: entry for entry in list_steps().steps if entry.kind == "transform"}
        assert [port.port for port in by_type["split"].outputs] == ["train", "val", "test"]
        assert [port.port for port in by_type["remove"].inputs] == ["input", "plans"]
        kfold_outputs = {port.port: port.is_list for port in by_type["kfold"].outputs}
        assert kfold_outputs == {"train": True, "val": True, "test": False}

    def test_a_check_names_the_evaluators_it_judges_and_an_evaluator_the_checks_that_judge_it(self) -> None:
        by_key = {(entry.kind, entry.type): entry for entry in list_steps().steps}
        assert by_key[("check", "image-duplicates")].judges == ["duplicates"]
        assert "image-duplicates" in by_key[("evaluator", "duplicates")].judged_by
        assert "class-outliers" in by_key[("combine", "outliers-by-class")].judged_by

    def test_the_catalog_serializes_to_json(self) -> None:
        payload = json.loads(list_steps().model_dump_json(by_alias=True))
        assert payload["format"] == 1
        assert {"dataset", "output", "export", "findings"} == set(payload["data_types"])
        assert len(payload["steps"]) == len(list_steps().steps)


class TestStepCatalogOnTheCommandLine:
    def test_steps_lists_a_line_per_step_with_its_kind_ports_and_description(self) -> None:
        proc = run_cli("steps")
        assert proc.returncode == 0, _output(proc)
        lines = [line.split() for line in proc.stdout.splitlines() if line.strip()]
        listed = {(parts[0], parts[1]) for parts in lines}
        expected = {(entry.kind, entry.type) for entry in list_steps().steps}
        assert listed == expected
        split_line = next(line for line in proc.stdout.splitlines() if line.split()[:2] == ["transform", "split"])
        assert "input -> train, val, test" in split_line

    def test_steps_json_prints_the_whole_catalog(self, cli) -> None:
        proc = cli("steps", "--json")
        assert proc.code == 0, proc.output
        payload = json.loads(proc.stdout)
        assert payload["format"] == 1
        assert {(entry["kind"], entry["type"]) for entry in payload["steps"]} == {
            (entry.kind, entry.type) for entry in list_steps().steps
        }

    def test_steps_with_a_name_prints_that_step_s_ports_title_and_settings_schema(self, cli) -> None:
        proc = cli("steps", "remove")
        assert proc.code == 0, proc.output
        entry = json.loads(proc.stdout)
        assert (entry["kind"], entry["type"], entry["title"]) == ("transform", "remove", "Remove")
        assert [port["port"] for port in entry["inputs"]] == ["input", "plans"]
        assert "plans" in entry["config_schema"]["properties"]

    def test_a_name_two_kinds_share_must_be_written_as_kind_and_name(self, cli) -> None:
        refused = cli("steps", "prioritization")
        assert "names 2 steps (evaluator, workflow); name one as KIND:NAME" in refused.output
        evaluator = json.loads(cli("steps", "evaluator:prioritization").stdout)
        workflow = json.loads(cli("steps", "workflow:prioritization").stdout)
        assert (evaluator["kind"], workflow["kind"]) == ("evaluator", "workflow")

    def test_an_unknown_name_is_reported(self, cli) -> None:
        proc = cli("steps", "does-not-exist")
        assert "'does-not-exist' names no step" in proc.output
