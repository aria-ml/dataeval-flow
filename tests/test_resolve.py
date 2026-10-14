"""Resolving a pipeline for an editor: graphs, origins, availability and located problems (builder spec §5.3, §5.4)."""

from typing import Any

import pytest

from dataeval_flow._cache import DatasetCache
from dataeval_flow._orchestrator import select_tasks
from dataeval_flow._service._resolve import resolve
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.evaluators.scope import LabelAlignmentConfig
from tests.chain_toys import ToyDetections, chain_pipeline, register_toys, run_chain_task
from tests.evaluator_toys import ToyImages
from tests.test_graph_diagnostics import _ONTOLOGY

pytestmark = pytest.mark.usefixtures("toys")


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    DatasetCache.clear_instances()
    yield plugins
    DatasetCache.clear_instances()


def _config(workflows: list[Any], tasks: list[dict[str, Any]], *, graphs: bool = False, **kwargs: Any):
    evaluators = [DuplicatesConfig(name="dupes"), *kwargs.pop("evaluators", [])]
    return chain_pipeline(workflows=workflows, tasks=tasks, evaluators=evaluators, graphs=graphs, **kwargs)


def _resolve(workflows: list[Any], tasks: list[dict[str, Any]], **kwargs: Any):
    config = _config(workflows, tasks, **kwargs)
    return resolve(config, select_tasks(config, None))


def _nodes(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["node_id"]: node for node in graph["nodes"]}


def _one(steps: list[dict[str, Any]], inputs: list[Any] | None = None) -> list[dict[str, Any]]:
    return [{"name": "w", "inputs": inputs or ["a"], "steps": steps}]


_TASK = [{"name": "t", "workflow": "w", "sources": "src"}]
_QUALITY = {"outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}


def test_a_chain_resolves_to_its_steps_ports_and_origins() -> None:
    steps = [
        {"name": "kept", "transform": "toy-keep", "input": "a"},
        {"name": "dupes", "evaluator": "dupes", "input": "kept"},
    ]
    found = _resolve(_one(steps), _TASK)
    assert found.diagnostics == []
    assert found.workflows == []
    (task,) = found.tasks
    assert (task["complete"], task["execution_order"]) == (True, ["t:kept", "t:dupes"])
    assert task["source_bindings"] == [
        {"slot": "a", "list": False, "sources": ["src"], "document_pointer": "/tasks/0/sources"}
    ]
    kept, dupes = _nodes(task)["t:kept"], _nodes(task)["t:dupes"]
    assert kept["path"] == ["t", "kept"]
    assert kept["origin"] == {"step_pointer": "/workflows/0/steps/0", "definition_pointer": None, "generated": False}
    assert dupes["origin"] == {
        "step_pointer": "/workflows/0/steps/1",
        "definition_pointer": "/evaluators/0",
        "generated": False,
    }
    assert [(port["port"], port["shape"], port["addresses"]) for port in kept["inputs"]] == [("input", "scalar", ["a"])]
    (output,) = dupes["outputs"]
    assert (output["address"], output["classes"]) == ("dupes", ["dataeval.quality.DuplicatesOutput"])
    assert (output["availability"], output["reason"], output["keys"]) == ("available", None, None)


def test_an_output_the_settings_leave_empty_is_unavailable() -> None:
    found = _resolve(_one([{"name": "s", "transform": "split", "input": "a", "test_frac": 0.2, "val_frac": 0}]), _TASK)
    outputs = {port["port"]: port for port in _nodes(found.tasks[0])["t:s"]["outputs"]}
    assert (outputs["val"]["address"], outputs["val"]["availability"]) == ("s.val", "unavailable")
    assert outputs["val"]["reason"] == "`s`'s settings leave `val` empty."
    assert outputs["train"]["availability"] == "available"


def test_a_list_slot_is_keyed_by_the_tasks_sources() -> None:
    steps = [
        {"name": "kept", "transform": "toy-keep", "input": "all"},
        {"name": "one", "transform": "toy-keep", "input": "kept[more]"},
    ]
    found = _resolve(
        _one(steps, [{"name": "all", "list": True}]),
        [{"name": "t", "workflow": "w", "sources": ["src", "more"]}],
        datasets={"src": ToyImages(), "more": ToyImages(seed=1)},
    )
    task = found.tasks[0]
    assert task["source_bindings"] == [
        {"slot": "all", "list": True, "sources": ["src", "more"], "document_pointer": "/tasks/0/sources"}
    ]
    kept = _nodes(task)["t:kept"]
    assert kept["broadcast"] is True
    assert (kept["outputs"][0]["list"], kept["outputs"][0]["keys"]) == (True, ["src", "more"])
    assert _nodes(task)["t:one"]["inputs"][0]["addresses"] == ["kept[more]"]


def test_a_preset_step_resolves_to_generated_steps_its_entry_defines() -> None:
    workflows = [
        *_one([{"name": "q", "workflow": "quality", "input": "a"}]),
        {"name": "quality", "type": "quality", **_QUALITY},
    ]
    found = _resolve(workflows, _TASK)
    assert found.diagnostics == []
    nodes = found.tasks[0]["nodes"]
    assert nodes
    assert all(node["path"][:2] == ["t", "q"] for node in nodes)
    assert all(
        node["origin"]
        == {"step_pointer": "/workflows/0/steps/0", "definition_pointer": "/workflows/1", "generated": True}
        for node in nodes
    )
    ids = {node["node_id"] for node in nodes}
    assert all(alias["node_id"] in ids for alias in found.tasks[0]["output_aliases"])
    assert [graph["name"] for graph in found.workflows] == ["quality"]


def test_a_preset_run_as_a_task_resolves_to_its_chain() -> None:
    found = _resolve(
        [{"name": "quality", "type": "quality", **_QUALITY}], [{"name": "t", "workflow": "quality", "sources": "src"}]
    )
    nodes = found.tasks[0]["nodes"]
    assert nodes
    assert all(
        node["origin"] == {"step_pointer": None, "definition_pointer": "/workflows/0", "generated": True}
        for node in nodes
    )


def test_an_evaluator_task_resolves_to_one_step() -> None:
    found = _resolve([], [{"name": "t", "evaluator": "dupes", "sources": "src"}])
    (node,) = found.tasks[0]["nodes"]
    assert node["node_id"] == "t:t"
    assert node["origin"] == {"step_pointer": None, "definition_pointer": "/evaluators/0", "generated": False}


def test_an_output_from_another_dataset_is_located_on_its_port() -> None:
    steps = [
        {"name": "few", "transform": "view", "input": "a", "operations": [{"type": "Limit", "params": {"size": 2}}]},
        {"name": "aligned", "evaluator": "align", "input": "few"},
        {"name": "c", "transform": "conform", "input": "a", "alignment": "aligned"},
    ]
    found = _resolve(
        _one(steps),
        _TASK,
        evaluators=[LabelAlignmentConfig(name="align", ontology="vehicles")],
        datasets={"src": ToyDetections([[0], [1]], {0: "car", 1: "person"}, dataset_id="d")},
        extra={"ontologies": [_ONTOLOGY]},
    )
    assert found.diagnostics == [
        {
            "code": "wrong_dataset",
            "severity": "error",
            "message": (
                "Step 'c' reads `aligned`, which was computed on `few`, not on `a`: it applies only to the Dataset it "
                "was computed on."
            ),
            "document_pointer": "/workflows/0/steps/2/alignment",
            "task": "t",
            "node_id": "t:c",
            "port": "alignment",
        }
    ]
    assert (found.tasks[0]["complete"], found.tasks[0]["execution_order"]) == (False, ["t:few", "t:aligned"])


def test_an_entry_error_is_reported_once_per_task_that_runs_it() -> None:
    found = _resolve(
        _one([{"name": "k", "transform": "toy-keep", "input": "nothere"}]),
        [{"name": "t1", "workflow": "w", "sources": "src"}, {"name": "t2", "workflow": "w", "sources": "src"}],
    )
    assert [(item["task"], item["node_id"], item["code"]) for item in found.diagnostics] == [
        ("t1", "t1:k", "unknown_address"),
        ("t2", "t2:k", "unknown_address"),
    ]
    assert all(task["complete"] is False and task["nodes"] == [] for task in found.tasks)
    assert found.workflows == []


def test_a_workflow_no_selected_task_runs_is_resolved_on_its_own() -> None:
    good = {"name": "w", "inputs": ["a"], "steps": [{"name": "k", "transform": "toy-keep", "input": "a"}]}
    draft = {"name": "draft", "inputs": ["a"], "steps": [{"name": "j", "transform": "toy-keep", "input": "k"}]}
    found = _resolve([good, draft], _TASK)
    (graph,) = found.workflows
    assert (graph["name"], graph["complete"], graph["nodes"]) == ("draft", False, [])
    assert [(item["task"], item["node_id"], item["document_pointer"]) for item in found.diagnostics] == [
        (None, "draft:j", "/workflows/1/steps/0/input")
    ]


def test_a_key_the_tasks_sources_do_not_bind_is_a_task_problem() -> None:
    found = _resolve(
        _one([{"name": "k", "transform": "toy-keep", "input": "all[nope]"}], [{"name": "all", "list": True}]),
        [{"name": "t", "workflow": "w", "sources": ["src"]}],
    )
    (problem,) = found.diagnostics
    assert (problem["code"], problem["task"], problem["document_pointer"]) == ("task_problem", "t", "/tasks/0")
    assert found.tasks[0]["complete"] is False


def test_a_matrix_task_resolves_once_and_lists_its_runs() -> None:
    found = _resolve(
        _one([{"name": "few", "transform": "toy-first", "input": "a", "n": 2}]),
        [{"name": "t", "workflow": "w", "sources": "src", "matrix": {"steps.few.n": [2, 3]}}],
    )
    task = found.tasks[0]
    assert task["execution_order"] == ["t:few"]
    assert [(run["number"], run["values"]) for run in task["matrix"]["runs"]] == [
        (1, {"steps.few.n": 2}),
        (2, {"steps.few.n": 3}),
    ]


def test_resolved_nodes_are_the_steps_a_run_records() -> None:
    steps = [
        {"name": "kept", "transform": "toy-keep", "input": "a"},
        {"name": "dupes", "evaluator": "dupes", "input": "kept"},
    ]
    config = _config(_one(steps), _TASK, graphs=True)
    nodes = resolve(config, select_tasks(config, None)).tasks[0]["nodes"]
    assert ["/".join(node["path"][1:]) for node in nodes] == list(run_chain_task(config, "t").steps)


def test_a_matrix_task_keeps_its_unkeyed_graph() -> None:
    found = _resolve(
        _one([{"name": "k", "transform": "toy-keep", "input": "all[more]"}], [{"name": "all", "list": True}]),
        [{"name": "t", "workflow": "w", "sources": ["src"], "matrix": {"sources": [["src", "more"]]}}],
        datasets={"src": ToyImages(), "more": ToyImages(seed=1)},
    )
    assert found.diagnostics == []
    assert found.tasks[0]["complete"] is True
