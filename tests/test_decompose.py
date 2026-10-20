"""Preparing a preset's decomposition: a whole candidate pipeline, or why not (preset decomposition spec §5)."""

from typing import Any

import pytest

from dataeval_flow._service import _contract
from dataeval_flow._service._decompose import (
    DecomposeError,
    DecomposeReady,
    DecomposeRequest,
    DecomposeUnsupported,
    prepare,
)
from dataeval_flow.config import PipelineConfig
from tests.chain_toys import chain_pipeline
from tests.test_audit_preset import _OUTLIERS
from tests.test_example_plugin import EXAMPLES
from tests.test_naming_conventions import _MINIMAL

_QUALITY = {"name": "q", "type": "quality", **_MINIMAL["quality"]}
_CLEAN = {"name": "clean", "workflow": "q", "sources": ["src"]}
_SWEEP = {
    "name": "sweep",
    "workflow": "q",
    "sources": ["src"],
    "matrix": {"outliers.outlier_threshold": ["zscore", "iqr"]},
}


# A dataset on disk, not in memory: a candidate is a JSON document, as a service request's pipeline is.
_ON_DISK = {
    "datasets": [{"name": "src_data", "format": "coco", "path": "src"}],
    "sources": [{"name": "src", "dataset": "src_data"}],
}


def _pipeline(workflows: list[Any] | None = None, tasks: list[Any] | None = None, **extra: Any) -> PipelineConfig:
    return chain_pipeline(
        workflows=workflows if workflows is not None else [_QUALITY],
        tasks=tasks if tasks is not None else [_CLEAN],
        datasets={},
        extra=_ON_DISK,
        graphs=False,
        **extra,
    )


def _copy(pipeline: PipelineConfig, **request: Any) -> DecomposeReady | DecomposeUnsupported:
    body = {
        "pipeline": {},
        "operation": "copy_definition",
        "target_pointer": "/workflows/0",
        "new_name": "qc",
        **request,
    }
    return prepare(pipeline, DecomposeRequest.model_validate(body))


def _ready(answer: DecomposeReady | DecomposeUnsupported) -> DecomposeReady:
    assert isinstance(answer, DecomposeReady), answer
    return answer


def test_a_copy_is_a_whole_valid_pipeline_beside_the_preset() -> None:
    pipeline = _pipeline()
    answer = _ready(_copy(pipeline))
    assert (answer.status, answer.valid, answer.diagnostics) == ("ready", True, [])
    assert answer.schema_fingerprint == _contract.schema_fingerprint()
    workflows = answer.pipeline["workflows"]
    assert [w["name"] for w in workflows] == ["q", "qc"]
    assert workflows[0] == pipeline.model_dump(mode="json", by_alias=True)["workflows"][0]
    copied = workflows[1]
    assert (copied["derived_from"][0]["type"], copied["derived_from"][0]["entry"]) == ("quality", "q")
    evaluators = {e["name"] for e in answer.pipeline["evaluators"]}
    assert set(answer.names.evaluators.values()) <= evaluators
    assert all(name.startswith("qc-") for name in answer.names.evaluators.values())
    assert {s["evaluator"] for s in copied["steps"] if "evaluator" in s} == set(answer.names.evaluators.values())
    assert answer.names.steps == {s["name"]: s["name"] for s in copied["steps"]}
    assert answer.pipeline["tasks"][0]["workflow"] == "q"
    assert [c.pointer for c in answer.changes][0] == "/workflows/1"


@pytest.mark.parametrize(
    "tasks",
    [{"tasks": []}, {}, {"tasks": [{**_CLEAN, "enabled": False}]}],
    ids=["no-tasks", "no-tasks-key", "all-disabled"],
)
def test_a_draft_running_no_task_still_gets_a_copy(tasks: dict[str, Any]) -> None:
    pipeline = PipelineConfig.model_validate({**_ON_DISK, "workflows": [_QUALITY], **tasks}, context={"graphs": False})
    answer = _ready(_copy(pipeline))
    assert (answer.status, answer.valid) == ("ready", True)


def test_a_disabled_task_reassigned_is_resolved_running_the_copy() -> None:
    answer = _ready(_copy(_pipeline(tasks=[{**_CLEAN, "enabled": False}]), reassign_tasks=["clean"]))
    assert answer.pipeline["tasks"][0]["workflow"] == "qc"
    assert answer.valid


def test_copied_evaluators_take_fresh_names_and_leave_shared_ones_alone() -> None:
    from dataeval_flow.evaluators.quality import OutliersConfig

    shared = [OutliersConfig(name="outliers", flags=["visual"]), OutliersConfig(name="qc-outliers", flags=["visual"])]
    pipeline = _pipeline(evaluators=shared)
    answer = _ready(_copy(pipeline))
    assert answer.names.evaluators["outliers"] == "qc-outliers-2"
    before = pipeline.model_dump(mode="json", by_alias=True)["evaluators"]
    assert answer.pipeline["evaluators"][: len(before)] == before


def test_a_reassigned_task_runs_the_copy() -> None:
    answer = _ready(_copy(_pipeline(), reassign_tasks=["clean"]))
    assert answer.pipeline["tasks"][0]["workflow"] == "qc"
    assert ("replace", "/tasks/0/workflow") in {(c.op, c.pointer) for c in answer.changes}
    assert answer.valid


def test_reassigning_a_task_that_sweeps_the_preset_is_refused_naming_each_key() -> None:
    sweep = {**_SWEEP, "matrix": {"outliers.outlier_threshold": ["zscore", "iqr"], "outliers.flags": [["pixel"]]}}
    answer = _copy(_pipeline(tasks=[_CLEAN, sweep]), reassign_tasks=["sweep"])
    assert isinstance(answer, DecomposeUnsupported)
    assert {(r.code, r.category) for r in answer.reasons} == {("unsupported_matrix_rewrite", "scope_limit")}
    assert {r.details["key"] for r in answer.reasons} == {"outliers.outlier_threshold", "outliers.flags"}
    assert {r.details["key"]: r.document_pointer for r in answer.reasons} == {
        "outliers.outlier_threshold": "/tasks/1/matrix/outliers.outlier_threshold",
        "outliers.flags": "/tasks/1/matrix/outliers.flags",
    }


def test_a_reason_for_a_key_in_a_list_of_grids_points_at_its_grid() -> None:
    sweep = {**_SWEEP, "matrix": [{"outliers.flags": [["pixel"]]}, {"outliers.outlier_threshold": ["iqr"]}]}
    answer = _copy(_pipeline(tasks=[sweep]), reassign_tasks=["sweep"])
    assert isinstance(answer, DecomposeUnsupported)
    assert [r.document_pointer for r in answer.reasons] == [
        "/tasks/0/matrix/0/outliers.flags",
        "/tasks/0/matrix/1/outliers.outlier_threshold",
    ]


def test_copying_without_reassigning_a_sweep_keeps_it() -> None:
    answer = _ready(_copy(_pipeline(tasks=[_CLEAN, _SWEEP])))
    assert (
        answer.pipeline["tasks"][1]["matrix"]
        == _pipeline(tasks=[_CLEAN, _SWEEP]).model_dump(mode="json", by_alias=True)["tasks"][1]["matrix"]
    )


def test_an_unrelated_problem_stays_a_diagnostic() -> None:
    broken = {
        "name": "broken",
        "inputs": ["data"],
        "steps": [{"name": "judge", "check": "image-duplicates", "input": "missing"}],
    }
    answer = _ready(_copy(_pipeline(workflows=[_QUALITY, broken])))
    assert answer.valid is False
    assert [d["document_pointer"] for d in answer.diagnostics] == ["/workflows/1/steps/0/input"]


def test_a_preset_run_as_a_step_elsewhere_stays_the_preset() -> None:
    outer = {"name": "outer", "inputs": ["data"], "steps": [{"name": "cleaning", "workflow": "q", "input": "data"}]}
    answer = _ready(_copy(_pipeline(workflows=[_QUALITY, outer])))
    assert answer.pipeline["workflows"][1]["steps"][0]["workflow"] == "q"


def test_a_pipeline_with_no_evaluators_gains_the_copies() -> None:
    pipeline = _pipeline()
    assert not pipeline.evaluators
    assert _ready(_copy(pipeline)).pipeline["evaluators"]


def test_a_copy_explains_its_structural_choices() -> None:
    audit = {"name": "a", "type": "audit", **_OUTLIERS}
    answer = _ready(_copy(_pipeline(workflows=[audit], tasks=[{"name": "t", "workflow": "a", "sources": ["src"]}])))
    method = next(d for d in answer.structure if d.setting == "coverage.method")
    assert (method.pointer, method.value, method.written, method.effective_value) == (
        "/workflows/0/coverage/method",
        None,
        False,
        "adaptive",
    )
    assert method.excluded_steps == ["uncovered-items"]
    assert "not included" in method.explanation
    assert answer.pipeline["workflows"][1]["verdict"]["blocking"] == ["leakage", "untrained-classes"]


def test_a_copy_gains_only_the_evaluators_its_steps_name() -> None:
    scope = {"name": "s", "type": "scope", "completeness": False}
    tasks = [{"name": "t", "workflow": "s", "sources": ["src"]}]
    answer = _ready(_copy(_pipeline(workflows=[scope], tasks=tasks)))
    assert "completeness" not in answer.names.evaluators
    assert not any("completeness" in e["name"] for e in answer.pipeline["evaluators"])
    assert not any("completeness" in c.summary for c in answer.changes)
    named = {s["evaluator"] for s in answer.pipeline["workflows"][1]["steps"] if "evaluator" in s}
    assert set(answer.names.evaluators.values()) == named


def test_a_copy_names_what_it_represents_differently() -> None:
    splits = {"name": "s", "type": "splits"}
    answer = _ready(_copy(_pipeline(workflows=[splits], tasks=[{"name": "t", "workflow": "s", "sources": ["src"]}])))
    assert {c.code for c in answer.representational_changes} == {"result_type", "outputs_not_carried"}


@pytest.mark.parametrize(
    ("request_", "loc", "message"),
    [
        ({"new_name": "x"}, ["body", "new_name"], "takes no `new_name`"),
        ({"reassign_tasks": ["clean"]}, ["body", "reassign_tasks"], "takes no `reassign_tasks`"),
        ({"target_pointer": "/workflows/1"}, ["body", "target_pointer"], "/steps/<index>"),
        ({"target_pointer": "/workflows/1/steps/00"}, ["body", "target_pointer"], "/steps/<index>"),
        ({"target_pointer": "/workflows/1/steps/0\n"}, ["body", "target_pointer"], "/steps/<index>"),
        ({"target_pointer": "/workflows/0/steps/0"}, ["body", "target_pointer"], "no step of a custom workflow"),
        ({"target_pointer": "/workflows/1/steps/1"}, ["body", "target_pointer"], "runs no preset"),
    ],
    ids=["new-name", "reassign", "no-step", "leading-zero", "trailing-newline", "a-preset", "no-preset-step"],
)
def test_an_inline_request_it_cannot_prepare_is_refused_where_it_is_wrong(
    request_: dict[str, Any], loc: list[Any], message: str
) -> None:
    outer = {
        "name": "outer",
        "inputs": ["data"],
        "steps": [
            {"name": "cleaning", "workflow": "q", "input": "data"},
            {"name": "judge", "check": "image-duplicates", "input": "data"},
        ],
    }
    body = {"pipeline": {}, "operation": "inline_step", "target_pointer": "/workflows/1/steps/0", **request_}
    with pytest.raises(DecomposeError, match=message) as raised:
        prepare(_pipeline(workflows=[_QUALITY, outer]), DecomposeRequest.model_validate(body))
    assert raised.value.loc == loc


@pytest.mark.parametrize(
    ("request_", "loc", "message"),
    [
        ({"target_pointer": "/workflows/9"}, ["body", "target_pointer"], "no workflow"),
        ({"target_pointer": "/tasks/0"}, ["body", "target_pointer"], "/workflows/<index>"),
        ({"target_pointer": "/workflows/00"}, ["body", "target_pointer"], "/workflows/<index>"),
        ({"new_name": None}, ["body", "new_name"], "needs `new_name`"),
        ({"new_name": "q"}, ["body", "new_name"], "exists"),
        ({"reassign_tasks": ["nope"]}, ["body", "reassign_tasks", 0], "No task"),
        ({"reassign_tasks": ["clean", "clean"]}, ["body", "reassign_tasks", 1], "more than once"),
    ],
    ids=["out-of-range", "wrong-shape", "leading-zero", "no-name", "name-taken", "unknown-task", "repeated-task"],
)
def test_a_request_it_cannot_prepare_is_refused_where_it_is_wrong(
    request_: dict[str, Any], loc: list[Any], message: str
) -> None:
    with pytest.raises(DecomposeError, match=message) as raised:
        _copy(_pipeline(), **request_)
    assert raised.value.loc == loc


def test_a_custom_workflow_target_is_refused() -> None:
    custom = {
        "name": "c",
        "inputs": ["data"],
        "steps": [{"name": "judge", "check": "image-duplicates", "input": "data"}],
    }
    with pytest.raises(DecomposeError, match="no preset: only a preset entry decomposes") as raised:
        _copy(_pipeline(workflows=[custom], tasks=[]))
    assert raised.value.loc == ["body", "target_pointer"]


def test_a_task_running_another_workflow_cant_be_reassigned() -> None:
    other = {"name": "q2", "type": "quality", **_MINIMAL["quality"]}
    with pytest.raises(DecomposeError, match="doesn't run workflow 'q'"):
        _copy(_pipeline(workflows=[_QUALITY, other], tasks=[{**_CLEAN, "workflow": "q2"}]), reassign_tasks=["clean"])


def test_a_candidate_over_the_node_limit_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_contract, "MAX_RESOLVED_NODES", 1)
    with pytest.raises(_contract.ContractError) as raised:
        _copy(_pipeline())
    assert raised.value.body["code"] == "too_many_nodes"


@pytest.fixture
def examples(plugins: dict[str, list[tuple[str, str]]]) -> dict[str, list[tuple[str, str]]]:
    for group, entries in EXAMPLES.items():
        plugins[group] = list(entries)
    plugins["dataeval_flow.workflows"].append(("example.refusing", "tests.example_plugin:RefusingWorkflow"))
    return plugins


@pytest.mark.usefixtures("examples")
def test_a_plugin_preset_that_declares_no_structure_is_a_plugin_gap() -> None:
    count = {"name": "c", "type": "example.count"}
    answer = _copy(_pipeline(workflows=[count], tasks=[{"name": "t", "workflow": "c", "sources": ["src"]}]))
    assert isinstance(answer, DecomposeUnsupported)
    assert {(r.code, r.category) for r in answer.reasons} == {("undeclared_structure", "plugin_contract_gap")}
    assert "dataeval_flow.workflows.Structural" in answer.reasons[0].remediation


@pytest.mark.usefixtures("examples")
def test_a_plugin_preset_that_refuses_in_python_is_a_plugin_gap() -> None:
    refusing = {"name": "r", "type": "example.refusing"}
    answer = _copy(_pipeline(workflows=[refusing], tasks=[{"name": "t", "workflow": "r", "sources": ["src"]}]))
    assert isinstance(answer, DecomposeUnsupported)
    assert {r.code for r in answer.reasons} == {"unrepresentable_preflight"}


@pytest.mark.parametrize(
    ("value", "written", "effective", "said"),
    [
        ({"mi_threshold": 0.1}, False, None, "`s` is unset, so it takes its default."),
        ({"mi_threshold": 0.1}, True, None, "`s` is set."),
        (0, False, None, "`s` is unset, so its default `0`."),
        (False, True, None, "`s` is `false`."),
        (None, False, None, "`s` is unset, so its default `null`."),
        (None, True, None, "`s` is `null`."),
        (None, False, "adaptive", "`s` is unset, so `adaptive`."),
        (None, True, "adaptive", "`s` is `null`, so `adaptive`."),
    ],
    ids=["default-mapping", "set-mapping", "default-0", "false", "default-null", "null", "unset-stands", "null-stands"],
)
def test_a_decision_says_whether_the_entry_set_its_value(value: Any, written: bool, effective: Any, said: str) -> None:
    from dataeval_flow._service._decompose import _decision
    from dataeval_flow.workflows._preset import StructuralChoice

    choice = StructuralChoice("s", value, written, effective if value is None else value, (), (), "Rule.")
    assert _decision(choice, "/workflows/0").explanation == f"{said} Rule."


@pytest.mark.usefixtures("examples")
def test_plugin_and_matrix_reasons_come_back_together() -> None:
    count = {"name": "c", "type": "example.count"}
    sweep = {"name": "t", "workflow": "c", "sources": ["src"], "matrix": {"minimum": [1, 2]}}
    answer = _copy(_pipeline(workflows=[count], tasks=[sweep]), reassign_tasks=["t"])
    assert isinstance(answer, DecomposeUnsupported)
    assert [(r.code, r.document_pointer) for r in answer.reasons] == [
        ("undeclared_structure", "/workflows/0"),
        ("unsupported_matrix_rewrite", "/tasks/0/matrix/minimum"),
    ]
