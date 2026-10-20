"""Preparing a preset step's inlining: its steps written into the parent (preset decomposition spec §6.2–§6.3)."""

from collections.abc import Sequence
from typing import Any

import pytest

from dataeval_flow._chain._graph import resolve_graph
from dataeval_flow._service._decompose import (
    DecomposeError,
    DecomposeReady,
    DecomposeRequest,
    DecomposeUnsupported,
    prepare,
)
from dataeval_flow.config import PipelineConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps._workflow import CustomWorkflowConfig
from dataeval_flow.workflows._preset import expand_preset, preset_of
from tests.test_audit_preset import _OUTLIERS
from tests.test_decompose import _QUALITY, _pipeline, examples  # noqa: F401 - `examples` is a fixture

_QUALITY_STEPS = [
    "outliers", "label-health", "outliers-by-class", "duplicates", "image-outliers", "target-outliers",
    "class-outliers", "image-duplicates", "clean",
]  # fmt: skip
_OUTER = {
    "name": "outer",
    "inputs": ["data"],
    "steps": [
        {"name": "c", "workflow": "q", "input": "data"},
        {"name": "dupes-clean", "evaluator": "dupes", "input": "c.clean"},
        {"name": "both", "transform": "collect", "input": ["data", "c.clean"]},
        {"name": "again", "workflow": "q", "input": "c.clean"},
    ],
}
_TASK = {"name": "t", "workflow": "outer", "sources": ["src"]}
_SPLITTING = {"name": "splitting", "type": "splits"}
_RELEASE = {"name": "release", "type": "audit", **_OUTLIERS}
_AUDITED = {
    "name": "outer",
    "inputs": ["data"],
    "steps": [
        {"name": "splits", "workflow": "splitting", "input": "data"},
        {"name": "evals", "transform": "collect", "input": ["splits.val", "splits.test"]},
        {"name": "audit", "workflow": "release", "input": ["splits.train", "evals"]},
    ],
}
# `c`, then an OOD detector and the factors behind its flags, wired as Shift's chain wires them.
_DEVIATION = {
    "name": "outer",
    "inputs": ["data"],
    "steps": [
        _OUTER["steps"][0],
        {"name": "ood", "evaluator": "knn", "input": ["c.clean", "data"], "extractor": "flat"},
        {"name": "dev", "combine": "factor-deviation", "ood": "ood", "reference": "c.clean", "input": "data"},
    ],
}


def _draft(
    *workflows: dict[str, Any], tasks: list[Any] | None = None, evaluators: Sequence[Any] = (), **extra: Any
) -> PipelineConfig:
    return _pipeline(
        workflows=list(workflows),
        tasks=tasks if tasks is not None else [_TASK],
        evaluators=[DuplicatesConfig(name="dupes"), *evaluators],
        **extra,
    )


def _inline(pipeline: PipelineConfig, pointer: str) -> DecomposeReady | DecomposeUnsupported:
    body = {"pipeline": {}, "operation": "inline_step", "target_pointer": pointer}
    return prepare(pipeline, DecomposeRequest.model_validate(body))


def _ready(answer: DecomposeReady | DecomposeUnsupported) -> DecomposeReady:
    assert isinstance(answer, DecomposeReady), answer
    return answer


def _steps(answer: DecomposeReady, workflow: int) -> dict[str, dict[str, Any]]:
    return {step["name"]: step for step in answer.pipeline["workflows"][workflow]["steps"]}


def test_an_inlined_step_becomes_its_steps_and_downstream_reads_follow() -> None:
    pipeline = _draft(_QUALITY, _OUTER)
    answer = _ready(_inline(pipeline, "/workflows/1/steps/0"))
    assert (answer.valid, answer.supported_operations) == (True, ["copy_definition", "inline_step"])
    steps = _steps(answer, 1)
    assert list(steps) == [*(f"c-{name}" for name in _QUALITY_STEPS), "dupes-clean", "both", "again"]
    assert steps["c-outliers"]["input"] == "data"
    assert steps["c-outliers-by-class"]["outliers"] == "c-outliers"
    expanded, _ = expand_preset(pipeline.workflows[0], preset_of(pipeline.workflows[0]))  # type: ignore[index,arg-type]
    plans = {step.name: step for step in expanded.steps}["clean"].model_dump(mode="json")["plans"]
    assert steps["c-clean"]["plans"] == {f"c-{address}": settings for address, settings in plans.items()}
    assert steps["dupes-clean"]["input"] == "c-clean"
    assert steps["both"]["input"] == ["data", "c-clean"]
    assert steps["both"]["keys"] == ["data", "clean"]  # the keys `collect` derived from `c.clean`, kept
    assert "keys" not in steps["c-outliers-by-class"]
    assert (steps["again"]["workflow"], steps["again"]["input"]) == ("q", "c-clean")
    assert answer.names.workflow == "outer"
    assert answer.names.steps == {name: f"c-{name}" for name in _QUALITY_STEPS}
    assert answer.names.addresses == {"c.clean": "c-clean"}
    assert set(answer.names.evaluators.values()) == {"outer-c-outliers", "outer-c-duplicates", "outer-c-label-health"}
    assert steps["c-outliers"]["evaluator"] == "outer-c-outliers"
    before = pipeline.model_dump(mode="json", by_alias=True)
    assert answer.pipeline["workflows"][0] == before["workflows"][0]
    assert answer.pipeline["tasks"] == before["tasks"]
    (origin,) = answer.pipeline["workflows"][1]["derived_from"]
    assert (origin["type"], origin["entry"], origin["steps"]) == ("quality", "q", list(answer.names.steps.values()))
    assert "scopes" not in answer.pipeline["workflows"][1]
    assert {c.code for c in answer.representational_changes} == {"step_names"}
    assert ("replace", "/workflows/1/steps/0") in {(c.op, c.pointer) for c in answer.changes}
    assert ("replace", "/workflows/1/steps/9/input") in {(c.op, c.pointer) for c in answer.changes}


def test_an_optional_instance_makes_every_inlined_step_optional() -> None:
    outer = {**_OUTER, "steps": [{**_OUTER["steps"][0], "optional": True}, *_OUTER["steps"][1:]]}
    steps = _steps(_ready(_inline(_draft(_QUALITY, outer), "/workflows/1/steps/0")), 1)
    assert all(steps[f"c-{name}"].get("optional") is True for name in _QUALITY_STEPS)


def test_a_step_extractor_lands_on_the_steps_that_embed() -> None:
    scope_entry = {"name": "sc", "type": "scope"}
    outer = {
        "name": "outer",
        "inputs": ["data"],
        "steps": [{"name": "c", "workflow": "sc", "input": "data", "extractor": "flat"}],
    }
    pipeline = _draft(scope_entry, outer, extractor=True)
    graph, _ = resolve_graph(pipeline.workflows[1], pipeline)  # type: ignore[index,arg-type]
    embedding = {spec.name.replace("c/", "c-") for spec in graph.steps if spec.extractor == "flat"}
    steps = _steps(_ready(_inline(pipeline, "/workflows/1/steps/0")), 1)
    assert embedding
    assert {name for name, step in steps.items() if step.get("extractor") == "flat"} == embedding


def test_an_element_read_of_an_inlined_output_keeps_its_key() -> None:
    outer = {
        "name": "outer",
        "inputs": ["train", {"name": "evals", "list": True}],
        "steps": [
            {"name": "c", "workflow": "q", "input": "evals"},
            {"name": "one", "evaluator": "dupes", "input": "c.clean[val]"},
        ],
    }
    steps = _steps(_ready(_inline(_draft(_QUALITY, outer, tasks=[]), "/workflows/1/steps/0")), 1)
    assert steps["one"]["input"] == "c-clean[val]"


def test_a_taken_name_is_numbered_and_reads_follow_it() -> None:
    outer = {**_OUTER, "steps": [{"name": "c-outliers", "evaluator": "dupes", "input": "data"}, *_OUTER["steps"]]}
    answer = _ready(_inline(_draft(_QUALITY, outer), "/workflows/1/steps/1"))
    steps = _steps(answer, 1)
    assert answer.names.steps["outliers"] == "c-outliers-2"
    assert steps["c-outliers-by-class"]["outliers"] == "c-outliers-2"
    assert set(steps["c-clean"]["plans"]) == {"c-outliers-2", "c-duplicates"}
    assert steps["c-outliers"]["evaluator"] == "dupes"


def test_inlining_a_second_instance_keeps_both_origins() -> None:
    first = _ready(_inline(_draft(_QUALITY, _OUTER), "/workflows/1/steps/0"))
    candidate = PipelineConfig.model_validate(first.pipeline, context={"graphs": False})
    again = len(_QUALITY_STEPS) + 2
    second = _ready(_inline(candidate, f"/workflows/1/steps/{again}"))
    origins = second.pipeline["workflows"][1]["derived_from"]
    assert [origin["steps"][0] for origin in origins] == ["c-outliers", "again-outliers"]
    assert _steps(second, 1)["again-outliers"]["input"] == "c-clean"


def test_an_inlined_audit_keeps_its_verdict_in_a_scope() -> None:
    release = {**_RELEASE, "accepted": {"image-outliers-train": "Planted.", "leakage": "Shared."}}
    answer = _ready(_inline(_draft(_SPLITTING, release, _AUDITED), "/workflows/2/steps/2"))
    assert answer.valid, answer.diagnostics
    (scope,) = answer.pipeline["workflows"][2]["scopes"]
    assert scope["name"] == "audit"
    assert scope["inputs"] == {"train": "splits.train", "evals": "evals"}
    assert scope["steps"] == list(answer.names.steps.values())
    assert (scope["reference"], scope["preflight"]) == ("train", {"nonempty": True, "one_kind": True})
    assert scope["verdict"]["blocking"] == ["leakage", "untrained-classes"]
    assert scope["verdict"]["accepted"] == {"audit-image-outliers-train": "Planted.", "leakage": "Shared."}
    assert "ontology" not in scope
    assert {c.code for c in answer.representational_changes} == {"step_names", "record_criteria"}
    steps = _steps(answer, 2)
    assert steps["audit-ood-kneighbors"]["optional"] is True  # the chain's own optional steps stay optional
    assert steps["audit-label-health-train"]["input"] == "splits.train"
    assert steps["audit-label-health-evals"]["input"] == "evals"


def test_a_preset_reading_a_step_s_output_keeps_its_start_in_a_bare_scope() -> None:
    # A step's output can hold nothing at run time, and then the splice skips its steps together; a scope does so too.
    def inlined(read: str) -> DecomposeReady:
        steps = [
            {"name": "splits", "workflow": "splitting", "input": "data"},
            {"name": "p", "workflow": "q", "input": read},
        ]
        outer = {"name": "outer", "inputs": ["data"], "steps": steps}
        return _ready(_inline(_draft(_SPLITTING, _QUALITY, outer), "/workflows/2/steps/1"))

    answer = inlined("splits.train")
    assert answer.valid, answer.diagnostics
    assert answer.pipeline["workflows"][2]["scopes"] == [
        {"name": "p", "inputs": {"data": "splits.train"}, "steps": [f"p-{name}" for name in _QUALITY_STEPS]}
    ]
    added = next(c for c in answer.changes if c.pointer == "/workflows/2/scopes/0")
    assert "skip together" in added.summary
    assert {c.code for c in answer.representational_changes} == {"step_names"}
    assert "scopes" not in inlined("data").pipeline["workflows"][2]


def test_inlining_upstream_of_a_scope_moves_its_inputs() -> None:
    first = _ready(_inline(_draft(_SPLITTING, _RELEASE, _AUDITED), "/workflows/2/steps/2"))
    candidate = PipelineConfig.model_validate(first.pipeline, context={"graphs": False})
    second = _ready(_inline(candidate, "/workflows/2/steps/0"))
    (scope,) = second.pipeline["workflows"][2]["scopes"]
    assert scope["inputs"] == {"train": "splits-split.train", "evals": "evals"}
    assert _steps(second, 2)["evals"]["input"] == ["splits-split.val", "splits-split.test"]
    assert ("replace", "/workflows/2/scopes/0/inputs") in {(c.op, c.pointer) for c in second.changes}
    assert second.valid, second.diagnostics


def test_a_read_of_an_output_the_settings_leave_empty_stays_a_diagnostic() -> None:
    splitting = {**_SPLITTING, "val_frac": 0}
    outer = {
        "name": "outer",
        "inputs": ["data"],
        "steps": [
            {"name": "s", "workflow": "splitting", "input": "data"},
            {"name": "v", "evaluator": "dupes", "input": "s.val"},
        ],
    }
    answer = _ready(_inline(_draft(splitting, outer), "/workflows/1/steps/0"))
    assert not answer.valid
    assert {d["code"] for d in answer.diagnostics} == {"unavailable_output"}
    assert _steps(answer, 1)["v"]["input"] == "s-split.val"


def test_a_parent_broken_after_the_target_still_gets_a_candidate() -> None:
    outer = {**_OUTER, "steps": [*_OUTER["steps"], {"name": "broken", "evaluator": "dupes", "input": "nope"}]}
    answer = _ready(_inline(_draft(_QUALITY, outer), "/workflows/1/steps/0"))
    assert not answer.valid
    assert any((d["node_id"] or "").endswith(":broken") for d in answer.diagnostics)


@pytest.mark.usefixtures("examples")
def test_a_target_that_does_not_resolve_is_refused_before_any_reason() -> None:
    count = {"name": "c", "type": "example.count"}  # undeclared structure: a plugin gap, were the target to resolve
    outer = {"name": "outer", "inputs": ["data"], "steps": [{"name": "n", "workflow": "c", "input": "nope"}]}
    with pytest.raises(DecomposeError, match="doesn't resolve") as raised:
        _inline(_draft(count, outer), "/workflows/1/steps/0")
    assert raised.value.loc == ["body", "target_pointer"]


def test_a_taken_evaluator_name_is_numbered_and_shared_entries_stay() -> None:
    from dataeval_flow.evaluators.quality import OutliersConfig

    shared = [OutliersConfig(name="outer-c-outliers", flags=["visual"]), OutliersConfig(name="outliers")]
    pipeline = _draft(_QUALITY, _OUTER, evaluators=shared)
    answer = _ready(_inline(pipeline, "/workflows/1/steps/0"))
    assert answer.names.evaluators["outliers"] == "outer-c-outliers-2"
    before = pipeline.model_dump(mode="json", by_alias=True)["evaluators"]
    assert answer.pipeline["evaluators"][: len(before)] == before


def test_a_scope_s_name_is_taken_too() -> None:
    scope = {"name": "c-outliers", "inputs": {"x": "data"}, "steps": ["dupes-clean"]}
    answer = _ready(_inline(_draft(_QUALITY, {**_OUTER, "scopes": [scope]}), "/workflows/1/steps/0"))
    assert answer.names.steps["outliers"] == "c-outliers-2"


def test_a_sweep_naming_nothing_the_inlining_moves_does_not_block_it() -> None:
    trim = {"name": "trim", "transform": "remove", "input": "c.clean", "plans": {"dupes-clean": {}}}
    outer = {**_OUTER, "steps": [*_OUTER["steps"][:2], trim]}
    tasks = [{**_TASK, "matrix": {"steps.trim.plans": [{"dupes-clean": {}}, {"dupes-clean": {"keep": "last"}}]}}]
    assert _ready(_inline(_draft(_QUALITY, outer, tasks=tasks), "/workflows/1/steps/0"))


def test_a_sweep_of_an_address_the_inlining_moves_is_refused() -> None:
    from dataeval_flow.evaluators.shift import OODKNeighborsConfig

    key = "steps.dev.reference"
    tasks = [{**_TASK, "matrix": {key: ["c.clean"]}}]
    pipeline = _draft(_QUALITY, _DEVIATION, tasks=tasks, evaluators=[OODKNeighborsConfig(name="knn")], extractor=True)
    answer = _inline(pipeline, "/workflows/1/steps/0")
    assert isinstance(answer, DecomposeUnsupported)
    (reason,) = answer.reasons
    assert (reason.code, reason.category, reason.document_pointer) == (
        "unsupported_matrix_rewrite",
        "scope_limit",
        f"/tasks/0/matrix/{key}",
    )
    assert reason.details == {"task": "t", "key": key}


def test_a_sweep_onto_an_address_the_inlining_moves_is_found() -> None:
    # Only a swept value reads `c.clean` here, beside a range, which holds numbers and never an address.
    from dataeval_flow._service._inline import _swept_moves
    from dataeval_flow.config import TaskConfig

    key = "steps.dev.reference"
    matrix = {key: ["data", "c.clean"], "steps.dev.max_items": {"from": 5, "to": 10, "step": 5}}
    task = TaskConfig.model_validate({**_TASK, "matrix": matrix})
    dev = {**_DEVIATION["steps"][2], "reference": "data"}
    parent = CustomWorkflowConfig.model_validate({**_DEVIATION, "steps": [*_DEVIATION["steps"][:2], dev]})
    assert _swept_moves(task, parent, {"c.clean": "c-clean"}) == [key]


@pytest.mark.parametrize("enabled", [True, False])
def test_a_task_sweeping_the_inlined_preset_is_refused(enabled: bool) -> None:
    key = "workflows.q.outliers.outlier_threshold"
    tasks = [{**_TASK, "enabled": enabled, "matrix": {key: ["zscore", "iqr"]}}]
    answer = _inline(_draft(_QUALITY, _OUTER, tasks=tasks), "/workflows/1/steps/0")
    assert isinstance(answer, DecomposeUnsupported)
    (reason,) = answer.reasons
    assert (reason.code, reason.category, reason.document_pointer) == (
        "unsupported_matrix_rewrite",
        "scope_limit",
        f"/tasks/0/matrix/{key}",
    )
    assert reason.details == {"task": "t", "key": key}


def test_an_unrelated_sweep_does_not_block_inlining() -> None:
    tasks = [{**_TASK, "matrix": {"evaluators.dupes.flags": [["hash_basic"], ["hash_d4"]]}}]
    assert _ready(_inline(_draft(_QUALITY, _OUTER, tasks=tasks), "/workflows/1/steps/0")).pipeline["tasks"][0]["matrix"]


def test_the_candidate_round_trips_as_yaml() -> None:
    import yaml

    answer = _ready(_inline(_draft(_SPLITTING, _RELEASE, _AUDITED), "/workflows/2/steps/2"))
    parent = CustomWorkflowConfig.model_validate(answer.pipeline["workflows"][2])
    reloaded = CustomWorkflowConfig.model_validate(yaml.safe_load(parent.to_yaml())["workflows"][0])
    assert reloaded == parent
    assert reloaded.derived_from[0].steps == list(answer.names.steps.values())
