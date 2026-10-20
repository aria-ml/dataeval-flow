"""Scopes: steps of a custom workflow judged together, as a preset run as a step was (decomposition spec §6.1)."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._graph import GraphError, build_graph
from dataeval_flow._chain._presets import Splice
from dataeval_flow._runner import _judged_tasks
from dataeval_flow.evaluators.quality import DuplicatesConfig, LabelHealthConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps._workflow import CustomWorkflowConfig
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import Items, ToyImages
from tests.golden.splitting import SplitImages
from tests.test_audit_preset import _OUTLIERS
from tests.test_chain_verdict_report import _outline
from tests.test_custom_workflow_verdict import _INPUTS, _STEPS

_NAMES = [step["name"] for step in _STEPS]
_OUTSIDE = [
    {"name": "dupes-outside", "evaluator": "dupes", "input": "train"},
    {"name": "outside", "check": "image-duplicates", "input": "dupes-outside"},
]
_SCOPE: dict[str, Any] = {
    "name": "s",
    "inputs": {"train": "train", "evals": "evals"},
    "steps": _NAMES,
    "verdict": {"blocking": ["image-duplicates"]},
}


def _workflow(*scopes: dict[str, Any], steps: list[dict[str, Any]] | None = None, **declared: Any) -> dict[str, Any]:
    return {
        "name": "w",
        "inputs": _INPUTS,
        "steps": steps if steps is not None else [*_STEPS, *_OUTSIDE],
        "scopes": list(scopes) if scopes else [_SCOPE],
        **declared,
    }


def test_a_scope_round_trips_as_written() -> None:
    written = {**_SCOPE, "reference": "train", "preflight": {"nonempty": True, "one_kind": False}}
    workflow = CustomWorkflowConfig.model_validate(_workflow(written))
    (dumped,) = workflow.model_dump(mode="json")["scopes"]
    assert list(dumped) == ["name", "inputs", "steps", "reference", "verdict", "preflight"]
    assert dumped["inputs"] == {"train": "train", "evals": "evals"}
    assert CustomWorkflowConfig.model_validate(workflow.model_dump(mode="json")) == workflow


def test_a_workflow_without_scopes_serializes_unchanged() -> None:
    dumped = CustomWorkflowConfig.model_validate(_workflow(steps=_STEPS) | {"scopes": []}).model_dump(mode="json")
    assert "scopes" not in dumped


def test_a_scope_declares_a_chain() -> None:
    (scope,) = CustomWorkflowConfig.model_validate(_workflow({**_SCOPE, "reference": "train"})).scopes
    chain = scope.declared_chain()
    assert (chain.blocking, chain.reference, chain.accepted) == (("image-duplicates",), "train", {})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"scopes": [_SCOPE, _SCOPE]}, "more than one scope named 's'"),
        ({"scopes": [{**_SCOPE, "name": "outside"}]}, "scope 'outside' shares its name with a step"),
        ({"scopes": [{**_SCOPE, "steps": ["nope"]}]}, "names step 'nope', which workflow 'w' has no"),
        ({"scopes": [_SCOPE, {**_SCOPE, "name": "t", "steps": ["imbalance"]}]}, "'imbalance' is in more than one"),
        ({"scopes": [{**_SCOPE, "inputs": {"bad name": "train"}}]}, "bad name"),
        ({"scopes": [{**_SCOPE, "inputs": {"train": "not an address!"}}]}, "is not an address"),
        ({"scopes": [{**_SCOPE, "reference": "nope"}]}, "`reference` names `nope`"),
        ({"scopes": [{**_SCOPE, "verdict": {"blocking": ["image-outliers"]}}]}, "`blocking` names `image-outliers`"),
        ({"scopes": [{**_SCOPE, "verdict": {"accepted": {"outside": "x"}}}]}, "`accepted` names `outside`"),
        (
            {"scopes": [{**_SCOPE, "verdict": None, "next_steps": {"by_check": {"class-imbalance": "x"}}}]},
            "needs `verdict:`",
        ),
        (
            {"scopes": [{**_SCOPE, "groups": [{"heading": "H", "checks": ["image-outliers"]}]}]},
            "names check `image-outliers`",
        ),
    ],
    ids=[
        "duplicate", "step-name", "unknown-step", "two-scopes", "bad-slot", "bad-address", "reference",
        "blocking", "accepted", "next-steps", "group",
    ],
)  # fmt: skip
def test_a_scope_naming_what_it_does_not_hold_is_refused(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        CustomWorkflowConfig.model_validate(_workflow() | change)


def test_a_check_only_outside_the_scope_is_not_the_scope_s() -> None:
    scope = {**_SCOPE, "steps": ["labels-train", "imbalance"]}
    with pytest.raises(ValidationError, match="`blocking` names `image-duplicates`"):
        CustomWorkflowConfig.model_validate(_workflow(scope))


def test_a_preset_step_in_a_scope_is_refused() -> None:
    steps = [*_STEPS, {"name": "q", "workflow": "q", "input": "train"}]
    with pytest.raises(ValidationError, match="step 'q' runs a preset"):
        CustomWorkflowConfig.model_validate(_workflow({**_SCOPE, "steps": [*_NAMES, "q"]}, steps=steps))


def test_a_scope_may_hold_optional_steps() -> None:
    steps = [{**_STEPS[0], "optional": True}, *_STEPS[1:], *_OUTSIDE]
    assert CustomWorkflowConfig.model_validate(_workflow(steps=steps)).scopes[0].steps == _NAMES


@pytest.mark.parametrize(
    ("key", "values"),
    [
        ("scopes", [[]]),
        ("scopes.s", [_SCOPE]),
        ("scopes.s.name", ["s"]),
        ("scopes.s.inputs.train", ["train"]),
        ("scopes.s.steps", [_NAMES]),
    ],
)
def test_a_matrix_may_not_rewire_a_scope(key: str, values: list[Any]) -> None:
    task = {"name": "t", "workflow": "w", "sources": ["train", "val"], "matrix": {key: values}}
    with pytest.raises(ValidationError, match="wires the chain every run shares"):
        chain_pipeline(
            workflows=[_workflow()],
            evaluators=[DuplicatesConfig(name="dupes"), LabelHealthConfig(name="labels")],
            tasks=[task],
            datasets={"train": ToyImages(), "val": ToyImages(seed=1)},
        )


def _graph_of(workflow: dict[str, Any], *, slot_keys=None, evaluators=None, extra=None, workflows=()):
    pipeline = chain_pipeline(
        workflows=[*workflows, workflow],
        evaluators=evaluators or [DuplicatesConfig(name="dupes"), LabelHealthConfig(name="labels")],
        extra=extra,
        graphs=False,
    )
    (loaded,) = [w for w in pipeline.workflows or () if isinstance(w, CustomWorkflowConfig) and w.name == "w"]
    return build_graph(loaded, pipeline, slot_keys)


def test_a_scope_is_a_splice_of_its_steps_in_run_order() -> None:
    graph = _graph_of(_workflow({**_SCOPE, "steps": list(reversed(_NAMES))}))
    (splice,) = graph.splices
    assert (splice.name, splice.preset, splice.entry, splice.prefix, splice.noun) == ("s", None, None, "", "scope")
    assert splice.steps == tuple(_NAMES)
    assert dict(splice.slots) == {"train": "train", "evals": "evals"}
    assert splice.gives_verdict
    with pytest.raises(TypeError, match="prefix"):
        Splice(splice.name, None, None, splice.chain, splice.slots, splice.steps)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "workflow",
    [
        _workflow(_SCOPE, {**_SCOPE, "name": "t", "steps": ["dupes-outside", "outside"]}),
        _workflow(verdict={}),
        _workflow(record={"title": "R", "steps": []}),
    ],
    ids=["two-scopes", "workflow-verdict", "workflow-record"],
)
def test_a_second_verdict_beside_a_scope_is_refused(workflow: dict[str, Any]) -> None:
    with pytest.raises(GraphError) as raised:
        _graph_of(workflow)
    assert raised.value.code == "too_many_verdicts"
    assert "scope `s`" in str(raised.value)


def test_a_verdict_scope_beside_a_preset_step_giving_one_is_refused() -> None:
    steps = [*_STEPS, *_OUTSIDE, {"name": "audit", "workflow": "release", "input": ["train", "evals"]}]
    with pytest.raises(GraphError) as raised:
        _graph_of(_workflow(steps=steps), workflows=[{"name": "release", "type": "audit", **_OUTLIERS}])
    assert raised.value.code == "too_many_verdicts"


@pytest.mark.parametrize(
    ("inputs", "code"),
    [
        ({"train": "nope", "evals": "evals"}, "unknown_address"),
        ({"train": "dupes-outside", "evals": "evals"}, "wrong_type"),
        ({"train": "late", "evals": "evals"}, "forward_address"),
    ],
    ids=["unknown", "an-output", "made-later"],
)
def test_a_scope_input_that_is_no_dataset_made_before_it_is_refused(inputs: dict[str, str], code: str) -> None:
    late = {"name": "late", "transform": "collect", "input": ["train"]}
    steps = [*_OUTSIDE, *_STEPS, late]
    with pytest.raises(GraphError) as raised:
        _graph_of(_workflow({**_SCOPE, "inputs": inputs}, steps=steps))
    assert raised.value.code == code
    assert (raised.value.scope, raised.value.step, raised.value.port) == ("s", None, "train")
    assert str(raised.value).startswith("Scope 's' input `train` reads ")
    assert "Step 's'" not in str(raised.value)


def test_a_scope_input_naming_nothing_is_refused_as_the_scope_s() -> None:
    with pytest.raises(GraphError) as raised:
        _graph_of(_workflow({**_SCOPE, "inputs": {"train": "nope", "evals": "evals"}}))
    assert (
        str(raised.value) == "Scope 's' input `train` reads `nope`, which is no input or earlier step of workflow 'w'."
    )
    assert (raised.value.code, raised.value.scope, raised.value.step, raised.value.port) == (
        "unknown_address",
        "s",
        None,
        "train",
    )


def test_a_scope_s_diagnostic_points_into_the_scope() -> None:
    from dataeval_flow._service._resolve import resolve

    pipeline = chain_pipeline(
        workflows=[_workflow({**_SCOPE, "inputs": {"train": "nope", "evals": "evals"}})],
        evaluators=[DuplicatesConfig(name="dupes"), LabelHealthConfig(name="labels")],
        graphs=False,
    )
    (diagnostic,) = resolve(pipeline, []).diagnostics
    assert (diagnostic["document_pointer"], diagnostic["node_id"]) == ("/workflows/0/scopes/0/inputs/train", None)


def test_an_accepted_element_of_a_step_that_runs_once_is_refused() -> None:
    with pytest.raises(GraphError) as raised:
        _graph_of(_workflow({**_SCOPE, "verdict": {"accepted": {"imbalance[val]": "x"}}}))
    assert (raised.value.code, raised.value.scope) == ("list_mismatch", "s")


def test_a_scope_reference_bound_to_a_list_is_refused() -> None:
    with pytest.raises(GraphError) as raised:
        _graph_of(_workflow({**_SCOPE, "reference": "evals"}))
    assert raised.value.code == "list_mismatch"


def test_an_accepted_element_no_bound_list_has_is_refused() -> None:
    scope = {**_SCOPE, "verdict": {"accepted": {"duplicates-evals[test]": "x"}}}
    with pytest.raises(GraphError) as raised:
        _graph_of(_workflow(scope), slot_keys={"evals": ("val",)})
    assert raised.value.code == "list_mismatch"


def test_an_element_keyed_like_a_single_slot_is_refused() -> None:
    with pytest.raises(GraphError, match="element `train`"):
        _graph_of(_workflow(), slot_keys={"evals": ("train",)})


def test_a_reference_beside_a_reference_split_policy_is_refused() -> None:
    evaluators = [DuplicatesConfig(name="dupes"), LabelHealthConfig(name="labels", metadata="pinned")]
    extra = {"metadata": [{"name": "pinned", "reference_split": "train"}]}
    with pytest.raises(GraphError, match="reference_split"):
        _graph_of(_workflow({**_SCOPE, "reference": "train"}), evaluators=evaluators, extra=extra)


def test_a_reference_beside_an_inline_step_s_reference_split_policy_is_refused() -> None:
    cut = {"name": "cut", "transform": "split", "input": "train", "metadata": "pinned"}
    scope = {**_SCOPE, "steps": [*_NAMES, "cut"], "reference": "train"}
    extra = {"metadata": [{"name": "pinned", "reference_split": "train"}]}
    with pytest.raises(GraphError, match="step 'cut', whose metadata policy 'pinned' names reference_split") as raised:
        _graph_of(_workflow(scope, steps=[*_STEPS, cut, *_OUTSIDE]), extra=extra)
    assert raised.value.scope == "s"


@pytest.fixture
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _config(workflow: dict[str, Any], datasets: dict[str, Any] | None = None, *, workflows=(), extra=None):
    datasets = datasets if datasets is not None else {"train": ToyImages(), "val": ToyImages(seed=1)}
    return chain_pipeline(
        workflows=[*workflows, workflow],
        evaluators=[DuplicatesConfig(name="dupes"), LabelHealthConfig(name="labels")],
        tasks=[{"name": "t", "workflow": "w", "sources": list(datasets)}],
        datasets=datasets,
        extra=extra,
    )


def _run(*args: Any, **kwargs: Any) -> ChainResult:
    result = run_tasks(_config(*args, **kwargs))["t"]
    assert isinstance(result, ChainResult)
    return result


@pytest.mark.usefixtures("_fresh_caches")
def test_a_scope_s_verdict_judges_only_its_steps() -> None:
    result = _run(_workflow())
    assert result.success, result.errors
    assert result.verdict is not None
    assert result.splice == "s"
    judged = {item.step.split("[", 1)[0] for item in [*result.verdict.blocking, *result.verdict.warnings]}
    assert judged
    assert judged <= set(_NAMES)
    assert result.steps["outside"].output  # the outside check warned, and the verdict left it out
    assert "presets" not in result.metadata.resolved_config


@pytest.mark.usefixtures("_fresh_caches")
def test_a_scope_whose_input_held_nothing_skips_its_steps_and_gives_no_verdict() -> None:
    splitting = {"name": "splitting", "type": "splits", "split_on": ["nope"]}
    steps = [
        {"name": "splits", "workflow": "splitting", "input": "data", "optional": True},
        {"name": "evals", "transform": "collect", "input": ["splits.val", "splits.test"]},
        *[{**step, "input": "splits.train" if step["input"] == "train" else step["input"]} for step in _STEPS],
    ]
    workflow = {
        "name": "w",
        "inputs": ["data"],
        "steps": steps,
        "scopes": [{**_SCOPE, "inputs": {"train": "splits.train", "evals": "evals"}}],
    }
    result = _run(workflow, {"data": SplitImages(90)}, workflows=[splitting], extra={"seed": 0})
    assert result.success, result.errors
    assert result.verdict is None
    assert (result.no_verdict or "").startswith("scope `s` did not run:")
    assert {result.steps[name].status for name in _NAMES} == {"skipped"}


@pytest.mark.usefixtures("_fresh_caches")
def test_a_scope_s_preflight_refusal_fails_its_first_step_and_skips_the_rest() -> None:
    result = _run(_workflow({**_SCOPE, "preflight": {"nonempty": True}}), {"train": ToyImages(), "val": Items([])})
    assert not result.success
    assert result.verdict is None
    first, *rest = _NAMES
    assert "Scope 's': Input `val` holds no items" in result.steps[first].errors[0]
    assert all("scope `s` failed as it started" in (result.steps[name].reason or "") for name in rest)
    assert result.steps["outside"].status == "ok"


@pytest.mark.usefixtures("_fresh_caches")
def test_a_scope_s_questions_come_before_the_workflow_s_groups() -> None:
    scope = {**_SCOPE, "groups": [{"heading": "Inside?", "checks": ["image-duplicates"]}]}
    outline = _outline(_run(_workflow(scope, groups=[{"heading": "Outside?", "checks": ["image-duplicates"]}])))
    assert outline.index("Inside?") < outline.index("Outside?")


def test_a_task_running_a_verdict_scope_is_judged() -> None:
    assert _judged_tasks(_config(_workflow()), None) == ["t"]
