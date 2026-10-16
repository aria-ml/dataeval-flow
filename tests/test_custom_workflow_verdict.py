"""A custom workflow's declared verdict, record, reference and preflight, run (preset decomposition spec §3.2)."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._graph import GraphError, build_graph
from dataeval_flow._runner import _judged_tasks
from dataeval_flow.evaluators.quality import DuplicatesConfig, LabelHealthConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps._workflow import CustomWorkflowConfig
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import Items, ToyImages
from tests.test_audit_preset import _OUTLIERS
from tests.test_chain_verdict_report import _outline

# Each input judged on its own, as audit's `_each_split` does: `*-evals` steps run once per element of `evals`.
_STEPS = [
    {"name": "dupes-train", "evaluator": "dupes", "input": "train"},
    {"name": "dupes-evals", "evaluator": "dupes", "input": "evals"},
    {"name": "duplicates-train", "check": "image-duplicates", "input": "dupes-train"},
    {"name": "duplicates-evals", "check": "image-duplicates", "input": "dupes-evals"},
    {"name": "labels-train", "evaluator": "labels", "input": "train"},
    {"name": "imbalance", "check": "class-imbalance", "input": "labels-train"},
]
_INPUTS = ["train", {"name": "evals", "list": True, "empty": "no evaluation split given"}]
_EVALUATORS = [DuplicatesConfig(name="dupes"), LabelHealthConfig(name="labels")]


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _config(datasets: dict[str, Any] | None = None, **declared: Any):
    datasets = datasets if datasets is not None else {"train": ToyImages(), "val": ToyImages(seed=1)}
    return chain_pipeline(
        workflows=[{"name": "w", "inputs": _INPUTS, "steps": _STEPS, **declared}],
        evaluators=_EVALUATORS,
        tasks=[{"name": "t", "workflow": "w", "sources": list(datasets)}],
        datasets=datasets,
    )


def _run(datasets: dict[str, Any] | None = None, **declared: Any) -> ChainResult:
    result = run_tasks(_config(datasets, **declared))["t"]
    assert isinstance(result, ChainResult)
    return result


def test_a_declared_verdict_judges_the_workflow_s_checks() -> None:
    result = _run(verdict={"blocking": ["image-duplicates"]})
    assert result.success, result.errors
    assert result.verdict is not None
    assert result.preset_chain is not None
    assert result.preset_chain.blocking == ("image-duplicates",)


def test_the_record_heads_the_report() -> None:
    result = _run(verdict={}, record={"title": "What was read", "steps": ["label-health"]})
    assert "What was read" in _outline(result)


def test_a_task_running_a_declared_verdict_is_judged() -> None:
    assert _judged_tasks(_config(verdict={}), None) == ["t"]


def test_a_declared_preflight_refuses_an_empty_input() -> None:
    with pytest.raises(GraphError, match=r"Input `val` holds no items"):
        _run({"train": ToyImages(), "val": Items([])}, preflight={"nonempty": True})


def test_without_a_declared_preflight_an_empty_input_runs() -> None:
    assert isinstance(_run({"train": ToyImages(), "val": Items([])}), ChainResult)


def test_an_acceptance_naming_no_element_of_the_task_is_refused_before_the_run() -> None:
    with pytest.raises(GraphError, match=r"`duplicates-evals\[test\]`, but this task has no `evals` element `test`"):
        _run(verdict={"accepted": {"duplicates-evals[test]": "x"}})


def test_an_acceptance_by_element_of_a_step_that_runs_once_is_refused_at_load() -> None:
    declared = {"verdict": {"accepted": {"duplicates-train[val]": "x"}}}
    with pytest.raises(ValidationError, match=r"step `duplicates-train` runs once, not once per element"):
        _config(**declared)
    pipeline = chain_pipeline(
        workflows=[{"name": "w", "inputs": _INPUTS, "steps": _STEPS, **declared}],
        evaluators=_EVALUATORS,
        tasks=[{"name": "t", "workflow": "w", "sources": ["train", "val"]}],
        datasets={"train": ToyImages(), "val": ToyImages(seed=1)},
        graphs=False,
    )
    workflow = pipeline.workflows[0] if pipeline.workflows else None
    assert isinstance(workflow, CustomWorkflowConfig)
    with pytest.raises(GraphError) as raised:
        build_graph(workflow, pipeline)
    assert raised.value.code == "list_mismatch"


def _over_a_collected_list(accepted: dict[str, str]):
    """Inputs `a` and `b`, collected into one list, whose duplicates are judged once per element."""
    return chain_pipeline(
        workflows=[
            {
                "name": "w",
                "inputs": ["a", "b"],
                "steps": [
                    {"name": "both", "transform": "collect", "input": ["a", "b"]},
                    {"name": "dupes", "evaluator": "dupes", "input": "both"},
                    {"name": "duplicates", "check": "image-duplicates", "input": "dupes"},
                ],
                "verdict": {"blocking": ["image-duplicates"], "accepted": accepted},
            }
        ],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "workflow": "w", "sources": ["a", "b"]}],
        datasets={"a": ToyImages(), "b": ToyImages(seed=1)},
    )


def test_an_acceptance_by_element_of_a_list_the_chain_collected_covers_that_element_s_warning() -> None:
    plain = run_tasks(_over_a_collected_list({}))["t"]
    assert isinstance(plain, ChainResult)
    assert plain.verdict is not None
    assert plain.verdict.level == "not-ready"
    result = run_tasks(_over_a_collected_list({"duplicates[a]": "x", "duplicates[b]": "y"}))["t"]
    assert isinstance(result, ChainResult)
    assert result.verdict is not None
    assert result.verdict.level == "ready-with-caveats"
    assert [(a.check, a.state) for a in result.verdict.accepted] == [
        ("duplicates[a]", "warned"),
        ("duplicates[b]", "warned"),
    ]


def test_an_acceptance_naming_no_element_of_a_collected_list_is_refused_before_the_run() -> None:
    with pytest.raises(GraphError, match=r"`duplicates\[c\]`, but step `duplicates` runs over no element `c`"):
        run_tasks(_over_a_collected_list({"duplicates[c]": "x"}))


def _with_audit_step(*, graphs: bool = True, **declared: Any):
    steps = [{"name": "audit", "workflow": "a", "input": ["train", "evals"]}, *_STEPS[:1], *_STEPS[2:3]]
    return chain_pipeline(
        workflows=[
            {"name": "a", "type": "audit", **_OUTLIERS},
            {"name": "w", "inputs": _INPUTS, "steps": steps, **declared},
        ],
        evaluators=_EVALUATORS,
        tasks=[{"name": "t", "workflow": "w", "sources": ["train", "val"]}],
        datasets={"train": ToyImages(), "val": ToyImages(seed=1)},
        graphs=graphs,
    )


_BESIDE_A_VERDICT_STEP = [
    ({"verdict": {}}, r"declares `verdict:` and runs step `audit`, which gives one; a workflow gives one verdict"),
    ({"reference": "train"}, r"runs step `audit`, whose verdict reports its own reference, record and next steps"),
]


@pytest.mark.parametrize(("declared", "message"), _BESIDE_A_VERDICT_STEP, ids=["verdict", "reference"])
def test_a_workflow_running_a_verdict_step_declares_no_verdict_of_its_own(declared: dict[str, Any], message: str):
    with pytest.raises(ValidationError, match=message):
        _with_audit_step(**declared)
    pipeline = _with_audit_step(graphs=False, **declared)
    workflow = pipeline.workflows[1] if pipeline.workflows else None
    assert isinstance(workflow, CustomWorkflowConfig)
    with pytest.raises(GraphError, match=message) as raised:
        build_graph(workflow, pipeline)
    assert raised.value.code == "too_many_verdicts"


def test_groups_beside_a_verdict_step_are_still_allowed() -> None:
    assert _with_audit_step(groups=[{"heading": "Train", "checks": ["image-duplicates"]}]) is not None
