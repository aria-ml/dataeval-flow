"""The graph a custom workflow makes: what refuses to load, and what a valid one resolves to (spec §4, §5.3)."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow._chain._graph import build_graph
from dataeval_flow.evaluators.bias import BalanceConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.evaluators.scope import CoverageConfig
from tests.chain_toys import chain_pipeline, register_toys
from tests.evaluator_toys import ToyImages
from tests.workflow_toys import ToyCountConfig, register_count

_POOLS: dict[str, Any] = {
    "evaluators": [DuplicatesConfig(name="dupes"), BalanceConfig(name="balance"), CoverageConfig(name="cov")],
}


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    register_count(plugins)
    return plugins


def _workflow(steps: list[dict[str, Any]], inputs: list[Any] | None = None) -> dict[str, Any]:
    return {"name": "w", "inputs": inputs or ["a"], "steps": steps}


def _load(steps: list[dict[str, Any]], inputs: list[Any] | None = None, **kwargs: Any):
    return chain_pipeline(workflows=[_workflow(steps, inputs)], **_POOLS, **kwargs)


@pytest.mark.usefixtures("toys")
@pytest.mark.parametrize(
    ("steps", "message"),
    [
        (
            [{"name": "k", "transform": "toy-keep", "input": "nothere"}],
            "`nothere`, which is no input or earlier step",
        ),
        (
            [
                {"name": "k", "transform": "toy-keep", "input": "later"},
                {"name": "later", "transform": "toy-keep", "input": "a"},
            ],
            "step 'later' runs later",
        ),
        (
            [
                {"name": "k", "transform": "toy-keep", "input": "a"},
                {"name": "j", "transform": "toy-keep", "input": "k.output"},
            ],
            "`k` has one output: name it `k`",
        ),
        (
            [
                {"name": "h", "transform": "toy-halves", "input": "a"},
                {"name": "j", "transform": "toy-keep", "input": "h"},
            ],
            "has outputs head, tail: name one, such as `h.head`",
        ),
        ([{"name": "k", "transform": "toy-keep", "input": "a[x]"}], "`a` is not a list"),
        (
            [
                {"name": "s", "transform": "toy-spread", "input": "a", "parts": 2},
                {"name": "j", "transform": "toy-keep", "input": "s[5]"},
            ],
            "has elements 0, 1, not `5`",
        ),
        (
            [
                {"name": "d", "evaluator": "dupes", "input": "a"},
                {"name": "j", "transform": "toy-keep", "input": "d"},
            ],
            "which takes a Dataset, but `d` is an Output",
        ),
        ([{"name": "g", "transform": "toy-gather", "input": "a"}], "takes a whole list, but `a` is one item"),
        (
            [{"name": "b", "evaluator": "balance", "input": ["a", "a"]}],
            "which takes exactly one source, but the step names 2",
        ),
        (
            [{"name": "p", "transform": "toy-pair", "input": ["a"]}],
            "Step 'p' runs transform 'toy-pair', whose `input` takes exactly two sources, but the step names 1.",
        ),
        ([{"name": "b", "evaluator": "nothere", "input": "a"}], "`evaluators:` does not define"),
        ([{"name": "b", "evaluator": "balance"}], "reads nothing: give it `input:`"),
        (
            [{"name": "b", "evaluator": "dupes", "input": "a", "extractor": "none"}],
            "extractor 'none', which `extractors:` does not define",
        ),
        ([{"name": "v", "transform": "view", "input": "a", "view": "nothere"}], "`views:` does not define"),
    ],
)
def test_a_workflow_whose_steps_do_not_connect_fails_the_load(steps: list, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _load(steps)


@pytest.mark.usefixtures("toys")
def test_a_transform_fed_as_many_datasets_as_its_port_counts_loads() -> None:
    _load([{"name": "p", "transform": "toy-pair", "input": ["a", "b"]}], inputs=["a", "b"])


@pytest.mark.usefixtures("toys")
def test_a_list_step_fed_a_list_would_nest_lists_and_is_refused() -> None:
    steps = [{"name": "s", "transform": "toy-spread", "input": "all"}]
    with pytest.raises(ValidationError, match="lists do not nest"):
        _load(steps, inputs=[{"name": "all", "list": True}])


@pytest.mark.usefixtures("toys")
def test_a_custom_workflow_cannot_run_as_a_step() -> None:
    inner = {"name": "inner", "inputs": ["x"], "steps": [{"name": "k", "transform": "toy-keep", "input": "x"}]}
    outer = _workflow([{"name": "i", "workflow": "inner", "input": "a"}])
    with pytest.raises(ValidationError, match="a custom workflow: only a workflow type"):
        chain_pipeline(workflows=[inner, outer], **_POOLS)


@pytest.mark.usefixtures("toys")
def test_a_step_that_needs_embeddings_needs_an_extractor_on_the_task_or_the_step() -> None:
    steps = [{"name": "c", "evaluator": "cov", "input": "a"}]
    task = {"name": "t", "workflow": "w", "sources": ["src"]}
    with pytest.raises(ValidationError, match="step 'c' needs an extractor to produce embeddings"):
        _load(steps, tasks=[task])
    _load(steps, tasks=[{**task, "extractor": "flat"}], extractor=True)
    _load([{**steps[0], "extractor": "flat"}], tasks=[task], extractor=True)


@pytest.mark.usefixtures("toys")
def test_a_transform_step_naming_an_extractor_fails_the_load() -> None:
    steps = [{"name": "k", "transform": "toy-keep", "input": "a", "extractor": "flat"}]
    message = "Step 'k' runs transform 'toy-keep', which embeds nothing; remove `extractor:`."
    with pytest.raises(ValidationError, match=message):
        _load(steps, extractor=True)


def test_the_schema_offers_extractor_only_on_the_steps_that_embed() -> None:
    from dataeval_flow.config import PipelineConfig

    steps = {name: branch for name, branch in PipelineConfig.model_json_schema()["$defs"].items() if "Step" in name}
    offered = sorted(name for name, branch in steps.items() if "extractor" in branch.get("properties", {}))
    assert offered == ["EvaluatorStep", "WorkflowStep"]


@pytest.mark.usefixtures("toys")
def test_a_valid_workflow_resolves_each_step_and_marks_broadcasts() -> None:
    steps = [
        {"name": "kept", "transform": "toy-keep", "input": "all"},
        {"name": "dupes", "evaluator": "dupes", "input": "kept"},
        {"name": "h", "transform": "toy-halves", "input": "one"},
        {"name": "clean", "workflow": "clean", "input": "h.head"},
        {"name": "v", "transform": "view", "input": "one", "operations": [{"type": "Limit", "params": {"size": 4}}]},
    ]
    config = chain_pipeline(
        workflows=[
            _workflow(steps, inputs=["one", {"name": "all", "list": True}]),
            ToyCountConfig(name="clean"),
        ],
        **_POOLS,
        datasets={"src": ToyImages(), "more": ToyImages(seed=1)},
    )
    graph = build_graph(config.workflows[0], config)  # type: ignore[arg-type,index]
    spec = {s.name: s for s in graph.steps}
    assert [s.kind for s in graph.steps] == ["transform", "evaluator", "transform", "workflow", "transform"]
    assert spec["kept"].broadcast
    assert spec["dupes"].broadcast
    assert not spec["h"].broadcast
    assert [str(a) for a in spec["clean"].addresses("input")] == ["h.head"]
    assert spec["h"].output_address(spec["h"].outputs[0]) == "h.head"
    assert spec["v"].config.operations[0].type == "Limit"  # type: ignore[attr-defined]


@pytest.mark.usefixtures("toys")
def test_a_view_step_naming_a_pool_view_is_keyed_on_its_operations() -> None:
    views = [{"name": "first4", "operations": [{"type": "Limit", "params": {"size": 4}}]}]
    config = _load([{"name": "v", "transform": "view", "input": "a", "view": "first4"}], extra={"views": views})
    (spec,) = build_graph(config.workflows[0], config).steps  # type: ignore[arg-type,index]
    assert spec.config.view is None  # type: ignore[attr-defined]
    assert [op.params for op in spec.config.operations] == [{"size": 4}]  # type: ignore[attr-defined]


@pytest.mark.usefixtures("toys")
def test_each_custom_workflows_graph_is_built_once_at_load() -> None:
    from unittest.mock import patch

    from dataeval_flow._chain import _graph

    steps = [{"name": "k", "transform": "toy-keep", "input": "a"}]
    with patch.object(_graph, "build_graph", wraps=_graph.build_graph) as built:
        _load(steps, tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}])
    assert built.call_count == 1
