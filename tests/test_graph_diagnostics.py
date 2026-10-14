"""Graph problems say what they are and where, and resolution keeps the steps that connect (builder spec §5.2)."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow._chain._graph import GraphError, build_graph, resolve_graph, task_problems
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.evaluators.scope import LabelAlignmentConfig
from tests.chain_toys import ToyDetections, chain_pipeline, register_toys

_ONTOLOGY = {
    "name": "vehicles",
    "concepts": [
        {"id": "Vehicle", "label": "Vehicle", "synonyms": ["car", "truck"]},
        {"id": "Person", "label": "Person", "synonyms": ["person", "pedestrian"]},
    ],
}


@pytest.fixture(autouse=True)
def toys(plugins):
    register_toys(plugins)
    return plugins


def _config(steps: list[dict[str, Any]], inputs: list[Any] | None = None, **kwargs: Any):
    evaluators = [DuplicatesConfig(name="dupes"), *kwargs.pop("evaluators", [])]
    workflow = {"name": "w", "inputs": inputs or ["a"], "steps": steps}
    return chain_pipeline(workflows=[workflow], evaluators=evaluators, graphs=False, **kwargs)


def _resolved(steps: list[dict[str, Any]], **kwargs: Any):
    config = _config(steps, **kwargs)
    return resolve_graph(config.workflows[0], config)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("steps", "code", "step", "port"),
    [
        ([{"name": "k", "transform": "toy-keep", "input": "nothere"}], "unknown_address", "k", "input"),
        (
            [
                {"name": "k", "transform": "toy-keep", "input": "later"},
                {"name": "later", "transform": "toy-keep", "input": "a"},
            ],
            "forward_address",
            "k",
            "input",
        ),
        (
            [
                {"name": "k", "transform": "toy-keep", "input": "a"},
                {"name": "j", "transform": "toy-keep", "input": "k.output"},
            ],
            "ambiguous_output",
            "j",
            "input",
        ),
        (
            [
                {"name": "d", "evaluator": "dupes", "input": "a"},
                {"name": "k", "transform": "toy-keep", "input": "d"},
            ],
            "wrong_type",
            "k",
            "input",
        ),
        (
            [
                {"name": "k", "transform": "toy-keep", "input": "a"},
                {"name": "j", "transform": "toy-keep", "input": "k[x]"},
            ],
            "list_mismatch",
            "j",
            "input",
        ),
        (
            [
                {"name": "s", "transform": "split", "input": "a", "test_frac": 0.2, "val_frac": 0},
                {"name": "k", "transform": "toy-keep", "input": "s.val"},
            ],
            "unavailable_output",
            "k",
            "input",
        ),
        ([{"name": "e", "evaluator": "nope", "input": "a"}], "unknown_entry", "e", None),
        ([{"name": "e", "evaluator": "dupes", "input": "a b"}], "bad_address", "e", "input"),
        ([{"name": "k", "transform": "toy-keep", "input": "a b"}], "bad_address", "k", "input"),
        ([{"name": "d", "evaluator": "dupes", "input": "a", "extractor": "x"}], "bad_extractor", "d", None),
    ],
)
def test_a_graph_problem_says_what_it_is_and_where(steps, code, step, port) -> None:
    _, error = _resolved(steps)
    assert error is not None
    assert (error.code, error.step, error.port) == (code, step, port)


def test_an_output_from_another_dataset_is_located_on_its_port() -> None:
    steps = [
        {"name": "few", "transform": "view", "input": "a", "operations": [{"type": "Limit", "params": {"size": 2}}]},
        {"name": "aligned", "evaluator": "align", "input": "few"},
        {"name": "c", "transform": "conform", "input": "a", "alignment": "aligned"},
    ]
    graph, error = _resolved(
        steps,
        evaluators=[LabelAlignmentConfig(name="align", ontology="vehicles")],
        datasets={"src": ToyDetections([[0], [1]], {0: "car", 1: "person"}, dataset_id="d")},
        extra={"ontologies": [_ONTOLOGY]},
    )
    assert error is not None
    assert (error.code, error.step, error.port) == ("wrong_dataset", "c", "alignment")
    assert [spec.name for spec in graph.steps] == ["few", "aligned"]


def test_resolution_keeps_the_steps_before_the_first_problem() -> None:
    graph, error = _resolved(
        [
            {"name": "k", "transform": "toy-keep", "input": "a"},
            {"name": "j", "transform": "toy-keep", "input": "nothere"},
            {"name": "m", "transform": "toy-keep", "input": "k"},
        ]
    )
    assert [spec.name for spec in graph.steps] == ["k"]
    assert error is not None
    assert error.step == "j"


def test_build_graph_raises_what_resolution_returns() -> None:
    config = _config([{"name": "k", "transform": "toy-keep", "input": "nothere"}])
    _, error = resolve_graph(config.workflows[0], config)  # type: ignore[arg-type]
    with pytest.raises(GraphError) as raised:
        build_graph(config.workflows[0], config)  # type: ignore[arg-type]
    assert (str(raised.value), raised.value.code) == (str(error), error.code)  # type: ignore[union-attr]


def test_only_a_service_request_skips_the_graph_check() -> None:
    steps = [{"name": "k", "transform": "toy-keep", "input": "nothere"}]
    with pytest.raises(ValidationError, match="no input or earlier step"):
        chain_pipeline(workflows=[{"name": "w", "inputs": ["a"], "steps": steps}])
    assert chain_pipeline(workflows=[{"name": "w", "inputs": ["a"], "steps": steps}], graphs=False).workflows


def test_task_problems_name_the_task_they_belong_to() -> None:
    config = chain_pipeline(
        workflows=[
            {
                "name": "w",
                "inputs": [{"name": "all", "list": True}],
                "steps": [{"name": "k", "transform": "toy-keep", "input": "all[nope]"}],
            }
        ],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        graphs=False,
    )
    problems = task_problems(config, {"w": build_graph(config.workflows[0], config)})  # type: ignore[arg-type]
    assert [task for task, _ in problems] == ["t"]
    assert "binds sources src to `all`" in problems[0][1]
