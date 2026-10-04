"""`pairs: true`: a step runs once per unordered pair of a list's elements, keyed `a_vs_b` (audit spec §9.2)."""

from typing import Any

import pytest

from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._graph import GraphError
from dataeval_flow._chain._nodes import Node, NodeList
from dataeval_flow.evaluators.bias import BalanceConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult, CustomWorkflowConfig
from tests.chain_toys import chain_pipeline, register_toys, run_chain_task, run_toy_chain
from tests.evaluator_toys import ToyImages

_CAMS = [{"name": "cams", "list": True}]
_PAIRS = {"name": "dupes", "evaluator": "dupes", "input": "cams", "pairs": True}


@pytest.fixture(autouse=True)
def _toys(plugins):
    register_toys(plugins)
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _config(steps: list[dict[str, Any]], sources: list[str], inputs: list[Any] | None = None) -> Any:
    return chain_pipeline(
        workflows=[{"name": "w", "inputs": inputs or _CAMS, "steps": steps}],
        evaluators=[DuplicatesConfig(name="dupes"), BalanceConfig(name="balance")],
        tasks=[{"name": "t", "workflow": "w", "sources": sources}],
        datasets={name: ToyImages(seed=index) for index, name in enumerate(sources)},
    )


def test_a_pairwise_step_runs_once_per_unordered_pair_in_list_order() -> None:
    run = run_toy_chain(_config([_PAIRS], ["s1", "s2", "s3"]), "w", ["s1", "s2", "s3"])
    elements = run.steps["dupes"].elements
    assert elements is not None
    assert list(elements) == ["s1_vs_s2", "s1_vs_s3", "s2_vs_s3"]
    assert elements["s1_vs_s3"].inputs == ["cams[s1]", "cams[s3]"]
    nodes = run.nodes["dupes"]
    assert isinstance(nodes, NodeList)
    node = nodes.elements["s2_vs_s3"]
    assert isinstance(node, Node)
    assert [on.address for on in node.computed_on] == ["cams[s2]", "cams[s3]"]


def test_two_elements_make_one_pair() -> None:
    run = run_toy_chain(_config([_PAIRS], ["s1", "s2"]), "w", ["s1", "s2"])
    elements = run.steps["dupes"].elements
    assert elements is not None
    assert list(elements) == ["s1_vs_s2"]


def test_one_element_makes_no_pair_and_one_record_says_so() -> None:
    result = run_chain_task(_config([_PAIRS], ["s1"]))
    assert isinstance(result, ChainResult)
    dupes = result.steps["dupes"]
    assert (dupes.status, dupes.not_assessed) == ("skipped", "`cams` holds one element, so it has no pair")


def test_a_pair_with_a_failed_member_is_skipped_and_the_rest_run() -> None:
    steps = [
        {"name": "boom", "transform": "toy-explode", "input": "cams", "only": "cams[s2]", "optional": True},
        {"name": "dupes", "evaluator": "dupes", "input": "boom", "pairs": True},
    ]
    run = run_toy_chain(_config(steps, ["s1", "s2", "s3"]), "w", ["s1", "s2", "s3"])
    elements = run.steps["dupes"].elements
    assert elements is not None
    assert elements["s1_vs_s3"].status == "ok"
    assert elements["s1_vs_s2"].status == "skipped"
    assert "boom[s2]" in (elements["s1_vs_s2"].reason or "")
    assert elements["s2_vs_s3"].status == "skipped"


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"name": "x", "evaluator": "dupes", "input": "a", "pairs": True}, "reads one list"),
        ({"name": "x", "evaluator": "dupes", "input": ["a", "cams"], "pairs": True}, "reads one list"),
        ({"name": "x", "transform": "toy-keep", "input": "cams", "pairs": True}, "reads one item, not two"),
        ({"name": "x", "evaluator": "balance", "input": "cams", "pairs": True}, "`pairs:` hands it two"),
    ],
)
def test_a_step_that_cannot_read_pairs_is_refused_at_load(step: dict[str, Any], message: str) -> None:
    with pytest.raises((GraphError, ValueError), match=message):
        _config([step], ["s0", "s1", "s2"], inputs=["a", {"name": "cams", "list": True}])


def test_an_evaluator_step_takes_pairs_and_it_round_trips() -> None:
    workflow = CustomWorkflowConfig.model_validate({"name": "w", "inputs": _CAMS, "steps": [_PAIRS]})
    assert workflow.steps[0].pairs is True
    assert "pairs: true" in workflow.to_yaml()


_CLASH = ["a", "b_vs_c", "a_vs_b", "c"]
_CLASH_MESSAGE = (
    r"Step 'dupes' pairs `cams`, whose elements `a` and `b_vs_c`, and `a_vs_b` and `c`, both give the pair key "
    r"`a_vs_b_vs_c`: rename a source"
)


def test_two_pairs_sharing_a_key_are_refused_at_load() -> None:
    with pytest.raises(ValueError, match=_CLASH_MESSAGE):
        _config([_PAIRS], _CLASH)


def test_two_pairs_sharing_a_key_fail_the_step_when_the_keys_were_unknown_at_load() -> None:
    """`run_toy_chain` builds its graph without the task's source names, as a list of unknown keys does."""
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": _CAMS, "steps": [_PAIRS]}],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "workflow": "w", "sources": ["a", "c"]}],
        datasets={name: ToyImages(seed=index) for index, name in enumerate(_CLASH)},
    )
    run = run_toy_chain(config, "w", _CLASH)
    dupes = run.steps["dupes"]
    assert dupes.status == "failed"
    assert "both give the pair key `a_vs_b_vs_c`" in dupes.errors[0]


_RATE = {"name": "rate", "check": "duplicate-rate", "input": "dupes"}


def test_a_check_over_a_pairwise_step_runs_once_per_pair() -> None:
    run = run_toy_chain(_config([_PAIRS, _RATE], ["s1", "s2", "s3"]), "w", ["s1", "s2", "s3"])
    elements = run.steps["rate"].elements
    assert elements is not None
    assert list(elements) == ["s1_vs_s2", "s1_vs_s3", "s2_vs_s3"]
    assert elements["s1_vs_s3"].inputs == ["dupes[s1_vs_s3]"]


def test_a_check_over_a_pairwise_step_of_one_element_says_it_has_no_pair() -> None:
    result = run_chain_task(_config([_PAIRS, _RATE], ["s1"]))
    assert isinstance(result, ChainResult)
    assert result.steps["rate"].not_assessed == "`cams` holds one element, so it has no pair"


def test_a_preset_step_with_pairs_is_refused(plugins) -> None:
    from tests.preset_toys import register_presets

    register_presets(plugins)
    step = {"name": "p", "workflow": "toy", "input": "cams", "pairs": True}
    with pytest.raises((GraphError, ValueError), match=r"a preset, which runs no pairs: remove `pairs:`\."):
        chain_pipeline(
            workflows=[
                {"name": "w", "inputs": _CAMS, "steps": [step]},
                {"name": "toy", "type": "toy-preset"},
            ],
            tasks=[{"name": "t", "workflow": "w", "sources": ["s1", "s2"]}],
            datasets={"s1": ToyImages(seed=1), "s2": ToyImages(seed=2)},
        )


def test_computed_on_resolves_a_pair_key_to_its_two_elements() -> None:
    from dataeval_flow._chain._graph import ValueType, _computed_on, build_graph
    from dataeval_flow.steps._address import parse_address
    from dataeval_flow.steps._port import DataType

    config = _config([_PAIRS, _RATE], ["s1", "s2", "s3"])
    graph = build_graph(config.workflows[0], config, slot_keys={"cams": ("s1", "s2", "s3")})
    specs = {spec.name: spec for spec in graph.steps}
    keys = ("s1", "s2", "s3")
    types = {
        "cams": ValueType(DataType.DATASET, is_list=True, keys=keys),
        "dupes": ValueType(DataType.OUTPUT, is_list=True, keys=("s1_vs_s2", "s1_vs_s3", "s2_vs_s3")),
    }
    assert _computed_on(parse_address("dupes[s1_vs_s3]"), specs, types) == ("cams[s1]", "cams[s3]")
    assert _computed_on(parse_address("rate[s2_vs_s3]"), specs, types) == ("cams[s2]", "cams[s3]")
