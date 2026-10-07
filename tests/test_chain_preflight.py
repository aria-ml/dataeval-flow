"""Preflight: each step's policies resolved as a task resolves them today, and Dataset kinds checked (spec §5.4)."""

import re

import pytest

from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._graph import GraphError, build_graph, one_step_graph
from dataeval_flow._chain._preflight import check_kinds, detect_kind, step_contexts
from dataeval_flow._chain._run import bind_inputs
from dataeval_flow._sources import resolve_source
from dataeval_flow.config import MetadataPolicyConfig, TaskConfig
from dataeval_flow.evaluators.bias import BalanceConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig, FactorLeakageConfig
from dataeval_flow.workflows._context import DatasetContext
from tests.chain_toys import ToyDetections, chain_pipeline, register_toys
from tests.evaluator_toys import ToyFactors, ToyImages


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    return plugins


def _contexts(config, names):
    contexts, resolved = {}, {}
    for name in names:
        source = resolve_source(name, config)
        resolved[name] = source
        contexts[name] = DatasetContext(
            name=name,
            dataset=source.dataset,
            cache=DatasetCache.get_or_create(None, source.cache_name, source.cache_key),
        )
    return contexts, resolved


def test_detect_kind_reads_the_first_datum() -> None:
    assert detect_kind(ToyImages()) == "classification"
    assert detect_kind(ToyDetections([[0], [1]], {0: "a", 1: "b"})) == "object_detection"


@pytest.mark.usefixtures("toys")
def test_a_step_given_a_kind_it_does_not_take_is_refused_before_running() -> None:
    workflow = {
        "name": "w",
        "inputs": ["a"],
        "steps": [{"name": "d", "transform": "toy-detections-only", "input": "a"}],
    }
    config = chain_pipeline(workflows=[workflow])
    graph = build_graph(config.workflows[0], config)  # type: ignore[arg-type,index]
    contexts, resolved = _contexts(config, ["src"])
    inputs = bind_inputs(graph, ["src"], contexts, resolved)
    with pytest.raises(GraphError, match="reads `a`, a classification Dataset, but `input` takes object_detection"):
        check_kinds(graph, inputs)


@pytest.mark.usefixtures("toys")
def test_a_view_whose_operations_take_any_target_keeps_its_input_kind() -> None:
    ops = [{"type": "Relabel", "params": {"class_remap": {"a": "x"}, "target": ["x"]}}]
    workflow = {
        "name": "w",
        "inputs": ["a"],
        "steps": [{"name": "v", "transform": "view", "input": "a", "operations": ops}],
    }
    config = chain_pipeline(workflows=[workflow])
    graph = build_graph(config.workflows[0], config)  # type: ignore[arg-type,index]
    contexts, resolved = _contexts(config, ["src"])
    kinds = check_kinds(graph, bind_inputs(graph, ["src"], contexts, resolved))
    assert kinds["v"] == "classification"


@pytest.mark.usefixtures("toys")
def test_kinds_flow_through_the_chain() -> None:
    workflow = {
        "name": "w",
        "inputs": ["a"],
        "steps": [
            {"name": "k", "transform": "toy-keep", "input": "a"},
            {"name": "d", "transform": "toy-detections-only", "input": "k"},
        ],
    }
    config = chain_pipeline(workflows=[workflow], datasets={"src": ToyDetections([[0], [1]], {0: "a", 1: "b"})})
    graph = build_graph(config.workflows[0], config)  # type: ignore[arg-type,index]
    contexts, resolved = _contexts(config, ["src"])
    assert check_kinds(graph, bind_inputs(graph, ["src"], contexts, resolved)) == {
        "a": "object_detection",
        "k": "object_detection",
        "d": "object_detection",
    }


@pytest.mark.usefixtures("toys")
def test_a_list_mixing_kinds_is_refused_on_the_kind_it_does_not_take() -> None:
    workflow = {
        "name": "w",
        "inputs": [{"name": "all", "list": True}],
        "steps": [{"name": "d", "transform": "toy-detections-only", "input": "all"}],
    }
    config = chain_pipeline(
        workflows=[workflow], datasets={"cls": ToyImages(), "det": ToyDetections([[0], [1]], {0: "a", 1: "b"})}
    )
    graph = build_graph(config.workflows[0], config)  # type: ignore[arg-type,index]
    contexts, resolved = _contexts(config, ["cls", "det"])
    inputs = bind_inputs(graph, ["cls", "det"], contexts, resolved)
    with pytest.raises(GraphError, match="reads `all`, a classification Dataset, but `input` takes object_detection"):
        check_kinds(graph, inputs)


@pytest.mark.usefixtures("toys")
@pytest.mark.parametrize("transform", ["merge", "toy-pair"])
@pytest.mark.parametrize(
    ("order", "listed"),
    [
        (["cls", "det"], "`a` is classification, `b` is object_detection"),
        (["det", "cls"], "`a` is object_detection, `b` is classification"),
    ],
)
def test_one_port_reading_datasets_of_different_kinds_is_refused_before_running(
    transform: str, order: list[str], listed: str
) -> None:
    workflow = {
        "name": "w",
        "inputs": ["a", "b"],
        "steps": [{"name": "m", "transform": transform, "input": ["a", "b"]}],
    }
    datasets = {"cls": ToyImages(), "det": ToyDetections([[0], [1]], {0: "a", 1: "b"})}
    config = chain_pipeline(workflows=[workflow], datasets={name: datasets[name] for name in order})
    graph = build_graph(config.workflows[0], config)  # type: ignore[arg-type,index]
    contexts, resolved = _contexts(config, order)
    inputs = bind_inputs(graph, order, contexts, resolved)
    with pytest.raises(GraphError, match=f"Step 'm' reads Datasets of different kinds on `input`: {listed}\\."):
        check_kinds(graph, inputs)


@pytest.mark.usefixtures("toys")
def test_a_one_step_graph_resolves_its_policy_exactly_as_a_task_does() -> None:
    policy = MetadataPolicyConfig(name="p", exclude=["angle"])
    balance = BalanceConfig(name="balance", metadata="p")
    config = chain_pipeline(evaluators=[balance], datasets={"src": ToyFactors()}, extra={"metadata": [policy]})
    task = TaskConfig(name="t", workflow="balance", kind="evaluator", sources="src")
    graph = one_step_graph(task, balance, ["src"])
    contexts, _ = _contexts(config, ["src"])
    (context,) = step_contexts(graph, config, None, {"src": [contexts["src"]]}).values()

    from dataeval_flow._orchestrator import _resolve_metadata_policy

    expected = _resolve_metadata_policy(balance, config, None)
    assert context.metadata_policy == expected
    assert context.stats_policy is None
    assert context.ontology is None


def _splits_step(ranges: tuple[tuple[float, float], tuple[float, float]]):
    """A graph of two steps reading two slots, metadata's and statistics', the pipeline it came from, and each slot's
    context over `ranges`."""
    from dataeval_flow.config import StatsMeasureConfig, StatsPolicyConfig

    leak = FactorLeakageConfig(name="leak", factors=["p"])
    dupes = DuplicatesConfig(name="dupes", stats="bands")
    measure = [
        StatsMeasureConfig(bands=None, families=["visual"]),
        StatsMeasureConfig(bands="rgb", families=["visual"]),
        StatsMeasureConfig(bands="ir", families=["visual"]),
    ]
    workflow = {
        "name": "w",
        "inputs": ["a", "b"],
        "steps": [
            {"name": "leak", "evaluator": "leak", "input": ["a", "b"]},
            {"name": "dupes", "evaluator": "dupes", "input": ["a", "b"]},
        ],
    }
    config = chain_pipeline(
        workflows=[workflow],
        evaluators=[leak, dupes],
        datasets={"one": ToyImages(), "two": ToyImages(seed=1)},
        extra={"stats": [StatsPolicyConfig(name="bands", measure=measure)]},
    )
    graph = build_graph(config.workflows[0], config)  # type: ignore[arg-type,index]
    contexts = {
        "a": [
            DatasetContext(
                name="one", dataset=ToyImages(), value_range=ranges[0], channel_groups={"rgb": ((0, 1, 2), None)}
            )
        ],
        "b": [
            DatasetContext(
                name="two", dataset=ToyImages(seed=1), value_range=ranges[1], channel_groups={"ir": ((3,), None)}
            )
        ],
    }
    return graph, config, contexts


def test_a_step_reading_two_sources_takes_their_value_range_and_both_their_band_groups() -> None:
    graph, config, contexts = _splits_step(((0.0, 255.0), (0.0, 255.0)))
    resolved = step_contexts(graph, config, None, contexts)
    metadata, stats = resolved["leak"].metadata_policy, resolved["dupes"].stats_policy
    assert metadata is not None
    assert stats is not None
    assert metadata.value_range == (0.0, 255.0)
    assert stats.channels == (("ir", ((3,), None)), ("rgb", ((0, 1, 2), None)))


def test_a_step_reading_two_sources_of_different_value_ranges_is_refused() -> None:
    graph, config, contexts = _splits_step(((0.0, 1.0), (0.0, 255.0)))
    message = (
        "Evaluator 'leak' reads datasets declaring different `value_range`s ((0.0, 1.0) and (0.0, 255.0)). "
        "Statistics measured on different pixel scales are not comparable, so there is no right answer to pick — give "
        "the datasets one range, or run them as separate tasks."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        step_contexts(graph, config, None, contexts)
