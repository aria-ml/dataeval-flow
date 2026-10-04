"""The `leakage` check: items in groups spanning two splits, and group values two splits share (audit spec §10.2)."""

from types import SimpleNamespace
from typing import Any

from dataeval_flow import run
from dataeval_flow.evaluators.quality import DuplicatesConfig, FactorLeakageOutput
from dataeval_flow.steps import CheckContext
from dataeval_flow.steps.checks import LeakageCheck, LeakageConfig
from tests.evaluator_toys import Items, ToyImages


def _duplicates_node() -> SimpleNamespace:
    """A `duplicates` Output over train and test, where test's item 3 is train's item 3."""
    train = ToyImages(12, seed=1, near_duplicate=True)
    test_items = [ToyImages(12, seed=2)[index] for index in range(12)]
    test_items[3] = train[3]
    sources: dict[str, Any] = {"train": train, "test": Items(test_items)}
    output = run(DuplicatesConfig(), sources).output
    on = (SimpleNamespace(address="train"), SimpleNamespace(address="evals[test]"))
    return SimpleNamespace(value=output, computed_on=on, address="dupes[test]", items=24)


def _factors_node(shared: bool) -> SimpleNamespace:
    counts = {"s0": [10, 3 if shared else 0], "s1": [0, 5]}
    output = FactorLeakageOutput({"sources": ["train", "test"], "items": [10, 8], "factors": {"scene": counts}}, None)
    on = (SimpleNamespace(address="train"), SimpleNamespace(address="evals[test]"))
    return SimpleNamespace(value=output, computed_on=on, address="groups[test]")


def _judge(inputs: dict[str, Any], **limits: Any) -> Any:
    config = LeakageConfig(duplicates="dupes", **({"factors": "groups"} if "factors" in inputs else {}), **limits)
    (finding,) = LeakageCheck().run(config, inputs, CheckContext("t", "s"))
    return finding


def test_items_in_groups_spanning_two_splits_warn() -> None:
    finding = _judge({"duplicates": _duplicates_node()})
    assert finding.severity == "warning"
    assert finding.title == "Leakage"
    assert "exact" in finding.brief
    assert "cross-split duplicates" in finding.brief


def test_groups_within_one_split_do_not_leak() -> None:
    train = ToyImages(12, seed=1)  # item 5 copies item 0, within train only
    # ToyImages plants its solid-white item from the 8th item on, so seven items hold none.
    test = Items([ToyImages(7, seed=3)[index] for index in range(7)])
    sources: dict[str, Any] = {"train": train, "test": test}
    output = run(DuplicatesConfig(), sources).output
    on = (SimpleNamespace(address="train"), SimpleNamespace(address="evals[test]"))
    node = SimpleNamespace(value=output, computed_on=on, address="dupes[test]")
    finding = _judge({"duplicates": node})
    assert finding.severity == "ok"
    assert finding.brief == "No cross-split duplicates"


def test_a_shared_group_value_warns_and_none_does_not() -> None:
    clean = _judge({"duplicates": [], "factors": _factors_node(shared=False)}, exact=None, near=None)
    leaked = _judge({"duplicates": [], "factors": _factors_node(shared=True)}, exact=None, near=None)
    assert clean.severity == "ok"
    assert leaked.severity == "warning"
    assert "1 shared group value" in leaked.brief


def test_with_every_limit_off_it_judges_nothing() -> None:
    finding = _judge({"duplicates": _duplicates_node()}, exact=None, near=None, groups=None)
    assert finding.severity == "info"


def test_it_reads_each_shape_the_engine_hands_it() -> None:
    node = _duplicates_node()
    listed = SimpleNamespace(present={"test": node}, elements={"test": node})
    for value in (node, listed, [listed], [listed, SimpleNamespace(present={}, elements={})]):
        assert _judge({"duplicates": value}).severity == "warning"


def _chain(steps: list[dict[str, Any]]) -> Any:
    """A chain over two splits that hold the same items, run through the engine."""
    from dataeval_flow._cache import DatasetCache
    from dataeval_flow.evaluators.quality import FactorLeakageConfig
    from dataeval_flow.steps import ChainResult
    from tests.chain_toys import chain_pipeline, run_chain_task
    from tests.evaluator_toys import ToyFactors

    DatasetCache.clear_instances()
    datasets = {"s1": ToyFactors(12), "s2": ToyFactors(12)}
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": [{"name": "cams", "list": True}], "steps": steps}],
        evaluators=[DuplicatesConfig(name="dupes"), FactorLeakageConfig(name="fl", factors=["site"])],
        tasks=[{"name": "t", "workflow": "w", "sources": list(datasets)}],
        datasets=datasets,
    )
    result = run_chain_task(config)
    assert isinstance(result, ChainResult)
    return result


_BOOM = {"name": "boom", "transform": "toy-explode", "input": "cams", "optional": True}
_JUDGE = {"name": "judge", "check": "leakage", "duplicates": "dupes", "factors": "groups"}


def test_a_factors_list_that_holds_nothing_leaves_the_duplicates_judged(plugins) -> None:
    from tests.chain_toys import register_toys

    register_toys(plugins)
    result = _chain(
        [
            {"name": "dupes", "evaluator": "dupes", "input": "cams", "pairs": True},
            _BOOM,
            {"name": "groups", "evaluator": "fl", "input": "boom", "pairs": True},
            _JUDGE,
        ]
    )
    judge = result.steps["judge"]
    assert judge.not_assessed is None
    (finding,) = (item for item in result.findings if item.step == "judge")
    assert (finding.severity, finding.brief) == ("warning", "24 exact cross-split duplicates")


def test_a_duplicates_list_that_holds_nothing_leaves_the_check_not_assessed(plugins) -> None:
    from tests.chain_toys import register_toys

    register_toys(plugins)
    result = _chain(
        [
            _BOOM,
            {"name": "dupes", "evaluator": "dupes", "input": "boom", "pairs": True},
            {"name": "groups", "evaluator": "fl", "input": "cams", "pairs": True},
            _JUDGE,
        ]
    )
    reason = "`dupes` holds no element; `dupes[s1_vs_s2]` was skipped: needs `boom[s1]`, which was skipped"
    assert result.steps["judge"].not_assessed == reason
