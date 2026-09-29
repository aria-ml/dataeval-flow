"""Node keys: what names a Dataset node's cache, and when two nodes share one (spec §5.7)."""

from dataeval.types import RemovalPlan, SourceIndex

from dataeval_flow._chain._identity import (
    element_key,
    indices_digest,
    output_key,
    plan_digest,
    settings_of,
    short_digest,
    step_key,
)
from tests.chain_toys import First, FirstConfig


def test_a_step_key_depends_on_type_settings_inputs_and_digest() -> None:
    base = step_key("remove", {"keep": "first"}, ["src:a"], "d1")
    assert base.startswith("step:")
    assert step_key("remove", {"keep": "first"}, ["src:a"], "d1") == base
    assert step_key("select", {"keep": "first"}, ["src:a"], "d1") != base
    assert step_key("remove", {"keep": "last"}, ["src:a"], "d1") != base
    assert step_key("remove", {"keep": "first"}, ["src:b"], "d1") != base
    assert step_key("remove", {"keep": "first"}, ["src:a"], "d2") != base


def test_two_tasks_binding_one_workflow_to_different_sources_key_apart() -> None:
    assert step_key("view", {}, ["cache:train"], None) != step_key("view", {}, ["cache:test"], None)


def test_the_same_plan_from_different_settings_digests_alike() -> None:
    first = RemovalPlan([SourceIndex(3), SourceIndex(1, 2)])
    second = RemovalPlan([SourceIndex(1, 2)]) | RemovalPlan([3])
    assert plan_digest(first) == plan_digest(second)
    assert plan_digest(first) != plan_digest(RemovalPlan([3]))


def test_a_detection_removal_digests_apart_from_removing_nothing_though_the_items_are_the_same() -> None:
    assert plan_digest(RemovalPlan([SourceIndex(0, 1)])) != plan_digest(RemovalPlan())


def test_indices_digest_follows_order_and_content() -> None:
    assert indices_digest([0, 1, 2]) == indices_digest((0, 1, 2))
    assert indices_digest([0, 1, 2]) != indices_digest([2, 1, 0])


def test_outputs_and_elements_key_apart_from_their_step() -> None:
    base = step_key("kfold", {"folds": 2}, ["k"], None)
    train, val = output_key(base, "train"), output_key(base, "val")
    assert len({base, train, val}) == 3
    assert element_key(train, "0") != element_key(train, "1")
    assert len(short_digest(train)) == 12


def test_settings_leave_out_the_addresses_a_step_reads() -> None:
    ports = First.input_ports()
    assert settings_of(FirstConfig(input="a", n=2), ports) == {"n": 2}
    assert settings_of(FirstConfig(input="b", n=2), ports) == settings_of(FirstConfig(input="a", n=2), ports)
