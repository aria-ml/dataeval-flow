"""Running a graph: order, failures, skips, broadcasting, and derivation per node (spec §4.2, §5.5, §5.6)."""

from typing import Any
from unittest.mock import patch

import pytest
from dataeval.quality import DuplicatesOutput

from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._nodes import Missing, Node, NodeList
from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesEvaluator
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig, DataCleaningResult
from tests.chain_toys import chain_pipeline, register_toys, run_toy_chain
from tests.evaluator_toys import ToyImages

_DUPES = [DuplicatesConfig(name="dupes")]

pytestmark = pytest.mark.usefixtures("toys")


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    DatasetCache.clear_instances()
    yield plugins
    DatasetCache.clear_instances()


def _run(
    steps: list[dict[str, Any]],
    inputs: list[Any] | None = None,
    datasets: dict[str, Any] | None = None,
    **pools: Any,
):
    datasets = datasets or {"src": ToyImages()}
    workflow = {"name": "w", "inputs": inputs or ["a"], "steps": steps}
    config = chain_pipeline(
        workflows=[workflow, *pools.pop("workflows", [])], evaluators=_DUPES, datasets=datasets, **pools
    )
    return run_toy_chain(config, "w", list(datasets))


def test_steps_run_in_order_each_reading_what_the_last_made() -> None:
    run = _run(
        [
            {"name": "kept", "transform": "toy-keep", "input": "a"},
            {"name": "few", "transform": "toy-first", "input": "kept", "n": 3},
            {"name": "dupes", "evaluator": "dupes", "input": "few"},
        ]
    )
    assert [record.status for record in run.steps.values()] == ["ok", "ok", "ok"]
    assert len(run.nodes["few"].value) == 3  # type: ignore[union-attr]
    assert isinstance(run.steps["dupes"].output, DuplicatesOutput)
    assert run.steps["dupes"].inputs == ["few"]


def test_a_failure_skips_what_reads_it_and_nothing_else() -> None:
    run = _run(
        [
            {"name": "boom", "transform": "toy-explode", "input": "a"},
            {"name": "after", "transform": "toy-keep", "input": "boom"},
            {"name": "dupes", "evaluator": "dupes", "input": "a"},
        ]
    )
    assert run.steps["boom"].status == "failed"
    assert run.steps["boom"].errors == ["RuntimeError: boom on a"]
    assert (run.steps["after"].status, run.steps["after"].reason) == ("skipped", "needs `boom`, which failed")
    assert run.steps["dupes"].status == "ok"


def test_an_optional_failure_is_a_skip_carrying_its_error() -> None:
    run = _run(
        [
            {"name": "boom", "transform": "toy-explode", "input": "a", "optional": True},
            {"name": "after", "transform": "toy-keep", "input": "boom"},
        ]
    )
    boom = run.steps["boom"]
    assert (boom.status, boom.errors) == ("skipped", ["RuntimeError: boom on a"])
    assert boom.reason == "failed: RuntimeError: boom on a"
    assert run.steps["after"].reason == "needs `boom`, which was skipped"


def test_a_list_runs_a_single_item_step_once_per_element() -> None:
    run = _run(
        [
            {"name": "kept", "transform": "toy-keep", "input": "all"},
            {"name": "few", "transform": "toy-first", "input": "kept", "n": 2},
        ],
        inputs=[{"name": "all", "list": True}],
        datasets={"src": ToyImages(), "more": ToyImages(seed=1)},
    )
    few = run.nodes["few"]
    assert isinstance(few, NodeList)
    assert list(few.elements) == ["src", "more"]
    assert [len(node.value) for node in few.present.values()] == [2, 2]
    assert few.elements["more"].address == "few[more]"  # type: ignore[union-attr]
    assert set(run.steps["few"].elements or {}) == {"src", "more"}


def test_a_failed_element_keeps_the_others_and_skips_only_its_own_descendants() -> None:
    original = DuplicatesEvaluator.run

    def fails_on_more(self: Any, config: Any, inputs: Any) -> Any:
        if inputs[0].source == "kept[more]":
            raise RuntimeError("no stats for more")
        return original(self, config, inputs)

    with patch.object(DuplicatesEvaluator, "run", fails_on_more):
        run = _run(
            [
                {"name": "kept", "transform": "toy-keep", "input": "all"},
                {"name": "dupes", "evaluator": "dupes", "input": "kept"},
            ],
            inputs=[{"name": "all", "list": True}],
            datasets={"src": ToyImages(), "more": ToyImages(seed=1)},
        )
    dupes = run.steps["dupes"]
    assert dupes.status == "failed"
    assert {key: element.status for key, element in (dupes.elements or {}).items()} == {"src": "ok", "more": "failed"}
    assert dupes.elements["more"].errors == ["RuntimeError: no stats for more"]  # type: ignore[index]
    assert isinstance(run.nodes["dupes"].elements["more"], Missing)  # type: ignore[union-attr]


def test_lists_zip_by_key_and_a_key_missing_from_one_skips_that_element() -> None:
    run = _run(
        [
            {"name": "parts", "transform": "toy-spread", "input": "one", "parts": 2},
            {"name": "dupes", "evaluator": "dupes", "input": ["parts", "all"]},
        ],
        inputs=["one", {"name": "all", "list": True}],
        datasets={"src": ToyImages(), "more": ToyImages(seed=1)},
    )
    elements = run.steps["dupes"].elements or {}
    assert list(elements) == ["0", "1", "more"]
    assert {element.status for element in elements.values()} == {"skipped"}
    assert elements["0"].reason == "`all` has no element `0`"
    assert elements["more"].reason == "`parts` has no element `more`"


def test_two_steps_reading_one_node_derive_its_statistics_once() -> None:
    from dataeval_flow import _cache

    with patch.object(_cache, "_do_compute_stats", wraps=_cache._do_compute_stats) as compute:
        run = _run(
            [
                {"name": "few", "transform": "toy-first", "input": "a", "n": 6},
                {"name": "d1", "evaluator": "dupes", "input": "few"},
                {"name": "d2", "evaluator": "dupes", "input": "few"},
            ]
        )
    assert {run.steps["d1"].status, run.steps["d2"].status} == {"ok"}
    assert compute.call_count == 1


def test_a_workflow_type_runs_as_a_step_on_an_intermediate_dataset() -> None:
    clean = DataCleaningConfig(name="clean", outlier_method="zscore", outlier_flags=["pixel"])
    run = _run(
        [
            {"name": "few", "transform": "toy-first", "input": "a", "n": 8},
            {"name": "cleaned", "workflow": "clean", "input": "few"},
        ],
        workflows=[clean],
    )
    assert run.steps["cleaned"].status == "ok"
    assert isinstance(run.steps["cleaned"].result, DataCleaningResult)


def test_lineage_records_each_dataset_with_where_it_came_from() -> None:
    run = _run(
        [
            {"name": "kept", "transform": "toy-keep", "input": "a"},
            {"name": "few", "transform": "toy-first", "input": "kept", "n": 3},
        ]
    )
    records = {record.name: record for record in run.lineage}
    assert list(records) == ["a", "kept", "few"]
    assert (records["a"].source, records["a"].step, records["a"].items) == ("src", None, 12)
    assert (records["few"].step, records["few"].type, records["few"].inputs, records["few"].items) == (
        "few",
        "toy-first",
        ["kept"],
        3,
    )
    assert len({record.digest for record in records.values()}) == 3


def test_lineage_over_a_list_records_each_datasets_elements_and_no_outputs() -> None:
    run = _run(
        [
            {"name": "kept", "transform": "toy-keep", "input": "all"},
            {"name": "dupes", "evaluator": "dupes", "input": "kept"},
        ],
        inputs=[{"name": "all", "list": True}],
        datasets={"src": ToyImages(), "more": ToyImages(seed=1)},
    )
    assert run.steps["dupes"].status == "ok"
    assert [record.name for record in run.lineage] == ["all[src]", "all[more]", "kept[src]", "kept[more]"]


def test_an_input_node_keeps_its_sources_cache_key() -> None:
    run = _run([{"name": "kept", "transform": "toy-keep", "input": "a"}])
    node = run.nodes["a"]
    assert isinstance(node, Node)
    assert node.key is not None
    assert node.key.startswith("src_data:maite:")
