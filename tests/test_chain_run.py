"""Running a graph: order, failures, skips, broadcasting, and derivation per node (spec §4.2, §5.5, §5.6)."""

import logging
from collections.abc import Mapping
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from dataeval.quality import DuplicatesOutput

from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._nodes import Missing, Node, NodeList
from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesEvaluator
from dataeval_flow.steps import DataType, Port, Transform, TransformConfig, TransformContext
from tests.chain_toys import chain_pipeline, register_toys, run_toy_chain
from tests.evaluator_toys import ToyImages
from tests.workflow_toys import ToyCountConfig, ToyCountResult, register_count

_DUPES = [DuplicatesConfig(name="dupes")]

pytestmark = pytest.mark.usefixtures("toys")


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    register_count(plugins)
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


def test_two_steps_reading_one_node_through_an_unseeded_shuffle_read_one_draw() -> None:
    from dataeval_flow.config import SourceConfig

    run = _run(
        [{"name": "d1", "evaluator": "dupes", "input": "a"}, {"name": "d2", "evaluator": "dupes", "input": "a"}],
        datasets={"src": ToyImages(count=40, near_duplicate=True)},
        extra={
            "views": [{"name": "shuffled", "operations": [{"type": "Shuffle", "params": {}}]}],
            "sources": [SourceConfig(name="src", dataset="src_data", view="shuffled")],
        },
    )
    first, second = run.steps["d1"].output, run.steps["d2"].output
    assert first.data().height > 0
    assert first.data().equals(second.data())


def test_a_workflow_type_runs_as_a_step_on_an_intermediate_dataset() -> None:
    clean = ToyCountConfig(name="clean")
    run = _run(
        [
            {"name": "few", "transform": "toy-first", "input": "a", "n": 8},
            {"name": "cleaned", "workflow": "clean", "input": "few"},
        ],
        workflows=[clean],
    )
    assert run.steps["cleaned"].status == "ok"
    assert isinstance(run.steps["cleaned"].result, ToyCountResult)


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
    assert (records["kept"].step, records["kept"].type, records["kept"].inputs, records["kept"].source) == (
        "kept",
        "toy-keep",
        ["a"],
        None,
    )
    assert run.nodes["kept"].value is run.nodes["a"].value  # type: ignore[union-attr]
    # `toy-keep` hands its input on as it is, so `kept` shares `a`'s key and digest; `few` is new (coverage spec §17).
    assert records["kept"].digest == records["a"].digest != records["few"].digest


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


def _failure_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "dataeval_flow._chain._run"]


def test_an_optional_failure_logs_a_warning_without_a_traceback(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="dataeval_flow"):
        _run([{"name": "boom", "transform": "toy-explode", "input": "a", "optional": True}])
    (record,) = _failure_records(caplog)
    assert (record.levelno, record.exc_info) == (logging.WARNING, None)
    assert record.getMessage() == "Optional step 'boom' failed, so it is skipped: RuntimeError: boom on a"


def test_a_required_failure_logs_an_error_with_its_traceback(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="dataeval_flow"):
        _run([{"name": "boom", "transform": "toy-explode", "input": "a"}])
    (record,) = _failure_records(caplog)
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert record.getMessage() == "Step 'boom' failed"


class NoLengthConfig(TransformConfig):
    input: str


class NoLength(Transform[NoLengthConfig]):
    """A faulty plugin: its Dataset output has no length."""

    name: ClassVar[str] = "toy-no-length"
    description: ClassVar[str] = "Returns an object with no length."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    def run(self, config: NoLengthConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        return {"output": object()}


def test_a_transform_returning_a_dataset_with_no_length_fails_its_step_and_the_chain_goes_on(plugins) -> None:
    plugins["dataeval_flow.transforms"].append(("toy-no-length", "tests.test_chain_run:NoLength"))
    run = _run(
        [
            {"name": "bad", "transform": "toy-no-length", "input": "a"},
            {"name": "after", "transform": "toy-keep", "input": "bad"},
            {"name": "other", "transform": "toy-keep", "input": "a"},
        ]
    )
    bad = run.steps["bad"]
    assert bad.status == "failed"
    assert bad.errors == [
        (
            "TypeError: transform 'toy-no-length' returned object for `output`, which has no length: a Dataset must "
            "have one."
        )
    ]
    assert (run.steps["after"].status, run.steps["after"].reason) == ("skipped", "needs `bad`, which failed")
    assert run.steps["other"].status == "ok"
    assert [record.name for record in run.lineage] == ["a", "other"]


_TWO_SOURCES = {"src": ToyImages(), "more": ToyImages(seed=1)}


@pytest.mark.parametrize(("optional", "reason"), [(False, "failed"), (True, "was skipped")])
def test_a_whole_list_port_gets_the_elements_that_succeeded_and_why_each_other_is_missing(
    optional: bool, reason: str
) -> None:
    from tests.chain_toys import Gather

    received: dict[str, str] = {}
    original = Gather.run

    def records_what_it_got(self: Any, config: Any, inputs: Any, context: Any) -> Any:
        received.update({key: getattr(value, "reason", "present") for key, value in inputs["input"].elements.items()})
        return original(self, config, inputs, context)

    with patch.object(Gather, "run", records_what_it_got):
        run = _run(
            [
                {"name": "boom", "transform": "toy-explode", "input": "all", "only": "all[more]", "optional": optional},
                {"name": "gathered", "transform": "toy-gather", "input": "boom"},
            ],
            inputs=[{"name": "all", "list": True}],
            datasets=_TWO_SOURCES,
        )
    assert received == {"src": "present", "more": reason}
    assert run.steps["gathered"].status == "ok"
    assert len(run.nodes["gathered"].value) == 12  # type: ignore[union-attr]


def test_an_element_whose_input_failed_is_skipped_with_that_elements_reason() -> None:
    run = _run(
        [
            {"name": "boom", "transform": "toy-explode", "input": "all", "only": "all[more]"},
            {"name": "after", "transform": "toy-keep", "input": "boom"},
        ],
        inputs=[{"name": "all", "list": True}],
        datasets=_TWO_SOURCES,
    )
    after = run.steps["after"]
    elements = after.elements or {}
    assert after.status == "ok"
    assert (elements["src"].status, elements["src"].reason) == ("ok", None)
    assert (elements["more"].status, elements["more"].reason) == ("skipped", "needs `boom[more]`, which failed")
    assert isinstance(run.nodes["after"].elements["more"], Missing)  # type: ignore[union-attr]
