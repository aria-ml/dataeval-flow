"""A chain computes each Dataset's statistics in one pass over every family its evaluator steps read (spec §5.5).

Each run goes through the orchestrator, so the preflight plans it, on a cold cache. The flags are what DataEval was
asked to compute, in order: on the toy images, `quality.outliers` with `flags: [pixel, visual]` reads
`PIXEL | VISUAL`, and `quality.duplicates` by default reads `HASH_DUPLICATES_BASIC`.
"""

from typing import Any
from unittest.mock import patch

import pytest
from dataeval.flags import ImageStats

import dataeval_flow._cache as cache_module
from dataeval_flow import PipelineConfig, run_tasks
from dataeval_flow._cache import DatasetCache, active_cache, selection_repr
from dataeval_flow._stats import ResolvedStatsPolicy
from dataeval_flow.config import DatasetProtocolConfig, SourceConfig, TaskConfig
from dataeval_flow.evaluators._producers import ProducerContext, produce_stats
from dataeval_flow.evaluators.quality import DuplicatesConfig, OutliersConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows import DatasetContext, WorkflowContext
from tests.chain_toys import chain_pipeline, register_toys
from tests.evaluator_toys import ToyImages

_EVALUATORS = [OutliersConfig(name="outliers", flags=["pixel", "visual"]), DuplicatesConfig(name="dupes")]
_BOTH = [
    {"name": "outliers", "evaluator": "outliers", "input": "data"},
    {"name": "dupes", "evaluator": "dupes", "input": "data"},
]


@pytest.fixture(autouse=True)
def _cold_cache(plugins):
    register_toys(plugins)
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _run(config: PipelineConfig, task: str = "t") -> tuple[ChainResult, list[tuple[int, dict[str | None, Any]]]]:
    """Run `task`: its result, and each computation of statistics it made, in order, as how many items it read and
    what it asked for."""
    with patch.object(cache_module, "_do_compute_stats", wraps=cache_module._do_compute_stats) as compute:
        result = run_tasks(config, task)[task]
    assert isinstance(result, ChainResult)
    return result, [(len(call.args[0]), call.args[1].request) for call in compute.call_args_list]


def _all_ok(result: ChainResult) -> None:
    assert result.success, result.errors
    assert {name: step.status for name, step in result.steps.items()} == dict.fromkeys(result.steps, "ok")


def _computed(config: PipelineConfig, task: str = "t") -> list[dict[str | None, ImageStats]]:
    """What each computation of statistics running `task` asked for, in order; every step must succeed."""
    result, computed = _run(config, task)
    _all_ok(result)
    return [request for _, request in computed]


def _custom(steps: list[dict[str, Any]], inputs: list[Any] | None = None, **datasets: Any) -> PipelineConfig:
    """Custom workflow `w` of `steps`, run by task `t` on `datasets` (one `src` of 12 toy images by default)."""
    datasets = datasets or {"src": ToyImages(count=12)}
    return chain_pipeline(
        workflows=[{"name": "w", "inputs": inputs or ["data"], "steps": steps}],
        evaluators=_EVALUATORS,
        tasks=[{"name": "t", "workflow": "w", "sources": list(datasets)}],
        datasets=datasets,
    )


def test_a_data_cleaning_task_computes_its_statistics_once() -> None:
    cleaning = {
        "name": "cleaning",
        "type": "data-cleaning",
        "outlier_method": "zscore",
        "outlier_flags": ["pixel", "visual"],
    }
    config = chain_pipeline(
        workflows=[cleaning],
        tasks=[{"name": "t", "workflow": "cleaning", "sources": ["src"]}],
        datasets={"src": ToyImages(count=12)},
    )
    assert _computed(config) == [{None: ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.HASH_DUPLICATES_BASIC}]


def test_outliers_and_duplicates_on_one_input_compute_once() -> None:
    assert _computed(_custom(_BOTH)) == [
        {None: ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.HASH_DUPLICATES_BASIC}
    ]


def test_each_evaluator_finds_what_it_finds_alone() -> None:
    config = _custom(_BOTH)
    alone = {}
    for name in ("outliers", "dupes"):
        DatasetCache.clear_instances()
        task = TaskConfig(name=name, workflow=name, kind="evaluator", sources="src")
        alone[name] = run_tasks(config.model_copy(update={"tasks": [task]}), name)[name]
    DatasetCache.clear_instances()
    chain = run_tasks(config, "t")["t"]
    assert isinstance(chain, ChainResult)
    for name, result in alone.items():
        assert chain.steps[name].result.to_dict()["output"] == result.to_dict()["output"]  # type: ignore[union-attr]


def test_a_step_reading_what_another_made_computes_on_that_node() -> None:
    steps = [
        {"name": "outliers", "evaluator": "outliers", "input": "data"},
        {"name": "clean", "transform": "remove", "input": "data", "plans": {"outliers": {"min_flags": 1}}},
        {"name": "dupes", "evaluator": "dupes", "input": "clean"},
    ]
    assert _computed(_custom(steps)) == [
        {None: ImageStats.PIXEL | ImageStats.VISUAL},
        {None: ImageStats.HASH_DUPLICATES_BASIC},
    ]


def test_each_element_of_a_list_slot_computes_once() -> None:
    config = _custom(_BOTH, [{"name": "data", "list": True}], src=ToyImages(count=12), more=ToyImages(count=12, seed=1))
    union = {None: ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.HASH_DUPLICATES_BASIC}
    assert _computed(config) == [union, union]


def test_each_element_of_a_list_a_step_made_computes_once() -> None:
    steps = [
        {"name": "parts", "transform": "toy-spread", "input": "data", "parts": 2},
        {"name": "outliers", "evaluator": "outliers", "input": "parts"},
        {"name": "dupes", "evaluator": "dupes", "input": "parts"},
    ]
    union = {None: ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.HASH_DUPLICATES_BASIC}
    assert _computed(_custom(steps)) == [union, union]


def test_an_element_one_step_names_alone_computes_what_the_list_s_readers_read_too() -> None:
    steps = [
        {"name": "parts", "transform": "toy-spread", "input": "data", "parts": 2},
        {"name": "outliers", "evaluator": "outliers", "input": "parts"},
        {"name": "dupes", "evaluator": "dupes", "input": "parts[0]"},
    ]
    assert _computed(_custom(steps)) == [
        {None: ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.HASH_DUPLICATES_BASIC},
        {None: ImageStats.PIXEL | ImageStats.VISUAL},
    ]


def test_a_step_whose_stats_request_is_refused_fails_alone_and_the_others_still_compute_once() -> None:
    steps = [{"name": "unhashed", "evaluator": "unhashed", "input": "data"}, *_BOTH]
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        evaluators=[*_EVALUATORS, DuplicatesConfig(name="unhashed", stats="pixels")],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        extra={"stats": [{"name": "pixels", "measure": [{"bands": None, "families": ["pixel"]}]}]},
    )
    result, computed = _run(config)
    assert computed == [(12, {None: ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.HASH_DUPLICATES_BASIC})]
    assert {name: step.status for name, step in result.steps.items()} == {
        "unhashed": "failed",
        "outliers": "ok",
        "dupes": "ok",
    }
    refused = (
        "ValueError: Stats policy 'pixels' does not measure dhash, phash, xxhash, which `flags` asks for on view '~'. "
        "`measure` is a complete statement, so add those families to its `{bands: ~}` entry, or stop asking for them."
    )
    assert result.steps["unhashed"].errors == [refused]


def test_steps_reading_a_preset_step_s_output_compute_once_on_the_node_it_names() -> None:
    cleaning = {
        "name": "cleaning",
        "type": "data-cleaning",
        "outlier_method": "zscore",
        "outlier_flags": ["pixel", "visual"],
    }
    steps = [
        {"name": "cleaning", "workflow": "cleaning", "input": "data"},
        {"name": "outliers", "evaluator": "outliers", "input": "cleaning.clean"},
        {"name": "dupes", "evaluator": "dupes", "input": "cleaning.clean"},
    ]
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}, cleaning],
        evaluators=_EVALUATORS,
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
    )
    result, computed = _run(config)
    _all_ok(result)
    union = {None: ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.HASH_DUPLICATES_BASIC}
    # `data` holds 12 toy images; `cleaning/clean`, the node `cleaning.clean` names, the 10 data-cleaning keeps.
    assert computed == [(12, union), (10, union)]


def _policies(toy_multiband_dataset: Any, outliers: dict[str, Any], dupes: dict[str, Any]) -> PipelineConfig:
    """Outliers and duplicates on one detection dataset, each under its own stats policy."""
    return PipelineConfig.model_validate(
        {
            "datasets": [DatasetProtocolConfig(name="src_data", format="maite", dataset=toy_multiband_dataset)],
            "sources": [SourceConfig(name="src", dataset="src_data")],
            "stats": [{"name": "for_outliers", **outliers}, {"name": "for_dupes", **dupes}],
            "evaluators": [
                OutliersConfig(name="outliers", flags=["pixel", "visual"], stats="for_outliers"),
                DuplicatesConfig(name="dupes", stats="for_dupes"),
            ],
            "workflows": [{"name": "w", "inputs": ["data"], "steps": _BOTH}],
            "tasks": [TaskConfig(name="t", workflow="w", sources="src")],
        }
    )


def test_policies_that_cannot_share_a_cache_entry_compute_separately(toy_multiband_dataset: Any) -> None:
    # `background` is in a policy's scope fragment, as its band groups are; an in-memory dataset declares no groups.
    config = _policies(
        toy_multiband_dataset,
        {"measure": [{"bands": None, "families": ["pixel", "visual"]}], "background": True},
        {"measure": [{"bands": None, "families": ["hash"]}]},
    )
    assert _computed(config) == [{None: ImageStats.PIXEL | ImageStats.VISUAL}, {None: ImageStats.HASH}]


def test_policies_that_share_a_cache_entry_compute_their_union_once(toy_multiband_dataset: Any) -> None:
    config = _policies(
        toy_multiband_dataset,
        {"measure": [{"bands": None, "families": ["pixel", "visual"]}], "background": True},
        {"measure": [{"bands": None, "families": ["visual", "hash"]}], "background": True},
    )
    assert _computed(config) == [{None: ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.HASH}]


def _producer(union: ResolvedStatsPolicy) -> ProducerContext:
    dataset = ToyImages(count=12)
    return ProducerContext(
        source="toy",
        dataset=dataset,  # type: ignore[arg-type]
        dataset_context=DatasetContext(name="toy", dataset=dataset),  # type: ignore[arg-type]
        workflow_context=WorkflowContext(),
        config=DuplicatesConfig(name="dupes"),
        stats_union=union,
    )


_UNION = ResolvedStatsPolicy.of_flags(ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.HASH_DUPLICATES_BASIC)


def test_under_a_cache_the_producer_asks_for_the_union_and_hands_on_its_own_policy() -> None:
    pc = _producer(_UNION)
    with (
        patch.object(cache_module, "_do_compute_stats", wraps=cache_module._do_compute_stats) as compute,
        active_cache(DatasetCache(None, "toy"), selection_repr(pc.dataset)),
    ):
        produced = produce_stats(pc)
    assert [call.args[1].request for call in compute.call_args_list] == [_UNION.request]
    assert produced["stats_policy"] == ResolvedStatsPolicy.of_flags(ImageStats.HASH_DUPLICATES_BASIC)


def test_without_a_cache_the_producer_computes_only_its_own_request() -> None:
    with patch.object(cache_module, "_do_compute_stats", wraps=cache_module._do_compute_stats) as compute:
        produced = produce_stats(_producer(_UNION))
    assert [call.args[1].request for call in compute.call_args_list] == [{None: ImageStats.HASH_DUPLICATES_BASIC}]
    assert produced["stats_policy"] == ResolvedStatsPolicy.of_flags(ImageStats.HASH_DUPLICATES_BASIC)
