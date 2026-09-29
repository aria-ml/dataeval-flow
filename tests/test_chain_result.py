"""A chain's result: what its JSON, report, health and CI files say about each step (spec §7)."""

import json
from typing import Any, cast
from unittest.mock import patch

import pytest

from dataeval_flow._blocks import Section
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._report import lineage_line
from dataeval_flow._ci_reports import junit_report, markdown_summary
from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesEvaluator
from dataeval_flow.steps import ChainResult
from tests.chain_toys import chain_pipeline, register_toys, run_toy_chain
from tests.evaluator_toys import ToyImages

pytestmark = pytest.mark.usefixtures("toys")


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    DatasetCache.clear_instances()
    yield plugins
    DatasetCache.clear_instances()


def _result(
    steps: list[dict[str, Any]],
    inputs: list[Any] | None = None,
    datasets: dict[str, Any] | None = None,
) -> ChainResult:
    datasets = datasets or {"src": ToyImages()}
    workflow = {"name": "w", "inputs": inputs or ["a"], "steps": steps}
    config = chain_pipeline(workflows=[workflow], evaluators=[DuplicatesConfig(name="dupes")], datasets=datasets)
    return ChainResult.from_run("w", run_toy_chain(config, "w", list(datasets)))


_MIXED = [
    {"name": "few", "transform": "toy-first", "input": "a", "n": 4},
    {"name": "dupes", "evaluator": "dupes", "input": "few"},
    {"name": "boom", "transform": "toy-explode", "input": "few"},
    {"name": "after", "transform": "toy-keep", "input": "boom"},
]


def test_a_chain_with_a_failed_step_fails_but_keeps_every_step() -> None:
    result = _result(_MIXED)
    assert not result.success
    assert result.failed_steps == ["boom"]
    assert [record.status for record in result.steps.values()] == ["ok", "ok", "failed", "skipped"]
    assert result.health == {"status": "failed", "warnings": 0, "findings": 0, "failed_steps": ["boom"]}
    with pytest.raises(RuntimeError):
        _ = result.output


def test_the_json_keys_each_step_by_name_with_what_it_made() -> None:
    payload = json.loads(json.dumps(_result(_MIXED).to_dict()))
    assert payload["kind"] == "workflow"
    assert list(payload["steps"]) == ["few", "dupes", "boom", "after"]
    few = payload["steps"]["few"]
    assert (few["kind"], few["type"], few["status"], few["inputs"]) == ("transform", "toy-first", "ok", ["a"])
    assert few["output"]["items"] == 4
    assert len(few["output"]["digest"]) == 12
    assert payload["steps"]["dupes"]["output"]["shape"] == "table"
    assert payload["steps"]["dupes"]["dataeval"]["version"]
    assert payload["steps"]["boom"]["errors"] == ["RuntimeError: boom on few"]
    assert payload["steps"]["after"]["reason"] == "needs `boom`, which failed"
    assert [record["name"] for record in payload["metadata"]["lineage"]] == ["a", "few"]
    assert payload["health"]["failed_steps"] == ["boom"]
    assert payload["findings"] == []


def test_a_failed_element_shows_its_error_in_the_json() -> None:
    original = DuplicatesEvaluator.run

    def fails_on_more(self: Any, config: Any, inputs: Any) -> Any:
        if inputs[0].source == "all[more]":
            raise RuntimeError("no stats for more")
        return original(self, config, inputs)

    with patch.object(DuplicatesEvaluator, "run", fails_on_more):
        result = _result(
            [{"name": "dupes", "evaluator": "dupes", "input": "all"}],
            inputs=[{"name": "all", "list": True}],
            datasets={"src": ToyImages(), "more": ToyImages(seed=1)},
        )
    payload = cast("dict[str, Any]", result.to_dict())
    elements = payload["steps"]["dupes"]["elements"]
    assert elements["src"]["status"] == "ok"
    assert (elements["more"]["status"], elements["more"]["errors"]) == ("failed", ["RuntimeError: no stats for more"])


def test_the_report_has_a_section_per_step_headed_by_its_lineage() -> None:
    result = _result(_MIXED)
    text = result.report(detailed=True, width=120)
    # The text renderer capitalizes a top-level section's heading, like every other report's (Configuration,
    # a workflow's Summary): a step's section is a peer of those, not nested under a synthetic wrapper.
    for heading in ("FEW (TOY-FIRST)", "DUPES (QUALITY.DUPLICATES)", "BOOM (TOY-EXPLODE)", "AFTER (TOY-KEEP)"):
        assert heading in text
    assert "`few` ← `a` (src)" in text
    assert "RuntimeError: boom on few" in text
    assert "needs `boom`, which failed" in text

    titles = ["few (toy-first)", "dupes (quality.duplicates)", "boom (toy-explode)", "after (toy-keep)"]
    top_level = result._document(detailed=True).blocks
    step_sections = [block for block in top_level if isinstance(block, Section) and block.title in titles]
    assert [section.title for section in step_sections] == titles


def test_a_lineage_line_walks_back_to_the_source() -> None:
    steps = [
        {"name": "k", "transform": "toy-keep", "input": "a"},
        {"name": "few", "transform": "toy-first", "input": "k", "n": 2},
    ]
    result = _result(steps)
    assert lineage_line("few", result.metadata.lineage) == "`few` ← `k` ← `a` (src)"


def test_junit_has_one_error_per_failed_step_and_markdown_names_them() -> None:
    result = _result(_MIXED)
    xml = junit_report({"t": result})
    assert 'name="step: boom"' in xml
    assert "RuntimeError: boom on few" in xml
    assert 'errors="1"' in xml
    assert "**Failed steps:** boom" in markdown_summary({"t": result})


def test_the_runner_prints_and_writes_a_failed_chain() -> None:
    from dataeval_flow._runner import _collect_results

    collected = _collect_results({"t": _result(_MIXED)}, verbosity=0)
    assert collected.failures == 1
    assert "t" in collected.merged
    assert list(collected.merged["t"]["steps"]) == ["few", "dupes", "boom", "after"]
