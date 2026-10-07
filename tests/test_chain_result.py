"""A chain's result: what its JSON, report, health and CI files say about each step (spec §7)."""

import json
import xml.etree.ElementTree as ET
from typing import Any, cast
from unittest.mock import patch

import pytest

from dataeval_flow import run_task, run_tasks
from dataeval_flow._blocks import Section, Table
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._report import lineage_line
from dataeval_flow._ci_reports import junit_report, markdown_summary
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesEvaluator
from dataeval_flow.steps import ChainResult
from tests.chain_toys import chain_pipeline, register_toys, run_toy_chain
from tests.evaluator_toys import ToyImages
from tests.preset_toys import register_presets

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
    assert payload["steps"]["after"]["reason"] == "needs `boom`, which failed: RuntimeError: boom on few"
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


def _reads(result: ChainResult) -> dict[str, str]:
    """What the report's Steps table says each step read, by step."""
    top_level = result._document(detailed=True).blocks
    (steps,) = [block for block in top_level if isinstance(block, Section) and block.title == "Steps"]
    (table,) = steps.blocks
    assert isinstance(table, Table)
    return {str(row["step"]): str(row["reads"]) for row in table.rows}


def test_the_report_has_a_section_per_step_and_names_where_each_read_came_from() -> None:
    result = _result(_MIXED)
    text = result.report(detailed=True, width=120)
    # The text renderer capitalizes a top-level section's heading, like every other report's (Configuration,
    # a workflow's Summary): a step's section is a peer of those, not nested under a synthetic wrapper.
    for heading in ("TOY-FIRST · FEW", "DUPLICATES · DUPES", "TOY-EXPLODE · BOOM", "TOY-KEEP · AFTER"):
        assert heading in text
    assert _reads(result)["dupes"] == "`few` ← `a` (src)"
    assert "RuntimeError: boom on few" in text
    assert "needs `boom`, which failed" in text

    titles = ["toy-first · few", "Duplicates · dupes", "toy-explode · boom", "toy-keep · after"]
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


def test_a_step_run_over_a_list_input_reads_each_elements_source() -> None:
    result = _result(
        [{"name": "kept", "transform": "toy-keep", "input": "all"}],
        inputs=[{"name": "all", "list": True}],
        datasets={"src": ToyImages(), "more": ToyImages(seed=1)},
    )
    lineage = result.metadata.lineage
    assert lineage_line("all", lineage) == "`all` (src, more)"
    assert lineage_line("all[more]", lineage) == "`all[more]` (more)"
    assert lineage_line("kept", lineage) == "`kept` ← `all` (src, more)"
    elements = result.steps["kept"].elements or {}
    assert [element.inputs for element in elements.values()] == [["all[src]"], ["all[more]"]]
    assert _reads(result) == {"kept": "`all` (src, more)"}


def test_a_step_run_over_a_list_output_reads_where_the_list_came_from() -> None:
    result = _result(
        [
            {"name": "parts", "transform": "toy-spread", "input": "a", "parts": 2},
            {"name": "kept", "transform": "toy-keep", "input": "parts"},
        ]
    )
    lineage = result.metadata.lineage
    assert lineage_line("parts", lineage) == "`parts` ← `a` (src)"
    assert lineage_line("kept", lineage) == "`kept` ← `parts` ← `a` (src)"
    payload = cast("dict[str, Any]", result.to_dict())
    elements = payload["steps"]["kept"]["elements"]
    assert (elements["0"]["inputs"], elements["1"]["inputs"]) == (["parts[0]"], ["parts[1]"])
    assert _reads(result) == {"parts": "`a` (src)", "kept": "`parts` ← `a` (src)"}


def test_junit_has_one_error_per_failed_step_and_markdown_names_them() -> None:
    result = _result(_MIXED)
    xml = junit_report({"t": result})
    assert 'name="step: boom"' in xml
    assert "RuntimeError: boom on few" in xml
    assert 'errors="1"' in xml
    assert "**Failed steps:** boom" in markdown_summary({"t": result})


_REFUSAL = "Task 't' runs workflow 'two', which takes one source for 'a' and one source for 'b', but the task names 1."


def _refused() -> ChainResult:
    """A chain refused before any step ran: its task binds one source to a workflow of two inputs."""
    two = {"name": "two", "inputs": ["a", "b"], "steps": [{"name": "k", "transform": "toy-keep", "input": "a"}]}
    result = run_task(chain_pipeline(workflows=[two]), TaskConfig(name="t", workflow="two", sources=["src"]))
    assert isinstance(result, ChainResult)
    return result


def test_a_chain_refused_before_any_step_ran_is_failed_in_its_health() -> None:
    result = _refused()
    assert (result.success, result.steps, result.errors) == (False, {}, [_REFUSAL])
    assert result.health == {"status": "failed", "warnings": 0, "findings": 0, "failed_steps": []}
    assert cast("dict[str, Any]", result.to_dict())["health"]["status"] == "failed"


def test_the_report_of_a_chain_refused_before_any_step_ran_says_why() -> None:
    import html

    result = _refused()
    text = result.report(width=200)
    assert f"  FAILED\n{'=' * 200}\n  {_REFUSAL}\n" in text
    assert "Steps:" not in text
    assert f"<p>{html.escape(_REFUSAL)}</p>" in result.to_html()


def test_junit_errors_a_chain_refused_before_any_step_ran() -> None:
    root = ET.fromstring(junit_report({"t": _refused()}))  # noqa: S314 - our own output
    (case,) = root.iter("testcase")
    error = case.find("error")
    assert case.get("name") == "run"
    assert error is not None
    assert (error.get("message"), error.text) == (_REFUSAL, _REFUSAL)
    assert (root.get("tests"), root.get("failures"), root.get("errors")) == ("1", "0", "1")


def test_markdown_shows_the_errors_of_a_chain_refused_before_any_step_ran() -> None:
    summary = markdown_summary({"t": _refused()})
    assert "**Health:** failed" in summary
    assert f"```\n{_REFUSAL}\n```" in summary
    assert "**Failed steps:**" not in summary


def test_the_ci_markdown_summary_gives_the_verdict_in_place_of_the_health_line(toys) -> None:
    register_presets(toys)
    entries = [
        {"name": "blocked", "type": "toy-verdict-preset"},
        {"name": "caveated", "type": "toy-verdict-preset", "exact": 50.0},
        {"name": "plain", "type": "toy-preset"},
    ]
    tasks = [{"name": entry["name"], "workflow": entry["name"], "sources": ["src"]} for entry in entries]
    results = run_tasks(chain_pipeline(workflows=entries, tasks=tasks))
    blocked, caveated, plain = (markdown_summary({name: results[name]}) for name in ("blocked", "caveated", "plain"))
    assert "## blocked\n\n**Verdict:** Not ready: Image Duplicates (2 exact (16.7%), 0 near (0.0%))\n" in blocked
    assert "## caveated\n\n**Verdict:** Ready with caveats: 1 not assessed\n" in caveated
    assert "**Health:**" not in blocked + caveated  # "passed" would contradict the caveats
    assert "**Health:** 1 warning" in plain
    assert "**Verdict:**" not in plain


def test_the_runner_prints_and_writes_a_failed_chain(caplog) -> None:
    import logging

    from dataeval_flow._runner import _collect_results

    with caplog.at_level(logging.INFO):
        collected = _collect_results({"t": _result(_MIXED)}, verbosity=0)
    assert collected.failures == 1
    assert "t" in collected.merged
    assert list(collected.merged["t"]["steps"]) == ["few", "dupes", "boom", "after"]
    assert "FAILED: t" in caplog.text
    assert "OK: t" not in caplog.text


def test_each_step_is_a_top_level_section_in_chain_order_with_its_elements_nested_in_it() -> None:
    result = _result(
        [
            {"name": "parts", "transform": "toy-spread", "input": "a", "parts": 2},
            {"name": "dupes", "evaluator": "dupes", "input": "parts"},
            {"name": "kept", "transform": "toy-keep", "input": "a"},
        ]
    )
    sections = [block for block in result._document(detailed=True).blocks if isinstance(block, Section)]
    assert [section.title for section in sections] == [
        "toy-spread · parts",
        "Duplicates · dupes",
        "toy-keep · kept",
        "Steps",
    ]
    nested = [[block.title for block in section.blocks if isinstance(block, Section)] for section in sections]
    assert nested == [[], ["[0]", "[1]"], [], []]


def test_a_chain_carries_a_thumbnail_of_each_item_its_steps_name_read_from_the_dataset_the_step_read() -> None:
    from dataeval_flow.workflows.quality import QualityConfig

    clean = QualityConfig(name="clean", outliers={"flags": ["pixel"], "outlier_threshold": "zscore"})  # type: ignore[arg-type]
    steps = [
        {"name": "few", "transform": "view", "input": "a", "operations": [{"type": "Limit", "params": {"size": 10}}]},
        {"name": "cleaned", "workflow": "clean", "input": "few"},
    ]
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["a"], "steps": steps}, clean],
        datasets={"src": ToyImages(count=24)},
    )
    task = TaskConfig(name="t", workflow="w", sources="src")
    result = run_task(config, task)
    assert isinstance(result, ChainResult)
    # The step is spliced, and `cleaned/dupes`' section names items 0 and 5: ToyImages' item 5 copies item 0.
    # `cleaned/outliers` flags none of these ten images on pixel statistics, so no other item is pictured. Each is
    # read from `few`, the Dataset the step read.
    thumbnails = [
        (asset.item.source, asset.item.index, asset.media_type, asset.width, asset.height) for asset in result.assets
    ]
    assert thumbnails == [("few", 0, "image/webp", 16, 16), ("few", 5, "image/webp", 16, 16)]
    payload = cast("dict[str, Any]", result.to_dict())
    assert [(asset["item"]["source"], asset["item"]["index"]) for asset in payload["assets"]] == [
        ("few", 0),
        ("few", 5),
    ]
    config.result.max_images = 0
    without = run_task(config, task)
    assert without.assets == []
    assert "assets" not in cast("dict[str, Any]", without.to_dict())
