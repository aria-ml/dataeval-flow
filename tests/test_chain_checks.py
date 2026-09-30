"""Running combine and check steps: findings, health, inputs that hold nothing, and lists (spec §5.6, §7, §9.1)."""

from typing import Any, cast
from unittest.mock import patch

import pytest

from dataeval_flow._blocks import Section, Summary
from dataeval_flow._cache import DatasetCache
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows import Finding
from tests.chain_toys import CountGroups, GroupLimit, chain_pipeline, register_toys, run_chain_task, run_toy_chain
from tests.evaluator_toys import ToyImages
from tests.workflow_toys import ToyCountConfig, register_count

pytestmark = pytest.mark.usefixtures("toys")

_DUPES = {"name": "dupes", "evaluator": "dupes", "input": "a"}
_COUNT = {"name": "count", "combine": "toy-count-groups", "input": "dupes"}
_CAMS = {"s1": ToyImages(), "s2": ToyImages(seed=1)}
_LIST = [{"name": "cams", "list": True}]


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    register_count(plugins)
    DatasetCache.clear_instances()
    yield plugins
    DatasetCache.clear_instances()


def _result(
    *steps: dict[str, Any],
    inputs: list[Any] | None = None,
    datasets: dict[str, Any] | None = None,
    workflows: tuple[Any, ...] = (),
) -> ChainResult:
    datasets = datasets or {"src": ToyImages()}
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": inputs or ["a"], "steps": list(steps)}, *workflows],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "workflow": "w", "sources": list(datasets)}],
        datasets=datasets,
    )
    result = run_chain_task(config)
    assert isinstance(result, ChainResult)
    return result


def test_a_combine_s_output_feeds_a_check_whose_findings_reach_the_result() -> None:
    result = _result(_DUPES, _COUNT, {"name": "judge", "check": "toy-at-most", "input": "count", "most": 0})
    assert result.findings == [Finding(severity="warning", title="Group count", brief="1 groups", step="judge")]
    assert result.health == {"status": "warning", "warnings": 1, "findings": 1, "failed_steps": []}
    payload = cast("dict[str, Any]", result.to_dict())
    assert payload["findings"] == [
        {
            "severity": "warning",
            "title": "Group count",
            "brief": "1 groups",
            "description": None,
            "blocks": [],
            "step": "judge",
        }
    ]
    assert payload["steps"]["count"]["output"] == {"groups": 1}
    assert payload["steps"]["judge"]["output"] == {"findings": 1, "warnings": 1}


def test_a_threshold_of_none_reports_its_finding_as_info_and_judges_nothing() -> None:
    result = _result(_DUPES, _COUNT, {"name": "judge", "check": "toy-at-most", "input": "count", "most": None})
    assert [(finding.severity, finding.brief) for finding in result.findings] == [("info", "1 groups")]
    assert result.health["status"] == "ok"


def test_a_check_whose_input_an_optional_failure_left_empty_is_not_assessed_and_the_task_stays_ok() -> None:
    result = _result(
        {"name": "boom", "transform": "toy-explode", "input": "a", "optional": True},
        {"name": "dupes", "evaluator": "dupes", "input": "boom"},
        _COUNT,
        {"name": "judge", "check": "toy-at-most", "input": "count"},
    )
    (finding,) = result.findings
    assert (finding.severity, finding.title, finding.brief, finding.step) == (
        "info",
        "Group count",
        "not assessed",
        "judge",
    )
    assert finding.description == "Not assessed: `count` was skipped: needs `dupes`, which was skipped."
    assert result.steps["judge"].status == "ok"
    assert result.health == {"status": "ok", "warnings": 0, "findings": 1, "failed_steps": []}


def test_a_check_whose_input_failed_says_what_failed() -> None:
    with patch.object(CountGroups, "run", side_effect=RuntimeError("no count")):
        result = _result(_DUPES, _COUNT, {"name": "judge", "check": "toy-at-most", "input": "count"})
    (finding,) = result.findings
    assert finding.description == "Not assessed: `count` failed: RuntimeError: no count."
    assert result.failed_steps == ["count"]
    assert result.health["status"] == "failed"


def test_a_check_run_once_per_element_names_each_element_it_judged() -> None:
    result = _result(
        {"name": "dupes", "evaluator": "dupes", "input": "cams"},
        _COUNT,
        {"name": "judge", "check": "toy-at-most", "input": "count", "most": 0},
        inputs=_LIST,
        datasets=_CAMS,
    )
    assert [(finding.step, finding.severity, finding.brief) for finding in result.findings] == [
        ("judge[s1]", "warning", "1 groups"),
        ("judge[s2]", "warning", "1 groups"),
    ]
    (section,) = result._summary_blocks()
    assert isinstance(section, Section)
    (summary,) = section.blocks
    assert isinstance(summary, Summary)
    assert [item.label for item in summary.items] == ["Group count [s1]", "Group count [s2]"]
    assert summary.warnings == 2


def test_one_element_whose_input_failed_leaves_the_others_judged() -> None:
    result = _result(
        {"name": "boom", "transform": "toy-explode", "input": "cams", "only": "cams[s2]"},
        {"name": "dupes", "evaluator": "dupes", "input": "boom"},
        _COUNT,
        {"name": "judge", "check": "toy-at-most", "input": "count", "most": 0},
        inputs=_LIST,
        datasets=_CAMS,
    )
    first, second = result.findings
    assert (first.step, first.severity, first.brief) == ("judge[s1]", "warning", "1 groups")
    assert (second.step, second.severity, second.brief) == ("judge[s2]", "info", "not assessed")
    assert second.description == "Not assessed: `count[s2]` was skipped: needs `dupes[s2]`, which was skipped."
    assert result.steps["judge"].status == "ok"


def test_a_check_that_takes_a_whole_list_judges_it_once() -> None:
    result = _result(
        {"name": "dupes", "evaluator": "dupes", "input": "cams"},
        _COUNT,
        {"name": "worst", "check": "toy-worst", "input": "count"},
        inputs=_LIST,
        datasets=_CAMS,
    )
    assert [(finding.step, finding.brief) for finding in result.findings] == [("worst", "worst: s1 (1 groups)")]


def test_the_json_lists_only_check_findings_while_health_counts_a_workflow_step_s_too() -> None:
    cleaning = ToyCountConfig(name="clean")
    result = _result(
        {"name": "cleaning", "workflow": "clean", "input": "a"},
        _DUPES,
        _COUNT,
        {"name": "judge", "check": "toy-at-most", "input": "count", "most": 5},
        workflows=(cleaning,),
    )
    workflow_findings = result.steps["cleaning"].result.findings  # type: ignore[union-attr]
    assert workflow_findings
    assert result.findings == [
        *workflow_findings,
        Finding(severity="ok", title="Group count", brief="1 groups", step="judge"),
    ]
    assert [finding["step"] for finding in result.to_dict()["findings"]] == ["judge"]  # type: ignore[index,union-attr]
    assert result.health["findings"] == len(workflow_findings) + 1


def test_a_check_that_returns_something_other_than_findings_fails_its_step() -> None:
    with patch.object(GroupLimit, "run", return_value=[1]):
        result = _result(_DUPES, _COUNT, {"name": "judge", "check": "toy-at-most", "input": "count"})
    assert result.failed_steps == ["judge"]
    assert result.steps["judge"].errors == ["TypeError: check 'toy-at-most' returned int, not findings."]


def test_a_check_that_returns_one_finding_instead_of_a_list_fails_its_step() -> None:
    with patch.object(GroupLimit, "run", return_value=Finding(title="x")):
        result = _result(_DUPES, _COUNT, {"name": "judge", "check": "toy-at-most", "input": "count"})
    assert result.failed_steps == ["judge"]
    assert result.steps["judge"].errors == [
        "TypeError: check 'toy-at-most' returned a Finding, not a list of findings."
    ]


def test_a_combine_that_omits_an_output_port_fails_its_step() -> None:
    with patch.object(CountGroups, "run", return_value={}):
        result = _result(_DUPES, _COUNT)
    assert result.failed_steps == ["count"]
    assert result.steps["count"].errors == [
        "TypeError: combine 'toy-count-groups' returned no `output`: a combine returns every output port it declares."
    ]


def test_a_check_step_s_report_section_shows_its_findings() -> None:
    result = _result(_DUPES, _COUNT, {"name": "judge", "check": "toy-at-most", "input": "count", "most": 0})
    # The text renderer capitalizes a top-level section's heading, and a step's section is one.
    section = result.report(detailed=True, width=120).split("GROUP COUNT · JUDGE", 1)[1]
    assert "Group count" in section
    assert "1 groups" in section


def test_an_output_knows_the_datasets_it_was_computed_on_and_their_size() -> None:
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["a"], "steps": [_DUPES, _COUNT]}],
        evaluators=[DuplicatesConfig(name="dupes")],
        datasets={"src": ToyImages(count=12)},
    )
    run = run_toy_chain(config, "w", ["src"])
    dupes, count = run.nodes["dupes"], run.nodes["count"]
    assert (dupes.items, dupes.inputs) == (12, ("a",))  # type: ignore[union-attr]
    assert (count.items, count.inputs) == (12, ("a",))  # type: ignore[union-attr]


def test_a_whole_list_check_whose_list_holds_nothing_is_not_assessed_and_the_task_stays_ok() -> None:
    result = _result(
        {"name": "boom", "transform": "toy-explode", "input": "cams", "optional": True},
        {"name": "dupes", "evaluator": "dupes", "input": "boom"},
        _COUNT,
        {"name": "worst", "check": "toy-worst", "input": "count"},
        inputs=_LIST,
        datasets=_CAMS,
    )
    (finding,) = result.findings
    assert (finding.severity, finding.brief, finding.step) == ("info", "not assessed", "worst")
    assert finding.description == (
        "Not assessed: `count` holds no element; `count[s1]` was skipped: needs `dupes[s1]`, which was skipped."
    )
    assert result.steps["worst"].status == "ok"
    assert result.health == {"status": "ok", "warnings": 0, "findings": 1, "failed_steps": []}


def test_the_one_step_path_builds_a_source_s_view_once() -> None:
    from dataeval_flow._view import build_view

    views = [{"name": "first4", "operations": [{"type": "Limit", "params": {"size": 4}}]}]
    config = chain_pipeline(
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "evaluator": "dupes", "sources": "src"}],
        extra={"views": views},
    )
    config.sources[0].view = "first4"  # type: ignore[index]
    with patch("dataeval_flow._view.build_view", wraps=build_view) as built:
        run_chain_task(config)
    assert built.call_count == 1
