"""Running combine and check steps: findings, health, inputs that hold nothing, and lists (spec §5.6, §7, §9.1)."""

import re
from typing import Any, cast
from unittest.mock import patch

import pytest

from dataeval_flow._blocks import Section, Summary
from dataeval_flow._cache import DatasetCache
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult, Finding
from tests.chain_toys import CountGroups, GroupLimit, chain_pipeline, register_toys, run_chain_task, run_toy_chain
from tests.evaluator_toys import ToyImages

pytestmark = pytest.mark.usefixtures("toys")

_DUPES = {"name": "dupes", "evaluator": "dupes", "input": "a"}
_COUNT = {"name": "count", "combine": "toy-count-groups", "input": "dupes"}
_CAMS = {"s1": ToyImages(), "s2": ToyImages(seed=1)}
_LIST = [{"name": "cams", "list": True}]


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    DatasetCache.clear_instances()
    yield plugins
    DatasetCache.clear_instances()


def _result(
    *steps: dict[str, Any],
    inputs: list[Any] | None = None,
    datasets: dict[str, Any] | None = None,
) -> ChainResult:
    datasets = datasets or {"src": ToyImages()}
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": inputs or ["a"], "steps": list(steps)}],
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
    assert finding.description == (
        "Not assessed: `count` was skipped: needs `dupes`, which was skipped: needs `boom`, which was skipped: "
        "failed: RuntimeError: boom on a."
    )
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
    assert [(item.group, item.label) for item in summary.items] == [("s1", "Group count"), ("s2", "Group count")]
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
    assert second.description == (
        "Not assessed: `count[s2]` was skipped: needs `dupes[s2]`, which was skipped: needs `boom[s2]`, which "
        "failed: RuntimeError: boom on cams[s2]."
    )
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


def test_a_check_s_finding_is_a_top_level_section_beside_the_evidence_it_judged() -> None:
    result = _result(_DUPES, _COUNT, {"name": "judge", "check": "toy-at-most", "input": "count", "most": 0})
    # The text renderer capitalizes a top-level section's heading, and a finding's section is one.
    assert re.search(r"\n  GROUP COUNT +1 groups\n", result.report(detailed=True, width=120))
    top_level = result._document(detailed=True).blocks
    (finding,) = [block for block in top_level if isinstance(block, Section) and block.severity is not None]
    assert (finding.title, finding.brief, finding.severity) == ("Group count", "1 groups", "warning")
    # `count` is a combine with nothing to show, so the evidence is the Duplicates it counted.
    assert [block.title for block in finding.blocks if isinstance(block, Section)] == ["From Duplicates · dupes"]


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
        "Not assessed: `count` holds no element; `count[s1]` was skipped: needs `dupes[s1]`, which was skipped: "
        "needs `boom[s1]`, which was skipped: failed: RuntimeError: boom on cams[s1]."
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


_ONE_OR_NONE = [
    "a",
    {"name": "rest", "list": True, "empty": "no other source given"},
]


def _spread(name: str, parts: int) -> list[dict[str, Any]]:
    """`name` spreads input `a` into `parts` parts (none at 0), and `count-<name>` counts each part's groups."""
    return [
        {"name": name, "transform": "toy-spread", "input": "a", "parts": parts},
        {"name": f"dupes-{name}", "evaluator": "dupes", "input": name},
        {"name": f"count-{name}", "combine": "toy-count-groups", "input": f"dupes-{name}"},
    ]


def test_a_check_fed_one_empty_and_one_full_list_is_assessed() -> None:
    # The usual two-split audit: train against test holds one element, the evaluation pairs none (spec §19 C1).
    result = _result(
        *_spread("none", 0),
        *_spread("two", 2),
        {"name": "worst", "check": "toy-worst-of", "input": ["count-none", "count-two"]},
    )
    worst = result.steps["worst"]
    assert worst.not_assessed is None
    assert worst.output[0].brief.startswith("2 counts")


def test_a_check_whose_every_list_is_empty_is_not_assessed_with_the_reason() -> None:
    result = _result(
        {"name": "against-a", "evaluator": "dupes", "input": ["a", "rest"]},
        {"name": "count-a", "combine": "toy-count-groups", "input": "against-a"},
        {"name": "worst", "check": "toy-worst-of", "input": ["count-a"]},
        inputs=_ONE_OR_NONE,
        datasets={"a": ToyImages(seed=0)},
    )
    worst = result.steps["worst"]
    assert worst.not_assessed == "no other source given"
    assert worst.output[0].description == "Not assessed: no other source given."


def test_lists_empty_for_no_carried_reason_say_which_holds_none() -> None:
    result = _result(
        *_spread("none", 0),
        {"name": "worst", "check": "toy-worst-of", "input": ["count-none"]},
    )
    assert result.steps["worst"].not_assessed == "`none` holds no element"


def test_a_port_that_may_be_empty_is_judged_empty() -> None:
    result = _result(
        {"name": "self", "evaluator": "dupes", "input": "a"},
        {"name": "against-a", "evaluator": "dupes", "input": ["a", "rest"]},
        {"name": "count-self", "combine": "toy-count-groups", "input": "self"},
        {"name": "count-a", "combine": "toy-count-groups", "input": "against-a"},
        {"name": "judge", "check": "toy-against", "reference": "count-self", "others": "count-a"},
        inputs=_ONE_OR_NONE,
        datasets={"a": ToyImages(seed=0)},
    )
    judge = result.steps["judge"]
    assert judge.not_assessed is None
    assert judge.output[0].brief.endswith("against 0 others")


def test_a_check_that_raises_step_skipped_is_recorded_as_not_assessed() -> None:
    result = _result(_DUPES, _COUNT, {"name": "judge", "check": "toy-unassessable", "input": "count"})
    judge = result.steps["judge"]
    assert judge.status == "ok"
    assert judge.not_assessed == "nothing to judge in count"
    (finding,) = result.findings
    assert (finding.severity, finding.brief, finding.step) == ("info", "not assessed", "judge")
    assert finding.description == "Not assessed: nothing to judge in count."


def test_a_broadcast_check_that_raises_step_skipped_records_it_on_that_element() -> None:
    result = _result(
        {"name": "dupes", "evaluator": "dupes", "input": "cams"},
        _COUNT,
        {"name": "judge", "check": "toy-unassessable", "input": "count", "only": "count[s2]"},
        inputs=_LIST,
        datasets=_CAMS,
    )
    elements = result.steps["judge"].elements
    assert elements is not None
    assert elements["s1"].not_assessed is None
    assert elements["s2"].not_assessed == "nothing to judge in count[s2]"
    assert [(finding.step, finding.brief) for finding in result.findings] == [
        ("judge[s1]", "judged"),
        ("judge[s2]", "not assessed"),
    ]


def test_a_transform_that_raises_step_skipped_is_still_skipped() -> None:
    result = _result({"name": "decline", "transform": "toy-yield", "input": "a"})
    decline = result.steps["decline"]
    assert decline.status == "skipped"
    assert decline.not_assessed is None


def test_an_unbound_optional_whole_list_port_leaves_the_check_run() -> None:
    result = _result(_DUPES, _COUNT, {"name": "judge", "check": "toy-opt", "input": "count"})
    assert result.steps["judge"].status == "ok"
    assert result.steps["judge"].not_assessed is None
    assert [(finding.brief, finding.step) for finding in result.findings] == [("ran", "judge")]
