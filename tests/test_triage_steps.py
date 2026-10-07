"""The `factor-triage` evaluator and the `factor-issues` check, run as steps of a custom workflow (spec §10.10)."""

import json
from typing import Any
from unittest.mock import patch

import pytest

from dataeval_flow import run_tasks
from dataeval_flow._blocks import Fields, ItemRef
from dataeval_flow._cache import DatasetCache
from dataeval_flow._triage_report import triage_section
from dataeval_flow.steps import ChainResult
from tests.chain_toys import chain_pipeline
from tests.finding_blocks import column, fields, tables
from tests.triage_toys import LatitudeDataset, MixedWeightDataset


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _chain(dataset: Any, *, triage: dict[str, Any] | None = None, issues: dict[str, Any] | None = None) -> ChainResult:
    steps = [
        {"name": "triage", "evaluator": "triage", "input": "data"},
        {"name": "issues", "check": "factor-issues", "input": "triage", **(issues or {})},
    ]
    config = chain_pipeline(
        evaluators=[{"name": "triage", "type": "factor-triage", **(triage or {})}],
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": dataset},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def _data(result: ChainResult) -> dict[str, Any]:
    return result.steps["triage"].output.data()


def test_triage_finds_a_mixed_column_and_verifies_the_correction_it_suggests() -> None:
    data = _data(_chain(MixedWeightDataset()))
    (weight,) = [f for f in data["findings"] if f.factor == "weight"]
    assert (weight.category, weight.severity) == ("unreadable", "blocking")
    assert "kind: parse_value" in data["suggested_policy_yaml"]
    assert [(v.factor, v.applied, v.recovered) for v in data["verification"]] == [("weight", True, True)]
    assert data["verification_error"] is None
    assert data["counts"] == {"unreadable": 1, "blocking": 1}


def test_metadata_issues_makes_the_findings_metadata_triage_made() -> None:
    result = _chain(MixedWeightDataset())
    assert [[f.severity, f.title, f.brief] for f in result.findings] == [
        ["warning", "Unreadable factors", "1 factors"],
        ["info", "Suggested policy", "add to configuration under `metadata:`"],
        ["info", "Verified", "1 recovered"],
        ["info", "Recommended policy", "pins 1 factor as read from this data"],
    ]
    assert {f.step for f in result.findings} == {"issues"}
    assert result.health["status"] == "warning"


def test_verify_off_leaves_verification_out() -> None:
    result = _chain(MixedWeightDataset(), triage={"verify": False})
    assert _data(result)["verification"] == []
    assert _data(result)["verification_error"] is None
    assert "Verified" not in [f.title for f in result.findings]


def test_a_verification_that_raises_is_a_warning_and_the_step_still_succeeds() -> None:
    with patch("dataeval_flow.evaluators.quality._triage.verify", side_effect=RuntimeError("boom")):
        result = _chain(MixedWeightDataset())
    assert result.steps["triage"].status == "ok"
    assert _data(result)["verification_error"] == "boom"
    (failed,) = [f for f in result.findings if f.title == "Verification failed"]
    assert failed.severity == "warning"


def test_max_examples_is_how_many_values_the_check_shows() -> None:
    result = _chain(LatitudeDataset(), issues={"max_examples": 1})
    unreadable = next(f for f in result.findings if f.title == "Unreadable factors")
    assert str(fields(unreadable)["text reads"]).endswith("(+1 more)")


def test_the_problem_values_are_pictured_by_the_chain_input() -> None:
    unreadable = next(f for f in _chain(LatitudeDataset()).findings if f.title == "Unreadable factors")
    (table,) = tables(unreadable)
    assert column(table, "image")[1] == [ItemRef(source="data", index=20)]


def test_the_triage_section_counts_the_factors_and_issues() -> None:
    step_result = _chain(MixedWeightDataset()).steps["triage"].result
    assert step_result is not None
    text = step_result.report()
    assert "Factors" in text
    assert "Issues" in text


def test_the_triage_section_names_its_counts_by_category_and_as_issues() -> None:
    (section,) = triage_section(
        {"data": {"factor_count": 3, "findings": [{}, {}], "counts": {"unbinned": 1, "warning": 1, "blocking": 1}}}
    )
    assert isinstance(section, Fields)
    assert section.items == [
        ("Factors", 3),
        ("Issues", 2),
        ("Unpinned continuous bins", 1),
        ("warning issues", 1),
        ("blocking issues", 1),
    ]


def test_the_triage_output_is_json() -> None:
    body = json.loads(json.dumps(_chain(MixedWeightDataset()).to_dict()))
    data = body["steps"]["triage"]["output"]["data"]
    assert data["findings"][0]["factor"] == "weight"
    assert data["verification"][0]["recovered"] is True
