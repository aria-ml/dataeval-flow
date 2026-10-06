"""A preset chain's verdict over the checks it names, and its preflight (audit spec §6, §9.5, §21).

`ToyImages()` plants one exact duplicate (item 5 is item 0), so `image-duplicates` warns at `exact: 0.0` and passes at
`exact: 50.0`. With no extractor, `completeness` is skipped and `dimensional-completeness` is not assessed.
"""

from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from pydantic import Field

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._graph import GraphError
from dataeval_flow._chain._verdict import (
    Acceptance,
    Unassessed,
    Verdict,
    VerdictItem,
    judge,
    moot_checks,
    next_step_lines,
)
from dataeval_flow.evaluators.quality import DuplicatesEvaluator
from dataeval_flow.steps import ChainResult, Finding, StepResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows import Workflow
from dataeval_flow.workflows._preset import NextSteps, Preset, PresetChain
from tests.chain_toys import chain_pipeline
from tests.preset_toys import ToyVerdictPreset, ToyVerdictPresetConfig, register_presets


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _run(entry: dict[str, Any], extractor: bool = False) -> ChainResult:
    task: dict[str, Any] = {"name": "t", "workflow": "w", "sources": ["src"]}
    if extractor:
        task["extractor"] = "flat"
    config = chain_pipeline(workflows=[{"name": "w", **entry}], tasks=[task], extractor=extractor)
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


@pytest.mark.parametrize(
    ("entry", "extractor", "level"),
    [
        ({}, True, "not-ready"),  # a blocking warning
        ({"accepted": {"image-duplicates": "Planted on purpose."}}, True, "ready-with-caveats"),  # accepted, fired
        ({"exact": 50.0}, False, "ready-with-caveats"),  # not assessed: no extractor
        ({"exact": 50.0}, True, "ready"),  # nothing warns, all assessed
        ({"exact": 50.0, "accepted": {"image-duplicates": "Fine."}}, True, "ready"),  # acceptance did not fire
        ({"blocking": []}, True, "ready-with-caveats"),  # a warning, not blocking
    ],
)
def test_the_verdict_follows_the_rule(plugins, entry, extractor, level) -> None:
    register_presets(plugins)
    result = _run({"type": "toy-verdict-preset", **entry}, extractor=extractor)
    assert result.verdict is not None
    assert result.verdict.level == level


def test_an_acceptance_that_did_not_fire_is_listed_with_its_state(plugins) -> None:
    register_presets(plugins)
    result = _run({"type": "toy-verdict-preset", "exact": 50.0, "accepted": {"image-duplicates": "Fine."}}, True)
    assert result.verdict is not None
    assert [(a.check, a.state) for a in result.verdict.accepted] == [("image-duplicates", "did-not-warn")]


def test_an_unassessed_check_is_listed_with_its_reason_and_its_next_step(plugins) -> None:
    register_presets(plugins)
    result = _run({"type": "toy-verdict-preset", "exact": 50.0}, extractor=False)
    assert result.verdict is not None
    assert result.preset_chain is not None
    assert [(u.check, u.step) for u in result.verdict.not_assessed] == [
        ("dimensional-completeness", "dimensional-completeness")
    ]
    assert "requires an extractor" in result.verdict.not_assessed[0].reason
    assert next_step_lines(result.verdict, result.preset_chain.next_steps) == [
        "Name an extractor. Not assessed: Dimensional Completeness (dimensional-completeness)."
    ]


def test_a_chain_without_blocking_has_no_verdict(plugins) -> None:
    register_presets(plugins)
    result = _run({"type": "toy-preset"})
    assert result.verdict is None
    assert "verdict" not in result.to_dict()


def test_the_verdict_lands_in_the_json(plugins) -> None:
    register_presets(plugins)
    data: Any = _run({"type": "toy-verdict-preset"}, extractor=True).to_dict()
    assert data["verdict"]["level"] == "not-ready"
    assert data["verdict"]["blocking"][0]["check"] == "image-duplicates"


def test_the_verdict_line_names_its_reasons() -> None:
    verdict = Verdict(
        level="ready-with-caveats",
        blocking=[],
        warnings=[VerdictItem(check="c", step="s", title="T", brief="b")] * 2,
        accepted=[Acceptance(check="a", reason="r", state="warned")],
        not_assessed=[Unassessed(check="u", step="u", reason="x")],
    )
    assert verdict.line() == "Ready with caveats: 2 warnings, 1 accepted risk, 1 not assessed"


class _RefusingConfig(ToyVerdictPresetConfig):
    type: str = Field(default="toy-refusing-preset", description="The workflow type this entry configures.")


class _Refusing(Preset, Workflow[_RefusingConfig, ChainResult]):
    """The toy verdict preset's chain, refused at preflight."""

    name: ClassVar[str] = "toy-refusing-preset"
    description: ClassVar[str] = "Refuses before any step runs."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: Any) -> PresetChain:
        return ToyVerdictPreset.chain(config)

    @classmethod
    def preflight(cls, config: Any, inputs: Any) -> None:  # noqa: ARG003
        raise GraphError("refused")


def test_a_preset_preflight_refusal_raises_before_the_run(plugins) -> None:
    register_presets(plugins)
    plugins["dataeval_flow.workflows"].append(("toy-refusing-preset", "tests.test_chain_verdict:_Refusing"))
    with patch.object(DuplicatesEvaluator, "run", autospec=True) as dupes, pytest.raises(GraphError, match="refused"):
        _run({"type": "toy-refusing-preset"}, extractor=True)
    dupes.assert_not_called()


def test_a_failed_task_has_no_verdict(plugins) -> None:
    register_presets(plugins)
    with patch.object(DuplicatesEvaluator, "run", side_effect=RuntimeError("no stats")):
        result = _run({"type": "toy-verdict-preset"}, extractor=True)
    assert not result.success
    assert result.verdict is None


def _record(
    name: str, check: str, *findings: Finding, not_assessed: str | None = None, kind: Any = "check", **fields: Any
) -> StepResult:
    """A step's record, built by hand: `findings` as its output, with no chain run."""
    return StepResult(
        name=name,
        kind=kind,
        type=check,
        inputs=[],
        status="ok",
        output=list(findings),
        not_assessed=not_assessed,
        **fields,
    )


def _warning(step: str, title: str, brief: str | None = "b") -> Finding:
    return Finding(severity="warning", title=title, brief=brief, step=step)


def test_judge_walks_each_element_of_a_check_run_once_per_element() -> None:
    elements = {
        "train": _record("class-imbalance", "class-imbalance", _warning("class-imbalance[train]", "Class Imbalance")),
        "val": _record("class-imbalance", "class-imbalance", not_assessed="no evaluation split given"),
    }
    steps = {"class-imbalance": _record("class-imbalance", "class-imbalance", elements=elements)}
    verdict = judge(steps, blocking=[], accepted={})
    assert [(w.check, w.step) for w in verdict.warnings] == [("class-imbalance", "class-imbalance[train]")]
    assert [(u.check, u.step, u.reason) for u in verdict.not_assessed] == [
        ("class-imbalance", "class-imbalance[val]", "no evaluation split given")
    ]


def test_judge_reads_not_assessed_from_check_records_only() -> None:
    pairs = _record("divergence", "divergence", kind="evaluator", not_assessed="`evals` holds one element, so no pair")
    verdict = judge({"divergence": pairs}, blocking=[], accepted={})
    assert verdict.not_assessed == []
    assert verdict.level == "ready"


def test_an_acceptance_s_state_is_not_assessed_or_did_not_warn_when_its_check_did_not_warn() -> None:
    steps = {"leakage": _record("leakage", "leakage", not_assessed="`pairs` failed: boom")}
    verdict = judge(steps, blocking=[], accepted={"leakage": "Known.", "image-outliers": "Expected."})
    assert [(a.check, a.state) for a in verdict.accepted] == [
        ("leakage", "not-assessed"),
        ("image-outliers", "did-not-warn"),
    ]


def test_a_blocking_check_not_assessed_is_only_a_caveat() -> None:
    steps = {"leakage": _record("leakage", "leakage", not_assessed="no evaluation split given")}
    verdict = judge(steps, blocking=["leakage"], accepted={})
    assert verdict.level == "ready-with-caveats"
    assert verdict.blocking == []
    assert [u.check for u in verdict.not_assessed] == ["leakage"]


def test_next_steps_put_blocking_types_first_give_their_advice_and_quote_an_unmatched_reason() -> None:
    steps = {
        "class-imbalance": _record(
            "class-imbalance", "class-imbalance", _warning("class-imbalance", "Class Imbalance", None)
        ),
        "dupes-train": _record("dupes-train", "image-duplicates", _warning("dupes-train", "Image Duplicates")),
        "leakage": _record("leakage", "leakage", not_assessed="`pairs` failed: boom."),
    }
    verdict = judge(steps, blocking=["image-duplicates"], accepted={})
    plan = NextSteps(by_check={"image-duplicates": "Remove them."}, by_reason={"requires an extractor": "Name one."})
    assert next_step_lines(verdict, plan) == [
        "Image Duplicates (dupes-train): Remove them.",
        "Class Imbalance (class-imbalance)",
        "Not assessed: Leakage (leakage): `pairs` failed: boom.",
    ]


def test_a_not_assessed_line_names_each_check_with_its_steps() -> None:
    steps = {
        "eval-coverage": _record("eval-coverage", "eval-coverage", not_assessed="no evaluation split given"),
        "leakage": _record("leakage", "leakage", not_assessed="no evaluation split given"),
        "imbalance": _record(
            "imbalance",
            "class-imbalance",
            elements={
                "val": _record("imbalance", "class-imbalance", not_assessed="no evaluation split given"),
                "test": _record("imbalance", "class-imbalance", not_assessed="no evaluation split given"),
            },
        ),
    }
    plan = NextSteps(by_reason={"no evaluation split": "Give one."})
    assert next_step_lines(judge(steps, blocking=[], accepted={}), plan) == [
        (
            "Give one. Not assessed: Eval Coverage (eval-coverage), Leakage (leakage), "
            "Class Imbalance (imbalance[val], imbalance[test])."
        )
    ]


def test_a_check_run_left_empty_beside_an_assessed_run_of_its_type_is_moot() -> None:
    empty = "no evaluation split given"
    steps = {
        "outliers-train": _record("outliers-train", "image-outliers", _warning("outliers-train", "Image Outliers")),
        "outliers-evals": _record("outliers-evals", "image-outliers", not_assessed=empty),
        "imbalance-train": _record("imbalance-train", "class-imbalance"),
        "imbalance-evals": _record(
            "imbalance-evals", "class-imbalance", elements={"val": _record("x", "class-imbalance", not_assessed=empty)}
        ),
        "leakage": _record("leakage", "leakage", not_assessed=empty),
        "count": _record("count", "dupes-count", kind="evaluator", not_assessed=empty),
    }
    assert moot_checks(steps) == {"outliers-evals"}
    verdict = judge(steps, blocking=[], accepted={})
    assert [(u.check, u.step) for u in verdict.not_assessed] == [
        ("class-imbalance", "imbalance-evals[val]"),
        ("leakage", "leakage"),
    ]
    assert verdict.line() == "Ready with caveats: 1 warning, 2 not assessed"


def test_a_check_not_assessed_for_a_failed_producer_is_kept_beside_an_assessed_run_of_its_type() -> None:
    reason = "`outliers-train` failed"
    steps = {
        "image-outliers-train": _record("image-outliers-train", "image-outliers", not_assessed=reason),
        "image-outliers-evals": _record(
            "image-outliers-evals", "image-outliers", elements={"test": _record("x", "image-outliers")}
        ),
    }
    assert moot_checks(steps) == set()
    verdict = judge(steps, blocking=[], accepted={})
    assert [(u.check, u.step, u.reason) for u in verdict.not_assessed] == [
        ("image-outliers", "image-outliers-train", reason)
    ]


def test_a_check_run_that_did_not_complete_is_not_assessed_for_its_reason_or_errors() -> None:
    skipped = _record("optional", "image-outliers", errors=["RuntimeError: boom"], reason="failed: RuntimeError: boom")
    skipped.status = "skipped"
    failed = _record("required", "leakage", errors=["ValueError: no"])
    failed.status = "failed"
    verdict = judge({"optional": skipped, "required": failed}, blocking=[], accepted={})
    assert [(u.step, u.reason) for u in verdict.not_assessed] == [
        ("optional", "failed: RuntimeError: boom"),
        ("required", "ValueError: no"),
    ]
    assert verdict.level == "ready-with-caveats"


def test_a_not_ready_line_names_each_blocking_warning_and_its_brief_if_any() -> None:
    verdict = Verdict(
        level="not-ready",
        blocking=[
            VerdictItem(check="leakage", step="leakage", title="Leakage", brief="2 exact duplicates across splits"),
            VerdictItem(check="untrained-classes", step="untrained-classes", title="Untrained Classes", brief=""),
        ],
        warnings=[],
        accepted=[],
        not_assessed=[],
    )
    assert verdict.line() == "Not ready: Leakage (2 exact duplicates across splits); Untrained Classes"


def _verdict(level: str, *, warning: bool = False, unassessed: bool = False) -> Verdict:
    """A verdict at `level`, with one unaccepted warning and one check not assessed where asked."""
    return Verdict(
        level=level,  # type: ignore[arg-type]
        blocking=[],
        warnings=[VerdictItem(check="c", step="c", title="C", brief="b")] if warning else [],
        accepted=[],
        not_assessed=[Unassessed(check="d", step="d", reason="r")] if unassessed else [],
    )


@pytest.mark.parametrize(
    ("verdict", "passes"),
    [
        (_verdict("not-ready"), {"ready-with-caveats": False, "ready-with-accepted-risks": False, "ready": False}),
        (
            _verdict("ready-with-caveats"),
            {"ready-with-caveats": True, "ready-with-accepted-risks": True, "ready": False},
        ),
        (
            _verdict("ready-with-caveats", warning=True),
            {"ready-with-caveats": True, "ready-with-accepted-risks": False, "ready": False},
        ),
        (
            _verdict("ready-with-caveats", unassessed=True),
            {"ready-with-caveats": True, "ready-with-accepted-risks": False, "ready": False},
        ),
        (_verdict("ready"), {"ready-with-caveats": True, "ready-with-accepted-risks": True, "ready": True}),
    ],
    ids=["not-ready", "only-accepted-risks", "a-warning", "not-assessed", "ready"],
)
def test_a_verdict_meets_each_requirement_its_level_and_caveats_allow(verdict: Verdict, passes: dict) -> None:
    assert {requirement: verdict.meets(requirement) for requirement in passes} == passes
