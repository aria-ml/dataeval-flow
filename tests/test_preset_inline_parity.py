"""A preset step's prepared inlining runs as the nested preset did (preset decomposition spec §6.4)."""

import json
import re
from typing import Any, NamedTuple

import pytest

from dataeval_flow._cache import DatasetCache
from dataeval_flow._service._decompose import DecomposeReady, DecomposeRequest, prepare
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult
from tests.evaluator_toys import Items, ToyImages
from tests.golden.coverage import CoverageImages
from tests.golden.splitting import SplitImages
from tests.test_audit_preset import _OUTLIERS
from tests.test_audit_run import _three
from tests.test_chain_verdict_report import _outline
from tests.test_decompose_inline import _AUDITED, _OUTER
from tests.test_naming_conventions import _MINIMAL
from tests.test_preset_copy_parity import _draft, _failed, _run, _selections
from tests.test_preset_expansion_parity import _outcome
from tests.triage_toys import MixedWeightDataset


class _Case(NamedTuple):
    workflows: list[dict[str, Any]]  # the preset entries, then `outer`
    datasets: dict[str, Any]
    target: str  # `/workflows/<outer>/steps/<j>`
    anchor: str | None  # the nested run's: a verdict level; "verdict" (any); "no-verdict"; "refused"; None (none given)
    task: dict[str, Any] = {}
    extra: dict[str, Any] = {}


_SPLITTING = {"name": "splitting", "type": "splits"}
_NEVER = {**_SPLITTING, "split_on": ["nope"]}
_AUDIT = {"name": "release", "type": "audit", **_OUTLIERS}
_QUALITY = {"name": "q", "type": "quality", **_MINIMAL["quality"]}
_COLLECT_AUDIT = [
    {"name": "evals", "transform": "collect", "input": ["splits.val", "splits.test"]},
    {"name": "audit", "workflow": "release", "input": ["splits.train", "evals"]},
]
_SPLICED_THREE = [  # `_spliced_three`'s steps over train, val and test
    {"name": "evals", "transform": "collect", "input": ["val", "test"]},
    {"name": "audit", "workflow": "release", "input": ["train", "evals"]},
]


def _outer(*steps: dict[str, Any], inputs: list[Any] | None = None, **declared: Any) -> dict[str, Any]:
    return {"name": "outer", "inputs": inputs or ["data"], "steps": list(steps), **declared}


def _p(entry: str, read: Any = "data", **settings: Any) -> dict[str, Any]:
    return {"name": "p", "workflow": entry, "input": read, **settings}


def _splits_case(settings: dict[str, Any], *steps: dict[str, Any]) -> _Case:
    entry = {"name": "s", "type": "splits", **settings}
    return _Case([entry, _outer(_p("s"), *steps)], {"data": ToyImages(count=40)}, "/workflows/1/steps/0", None)


_CASES: dict[str, _Case] = {
    "audit-split": _Case(
        [_SPLITTING, _AUDIT, _AUDITED], {"data": SplitImages(90)}, "/workflows/2/steps/2", "verdict", extra={"seed": 0}
    ),
    "audit-outside-check": _Case(
        [
            _AUDIT,
            _outer(
                {"name": "dupes-all", "evaluator": "dupes", "input": "train"},
                {"name": "dupes-check", "check": "image-duplicates", "input": "dupes-all"},
                *_SPLICED_THREE,
                inputs=["train", "val", "test"],
            ),
        ],
        _three(),
        "/workflows/1/steps/3",
        "verdict",
    ),
    "audit-never-started": _Case(
        [
            _NEVER,
            _AUDIT,
            _outer({"name": "splits", "workflow": "splitting", "input": "data", "optional": True}, *_COLLECT_AUDIT),
        ],
        {"data": SplitImages(90)},
        "/workflows/2/steps/2",
        "no-verdict",
        extra={"seed": 0},
    ),
    "audit-empty-part": _Case(
        [
            _AUDIT,
            _outer(
                {"name": "first", "evaluator": "dupes", "input": "train"},
                {"name": "audit", "workflow": "release", "input": ["train", "evals"]},
                inputs=["train", {"name": "evals", "list": True}],
            ),
        ],
        {"train": ToyImages(), "val": Items([]), "test": ToyImages(seed=2)},
        "/workflows/1/steps/1",
        "refused",
    ),
    "audit-accepted-step": _Case(
        [
            {**_AUDIT, "accepted": {"image-outliers-train": "Planted."}},
            _outer(*_SPLICED_THREE, inputs=["train", "val", "test"]),
        ],
        _three(),
        "/workflows/1/steps/1",
        "not-ready",
    ),
    "audit-step-extractor": _Case(
        [
            _AUDIT,
            _outer(_SPLICED_THREE[0], {**_SPLICED_THREE[1], "extractor": "flat"}, inputs=["train", "val", "test"]),
        ],
        _three(),
        "/workflows/1/steps/1",
        "not-ready",
    ),
    "bias": _Case(
        [{"name": "b", "type": "bias"}, _outer(_p("b"))], {"data": CoverageImages()}, "/workflows/1/steps/0", None
    ),
    "prioritization": _Case(
        [
            {"name": "prio", "type": "prioritization"},
            _outer(
                _p("prio", ["ref", "pools"]),
                {"name": "after", "evaluator": "dupes", "input": "p.selected"},
                inputs=["ref", {"name": "pools", "list": True}],
            ),
        ],
        {"ref": ToyImages(count=16), "pool": ToyImages(count=20, seed=1)},
        "/workflows/1/steps/0",
        None,
        task={"extractor": "flat"},
    ),
    "quality-downstream": _Case(
        [_QUALITY, {**_OUTER, "steps": _OUTER["steps"][:3]}],
        {"data": ToyImages(count=20)},
        "/workflows/1/steps/0",
        None,
    ),
    "quality-two-instances": _Case(
        [_QUALITY, _outer(_p("q"), {**_p("q"), "name": "p2"})],
        {"data": ToyImages(count=20)},
        "/workflows/1/steps/0",
        None,
    ),
    "quality-optional-skipped": _Case(
        [
            _NEVER,
            _QUALITY,
            _outer(
                {"name": "splits", "workflow": "splitting", "input": "data", "optional": True},
                _p("q", "splits.train", optional=True),
            ),
        ],
        {"data": SplitImages(90)},
        "/workflows/2/steps/1",
        None,
        extra={"seed": 0},
    ),
    "quality-under-parent-verdict": _Case(
        [_QUALITY, _outer(_p("q"), verdict={})], {"data": ToyImages(count=20)}, "/workflows/1/steps/0", "verdict"
    ),
    "scope-step-extractor": _Case(
        [{"name": "sc", "type": "scope"}, _outer(_p("sc", extractor="flat"))],
        {"data": CoverageImages()},
        "/workflows/1/steps/0",
        None,
    ),
    "shift": _Case(
        [
            {"name": "sh", "type": "shift", **_MINIMAL["shift"]},
            _outer(_p("sh", ["reference", "tests"]), inputs=["reference", {"name": "tests", "list": True}]),
        ],
        {"reference": ToyImages(count=20), "test": ToyImages(count=20, seed=1)},
        "/workflows/1/steps/0",
        None,
        task={"extractor": "flat"},
    ),
    "splits-downstream": _splits_case(
        {},
        {"name": "evals", "transform": "collect", "input": ["p.val", "p.test"]},
        {"name": "dupes-train", "evaluator": "dupes", "input": "p.train"},
    ),
    "splits-folds": _splits_case({"folds": 2}, {"name": "dupes-train", "evaluator": "dupes", "input": "p.train"}),
    "splits-rebalance-collect": _splits_case(
        {"rebalance": "interclass"},
        {"name": "both", "transform": "collect", "input": ["p.train", "p.test"]},
        {"name": "dupes-train", "evaluator": "dupes", "input": "both[train]"},
    ),
    "taxonomy": _Case(
        [{"name": "tx", "type": "taxonomy", **_MINIMAL["taxonomy"]}, _outer(_p("tx"))],
        {"data": ToyImages(count=20)},
        "/workflows/1/steps/0",
        None,
    ),
    "triage": _Case(
        [{"name": "tr", "type": "triage"}, _outer(_p("tr"))],
        {"data": MixedWeightDataset()},
        "/workflows/1/steps/0",
        None,
    ),
}


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _renamed(value: Any, answer: DecomposeReady, step: str) -> Any:
    """`value`, from the nested run, named as the inlined candidate names it: each spliced name and declared output
    read, matched whole (not inside a longer name), and the splice's messages as the scope's."""
    table = {f"{step}/{inner}": new for inner, new in answer.names.steps.items()} | answer.names.addresses
    table |= {f"step `{step}`": f"scope `{step}`", f"Step '{step}'": f"Scope '{step}'"}
    pattern = re.compile(
        r"(?<![\w/.-])(" + "|".join(re.escape(old) for old in sorted(table, key=len, reverse=True)) + r")(?![\w-])"
    )
    return json.loads(pattern.sub(lambda match: table[match[1]], json.dumps(value, default=str)))


def _json(value: Any) -> Any:
    """`value` as `_renamed` reads it back: tuples as lists, and anything else JSON can't hold as its text."""
    return json.loads(json.dumps(value, default=str))


def _accepted_renamed(outcome: dict[str, Any], answer: DecomposeReady, kept: set[str]) -> dict[str, Any]:
    """`outcome` with each acceptance the candidate's scope keys by a renamed check step keyed so; `kept` holds the
    scope's `accepted` keys, so a key that is a check type (`leakage`) stays as it is."""
    verdict = outcome.get("verdict")
    for acceptance in verdict["accepted"] if verdict else ():
        if acceptance["check"] in kept:
            continue
        bare, bracket, element = acceptance["check"].partition("[")
        acceptance["check"] = answer.names.steps.get(bare, bare) + bracket + element
    return outcome


def _digests(result: ChainResult) -> dict[str, str]:
    binning = result.metadata.metadata_binning or {}
    return {key.partition(" (")[0]: record["encoding_digest"] for key, record in binning.get("per_split", {}).items()}


def _anchored(result: ChainResult, anchor: str | None, *, skipped: bool) -> None:
    """Assert that `result`, the nested run, concluded as `anchor` says, so two runs failing alike can't pass a case;
    `skipped` where an optional step is meant to fail and skip what reads it."""
    if anchor == "refused":
        assert not result.success
        assert any(record.status == "failed" for record in result.steps.values())
        return
    assert result.success, result.errors
    if anchor is None:
        assert skipped or not [name for name, record in result.steps.items() if _failed(record)]
    elif anchor == "no-verdict":
        assert (result.verdict, bool(result.no_verdict)) == (None, True)
    else:
        assert result.verdict is not None
        assert anchor == "verdict" or result.verdict.level == anchor


@pytest.mark.parametrize("case", sorted(_CASES))
def test_an_inlined_step_runs_as_the_nested_preset_did(case: str) -> None:
    workflows, datasets, target, anchor, task, extra = _CASES[case]
    parts = {
        "workflows": workflows,
        "evaluators": [DuplicatesConfig(name="dupes")],
        "tasks": [{"name": "t", "workflow": "outer", "sources": list(datasets), **task}],
        "extractor": True,
    }
    request = DecomposeRequest.model_validate({"pipeline": {}, "operation": "inline_step", "target_pointer": target})
    answer = prepare(_draft(datasets, stand_in=True, graphs=False, extra=extra, **parts), request)
    assert isinstance(answer, DecomposeReady), answer
    assert answer.valid, answer.diagnostics

    nested = _run(_draft(datasets, extra=extra, **parts))
    _anchored(nested, anchor, skipped=case == "quality-optional-skipped")

    candidate = {key: value for key, value in answer.pipeline.items() if key not in ("datasets", "sources")}
    inlined = _run(_draft(datasets, extra={**extra, **candidate}, **parts))
    step = workflows[-1]["steps"][int(target.rpartition("/")[2])]["name"]
    assert not any(name.startswith(f"{step}/") for name in inlined.steps)  # the inlined steps ran, not the splice
    parent = next(w for w in answer.pipeline["workflows"] if w["name"] == "outer")
    kept = {key for scope in parent.get("scopes", []) for key in (scope.get("verdict") or {}).get("accepted", {})}
    assert _accepted_renamed(_renamed(_outcome(nested), answer, step), answer, kept) == _json(_outcome(inlined))
    assert _renamed(_outline(nested), answer, step) == _outline(inlined)
    assert _renamed(_selections(nested), answer, step) == _selections(inlined)
    assert _renamed(_digests(nested), answer, step) == _digests(inlined)
    recorded = inlined.metadata.resolved_config["workflow"]
    assert recorded["derived_from"][-1]["steps"] == list(answer.names.steps.values())

    if case == "audit-accepted-step":
        assert inlined.verdict is not None
        states = {acceptance.check: acceptance.state for acceptance in inlined.verdict.accepted}
        assert states["audit-image-outliers-train"] == "warned"
    if case == "audit-outside-check":
        assert inlined.verdict is not None
        assert all(item.step != "dupes-check" for item in [*inlined.verdict.warnings, *inlined.verdict.blocking])
    if case == "quality-optional-skipped":  # never started, as the nested splice: every step skipped, checks too
        inner = {name: record for name, record in inlined.steps.items() if name.startswith("p-")}
        assert any("check" in written for written in parent["steps"] if written["name"] in inner)
        assert {record.status for record in inner.values()} == {"skipped"}
