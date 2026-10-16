"""What a custom workflow declares beyond its steps, as a preset's chain does (preset decomposition spec §3.1)."""

from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow.steps import CustomWorkflowConfig
from dataeval_flow.workflows._preset import NextSteps, Preflight, PresetChain, Record, ReportGroup

_STEPS = [
    {"name": "dupes", "evaluator": "dupes", "input": ["train", "evals"]},
    {"name": "labels", "evaluator": "labels", "input": ["train", "evals"]},
    {"name": "duplicates", "check": "image-duplicates", "input": "dupes"},
    {"name": "imbalance", "check": "class-imbalance", "input": "labels"},
]
_INPUTS = ["train", {"name": "evals", "list": True, "empty": "no evaluation split given"}]
_DECLARED: dict[str, Any] = {
    "ontology": {"animal": {"cat": None}},
    "groups": [{"heading": "Is the data clean?", "checks": ["image-duplicates"], "evidence": ["label-health"]}],
    "reference": "train",
    "record": {"title": "What was read", "steps": ["label-health"]},
    "verdict": {"blocking": ["image-duplicates"], "accepted": {"imbalance[val]": "Rare by design."}},
    "next_steps": {"by_check": {"image-duplicates": "Remove them."}, "by_reason": {"failed": "See the Steps table."}},
    "preflight": {"nonempty": True, "one_kind": True},
}


def _workflow(**declared: Any) -> CustomWorkflowConfig:
    return CustomWorkflowConfig.model_validate({"name": "w", "inputs": _INPUTS, "steps": _STEPS, **declared})


def test_every_declaration_round_trips_through_to_yaml() -> None:
    written = yaml.safe_load(_workflow(**_DECLARED).to_yaml())["workflows"][0]
    assert {key: written[key] for key in _DECLARED} == _DECLARED


def test_a_groups_only_workflow_writes_as_before() -> None:
    groups = [{"heading": "Is the data clean?", "checks": ["image-duplicates"]}]
    written = yaml.safe_load(_workflow(groups=groups).to_yaml())["workflows"][0]
    assert list(written) == ["name", "inputs", "steps", "groups"]
    assert written["groups"] == groups


def test_a_workflow_declaring_nothing_declares_no_chain() -> None:
    assert _workflow().declared_chain() is None


def test_its_declarations_are_the_chain_a_preset_would_declare() -> None:
    chain = _workflow(**_DECLARED).declared_chain()
    assert chain == PresetChain(
        steps=chain.steps if chain is not None else (),
        groups=(ReportGroup("Is the data clean?", ("image-duplicates",), ("label-health",)),),
        reference="train",
        record=Record("What was read", ("label-health",)),
        blocking=("image-duplicates",),
        accepted={"imbalance[val]": "Rare by design."},
        next_steps=NextSteps({"image-duplicates": "Remove them."}, {"failed": "See the Steps table."}),
        preflight=Preflight(nonempty=True, one_kind=True),
    )


def test_a_verdict_blocking_nothing_still_gives_one() -> None:
    chain = _workflow(verdict={}).declared_chain()
    assert chain is not None
    assert chain.blocking == ()


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        (
            {"reference": "evals"},
            r"`reference` names `evals`, which is no input taking one source\. Its inputs taking ",
        ),
        ({"reference": "test"}, r"`reference` names `test`, which is no input taking one source"),
        (
            {"verdict": {"blocking": ["leakage"]}},
            r"`blocking` names `leakage`, which no step runs\. Its steps' checks: ",
        ),
        (
            {"verdict": {"accepted": {"leakage": "x"}}},
            r"`accepted` names `leakage`, which is no check type or check step of it\. Its checks: ",
        ),
        ({"verdict": {"accepted": {"dupes[val]": "x"}}}, r"`accepted` names `dupes\[val\]`, which is no check type"),
        ({"next_steps": {"by_check": {"leakage": "x"}}}, r"`next_steps` names check `leakage`, which no step runs"),
        (
            {"next_steps": {"by_check": {"image-duplicates": "x"}}},
            (
                r"`next_steps\.by_check` advises on the warnings a verdict lists, so it needs `verdict:`; add "
                r"`verdict: \{\}` to give one that blocks on nothing\."
            ),
        ),
    ],
    ids=[
        "list-reference",
        "unknown-reference",
        "blocking",
        "accepted",
        "accepted-evaluator",
        "next-steps",
        "next-steps-without-verdict",
    ],
)
def test_a_declaration_naming_what_the_workflow_lacks_is_refused(declared: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _workflow(**declared)


def test_next_steps_by_reason_need_no_verdict() -> None:
    assert _workflow(next_steps={"by_reason": {"failed": "See the Steps table."}}).next_steps is not None


def test_an_accepted_key_may_name_a_check_type_or_a_check_step() -> None:
    accepted = {"class-imbalance": "x", "duplicates": "y", "imbalance[val]": "z"}
    assert _workflow(verdict={"accepted": accepted}).verdict is not None
