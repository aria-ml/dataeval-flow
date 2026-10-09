"""TC-18-4 — audit run as a step of a custom workflow, on splits a chain made."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow.config import PipelineConfig
from dataeval_flow.steps import ChainResult
from verification.functional.audit._helpers import OUTLIERS, QUESTIONS, section
from verification.functional.chains._toys import Images, pipeline

pytestmark = pytest.mark.required

_COLLECT = {"name": "evals", "transform": "collect", "input": ["splits.val", "splits.test"]}
_AUDIT = {"name": "audit", "workflow": "release", "input": ["splits.train", "evals"]}


def _split_and_audit(
    steps: list[dict[str, Any]] | None = None,
    *,
    splitting: dict[str, Any] | None = None,
    audit: dict[str, Any] | None = None,
    optional_split: bool = False,
) -> PipelineConfig:
    """One source `data`, a `splits` step, then *steps* (by default collect the evaluation parts, then audit)."""
    return pipeline(
        {"data": Images(90, planted=False)},
        workflows=[
            {"name": "splitting", "type": "splits", **(splitting or {})},
            {"name": "release", "type": "audit", **OUTLIERS, **(audit or {})},
            {
                "name": "outer",
                "inputs": ["data"],
                "steps": [
                    {
                        "name": "splits",
                        "workflow": "splitting",
                        "input": "data",
                        **({"optional": True} if optional_split else {}),
                    },
                    *(steps or [_COLLECT, _AUDIT]),
                ],
            },
        ],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["data"]}],
        extra={"seed": 0},
    )


def _run(config: PipelineConfig) -> ChainResult:
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


class TestAuditAsAStep:
    def test_split_collect_audit_gives_the_task_a_verdict_and_the_five_questions(self) -> None:
        result = _run(_split_and_audit())
        assert result.success, result.errors
        assert result.verdict is not None
        assert result.verdict.level in {"not-ready", "ready-with-caveats", "ready"}
        assert result.preset_chain is not None
        assert [group.heading for group in result.preset_chain.groups] == QUESTIONS
        assert result.to_dict()["verdict"]["level"] == result.verdict.level  # type: ignore[index]

    def test_the_audit_s_steps_run_under_the_step_s_name(self) -> None:
        result = _run(_split_and_audit())
        audit_steps = [name for name in result.steps if name.startswith("audit/")]
        assert {"audit/leakage", "audit/content-digest-train"} <= set(audit_steps)
        assert all(item.step.startswith("audit/") for item in result.verdict.warnings)  # type: ignore[union-attr]

    def test_the_record_has_a_column_for_train_and_each_collected_part(self) -> None:
        record = section(_run(_split_and_audit()).report(), "What was audited")
        header = next(line for line in record.splitlines() if line.strip().startswith("train"))
        assert header.split() == ["train", "val", "test"]

    def test_every_part_is_encoded_like_train(self) -> None:
        binning = _run(_split_and_audit()).metadata.metadata_binning
        assert binning is not None
        digests = {entry["encoding_digest"] for entry in binning["per_split"].values()}
        assert len(digests) == 1

    def test_an_acceptance_is_written_without_the_step_s_prefix(self) -> None:
        plain = _run(_split_and_audit()).verdict
        assert plain is not None
        warned = {item.step for item in plain.warnings}
        assert "audit/image-outliers-train" in warned
        accepted = _run(_split_and_audit(audit={"accepted": {"image-outliers-train": "Known."}})).verdict
        assert accepted is not None
        assert "audit/image-outliers-train" not in {item.step for item in accepted.warnings}
        assert [(a.check, a.state) for a in accepted.accepted] == [("image-outliers-train", "warned")]

    def test_a_train_test_split_audits_with_the_test_part_alone(self) -> None:
        steps = [
            {"name": "evals", "transform": "collect", "input": ["splits.test"]},
            _AUDIT,
        ]
        result = _run(_split_and_audit(steps, splitting={"val_frac": 0.0}))
        assert result.verdict is not None
        assert set(result.steps["audit/duplicates-evals"].elements or {}) == {"test"}

    def test_an_audit_step_that_never_started_gives_no_verdict_and_says_why(self) -> None:
        # The split step is optional and fails (no such factor), so the audit's train is skipped.
        result = _run(_split_and_audit(splitting={"split_on": ["nope"]}, optional_split=True))
        assert result.success, result.errors
        assert result.verdict is None
        assert result.no_verdict is not None
        assert result.no_verdict.startswith("step `audit` did not run:")
        assert result.to_dict()["no_verdict"] == result.no_verdict
        assert f"No verdict: {result.no_verdict}" in result.report()
        assert "verdict" not in result.to_dict()

    def test_an_audit_step_may_not_be_optional(self) -> None:
        steps = [_COLLECT, {**_AUDIT, "optional": True}]
        with pytest.raises(ValidationError, match="gives a verdict, so it may not be optional"):
            _split_and_audit(steps)

    def test_a_workflow_runs_one_audit_step(self) -> None:
        steps = [_COLLECT, _AUDIT, {**_AUDIT, "name": "audit2"}]
        with pytest.raises(ValidationError, match="a workflow gives one verdict"):
            _split_and_audit(steps)

    def test_a_list_of_folds_bound_to_the_single_train_slot_is_refused_at_load(self) -> None:
        steps = [{"name": "evals", "transform": "collect", "input": ["splits.test"]}, _AUDIT]
        with pytest.raises(ValidationError, match=r"name one element, such as `splits\.train\[0\]`"):
            _split_and_audit(steps, splitting={"folds": 3})
