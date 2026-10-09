"""TC-18-3 — audit preset: which warnings block (`blocking:`) and which are accepted (`accepted:`)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._chain._graph import GraphError
from dataeval_flow.workflows.audit import AuditConfig
from verification.functional.audit._helpers import OUTLIERS, audit, audit_pipeline, leaky_pair, three_splits
from verification.functional.chains._toys import Images

pytestmark = pytest.mark.required


class TestBlocking:
    def test_leakage_and_untrained_classes_block_by_default(self) -> None:
        config = AuditConfig.model_validate({"name": "w", **OUTLIERS})
        assert config.blocking == ["leakage", "untrained-classes"]

    def test_an_empty_blocking_list_leaves_a_leaking_audit_ready_with_caveats(self) -> None:
        verdict = audit(leaky_pair(), {"blocking": []}).verdict
        assert verdict is not None
        assert verdict.level == "ready-with-caveats"
        assert verdict.blocking == []
        assert "leakage" in {item.check for item in verdict.warnings}

    def test_naming_another_check_makes_its_warning_block(self) -> None:
        # A planted duplicate in train alone (two splits that both plant the white image would leak it).
        splits = {"train": Images(12, seed=0), "test": Images(12, seed=1, planted=False)}
        assert audit(splits).verdict.level == "ready-with-caveats"  # type: ignore[union-attr]
        verdict = audit(splits, {"blocking": ["image-duplicates"]}).verdict
        assert verdict is not None
        assert verdict.level == "not-ready"
        assert [item.step for item in verdict.blocking] == ["image-duplicates-train"]

    def test_a_blocking_check_that_could_not_run_is_a_caveat_not_a_block(self) -> None:
        # Naming a check that needs an extractor, and running without one.
        verdict = audit(
            leaky_pair() | {"test": Images(12, seed=5, planted=False)}, {"blocking": ["eval-coverage"]}
        ).verdict
        assert verdict is not None
        assert verdict.blocking == []
        assert "eval-coverage" in {item.check for item in verdict.not_assessed}
        assert verdict.level == "ready-with-caveats"

    def test_a_blocking_entry_that_names_no_check_of_the_chain_is_refused_at_load(self) -> None:
        with pytest.raises(ValidationError, match="`blocking` names `nope`, which this audit's chain has no check for"):
            audit_pipeline(leaky_pair(), {"blocking": ["nope"]})

    def test_a_check_the_settings_leave_out_cannot_be_named(self) -> None:
        # label-conformance runs only where an ontology is set.
        with pytest.raises(ValidationError, match="label-conformance"):
            audit_pipeline(leaky_pair(), {"blocking": ["label-conformance"]})


class TestAccepted:
    def test_an_accepted_blocking_warning_leaves_the_audit_ready_with_caveats(self) -> None:
        result = audit(leaky_pair(), {"accepted": {"leakage": "Shared calibration frames."}})
        verdict = result.verdict
        assert verdict is not None
        assert verdict.blocking == []
        assert [(a.check, a.reason, a.state) for a in verdict.accepted] == [
            ("leakage", "Shared calibration frames.", "warned")
        ]
        assert verdict.level == "ready-with-caveats"

    def test_an_accepted_warning_keeps_its_finding_and_still_counts_in_health(self) -> None:
        plain, accepted = audit(leaky_pair()), audit(leaky_pair(), {"accepted": {"leakage": "x"}})
        assert accepted.health["warnings"] == plain.health["warnings"]
        leakage = [finding for finding in accepted.findings if finding.step == "leakage"]
        assert [finding.severity for finding in leakage] == ["warning"]

    def test_the_acceptance_is_in_the_json_verdict(self) -> None:
        payload = audit(leaky_pair(), {"accepted": {"leakage": "Shared frames."}}).to_dict()
        assert payload["verdict"]["accepted"] == [  # type: ignore[index]
            {"check": "leakage", "reason": "Shared frames.", "state": "warned"}
        ]

    def test_an_acceptance_of_a_check_that_did_not_warn_is_recorded_as_such(self) -> None:
        verdict = audit(leaky_pair(), {"accepted": {"image-duplicates": "Expected."}}).verdict
        assert verdict is not None
        assert [(a.check, a.state) for a in verdict.accepted] == [("image-duplicates", "did-not-warn")]

    def test_an_acceptance_keyed_by_a_check_type_covers_every_split(self) -> None:
        result = audit(three_splits(), {"accepted": {"image-outliers": "Night shots, by design."}})
        verdict = result.verdict
        assert verdict is not None
        assert "image-outliers" not in {item.check for item in verdict.warnings}
        assert [a.state for a in verdict.accepted] == ["warned"]

    def test_an_acceptance_keyed_by_one_split_leaves_the_other_split_s_warning(self) -> None:
        result = audit(three_splits(), {"accepted": {"image-outliers-evals[test]": "Night shots, by design."}})
        verdict = result.verdict
        assert verdict is not None
        steps = {item.step for item in verdict.warnings if item.check == "image-outliers"}
        assert steps == {"image-outliers-train", "image-outliers-evals[val]"}
        assert [a.check for a in verdict.accepted] == ["image-outliers-evals[test]"]

    def test_an_accepted_key_naming_no_check_or_check_step_is_refused_at_load(self) -> None:
        with pytest.raises(ValidationError, match="`accepted` names `nope`, which this audit's chain has no check or"):
            audit_pipeline(leaky_pair(), {"accepted": {"nope": "x"}})

    def test_a_blank_reason_is_refused_at_load(self) -> None:
        with pytest.raises(ValidationError, match="at least 1 character"):
            audit_pipeline(leaky_pair(), {"accepted": {"leakage": "  "}})

    def test_a_split_on_a_step_that_runs_once_is_refused_at_load(self) -> None:
        with pytest.raises(ValidationError, match=r"`leakage\[test\]`, but `leakage` runs once"):
            audit_pipeline(leaky_pair(), {"accepted": {"leakage[test]": "x"}})

    def test_a_split_the_task_does_not_have_is_refused_before_any_step_runs(self) -> None:
        config = audit_pipeline(three_splits(), {"accepted": {"image-outliers-evals[tset]": "x"}})
        with pytest.raises(GraphError, match=r"`image-outliers-evals\[tset\]`, but this task has no evaluation split"):
            run_tasks(config)
