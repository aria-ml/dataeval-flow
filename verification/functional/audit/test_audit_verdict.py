"""TC-18-1 and TC-18-2 — audit preset: verdict levels, findings under five questions, and the record."""

from __future__ import annotations

import pytest

from dataeval_flow import dataset_digest, run_tasks
from verification.functional.audit._helpers import (
    QUESTIONS,
    READY_CHECKS,
    audit,
    audit_pipeline,
    clean_pair,
    leaky_pair,
    section,
    three_splits,
)
from verification.functional.chains._toys import Images

pytestmark = pytest.mark.required


class TestAuditVerdict:
    def test_clean_splits_with_an_extractor_are_ready(self) -> None:
        result = audit(clean_pair(), {"checks": READY_CHECKS}, extractor=True)
        verdict = result.verdict
        assert verdict is not None
        assert verdict.level == "ready"
        assert (verdict.blocking, verdict.warnings, verdict.accepted, verdict.not_assessed) == ([], [], [], [])
        assert "Verdict: Ready" in result.report()
        assert result.health["status"] == "ok"

    def test_an_image_in_two_splits_makes_the_audit_not_ready(self) -> None:
        result = audit(leaky_pair())
        verdict = result.verdict
        assert verdict is not None
        assert verdict.level == "not-ready"
        assert [(item.check, item.brief) for item in verdict.blocking] == [
            ("leakage", "2 exact cross-split duplicates")
        ]
        assert "Verdict: Not ready: Leakage (2 exact cross-split duplicates)" in result.report()

    def test_a_warning_from_a_check_that_does_not_block_is_ready_with_caveats(self) -> None:
        # A planted duplicate in train alone: image-duplicates warns, and by default it does not block.
        splits = {"train": Images(30, seed=1), "test": Images(30, seed=2, planted=False)}
        result = audit(splits)
        verdict = result.verdict
        assert verdict is not None
        assert verdict.blocking == []
        assert "image-duplicates" in {item.check for item in verdict.warnings}
        assert verdict.level == "ready-with-caveats"
        assert "Verdict: Ready with caveats" in result.report()

    def test_a_check_that_could_not_run_is_a_caveat_and_never_a_block(self) -> None:
        # Clean data, no extractor: nothing warns, but the embedding checks cannot be assessed.
        result = audit(clean_pair())
        verdict = result.verdict
        assert verdict is not None
        assert (verdict.warnings, verdict.blocking) == ([], [])
        assert {item.check for item in verdict.not_assessed} >= {
            "eval-coverage",
            "embedding-divergence",
            "class-coverage",
            "dimensional-completeness",
        }
        assert all("requires an extractor" in item.reason for item in verdict.not_assessed)
        assert verdict.level == "ready-with-caveats"
        assert any(line.lstrip(" -").startswith("Name an extractor") for line in result.report().splitlines())

    def test_a_one_split_audit_cannot_assess_the_evaluation_checks(self) -> None:
        result = audit({"train": Images(80, seed=10, planted=False)}, {"checks": READY_CHECKS}, extractor=True)
        verdict = result.verdict
        assert verdict is not None
        reasons = {item.reason for item in verdict.not_assessed}
        assert reasons == {"no evaluation split given"}
        assert verdict.level == "ready-with-caveats"
        # Train alone is still judged.
        assert any(step.startswith("class-imbalance-train") for step in result.steps)
        assert result.steps["class-sufficiency"].status == "ok"

    def test_a_failed_audit_has_no_verdict(self) -> None:
        # A factor the splits' metadata lacks fails the audit as a whole.
        result = run_tasks(audit_pipeline(leaky_pair(), {"factor-leakage": {"factors": ["missing"]}}))["t"]
        assert not result.success
        assert result.verdict is None
        assert "verdict" not in result.to_dict()
        assert result.health["status"] == "failed"


class TestAuditFindings:
    def test_the_findings_are_grouped_under_the_five_questions_in_order(self) -> None:
        result = audit(leaky_pair())
        assert result.preset_chain is not None
        assert [group.heading for group in result.preset_chain.groups] == QUESTIONS
        report = result.report().upper()
        positions = [report.index(f"  {question.upper()}") for question in QUESTIONS]
        assert positions == sorted(positions)

    def test_each_finding_sits_under_the_question_its_check_answers(self) -> None:
        report = audit(leaky_pair()).report()
        assert "Leakage" in section(report, "Are the splits fit to evaluate on?")
        assert "Class Sufficiency" in section(report, "Are the labels sound?")
        assert "Image Outliers" in section(report, "Is the data clean?")
        assert "Leakage" not in section(report, "Is the data clean?")

    def test_every_check_the_chain_runs_belongs_to_a_question(self) -> None:
        result = audit(three_splits(), extractor=True)
        assert result.preset_chain is not None
        in_questions = {check for group in result.preset_chain.groups for check in group.checks}
        ran = {record.type for record in result.steps.values() if record.kind == "check"}
        assert ran <= in_questions

    def test_the_report_reads_verdict_then_record_then_questions_then_next_steps(self) -> None:
        report = audit(three_splits()).report().upper()
        order = [
            report.index(marker)
            for marker in ("  VERDICT", "  WHAT WAS AUDITED", "  IS THE DATA CLEAN?", "  NEXT STEPS")
        ]
        assert order == sorted(order)

    def test_the_next_steps_say_what_to_do_about_each_warning(self) -> None:
        text = section(audit(leaky_pair()).report(), "Next steps")
        assert "Leakage (leakage)" in text
        assert "Class Sufficiency (class-sufficiency)" in text

    def test_three_splits_pair_their_evaluation_splits(self) -> None:
        result = audit(three_splits())
        pairs = result.steps["duplicates-pairs"]
        assert pairs.elements is not None
        assert list(pairs.elements) == ["val_vs_test"]
        assert result.steps["image-outliers-evals"].elements is not None
        assert set(result.steps["image-outliers-evals"].elements) == {"val", "test"}


class TestAuditRecord:
    def test_the_record_lists_every_split_with_its_content_and_metadata_digests(self) -> None:
        datasets = three_splits()
        report = audit(datasets).report()
        record = section(report, "What was audited")
        for name, dataset in datasets.items():
            assert f"Content digest ({name}):" in record
            assert dataset_digest(dataset).content in record
        assert all(label in record for label in ("Items", "Labels", "Classes", "Metadata factors"))

    def test_the_digests_in_the_record_come_from_the_content_digest_steps(self) -> None:
        datasets = leaky_pair()
        result = audit(datasets)
        recorded = result.steps["content-digest-train"].output.data()
        assert recorded["content"] == dataset_digest(datasets["train"]).content
        evals = result.steps["content-digest-evals"].elements["test"].output.data()
        assert evals["content"] == dataset_digest(datasets["test"]).content

    def test_the_record_lists_the_criteria_the_checks_were_judged_by(self) -> None:
        record = section(
            audit(leaky_pair(), {"checks": {"class-sufficiency": {"eval": 50}}}).report(), "What was audited"
        )
        assert "class-sufficiency:        train 20, eval 50" in record
        assert "leakage:                  exact 0, near 0, groups 0" in record
        assert "Blocking:                 leakage, untrained-classes" in record

    def test_every_split_is_encoded_like_train(self) -> None:
        result = audit(three_splits())
        binning = result.metadata.metadata_binning
        assert binning is not None
        digests = {split: entry["encoding_digest"] for split, entry in binning["per_split"].items()}
        assert set(digests) == {"train", "evals[val]", "evals[test]"}
        assert len(set(digests.values())) == 1
        assert result.metadata.encoding_digest in digests.values()
