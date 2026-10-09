"""TC-7-3 — the triage preset: unreadable and unpinned metadata factors, and the policy that repairs them."""

from __future__ import annotations

from typing import Any

import pytest

from dataeval_flow.config import MetadataPolicyConfig
from dataeval_flow.workflows.triage import TriageConfig
from verification.functional.workflows._synthetic import Images, Readings, by_title, run_preset

pytestmark = pytest.mark.required

TRIAGE: dict[str, Any] = {"type": "triage"}


def triage_output(kind: str) -> dict[str, Any]:
    return run_preset(TRIAGE, Readings(kind)).steps["factor-triage"].output.data()


class TestMetadataTriage:
    def test_a_column_mixing_numbers_and_text_is_reported_unreadable_and_a_repair_is_suggested(self) -> None:
        result = run_preset(TRIAGE, Readings("commas"))

        assert result.success, result.errors
        assert result.type == "triage"
        assert list(result.steps) == ["factor-triage", "factor-issues"]
        found = by_title(result)
        assert found["Unreadable factors"].severity == "warning"
        assert found["Unreadable factors"].brief == "1 factors"
        assert result.health["status"] == "warning"
        output = result.steps["factor-triage"].output.data()
        (finding,) = output["findings"]
        assert (finding.factor, finding.category, finding.reasons) == ("weight", "unreadable", ("mixed_types",))
        assert finding.detail["counts"] == {"text": 6, "numeric": 54}
        assert output["suggested_policy"] == {
            "corrections": [{"kind": "parse_value", "factor": "weight", "drop": [","]}]
        }
        assert "metadata:" in output["suggested_policy_yaml"]

    def test_the_suggested_repair_is_verified_by_reading_the_data_again(self) -> None:
        result = run_preset(TRIAGE, Readings("commas"))

        (entry,) = result.steps["factor-triage"].output.data()["verification"]
        assert (entry.factor, entry.applied, entry.recovered) == ("weight", True, True)
        assert by_title(result)["Verified"].brief == "1 recovered"

    def test_verify_false_skips_the_second_reading(self) -> None:
        result = run_preset({**TRIAGE, "verify": False}, Readings("commas"))

        assert "Verified" not in by_title(result)
        assert result.steps["factor-triage"].output.data()["verification"] == []

    def test_marker_values_in_a_numeric_column_are_located_and_left_for_the_user_to_answer(self) -> None:
        output = triage_output("markers")

        (finding,) = output["findings"]
        assert (finding.factor, finding.category) == ("latitude", "unreadable")
        assert [(value, count) for value, count, _ in output["places"]["latitude"]] == [("N", 9), ("S", 1)]
        # The suggested remap leaves each marker's answer blank: only the user knows if "N" means missing
        rules = output["suggested_policy"]["corrections"][0]["rules"]
        assert [(rule["match"], rule["to"]) for rule in rules] == [("N", None), ("S", None)]
        (entry,) = output["verification"]
        assert not entry.applied
        assert "still need codes" in entry.detail

    def test_a_continuous_factor_without_bins_is_reported_with_a_bin_count_suggestion(self) -> None:
        result = run_preset(TRIAGE, Readings("continuous"))

        found = by_title(result)
        assert found["Unpinned continuous bins"].brief == "1 factors"
        assert result.health["status"] == "ok"  # information, not a defect
        output = result.steps["factor-triage"].output.data()
        assert output["suggested_policy"] == {"continuous_factor_bins": {"altitude": 4}}

    def test_the_recommended_policy_pins_the_cuts_and_levels_read_from_the_data(self) -> None:
        continuous = triage_output("continuous")["recommended_policy"]["continuous_factor_bins"]["altitude"]
        vocabulary = triage_output("clean")["recommended_policy"]["factor_levels"]["weather"]

        assert continuous[0] == "-inf"
        assert continuous[-1] == "inf"
        assert len(continuous) == 5  # four bins, open at both ends
        assert vocabulary == ["clear", "fog", "rain"]
        report = " ".join(run_preset(TRIAGE, Readings("clean")).report().split())
        assert "can be invalid or misleading" in report  # the caveat that comes with a policy read off one sample

    @pytest.mark.parametrize("kind", ["markers", "continuous", "clean"])
    def test_a_run_under_the_recommended_policy_has_nothing_left_to_report(self, kind: str) -> None:
        recommended = triage_output(kind)["recommended_policy"]
        policy = MetadataPolicyConfig.model_validate({"name": "pinned", **recommended})

        result = run_preset({**TRIAGE, "metadata": "pinned"}, Readings(kind), extra={"metadata": [policy]})

        assert result.success, result.errors
        assert result.findings == []
        binning = result.metadata.metadata_binning
        assert binning is not None
        assert binning["unusable"] == {}
        assert len(binning["factors"]) == 1  # the factor is now read

    def test_the_suggested_policy_makes_a_mixed_column_readable(self) -> None:
        suggested = triage_output("commas")["suggested_policy"]
        policy = MetadataPolicyConfig.model_validate({"name": "fixed", **suggested})

        result = run_preset({**TRIAGE, "metadata": "fixed"}, Readings("commas"), extra={"metadata": [policy]})

        assert "Unreadable factors" not in by_title(result)
        assert result.health["status"] == "ok"
        assert "weight" in result.metadata.metadata_binning["factors"]  # type: ignore[index]

    def test_the_result_records_the_encoding_triage_read(self) -> None:
        result = run_preset(TRIAGE, Readings("commas"))

        binning = result.metadata.metadata_binning
        assert binning is not None
        assert "weight" in binning["unusable"]
        assert result.metadata.encoding_digest == binning["encoding_digest"]

    def test_a_dataset_with_no_metadata_makes_no_findings(self) -> None:
        result = run_preset(TRIAGE, Images(30))

        assert result.success, result.errors
        assert result.findings == []
        assert result.health["status"] == "ok"

    def test_max_examples_limits_the_values_shown_per_factor(self) -> None:
        shown = run_preset({**TRIAGE, "checks": {"factor-issues": {"max_examples": 3}}}, Readings("commas"))
        default = run_preset(TRIAGE, Readings("commas"))

        assert "(+51 more)" in shown.report()  # 54 distinct numeric reads, 3 shown
        assert "(+34 more)" in default.report()  # 20 shown by default
        assert shown.findings[0].severity == default.findings[0].severity  # display only

    def test_settings_that_were_removed_are_refused(self) -> None:
        with pytest.raises(ValueError, match="metadata_exclude"):
            TriageConfig.model_validate({"metadata_exclude": ["id"]})
