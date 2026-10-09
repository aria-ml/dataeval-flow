"""TC-7-2 — the bias preset: class balance and metadata factors judged from labels and metadata alone."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from dataeval_flow.workflows.bias import BiasConfig
from verification.functional.workflows._synthetic import Images, by_title, run_preset

pytestmark = pytest.mark.required

BIAS = {"type": "bias"}
TITLES = ["Class Imbalance", "Shortcut Risk", "Factor Parity", "Factor Coverage Gaps"]


def shortcut_data() -> Images:
    """90 items over cat, dog and bird (5:3:2); `site` names the class except on every tenth item, `angle` does not."""
    return Images(90, factors=True)


class TestBiasPreset:
    def test_balance_shortcuts_parity_and_gaps_are_judged_without_an_extractor(self) -> None:
        result = run_preset(BIAS, shortcut_data())  # no extractor is named

        assert result.success, result.errors
        assert result.type == "bias"
        assert [f.title for f in result.findings] == TITLES
        found = by_title(result)
        assert found["Class Imbalance"].brief == "3 classes, 90 items, imbalance 2.5:1"
        assert found["Class Imbalance"].severity == "info"  # 2.5:1 is under the warning ratio of 5.0
        for title in TITLES[1:]:
            assert found[title].severity == "warning", title
        assert result.report().strip()

    def test_only_the_factor_that_follows_the_class_is_named(self) -> None:
        result = run_preset(BIAS, shortcut_data())

        found = by_title(result)
        assert found["Shortcut Risk"].brief.startswith("1 of 2 factors tied to the class: site")
        assert found["Factor Parity"].brief.startswith("1 of 2 factors associated with the class: site")
        assert "angle" not in found["Shortcut Risk"].brief
        mutual_information = result.steps["shortcut-risk"].inputs
        assert mutual_information == ["balance"]
        gaps = result.steps["factor-gaps"].output.gaps
        assert gaps
        assert {gap.factor_name for gap in gaps} == {"site"}
        assert all(gap.class_count == 0 for gap in gaps)  # a site no item of that class was seen at

    def test_class_imbalance_warns_past_its_ratio(self) -> None:
        lopsided = Images(100, pattern=(0,) * 9 + (1,), index2label={0: "cat", 1: "dog"})

        warned = run_preset(BIAS, lopsided)
        tolerated = run_preset({**BIAS, "checks": {"class-imbalance": {"warning": 10.0}}}, lopsided)

        assert by_title(warned)["Class Imbalance"].brief == "2 classes, 100 items, imbalance 9.0:1"
        assert by_title(warned)["Class Imbalance"].severity == "warning"
        assert by_title(tolerated)["Class Imbalance"].severity != "warning"

    def test_a_declared_class_with_no_labels_warns_unless_empty_is_false(self) -> None:
        data = Images(60, index2label={0: "cat", 1: "dog", 2: "bird", 3: "owl"})

        assert by_title(run_preset(BIAS, data))["Class Imbalance"].severity == "warning"
        relaxed = run_preset({**BIAS, "checks": {"class-imbalance": {"empty": False}}}, data)
        assert by_title(relaxed)["Class Imbalance"].severity == "info"

    def test_thresholds_in_checks_decide_whether_a_factor_warns(self) -> None:
        lenient = {
            "shortcut-risk": {"warning": 0.9},
            "factor-parity": {"warning": 0.95},
            "factor-coverage-gaps": {"warning": 10},
        }

        result = run_preset({**BIAS, "checks": lenient}, shortcut_data())

        found = by_title(result)
        assert "warning" not in {found[title].severity for title in TITLES[1:]}
        assert result.health["status"] == "ok"

    def test_a_dataset_with_no_metadata_factors_still_judges_its_class_balance(self) -> None:
        result = run_preset(BIAS, Images(90))

        assert result.success, result.errors
        found = by_title(result)
        assert found["Class Imbalance"].brief != "not assessed"
        for title in TITLES[1:]:
            assert found[title].brief == "not assessed", title
            assert found[title].severity == "info"

    def test_factor_gaps_false_leaves_out_the_gap_analysis_and_its_check(self) -> None:
        result = run_preset({**BIAS, "factor-gaps": False}, shortcut_data())

        assert [f.title for f in result.findings] == TITLES[:3]
        assert "factor-gaps" not in result.steps
        assert "factor-coverage-gaps" not in result.steps

    def test_crossed_class_imbalance_bands_are_refused_where_they_were_written(self) -> None:
        message = "`info` (3.0) must not exceed `warning` (2.0)."

        with pytest.raises(ValidationError, match=re.escape(message)):
            BiasConfig.model_validate({"checks": {"class-imbalance": {"warning": 2.0, "info": 3.0}}})
