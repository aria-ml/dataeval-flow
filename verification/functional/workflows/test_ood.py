"""TC-9-2 — the shift preset's out-of-distribution detectors: per-image flags, agreement, and their factors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from verification.functional.workflows._synthetic import Concat, Images, run_preset

pytestmark = pytest.mark.required


def exposure(base: float) -> Any:
    """An `exposure` reading per item: `base` plus 0 to 4."""
    return lambda index: base + index % 5


def with_a_half_off_distribution() -> dict[str, Any]:
    """A reference, and a test source: its first 30 images match it, its last 30 have a cast and more exposure."""
    mixed = Concat(
        Images(30, seed=1, extra={"exposure": exposure(1.0)}),
        Images(30, seed=2, tint=True, extra={"exposure": exposure(9.0)}),
    )
    return {"reference": Images(60, seed=0, extra={"exposure": exposure(1.0)}), "mixed": mixed}


TWO_DETECTORS = {
    "type": "shift",
    "detectors": [{"name": "knn", "type": "ood-kneighbors"}, {"name": "dc", "type": "ood-domain-classifier"}],
}


class TestOutOfDistributionDetectors:
    def test_every_test_image_gets_a_score_and_a_flag(self) -> None:
        data = {"reference": Images(60, seed=0), "same": Images(60, seed=1), "shifted": Images(60, seed=2, tint=True)}

        result = run_preset({"type": "shift", "detectors": [{"type": "ood-kneighbors", "k": 3}]}, data, extractor=True)

        assert result.success, result.errors
        for source in ("same", "shifted"):
            output = result.steps["ood-kneighbors"].elements[source].output
            assert len(output.is_ood) == 60
            assert len(output.instance_score) == 60
            assert output.is_ood.dtype == np.bool_
        assert result.steps["ood-kneighbors"].elements["shifted"].output.is_ood.all()
        assert result.steps["ood-kneighbors"].elements["same"].output.is_ood.mean() < 0.15

    def test_the_check_reports_the_share_of_images_flagged_per_test_source(self) -> None:
        data = {"reference": Images(60, seed=0), "same": Images(60, seed=1), "shifted": Images(60, seed=2, tint=True)}

        result = run_preset({"type": "shift", "detectors": [{"type": "ood-kneighbors"}]}, data, extractor=True)

        found = {f.step: (f.severity, f.brief) for f in result.findings}
        assert found == {
            "ood-kneighbors-check[same]": ("info", "3/60 images OOD (5.0%)"),  # over 1.0%, under the warning of 10.0%
            "ood-kneighbors-check[shifted]": ("warning", "60/60 images OOD (100.0%)"),
        }

    def test_the_distance_metric_decides_what_counts_as_different(self) -> None:
        """Dark noise and bright noise point the same way, so the default cosine distance cannot tell them apart."""
        data = {
            "reference": Images(60, seed=0, value_range=(0, 60)),
            "bright": Images(60, seed=1, value_range=(195, 255)),
        }

        def flagged(metric: str) -> int:
            detector = {"name": "knn", "type": "ood-kneighbors", "distance_metric": metric}
            result = run_preset({"type": "shift", "detectors": [detector]}, data, extractor=True)
            return int(result.steps["knn"].elements["bright"].output.is_ood.sum())

        assert flagged("cosine") == 0
        assert flagged("euclidean") == 60

    def test_the_domain_classifier_flags_the_images_it_tells_apart_from_the_reference(self) -> None:
        detector = {"name": "dc", "type": "ood-domain-classifier"}

        result = run_preset({"type": "shift", "detectors": [detector]}, with_a_half_off_distribution(), extractor=True)

        flags = result.steps["dc"].elements["mixed"].output.is_ood
        assert flags[30:].all()  # the colour-cast half
        assert not flags[:30].any()
        assert result.findings[0].brief == "30/60 images OOD (50.0%)"


class TestOutOfDistributionAgreement:
    def test_detectors_are_compared_on_which_images_they_flag(self) -> None:
        result = run_preset(TWO_DETECTORS, with_a_half_off_distribution(), extractor=True)

        assert result.success, result.errors
        union = result.steps["ood-union"].elements["mixed"].output
        assert union.detectors == ["knn", "dc"]
        assert union.mutual == list(range(30, 60))  # flagged by both
        assert union.unique == {"knn": [0], "dc": []}  # one image only the first detector flagged
        assert union.partial == []
        found = {(f.title, f.severity): f.brief for f in result.findings if f.step == "ood-agreement[mixed]"}
        assert found == {
            ("OOD Agreement", "warning"): "30/31 OOD images agreed by all detectors (50.0%)",
            ("OOD Agreement", "info"): "1 image(s) flagged by only one detector",
        }

    def test_agreement_needs_two_detectors_and_the_union_needs_an_ood_detector(self) -> None:
        one = run_preset(
            {"type": "shift", "detectors": [{"name": "knn", "type": "ood-kneighbors"}]},
            with_a_half_off_distribution(),
            extractor=True,
        )
        drift_only = run_preset(
            {"type": "shift", "detectors": [{"name": "ks", "type": "drift-univariate"}]},
            with_a_half_off_distribution(),
            extractor=True,
        )

        assert list(one.steps) == ["knn", "knn-check", "ood-union", "factor-predictors", "factor-deviation"]
        assert list(drift_only.steps) == ["ks", "ks-check"]


class TestFactorsBehindOutOfDistributionImages:
    def test_factor_predictors_ranks_the_factors_that_separate_flagged_images_from_the_rest(self) -> None:
        result = run_preset(TWO_DETECTORS, with_a_half_off_distribution(), extractor=True)

        output = result.steps["factor-predictors"].elements["mixed"].output
        assert output.flagged == 31
        scores = output.factors
        assert list(scores.values()) == sorted(scores.values(), reverse=True)
        assert scores["exposure"] > 0.9  # a metadata factor, ranked beside the image statistics (`f_*`)
        assert scores["f_brightness"] > 0.9
        assert scores["class_label"] < 0.5  # the class tells nothing about which images are flagged

    def test_factor_deviation_says_how_far_each_agreed_image_sits_from_the_reference_on_each_factor(self) -> None:
        result = run_preset(TWO_DETECTORS, with_a_half_off_distribution(), extractor=True)

        output = result.steps["factor-deviation"].elements["mixed"].output
        assert {item.index for item in output.items} == set(range(30, 60))
        assert all(item.deviations["exposure"] > 3 for item in output.items)  # exposure 9 to 13 against 1 to 5

    def test_factor_steps_can_be_left_out(self) -> None:
        entry = {**TWO_DETECTORS, "factor-predictors": False, "factor-deviation": False}

        result = run_preset(entry, with_a_half_off_distribution(), extractor=True)

        assert list(result.steps) == ["knn", "knn-check", "dc", "dc-check", "ood-union", "ood-agreement"]

    def test_a_test_source_with_no_agreed_image_explains_nothing(self) -> None:
        data = {"reference": Images(60, seed=0), "same": Images(60, seed=1)}

        result = run_preset(TWO_DETECTORS, data, extractor=True)

        output = result.steps["factor-deviation"].elements["same"].output
        assert output.items == []
        assert output.reason == "No image was flagged by every detector, so none is explained."
