"""TC-9-1 — the shift preset's drift detectors: whole-batch, chunked and per-class drift against a reference."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow.workflows.shift import ShiftConfig
from verification.functional.workflows._synthetic import Concat, Images, run_preset

pytestmark = pytest.mark.required


def sources() -> dict[str, Any]:
    """A reference, a source drawn from the same distribution, and one with a colour cast the reference never shows."""
    return {"reference": Images(60, seed=0), "same": Images(60, seed=1), "shifted": Images(60, seed=2, tint=True)}


def drift_findings(result: Any) -> dict[str, str]:
    """Each drift finding's brief, by the test source it judged."""
    return {f.step.split("[")[1].rstrip("]"): f.brief for f in result.findings if f.step and "-check[" in f.step}


class TestDriftDetectors:
    @pytest.mark.parametrize("detector", ["drift-univariate", "drift-mmd", "drift-kneighbors"])
    def test_a_source_like_the_reference_is_not_drifted_and_a_colour_cast_source_is(self, detector: str) -> None:
        result = run_preset({"type": "shift", "detectors": [{"type": detector}]}, sources(), extractor=True)

        assert result.success, result.errors
        assert drift_findings(result) == {"same": "no drift", "shifted": "drift"}
        severities = {f.step: f.severity for f in result.findings}
        assert severities == {f"{detector}-check[same]": "ok", f"{detector}-check[shifted]": "warning"}
        outputs = {key: element.output for key, element in result.steps[detector].elements.items()}
        assert not outputs["same"].drifted
        assert outputs["shifted"].drifted
        assert result.health["status"] == "warning"

    def test_the_domain_classifier_tells_the_colour_cast_source_apart_perfectly(self) -> None:
        result = run_preset(
            {"type": "shift", "detectors": [{"type": "drift-domain-classifier"}]}, sources(), extractor=True
        )

        output = result.steps["drift-domain-classifier"].elements["shifted"].output
        assert output.drifted
        assert output.metric_name == "auroc"
        assert float(output.distance) == 1.0

    def test_each_detector_judges_every_test_source_against_the_first_source(self) -> None:
        detectors = [{"name": "mmd", "type": "drift-mmd"}, {"name": "ks", "type": "drift-univariate"}]

        result = run_preset({"type": "shift", "detectors": detectors}, sources(), extractor=True)

        assert [f.step for f in result.findings] == [
            "mmd-check[same]",
            "mmd-check[shifted]",
            "ks-check[same]",
            "ks-check[shifted]",
        ]
        assert [f.title for f in result.findings] == ["Drift (MMD) · mmd"] * 2 + ["Drift (Univariate) · ks"] * 2
        assert list(result.steps["mmd"].elements) == ["same", "shifted"]
        assert result.steps["mmd"].inputs == ["reference", "tests"]

    def test_the_first_source_is_the_reference(self) -> None:
        swapped = {"reference": Images(60, seed=2, tint=True), "other": Images(60, seed=0)}

        result = run_preset({"type": "shift", "detectors": [{"type": "drift-univariate"}]}, swapped, extractor=True)

        assert drift_findings(result) == {"other": "drift"}  # the plain images are what drifted from the cast ones

    def test_without_detectors_it_runs_univariate_drift_and_k_neighbors_ood(self) -> None:
        result = run_preset({"type": "shift"}, sources(), extractor=True)

        assert list(result.steps) == [
            "drift-univariate",
            "drift-univariate-check",
            "ood-kneighbors",
            "ood-kneighbors-check",
            "ood-union",
            "factor-predictors",
            "factor-deviation",
        ]
        titles = {f.title for f in result.findings}
        assert titles == {"Drift (Univariate)", "OOD (K-Neighbors)"}

    def test_a_shift_task_needs_an_extractor_and_two_sources(self) -> None:
        with pytest.raises(ValidationError, match="extractor"):
            run_preset({"type": "shift"}, sources())
        with pytest.raises(ValidationError, match="two or more sources"):
            run_preset({"type": "shift"}, {"reference": Images(60)}, extractor=True)

    def test_warn_on_drift_false_reports_drift_without_warning(self) -> None:
        entry = {
            "type": "shift",
            "detectors": [{"type": "drift-mmd"}],
            "checks": {"drift": {"warn_on_drift": False}},
        }

        result = run_preset(entry, sources(), extractor=True)

        assert drift_findings(result)["shifted"] == "drift"
        assert {f.severity for f in result.findings} == {"ok", "info"}
        assert result.health["status"] == "ok"


class TestChunkedDrift:
    """A test source of 90 images read as three chunks of 30: the first 60 like the reference, the last 30 not."""

    @staticmethod
    def run(checks: dict[str, Any] | None = None, chunking: dict[str, int] | None = None) -> Any:
        detector = {"name": "mmd", "type": "drift-mmd", "chunking": chunking or {"chunk_count": 3}}
        test = Concat(Images(60, seed=1), Images(30, seed=2, tint=True))
        entry = {"type": "shift", "detectors": [detector], "checks": {"drift": checks or {}}}
        return run_preset(entry, {"reference": Images(90, seed=0), "test": test}, extractor=True)

    def test_each_chunk_has_its_own_verdict(self) -> None:
        result = self.run()

        output = result.steps["mmd"].elements["test"].output
        chunks = output.details.to_dicts()
        assert [(c["start_index"], c["end_index"], c["drifted"]) for c in chunks] == [
            (0, 29, False),
            (30, 59, False),
            (60, 89, True),
        ]
        (finding,) = result.findings
        assert finding.brief == "1/3 chunks drifted"
        assert finding.severity == "warning"  # 33% of the chunks, over the default 10%

    def test_chunk_size_sets_the_chunks_by_their_length(self) -> None:
        result = self.run(chunking={"chunk_size": 10})

        assert len(result.steps["mmd"].elements["test"].output.details) == 9
        assert result.findings[0].brief == "3/9 chunks drifted"

    def test_chunk_percent_is_the_share_of_drifting_chunks_that_warns(self) -> None:
        tolerant = self.run({"chunk_percent": 50.0})

        assert tolerant.findings[0].brief == "1/3 chunks drifted"
        assert tolerant.findings[0].severity == "info"  # a third of the chunks, under 50%

    def test_consecutive_chunks_warns_on_a_run_of_drifting_chunks(self) -> None:
        test = Concat(Images(30, seed=1), Images(60, seed=2, tint=True))  # chunks 2 and 3 drift
        entry = {
            "type": "shift",
            "detectors": [{"name": "mmd", "type": "drift-mmd", "chunking": {"chunk_count": 3}}],
        }

        def severity(**checks: float) -> str:
            result = run_preset(
                {**entry, "checks": {"drift": {"chunk_percent": 90.0, **checks}}},
                {"reference": Images(90, seed=0), "test": test},
                extractor=True,
            )
            return result.findings[0].severity

        assert severity() == "info"  # 2 chunks in a row is not more than the default 2
        assert severity(consecutive_chunks=1) == "warning"


class TestClasswiseDrift:
    @staticmethod
    def run(classwise: dict[str, Any]) -> Any:
        data = {"reference": Images(90, seed=0), "test": Images(90, seed=2, tint=True)}
        entry = {"type": "shift", "detectors": [{"name": "mmd", "type": "drift-mmd"}], "classwise": classwise}
        return run_preset(entry, data, extractor=True)

    def test_a_detector_also_runs_once_per_class_and_the_verdicts_roll_up(self) -> None:
        result = self.run({"mmd": "class"})

        assert list(result.steps) == ["mmd", "mmd-check", "mmd-by-class", "mmd-by-class-check"]
        by_class = {f.step: f for f in result.findings}
        assert by_class["mmd-check[test]"].brief == "drift"
        assert by_class["mmd-by-class-check[test]"].brief == "3/3 classes warn"
        assert set(result.steps["mmd-by-class"].elements["test"].output.outputs) == {"cat", "dog", "bird"}

    def test_a_class_with_too_few_items_is_left_out_and_the_output_says_why(self) -> None:
        result = self.run({"mmd": {"class": {"min_items": 30}}})

        output = result.steps["mmd-by-class"].elements["test"].output
        assert set(output.outputs) == {"cat"}  # 45 of 90 items; dog has 27 and bird 18
        assert output.skipped == {
            "dog": "27 items in `reference`, fewer than `min_items` 30",
            "bird": "18 items in `reference`, fewer than `min_items` 30",
        }
        by_step = {f.step: f.brief for f in result.findings}
        assert by_step["mmd-by-class-check[test]"] == "1/1 classes warn"

    def test_classes_can_be_grouped(self) -> None:
        result = self.run({"mmd": {"class": {"groups": {"pets": ["cat", "dog"], "wild": ["bird"]}}}})

        assert set(result.steps["mmd-by-class"].elements["test"].output.outputs) == {"pets", "wild"}
        assert {f.step: f.brief for f in result.findings}["mmd-by-class-check[test]"] == "2/2 groups warn"

    def test_only_drift_detectors_may_run_by_class(self) -> None:
        with pytest.raises(ValidationError, match="`classwise` names `knn`, an OOD detector"):
            ShiftConfig.model_validate(
                {"detectors": [{"name": "knn", "type": "ood-kneighbors"}], "classwise": {"knn": "class"}}
            )
        with pytest.raises(ValidationError, match="`classwise` names `nope`, which no detector is"):
            ShiftConfig.model_validate({"classwise": {"nope": "class"}})

    def test_two_detectors_may_not_share_a_name(self) -> None:
        with pytest.raises(ValidationError, match="two detectors named `x`"):
            ShiftConfig.model_validate(
                {"detectors": [{"name": "x", "type": "drift-mmd"}, {"name": "x", "type": "drift-univariate"}]}
            )
