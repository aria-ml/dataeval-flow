"""TC-8-1 — the scope preset: embedding coverage, dimensional completeness and the class worklist."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow.workflows.scope import ScopeConfig
from verification.functional.workflows._synthetic import ANIMALS, Detections, Images, by_title, run_preset

pytestmark = pytest.mark.required

SCOPE: dict[str, Any] = {"type": "scope"}


def animals(count: int = 90, **kwargs: Any) -> Images:
    """Cat, dog and bird in proportion 5:3:2, `owl` declared and never labelled."""
    return Images(count, index2label=ANIMALS, **kwargs)


class TestScopePreset:
    def test_embedding_coverage_completeness_and_the_class_worklist_are_judged_with_an_extractor(self) -> None:
        result = run_preset(SCOPE, animals(), extractor=True)

        assert result.success, result.errors
        assert result.type == "scope"
        assert [f.title for f in result.findings] == ["Class Coverage", "Dimensional Completeness", "Class Shortfall"]
        found = by_title(result)
        assert found["Class Coverage"].brief == "1 uncovered (1.1%)"
        assert found["Dimensional Completeness"].brief == "Completeness: 0.368"
        assert found["Dimensional Completeness"].severity == "warning"  # under the 0.5 default
        assert {name: step.status for name, step in result.steps.items()} == {
            "crops": "ok",
            "coverage": "ok",
            "class-coverage": "ok",
            "completeness": "ok",
            "dimensional-completeness": "ok",
            "representation": "ok",
            "class-shortfall": "ok",
        }
        assert result.report().strip()

    def test_without_an_extractor_the_embedding_findings_are_not_assessed_and_the_worklist_still_runs(self) -> None:
        result = run_preset(SCOPE, animals())

        assert result.success, result.errors
        found = by_title(result)
        assert found["Class Coverage"].brief == "not assessed"
        assert found["Dimensional Completeness"].brief == "not assessed"
        assert found["Class Shortfall"].brief == "2 classes short · deficit 26"
        assert result.steps["coverage"].status == "skipped"
        assert result.steps["coverage"].reason == "requires an extractor"
        assert result.steps["completeness"].status == "skipped"
        assert result.steps["representation"].status == "ok"

    def test_the_worklist_names_each_class_short_of_an_even_spread_and_what_to_do_about_it(self) -> None:
        result = run_preset(SCOPE, animals())

        rows = result.steps["representation"].output.data().to_dicts()
        assert [(r["concept"], r["action"], r["count"], r["target"], r["deficit"]) for r in rows] == [
            ("owl", "acquire", 0, 22, 22),  # declared and never collected
            ("bird", "augment", 18, 22, 4),  # collected, too few
        ]

    def test_expected_shares_replace_the_even_spread_for_the_classes_they_name(self) -> None:
        result = run_preset({**SCOPE, "representation": {"expected": {"bird": 0.4, "nope": 0.1}}}, Images(90))

        rows = result.steps["representation"].output.data().to_dicts()
        assert [(r["concept"], r["count"], r["target"], r["deficit"]) for r in rows] == [
            ("bird", 18, 36, 18),  # 40% of 90 items
            ("dog", 27, 30, 3),  # the others keep an even share
        ]
        assert by_title(result)["Class Shortfall"].severity == "warning"

    def test_a_class_whose_images_barely_vary_is_flagged_as_clustered(self) -> None:
        varied = run_preset(SCOPE, animals(150), extractor=True)
        clustered = run_preset(SCOPE, animals(150, tight=(2,)), extractor=True)

        assert by_title(varied)["Class Coverage"].severity == "info"
        assert by_title(clustered)["Class Coverage"].severity == "warning"
        assert by_title(clustered)["Class Coverage"].brief.endswith("1 clustered")
        rows = {r["class"]: r for r in clustered.steps["coverage"].output.data().to_dicts()}
        assert rows["bird"]["dispersion"] < 0.5 <= rows["cat"]["dispersion"]

    def test_thresholds_in_checks_decide_whether_a_finding_warns(self) -> None:
        lenient = {"dimensional-completeness": {"warning": 0.1, "info": 0.2}}

        default = run_preset(SCOPE, animals(), extractor=True)
        relaxed = run_preset({**SCOPE, "checks": lenient}, animals(), extractor=True)

        assert by_title(default)["Dimensional Completeness"].severity == "warning"
        assert by_title(relaxed)["Dimensional Completeness"].severity == "ok"

    def test_naive_coverage_adds_the_uncovered_items_check(self) -> None:
        result = run_preset({**SCOPE, "coverage": {"method": "naive"}}, animals(), extractor=True)

        assert "Uncovered Items" in by_title(result)
        assert "uncovered-items" in result.steps
        assert "uncovered-items" not in run_preset(SCOPE, animals(), extractor=True).steps

    def test_completeness_false_leaves_out_the_completeness_steps(self) -> None:
        result = run_preset({**SCOPE, "completeness": False}, animals(), extractor=True)

        assert "completeness" not in result.steps
        assert "Dimensional Completeness" not in by_title(result)

    def test_detections_are_cropped_and_coverage_is_measured_on_the_crops(self) -> None:
        # Of the 90 boxes, 30 are 2 pixels wide; `min_size` drops them, leaving 60 crops of one size
        entry = {**SCOPE, "wrap": {"params": {"min_size": 4}}}

        result = run_preset(entry, Detections(40), extractor=True)

        assert result.success, result.errors
        crops = result.steps["crops"]
        assert crops.details == {"wrapped": True, "kind": "object_detection", "items": 60, "dropped": 30}
        assert len(crops.output) == 60
        found = by_title(result)
        assert found["Class Coverage"].brief != "not assessed"
        assert found["Class Shortfall"].brief == "1 classes short · deficit 30"  # `bus`, declared and never boxed

    def test_without_a_minimum_size_every_box_is_cropped(self) -> None:
        result = run_preset(SCOPE, Detections(40))

        assert result.steps["crops"].details == {"wrapped": True, "kind": "object_detection", "items": 90, "dropped": 0}

    def test_classification_data_passes_through_the_crop_step_unchanged(self) -> None:
        result = run_preset(SCOPE, animals(60))

        assert result.steps["crops"].status == "ok"
        assert len(result.steps["crops"].output) == 60

    def test_crop_padding_widens_each_crop(self) -> None:
        def crop_shapes(padding: float) -> set[tuple[int, ...]]:
            entry = {**SCOPE, "wrap": {"params": {"padding": padding, "min_size": 4}}}
            crops = run_preset(entry, Detections(40)).steps["crops"].output
            return {tuple(crops[i][0].shape) for i in range(len(crops))}

        assert crop_shapes(0.0) == {(3, 8, 8)}
        assert crop_shapes(0.25) == {(3, 12, 12)}  # a quarter of the box's side on each edge

    def test_an_ontology_is_refused_and_points_to_the_taxonomy_preset(self) -> None:
        with pytest.raises(ValidationError, match="taxonomy"):
            ScopeConfig.model_validate({"ontology": {"animal": {"cat": None}}})
