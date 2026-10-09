"""TC-7-1 — the quality preset: outliers, duplicates, and the dataset without them."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.quality import QualityConfig
from verification.functional.workflows._synthetic import Detections, Images, by_title, run_preset

pytestmark = pytest.mark.required

QUALITY: dict[str, Any] = {"type": "quality", "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}
"""Image 7 of `planted_images` is solid white among pixel values of 100 to 149, so one zscore pixel outlier; image 5
is a copy of image 0, so one exact duplicate group, [0, 5]."""


def planted_images(**kwargs: Any) -> Images:
    return Images(60, planted=True, shape=(3, 16, 16), value_range=(100, 150), **kwargs)


def kept_ids(result: ChainResult) -> set[int]:
    clean = result.steps["clean"].output
    return {clean[i][2]["id"] for i in range(len(clean))}


class TestQualityPreset:
    def test_planted_outlier_and_duplicate_are_found_and_reported(self) -> None:
        result = run_preset(QUALITY, planted_images())

        assert result.success, result.errors
        assert result.type == "quality"
        found = by_title(result)
        assert {"Image Outliers", "Class Outliers", "Image Duplicates"} <= set(found)
        assert found["Image Outliers"].brief == "1 images (1.7%)"
        assert found["Image Duplicates"].brief == "2 exact (3.3%), 0 near (0.0%)"
        assert found["Image Duplicates"].severity == "warning"
        # The evaluators' own outputs name the items
        assert set(result.steps["outliers"].output.data()["item_index"].to_list()) == {7}
        assert result.steps["duplicates"].output.data()["item_indices"].to_list() == [[0, 5]]
        assert result.report().strip()

    def test_clean_removes_flagged_images_and_all_but_the_first_duplicate(self) -> None:
        result = run_preset(QUALITY, planted_images())

        clean = result.steps["clean"]
        assert kept_ids(result) == set(range(60)) - {5, 7}
        assert clean.details["removed"] == {"items": 2, "detections": 0, "tracks": 0, "frames": 0}
        assert clean.details["by_plan"] == {"duplicates": {"items": 1}, "outliers": {"items": 1}}

    def test_near_duplicates_are_found_and_removed_with_the_exact_ones(self) -> None:
        result = run_preset(QUALITY, planted_images(near_duplicate=True))

        assert by_title(result)["Image Duplicates"].brief == "2 exact (3.3%), 2 near (3.3%)"
        assert kept_ids(result) == set(range(60)) - {5, 7, 9}

    def test_run_returns_the_chain_and_the_cleaned_dataset(self) -> None:
        result = run(QualityConfig.model_validate(QUALITY), planted_images())

        assert isinstance(result, ChainResult)
        assert len(result.steps["clean"].output) == 58

    def test_a_threshold_set_in_checks_decides_whether_a_finding_warns(self) -> None:
        strict = run_preset({**QUALITY, "checks": {"image-outliers": {"warning": 1.0}}}, planted_images())
        silent = run_preset(
            {
                **QUALITY,
                "checks": {
                    "image-outliers": {"warning": None},
                    "image-duplicates": {"exact": None, "near": None},
                    "class-outliers": {"warning": None},
                },
            },
            planted_images(),
        )

        assert by_title(strict)["Image Outliers"].severity == "warning"  # 1.7% of images, over 1.0
        assert by_title(run_preset(QUALITY, planted_images()))["Image Outliers"].severity == "info"  # under 3.0
        assert {f.severity for f in silent.findings} == {"info"}  # a null threshold judges nothing
        assert silent.health["status"] == "ok"
        assert len(kept_ids(silent)) == 58  # and removes just the same

    def test_detection_data_is_judged_per_box_and_clean_removes_boxes_not_images(self) -> None:
        data = Detections(40)
        result = run_preset(QUALITY, data)

        assert result.success, result.errors
        found = by_title(result)
        assert found["Target Outliers"].severity == "warning"
        removed = result.steps["clean"].details["removed"]
        assert removed["items"] == 0
        assert removed["detections"] > 0
        clean = result.steps["clean"].output
        boxes_before = sum(len(data[i][1].boxes) for i in range(len(data)))
        boxes_after = sum(len(clean[i][1].boxes) for i in range(len(clean)))
        assert len(clean) == len(data)
        assert boxes_after == boxes_before - removed["detections"]

    def test_settings_reach_the_evaluators_they_name(self) -> None:
        from dataeval_flow.workflows.quality import QualityWorkflow

        config = QualityConfig.model_validate(
            {
                "outliers": {"flags": ["dimension"], "outlier_threshold": ["modzscore", 4.0]},
                "duplicates": {"flags": ["hash_d4"], "merge_near_duplicates": False},
                "checks": {"image-outliers": {"warning": 6.0}, "image-duplicates": {"exact": 1.0, "near": 9.0}},
            }
        )
        chain = QualityWorkflow.chain(config)
        outliers, duplicates, _ = chain.evaluators or ()

        assert (outliers.flags, outliers.outlier_threshold) == (["dimension"], ("modzscore", 4.0))  # type: ignore[attr-defined]
        assert (duplicates.flags, duplicates.merge_near_duplicates) == (["hash_d4"], False)  # type: ignore[attr-defined]
        steps = {step["name"]: step for step in chain.steps}  # type: ignore[index]
        assert steps["image-outliers"]["warning"] == 6.0
        assert (steps["image-duplicates"]["exact"], steps["image-duplicates"]["near"]) == (1.0, 9.0)

    def test_outlier_settings_are_required_and_unknown_settings_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="outliers"):
            QualityConfig.model_validate({"name": "w"})
        with pytest.raises(ValidationError, match="metadata_exclude"):
            QualityConfig.model_validate({**QUALITY, "metadata_exclude": ["id"]})
        with pytest.raises(ValidationError, match="class-imbalance"):
            QualityConfig.model_validate({**QUALITY, "checks": {"class-imbalance": {"warning": 2.0}}})

    def test_cluster_outliers_without_an_extractor_are_refused_when_the_config_loads(self) -> None:
        entry = {**QUALITY, "outliers": {**QUALITY["outliers"], "cluster_threshold": 3.0}}

        with pytest.raises(ValidationError, match="needs an extractor"):
            run_preset(entry, planted_images())
