"""The names a preset's chain gives its steps, which reach users as Finding.step and result keys (naming spec §5.3)."""

import pytest
from pydantic import ValidationError

from dataeval_flow.workflows.drift_monitoring import DriftMonitoringConfig, DriftMonitoringWorkflow
from dataeval_flow.workflows.ood_detection import OODDetectionConfig
from dataeval_flow.workflows.quality import QualityConfig, QualityWorkflow


def test_data_cleanings_steps() -> None:
    config = QualityConfig.model_validate({"outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}})
    assert [step["name"] for step in QualityWorkflow.chain(config).steps] == [  # type: ignore[index]
        "outliers",
        "label-health",
        "outliers-by-class",
        "duplicates",
        "image-outliers",
        "target-outliers",
        "class-outliers",
        "image-duplicates",
        "clean",
    ]


def test_drift_monitorings_per_detector_steps_take_suffixes() -> None:
    config = DriftMonitoringConfig.model_validate(
        {"detectors": [{"name": "mmd", "type": "drift-mmd"}], "classwise": {"mmd": "class"}}
    )
    assert [step["name"] for step in DriftMonitoringWorkflow.chain(config).steps] == [  # type: ignore[index]
        "mmd",
        "mmd-check",
        "mmd-by-class",
        "mmd-by-class-check",
    ]


@pytest.mark.parametrize("name", ["cam-by-class", "cam-check", "cam-unchunked"])
def test_drift_monitoring_refuses_a_detector_name_its_own_steps_would_take(name: str) -> None:
    with pytest.raises(ValidationError, match="rename it"):
        DriftMonitoringConfig.model_validate({"detectors": [{"name": name, "type": "drift-mmd"}]})


@pytest.mark.parametrize("name", ["ood-union", "ood-agreement", "factor-predictors", "factor-deviation", "knn-check"])
def test_ood_detection_refuses_a_detector_name_its_own_steps_take(name: str) -> None:
    with pytest.raises(ValidationError, match="rename it"):
        OODDetectionConfig.model_validate(
            {"detectors": [{"name": name, "type": "ood-kneighbors", "distance_metric": "euclidean"}]}
        )


def test_ood_detection_takes_a_detector_named_for_its_type() -> None:
    config = OODDetectionConfig.model_validate(
        {"detectors": [{"name": "ood-kneighbors", "type": "ood-kneighbors", "distance_metric": "euclidean"}]}
    )
    assert config.detectors[0].name == "ood-kneighbors"
