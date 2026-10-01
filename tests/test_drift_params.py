"""The drift-monitoring config: drift evaluator entries as detectors, `classwise` by name, thresholds by check type."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow.evaluators.shift import (
    DriftDomainClassifierConfig,
    DriftKNeighborsConfig,
    DriftMMDConfig,
    DriftUnivariateConfig,
)
from dataeval_flow.steps.checks import DriftThresholds
from dataeval_flow.workflows.drift_monitoring import DriftMonitoringConfig

pytestmark = pytest.mark.required


def _config(**settings: Any) -> DriftMonitoringConfig:
    return DriftMonitoringConfig.model_validate({"name": "drift", **settings})


@pytest.mark.parametrize(
    ("type_id", "config_class"),
    [
        ("drift-univariate", DriftUnivariateConfig),
        ("drift-mmd", DriftMMDConfig),
        ("drift-kneighbors", DriftKNeighborsConfig),
        ("drift-domain-classifier", DriftDomainClassifierConfig),
    ],
)
def test_each_detector_validates_as_the_evaluator_config_its_type_names(type_id: str, config_class: type) -> None:
    (detector,) = _config(detectors=[{"type": type_id}]).detectors
    assert type(detector) is config_class
    assert detector.name == type_id


def test_a_detector_takes_its_evaluator_s_settings() -> None:
    entry = {"name": "ks", "type": "drift-univariate", "method": "cvm", "p_val": 0.01, "chunking": {"chunk_count": 5}}
    (detector,) = _config(detectors=[entry]).detectors
    assert isinstance(detector, DriftUnivariateConfig)
    assert (detector.name, detector.method, detector.p_val) == ("ks", "cvm", 0.01)
    assert detector.chunking is not None
    assert detector.chunking.chunk_count == 5


def test_a_detector_refuses_another_evaluator_s_settings() -> None:
    with pytest.raises(ValidationError, match="n_permutations"):
        _config(detectors=[{"type": "drift-kneighbors", "n_permutations": 10}])


def test_a_config_object_is_taken_as_it_is() -> None:
    mmd = DriftMMDConfig(n_permutations=10)
    assert DriftMonitoringConfig(detectors=[mmd]).detectors == [mmd]


def test_defaults() -> None:
    config = _config(detectors=[{"type": "drift-mmd"}])
    assert config.type == "drift-monitoring"
    assert config.classwise == []
    assert config.health_thresholds.drift == DriftThresholds()


def test_a_dump_keeps_each_detector_s_own_settings_and_validates_back() -> None:
    config = _config(
        detectors=[{"name": "ks", "type": "drift-univariate", "chunking": {"chunk_count": 5}}, {"type": "drift-mmd"}],
        classwise=["ks"],
    )
    dumped = config.model_dump()
    assert dumped["detectors"][0]["chunking"]["chunk_count"] == 5
    assert dumped["detectors"][1]["n_permutations"] is None
    assert DriftMonitoringConfig.model_validate(dumped) == config


def test_health_thresholds_hold_the_drift_check_s_fields() -> None:
    drift = {"warn_on_drift": False, "chunk_percent": None, "consecutive_chunks": 2}
    config = _config(detectors=[{"type": "drift-mmd"}], health_thresholds={"drift": drift})
    assert config.health_thresholds.drift.model_dump() == drift


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"detectors": []}, "at least 1 item"),
        ({"detectors": [{"type": "drift-mmd"}, {"type": "drift-mmd"}]}, "two detectors named `drift-mmd`"),
        ({"detectors": [{"name": "x-check", "type": "drift-mmd"}]}, "-check"),
        ({"detectors": [{"name": "x-classes", "type": "drift-mmd"}]}, "`x-classes` ends in"),
        ({"detectors": [{"type": "drift-wasserstein"}]}, "validation set"),
        ({"detectors": [{"type": "outliers"}]}, "drift-univariate"),
        ({"detectors": [{"type": "drift-mmd"}], "classwise": ["ks"]}, "`ks`"),
        ({"detectors": [{"type": "drift-mmd"}], "update_strategy": {"type": "last_seen", "n": 5}}, "update_strategy"),
        ({"detectors": [{"type": "drift-mmd"}], "health_thresholds": {"any_drift_is_warning": True}}, "any_drift"),
    ],
)
def test_load_refuses(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**settings)
