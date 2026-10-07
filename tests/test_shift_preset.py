"""`shift`: one detector list, drift and OOD, each detector judged by its family's check (preset naming spec §4b)."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow.workflows._registry import get_workflow


def _chain(settings: dict[str, Any]) -> tuple[list[str], list[str]]:
    preset: Any = get_workflow("shift")
    chain = preset.chain(preset.config_type.model_validate(settings))
    return [step["name"] for step in chain.steps], [entry.name for entry in chain.evaluators or ()]


def test_with_no_settings_it_runs_univariate_drift_and_k_neighbors_ood() -> None:
    steps, evaluators = _chain({})
    assert evaluators == ["drift-univariate", "ood-kneighbors"]
    assert steps == [
        "drift-univariate",
        "drift-univariate-check",
        "ood-kneighbors",
        "ood-kneighbors-check",
        "ood-union",
        "factor-predictors",
        "factor-deviation",
    ]


def test_drift_detectors_alone_build_drift_monitoring_s_chain() -> None:
    steps, evaluators = _chain(
        {"detectors": [{"type": "drift-mmd", "chunking": {"chunk_count": 5}}], "classwise": {"drift-mmd": "class"}}
    )
    assert evaluators == ["drift-mmd", "drift-mmd-unchunked"]
    assert steps == ["drift-mmd", "drift-mmd-check", "drift-mmd-by-class", "drift-mmd-by-class-check"]


def test_ood_detectors_alone_build_ood_detection_s_chain() -> None:
    steps, _ = _chain(
        {"detectors": [{"name": "knn", "type": "ood-kneighbors"}, {"name": "dc", "type": "ood-domain-classifier"}]}
    )
    assert steps == [
        "knn",
        "knn-check",
        "dc",
        "dc-check",
        "ood-union",
        "ood-agreement",
        "factor-predictors",
        "factor-deviation",
    ]


def test_a_mixed_list_keeps_detector_order_and_judges_each_by_its_family() -> None:
    preset: Any = get_workflow("shift")
    config = preset.config_type.model_validate(
        {"detectors": [{"name": "knn", "type": "ood-kneighbors"}, {"name": "ks", "type": "drift-univariate"}]}
    )
    checks = {step["name"]: step["check"] for step in preset.chain(config).steps if "check" in step}
    assert checks == {"knn-check": "ood", "ks-check": "drift"}


def test_classwise_names_only_drift_detectors() -> None:
    with pytest.raises(ValidationError, match="`classwise` names `knn`, an OOD detector"):
        get_workflow("shift").config_type.model_validate(
            {"detectors": [{"name": "knn", "type": "ood-kneighbors"}], "classwise": {"knn": "class"}}
        )


@pytest.mark.parametrize("name", ["ood-union", "factor-deviation", "mmd-check", "mmd-by-class", "mmd-unchunked"])
def test_a_detector_may_not_take_a_name_the_preset_s_own_steps_use(name: str) -> None:
    with pytest.raises(ValidationError, match="the preset's own steps use"):
        get_workflow("shift").config_type.model_validate({"detectors": [{"name": name, "type": "drift-mmd"}]})
