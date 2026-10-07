"""Keys that earlier versions took fail as unknown, with no per-key migration message (preset naming spec §4g)."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow.workflows._registry import get_workflow
from tests.test_naming_conventions import _MINIMAL

_RETIRED = [
    ("audit", {"outlier_method": "zscore"}),
    ("audit", {"health_thresholds": {}}),
    ("scope", {"coverage_method": "naive"}),
    ("scope", {"balance": True}),
    ("scope", {"health_thresholds": {}}),
    ("quality", {"checks": {"class-imbalance": {"warning": 3.0}}}),
    ("splits", {"checks": {"class-imbalance": {"warning": 3.0}}}),
]


@pytest.mark.parametrize(("preset", "extra"), _RETIRED, ids=lambda value: str(value))
def test_a_retired_key_fails_as_an_unknown_key(preset: str, extra: dict[str, Any]) -> None:
    config_type = get_workflow(preset).config_type
    with pytest.raises(ValidationError, match="Extra inputs are not permitted") as raised:
        config_type.model_validate({**_MINIMAL[preset], **extra})
    assert "is refused" not in str(raised.value)
    assert "is now" not in str(raised.value)


def test_data_analysis_is_an_unknown_workflow() -> None:
    with pytest.raises(ValueError, match=r"Unknown workflow: 'data-analysis'\. Installed: "):
        get_workflow("data-analysis")


def test_a_detector_without_a_type_names_the_types_with_no_legacy_hint() -> None:
    config_type = get_workflow("shift").config_type
    with pytest.raises(ValidationError, match="Each detector needs a `type`") as raised:
        config_type.model_validate({"detectors": [{"method": "mmd"}]})
    assert "Legacy" not in str(raised.value)
