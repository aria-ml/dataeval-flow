"""Each preset's check settings sit under `checks:`, keyed by check type (naming spec §5.1)."""

import pytest
from pydantic import ValidationError

from dataeval_flow.workflows.data_cleaning import DataCleaningConfig
from dataeval_flow.workflows.metadata_triage import MetadataTriageConfig

_CLEANING = {"outlier_method": "zscore", "outlier_flags": ["pixel"]}  # Task 5 reshapes these into `outliers:`


def test_data_cleaning_keys_its_checks_by_type_with_todays_defaults() -> None:
    checks = DataCleaningConfig(**_CLEANING).model_dump(by_alias=True)["checks"]
    assert checks == {
        "image-outliers": {"warning": 3.0},
        "target-outliers": {"warning": 3.0},
        "classwise-outliers": {"warning": 3.0},
        "image-duplicates": {"exact": 0.0, "near": 5.0},
        "class-imbalance": {"warning": 5.0},
    }


def test_health_thresholds_is_refused() -> None:
    with pytest.raises(ValidationError, match="health_thresholds"):
        DataCleaningConfig(**_CLEANING, health_thresholds={"image_outliers": 1.0})  # type: ignore[call-arg]


def test_metadata_triage_takes_max_examples_under_its_check() -> None:
    config = MetadataTriageConfig.model_validate({"checks": {"metadata-issues": {"max_examples": 5}}})
    assert config.checks.metadata_issues.max_examples == 5
    with pytest.raises(ValidationError, match="max_examples"):
        MetadataTriageConfig.model_validate({"max_examples": 5})


def test_a_partial_checks_block_keeps_the_other_defaults() -> None:
    checks = DataCleaningConfig.model_validate({**_CLEANING, "checks": {"image-duplicates": {"near": 1.0}}}).checks
    assert (checks.image_duplicates.exact, checks.image_duplicates.near) == (0.0, 1.0)
    assert checks.image_outliers.warning == 3.0
    assert checks.class_imbalance.warning == 5.0
