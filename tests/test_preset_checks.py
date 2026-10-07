"""Each preset's check settings sit under `checks:`, keyed by check type (naming spec §5.1)."""

import pytest
from pydantic import ValidationError

from dataeval_flow.workflows.quality import QualityConfig
from dataeval_flow.workflows.triage import TriageConfig

_CLEANING = {"outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}


def test_data_cleaning_keys_its_checks_by_type_with_todays_defaults() -> None:
    checks = QualityConfig(**_CLEANING).model_dump(by_alias=True)["checks"]  # type: ignore[arg-type]
    assert checks == {
        "image-outliers": {"warning": 3.0},
        "target-outliers": {"warning": 3.0},
        "class-outliers": {"warning": 3.0},
        "image-duplicates": {"exact": 0.0, "near": 5.0},
    }


def test_metadata_triage_takes_max_examples_under_its_check() -> None:
    config = TriageConfig.model_validate({"checks": {"factor-issues": {"max_examples": 5}}})
    assert config.checks.factor_issues.max_examples == 5
    with pytest.raises(ValidationError, match="max_examples"):
        TriageConfig.model_validate({"max_examples": 5})


def test_a_partial_checks_block_keeps_the_other_defaults() -> None:
    checks = QualityConfig.model_validate({**_CLEANING, "checks": {"image-duplicates": {"near": 1.0}}}).checks
    assert (checks.image_duplicates.exact, checks.image_duplicates.near) == (0.0, 1.0)
    assert checks.image_outliers.warning == 3.0
    assert checks.target_outliers.warning == 3.0
