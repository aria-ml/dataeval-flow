"""Unit tests for cleaning workflow builder helpers (_build_outliers, _build_duplicates)."""

import pytest
from dataeval.quality import Duplicates, Outliers

from dataeval_flow.workflows.data_cleaning import DataCleaningConfig
from dataeval_flow.workflows.data_cleaning._workflow import (
    _build_duplicates,
    _build_outliers,
)

pytestmark = pytest.mark.required


def _make_params(**overrides: object) -> DataCleaningConfig:
    """Build DataCleaningConfig with defaults for testing."""
    defaults: dict[str, object] = {
        "outlier_method": "adaptive",
        "outlier_flags": ["dimension", "pixel", "visual"],
        "outlier_threshold": None,
    }
    defaults.update(overrides)
    return DataCleaningConfig(**defaults)  # type: ignore[arg-type]


class TestBuildOutliers:
    """Test _build_outliers factory helper."""

    def test_default_parameters(self):
        """Outliers created with default parameters."""
        params = _make_params()
        evaluator = _build_outliers(params)
        assert isinstance(evaluator, Outliers)

    def test_custom_method(self):
        """Outliers created with custom method."""
        params = _make_params(outlier_method="iqr")
        evaluator = _build_outliers(params)
        assert isinstance(evaluator, Outliers)

    def test_custom_threshold(self):
        """Outliers created with custom threshold."""
        params = _make_params(outlier_threshold=2.5)
        evaluator = _build_outliers(params)
        assert isinstance(evaluator, Outliers)

    def test_subset_flags(self):
        """Outliers created with subset of flags."""
        params = _make_params(outlier_flags=["visual"])
        evaluator = _build_outliers(params)
        assert isinstance(evaluator, Outliers)

    # A cluster param without an extractor is refused when the task's config loads (see
    # test_workflow_inputs.py); the builder itself no longer guards against it.


class TestBuildOutliersFromParams:
    """Test building Outliers from DataCleaningConfig."""

    def test_from_params(self):
        """Outliers created from DataCleaningConfig."""
        params = _make_params()
        evaluator = _build_outliers(params)
        assert isinstance(evaluator, Outliers)

    def test_from_params_with_threshold(self):
        """Outliers created from params with custom threshold."""
        params = _make_params(outlier_method="iqr", outlier_flags=["pixel"], outlier_threshold=3.0)
        evaluator = _build_outliers(params)
        assert isinstance(evaluator, Outliers)


class TestBuildDuplicates:
    """Test _build_duplicates factory helper."""

    def test_default_parameters(self):
        """Duplicates created with default parameters."""
        params = _make_params()
        evaluator = _build_duplicates(params)
        assert isinstance(evaluator, Duplicates)

    def test_with_merge_near_false(self):
        """Duplicates created with merge_near=False."""
        params = _make_params(duplicate_merge_near=False)
        evaluator = _build_duplicates(params)
        assert isinstance(evaluator, Duplicates)

    # A cluster param without an extractor is refused when the task's config loads (see
    # test_workflow_inputs.py); the builder itself no longer guards against it.
