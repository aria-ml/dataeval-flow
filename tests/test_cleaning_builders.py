"""Unit tests for cleaning workflow builder helpers (_build_outliers, _build_duplicates)."""

import pytest
from dataeval.quality import Duplicates, Outliers

from dataeval_app.workflows.cleaning.params import DataCleaningParameters
from dataeval_app.workflows.cleaning.workflow import (
    _build_duplicates,
    _build_outliers,
)


def _make_params(**overrides: object) -> DataCleaningParameters:
    """Build DataCleaningParameters with defaults for testing."""
    defaults: dict[str, object] = {
        "outlier_method": "adaptive",
        "outlier_flags": ["dimension", "pixel", "visual"],
        "outlier_threshold": None,
    }
    defaults.update(overrides)
    return DataCleaningParameters(**defaults)  # type: ignore[arg-type]


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

    def test_cluster_without_extractor_raises(self):
        """Cluster params without extractor raises ValueError."""
        params = _make_params(outlier_cluster_threshold=2.5)
        with pytest.raises(ValueError, match="requires an extractor"):
            _build_outliers(params)


class TestBuildOutliersFromParams:
    """Test building Outliers from DataCleaningParameters."""

    def test_from_params(self):
        """Outliers created from DataCleaningParameters."""
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

    def test_cluster_without_extractor_raises(self):
        """Cluster params without extractor raises ValueError."""
        params = _make_params(duplicate_cluster_sensitivity=2.5)
        with pytest.raises(ValueError, match="requires an extractor"):
            _build_duplicates(params)
