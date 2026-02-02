"""Unit tests for simplified factory module."""

from dataeval.quality import Duplicates, Outliers

from dataeval_app.factory import (
    create_duplicates,
    create_outliers,
    create_outliers_from_params,
)
from dataeval_app.ingest import DataCleaningParameters


class TestCreateOutliers:
    """Test create_outliers factory function."""

    def test_default_parameters(self):
        """Outliers created with default parameters."""
        detector = create_outliers()
        assert isinstance(detector, Outliers)

    def test_custom_method(self):
        """Outliers created with custom method."""
        detector = create_outliers(outlier_method="iqr")
        assert isinstance(detector, Outliers)

    def test_custom_threshold(self):
        """Outliers created with custom threshold."""
        detector = create_outliers(outlier_threshold=2.5)
        assert isinstance(detector, Outliers)

    def test_disable_flags(self):
        """Outliers created with disabled flags."""
        detector = create_outliers(
            outlier_use_dimension=False,
            outlier_use_pixel=False,
            outlier_use_visual=True,
        )
        assert isinstance(detector, Outliers)


class TestCreateOutliersFromParams:
    """Test create_outliers_from_params factory function."""

    def test_from_params(self):
        """Outliers created from DataCleaningParameters."""
        params = DataCleaningParameters(
            outlier_method="modzscore",
            outlier_use_dimension=True,
            outlier_use_pixel=True,
            outlier_use_visual=True,
            outlier_threshold=None,
        )
        detector = create_outliers_from_params(params)
        assert isinstance(detector, Outliers)

    def test_from_params_with_threshold(self):
        """Outliers created from params with custom threshold."""
        params = DataCleaningParameters(
            outlier_method="iqr",
            outlier_use_dimension=False,
            outlier_use_pixel=True,
            outlier_use_visual=False,
            outlier_threshold=3.0,
        )
        detector = create_outliers_from_params(params)
        assert isinstance(detector, Outliers)


class TestCreateDuplicates:
    """Test create_duplicates factory function."""

    def test_default_parameters(self):
        """Duplicates created with default parameters."""
        detector = create_duplicates()
        assert isinstance(detector, Duplicates)
