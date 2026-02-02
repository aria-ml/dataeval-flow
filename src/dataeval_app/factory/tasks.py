"""Task factory functions - simplified for internal use.

This module provides factory functions that create DataEval task instances.
The registry pattern has been removed in favor of direct instantiation.

Factory Functions:
    - create_outliers() -> Outliers detector
    - create_outliers_from_params() -> Outliers from DataCleaningParameters
    - create_duplicates() -> Duplicates detector
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dataeval.flags import ImageStats
from dataeval.quality import Duplicates, Outliers

if TYPE_CHECKING:
    from dataeval_app.ingest import DataCleaningParameters


def create_outliers(
    outlier_method: Literal["zscore", "modzscore", "iqr"] = "modzscore",
    outlier_use_dimension: bool = True,
    outlier_use_pixel: bool = True,
    outlier_use_visual: bool = True,
    outlier_threshold: float | None = None,
) -> Outliers:
    """Create Outliers detector with specified configuration.

    Parameters
    ----------
    outlier_method : Literal["zscore", "modzscore", "iqr"]
        Outlier detection method.
    outlier_use_dimension : bool
        Use dimension statistics for outlier detection.
    outlier_use_pixel : bool
        Use pixel statistics for outlier detection.
    outlier_use_visual : bool
        Use visual statistics for outlier detection.
    outlier_threshold : float | None
        Custom threshold. Uses DataEval default if None.

    Returns
    -------
    Outliers
        Configured Outliers detector instance.
    """
    flags = ImageStats.NONE
    if outlier_use_dimension:
        flags |= ImageStats.DIMENSION
    if outlier_use_pixel:
        flags |= ImageStats.PIXEL
    if outlier_use_visual:
        flags |= ImageStats.VISUAL

    return Outliers(
        flags=flags,
        outlier_method=outlier_method,
        outlier_threshold=outlier_threshold,
    )


def create_outliers_from_params(params: DataCleaningParameters) -> Outliers:
    """Create Outliers detector from DataCleaningParameters.

    Parameters
    ----------
    params : DataCleaningParameters
        Validated data cleaning parameters.

    Returns
    -------
    Outliers
        Configured Outliers detector instance.
    """
    return create_outliers(
        outlier_method=params.outlier_method,
        outlier_use_dimension=params.outlier_use_dimension,
        outlier_use_pixel=params.outlier_use_pixel,
        outlier_use_visual=params.outlier_use_visual,
        outlier_threshold=params.outlier_threshold,
    )


def create_duplicates() -> Duplicates:
    """Create Duplicates detector.

    Note: No `create_duplicates_from_params` is provided because duplicate
    detection uses sensible defaults (detect both exact and near duplicates)
    that don't require user configuration in DataCleaningParameters.

    Returns
    -------
    Duplicates
        Configured Duplicates detector instance.
    """
    return Duplicates()
