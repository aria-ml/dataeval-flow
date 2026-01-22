"""Factory module - simplified, internal use only."""

from dataeval_app.factory.tasks import (
    create_duplicates,
    create_outliers,
    create_outliers_from_params,
)

__all__ = [
    "create_duplicates",
    "create_outliers",
    "create_outliers_from_params",
]
