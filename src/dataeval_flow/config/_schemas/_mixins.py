"""Config mixins any workflow or evaluator config can take: a metadata policy, a stats policy."""

__all__ = ["MetadataConfigMixin", "StatsConfigMixin"]

from pydantic import BaseModel, Field


class MetadataConfigMixin(BaseModel):
    """Mixin for configs that read dataset metadata: which metadata policy they read it under.

    Mix into any workflow or evaluator config whose runs build metadata (``quality``'s and ``balance``'s
    do). Flow resolves the named policy before the dataset is read, and builds the metadata under it.
    """

    metadata: str | None = Field(
        default=None,
        description=(
            "Name of a policy defined under the top-level `metadata:` key. A policy is defined once and shared, so "
            "entries meant to be compared read their factors under one encoding. Leave unset for DataEval's "
            "defaults."
        ),
    )


class StatsConfigMixin(BaseModel):
    """Mixin for configs that measure image statistics: which stats policy they measure under."""

    stats: str | None = Field(
        default=None,
        description=(
            "Name of a policy defined under the top-level `stats:` key. Declare one to measure named band "
            "groups or the image background, and to name the views outlier detection reads. Leave unset to "
            "measure the whole image."
        ),
    )
