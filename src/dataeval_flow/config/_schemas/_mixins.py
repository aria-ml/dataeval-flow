"""Config mixins any workflow or evaluator config can take: a metadata policy, a stats policy."""

__all__ = ["MetadataConfigMixin", "StatsConfigMixin"]

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from dataeval_flow.config._schemas._task import AutoBinMethod, FactorSource


class MetadataConfigMixin(BaseModel):
    """Mixin for configs that read dataset metadata: which metadata policy they read it under.

    Provides binning and exclusion settings for metadata analysis. Mix into any workflow or
    evaluator config whose runs build metadata (``data-cleaning``'s does).
    """

    metadata: str | None = Field(
        default=None,
        description=(
            "Name of a policy defined under the top-level `metadata:` key. Preferred over "
            "the `metadata_*` fields below, which are kept for compatibility: a policy is "
            "defined once and shared, so workflows meant to be compared read their factors "
            "under one encoding rather than each spelling out its own."
        ),
    )

    metadata_auto_bin_method: AutoBinMethod | None = Field(
        default=None,
        description=(
            "How a continuous factor with no declared bins is cut: `uniform_width`, `uniform_count` or "
            "`clusters`. Kept for compatibility, and refused alongside `metadata`: prefer a policy's "
            "`auto_bin_method`."
        ),
    )
    metadata_exclude: Sequence[str] = Field(
        default_factory=list,
        description=(
            "Factor names removed before any evaluator sees them. Kept for compatibility, and refused alongside "
            "`metadata`: prefer a policy's `exclude`."
        ),
    )
    metadata_continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = Field(
        default=None,
        description=(
            "Bin count (int) or explicit edges (list) per continuous factor. Kept for compatibility, and refused "
            "alongside `metadata`: prefer a policy's `continuous_factor_bins`."
        ),
    )
    metadata_factor_source: FactorSource | None = Field(
        default=None,
        description=(
            "Which representation of each factor the bias statistics read. `coded` reads "
            "the integer codes binning produced; `values` reads the measurements, which "
            "recovers resolution a cut threw away at roughly 11x the cost; `auto` decides "
            "per factor, keeping codes wherever somebody declared or ratified the cut and "
            "reading values where nobody did. Leave unset for DataEval's default (`auto`). "
            "It governs every bias number a workflow reports, so two workflows meant to be "
            "compared want the same one."
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
