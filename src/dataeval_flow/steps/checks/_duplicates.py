"""The duplicate check: data-cleaning's Duplicates finding, as a step (spec §9.2)."""

__all__ = ["DuplicateRateCheck", "DuplicateRateConfig"]

from collections.abc import Mapping
from typing import Any, ClassVar

import polars as pl
from dataeval.quality import DuplicatesOutput
from pydantic import Field

from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import exceeds
from dataeval_flow.workflows._base import Finding


class DuplicateRateConfig(CheckConfig):
    """A `duplicate-rate` step's input, and the shares of a Dataset that may be exact and near duplicates."""

    input: str = Field(description="A `quality.duplicates` Output.")
    exact: float | None = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Most images, as a percentage of the Dataset, that may sit in exact-duplicate groups before the finding "
            "warns; `null` judges nothing. data-cleaning's `health_thresholds.exact_duplicates`."
        ),
    )
    near: float | None = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description=(
            "Most images, as a percentage of the Dataset, that may sit in near-duplicate groups before the finding "
            "warns; `null` judges nothing. data-cleaning's `health_thresholds.near_duplicates`."
        ),
    )


class DuplicateRateCheck(Check[DuplicateRateConfig]):
    """``duplicate-rate``: warns when too many of a Dataset's images sit in exact or near duplicate groups.

    Counts the item-level ``exact`` and ``near`` groups, as data-cleaning does, and makes no finding where there are
    none.
    """

    name: ClassVar[str] = "duplicate-rate"
    description: ClassVar[str] = "Warns when more than `exact` or `near` percent of the images are duplicates."
    title: ClassVar[str] = "Duplicates"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(DuplicatesOutput,)),)

    def run(self, config: DuplicateRateConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The shares of the Dataset's images in exact and in near duplicate groups."""
        node = inputs["input"]
        items = node.value.data().filter(pl.col("level") == "item")
        exact = items.filter(pl.col("dup_type") == "exact")["item_indices"].to_list()
        near = items.filter(pl.col("dup_type") == "near")["item_indices"].to_list()
        if not exact and not near:
            return []
        exact_affected = sum(len(group) for group in exact)
        near_affected = sum(len(group) for group in near)
        size = node.items or 0
        exact_pct = exact_affected / size * 100 if size else 0.0
        near_pct = near_affected / size * 100 if size else 0.0
        warns = exceeds(exact_pct, config.exact) or exceeds(near_pct, config.near)
        return [
            Finding(
                severity="warning" if warns else "info",
                title=self.title,
                brief=f"{exact_affected} exact ({round(exact_pct, 1)}%), {near_affected} near ({round(near_pct, 1)}%)",
                description=f"{len(exact)} exact duplicate groups, {len(near)} near-duplicate groups found.",
            )
        ]
