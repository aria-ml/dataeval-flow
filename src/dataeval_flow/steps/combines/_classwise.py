"""`classwise-outliers`: an Outliers Output pivoted by the labels of the Dataset it was computed on."""

__all__ = ["ClasswiseOutliers", "ClasswiseOutliersCombine", "ClasswiseOutliersConfig", "ClasswiseRow"]

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from dataeval.quality import OutliersOutput
from pydantic import BaseModel, Field

from dataeval_flow._classwise import classwise_pivot, split_outlier_issues
from dataeval_flow._input_spec import InputKind
from dataeval_flow.steps._combine import Combine, CombineConfig, CombineContext
from dataeval_flow.steps._port import DataType, Port


class ClasswiseRow(BaseModel):
    """One class's flagged items or boxes: how many, and what share of the class they are."""

    class_name: str = Field(description="The class, or `Total` for the row over every class.")
    count: int = Field(description="How many of its items, or boxes, were flagged.")
    pct: float = Field(description="That count as a percentage of the class's labels, to one decimal.")


class ClasswiseOutliers(BaseModel):
    """Outliers per class: how many of each class's items, or boxes, an Outliers Output flagged."""

    count_basis: Literal["image", "annotation"] = Field(
        description="What a row counts: `image` for classification, `annotation` (boxes) for detection."
    )
    rows: list[ClasswiseRow] = Field(description="One row per class with a flag, most flagged first.")
    total: ClasswiseRow | None = Field(description="The row over every class; `null` where nothing was flagged.")

    @classmethod
    def from_pivot(cls, pivot: Mapping[str, Any] | None, *, multi_target: bool) -> "ClasswiseOutliers":
        """The pivot :func:`~dataeval_flow._classwise.classwise_pivot` built, its last row the total."""
        if pivot is None:
            return cls(count_basis="annotation" if multi_target else "image", rows=[], total=None)
        *rows, total = pivot["rows"]
        return cls(
            count_basis=pivot["count_basis"],
            rows=[ClasswiseRow(**row) for row in rows],
            total=ClasswiseRow(**total),
        )


class ClasswiseOutliersConfig(CombineConfig):
    """A `classwise-outliers` step's inputs: the Dataset, and the Outliers Output computed on it."""

    input: str = Field(description="The Dataset the outliers were found in; its labels name each item's class.")
    outliers: str = Field(
        description=(
            "An `outliers` Output computed on exactly `input`; for a detection Dataset, with `per_target: true`."
        )
    )


class ClasswiseOutliersCombine(Combine[ClasswiseOutliersConfig]):
    """``classwise-outliers``: how many of each class's items, or boxes for detection, an Outliers Output flagged."""

    name: ClassVar[str] = "classwise-outliers"
    title: ClassVar[str] = "Outliers by Class"
    description: ClassVar[str] = "Pivots an Outliers Output by class: each class's flagged items or boxes."
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("input", DataType.DATASET, derives=frozenset({InputKind.METADATA})),
        Port("outliers", DataType.OUTPUT, classes=(OutliersOutput,)),
    )
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(ClasswiseOutliers,)),)
    same_node: ClassVar[tuple[str, ...]] = ("outliers",)

    def run(
        self,
        config: ClasswiseOutliersConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],
        context: CombineContext,
    ) -> Mapping[str, Any]:
        """The Outliers Output's flags, counted once per item or box, per class."""
        metadata = context.derive_metadata(inputs["input"])
        producer = inputs["outliers"].config
        if metadata.multi_target and producer is not None and getattr(producer, "per_target", True) is not True:
            raise ValueError(
                "classwise-outliers counts a detection Dataset's boxes, but `outliers` was not computed per box: "
                "set `per_target: true` on its `outliers` entry."
            )
        img_issues, target_issues = split_outlier_issues(inputs["outliers"].value.data())
        pivot = classwise_pivot(target_issues, img_issues, metadata)
        return {"output": ClasswiseOutliers.from_pivot(pivot, multi_target=metadata.multi_target)}
