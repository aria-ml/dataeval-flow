"""The `factor-gaps` combine: class-factor-value combinations under-represented among the factors balance ties to the
class, legacy data-coverage's gap analysis as a step (coverage spec §6.1)."""

__all__ = ["FactorGap", "FactorGapsCombine", "FactorGapsConfig", "FactorGapsOutput", "find_gaps", "mi_from_balance"]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl
from dataeval.bias import BalanceOutput
from pydantic import BaseModel, Field

from dataeval_flow._input_spec import InputKind
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin
from dataeval_flow.steps._combine import Combine, CombineConfig, CombineContext
from dataeval_flow.steps._port import DataType, Port

if TYPE_CHECKING:
    from dataeval import Metadata

    from dataeval_flow._blocks import Block


class FactorGap(BaseModel):
    """One class-factor-value combination with too few items: legacy data-coverage's ``ClassMetadataGap``."""

    class_name: str = Field(description="The class short of the factor's value.")
    factor_name: str = Field(description="The factor.")
    factor_value: str = Field(description="The factor's value, as text.")
    class_count: int = Field(description="How many of the class's items have the value.")
    expected_count: float = Field(description="How many the factor's overall spread leads to expect, to one decimal.")
    deficit: float = Field(description="The share of the expected count missing, from 0 to 1, to three decimals.")


class FactorGapsOutput(BaseModel):
    """Each factor's mutual information with the class, and the gaps among the factors at or over `mi_threshold`,
    largest deficit first."""

    mutual_information: dict[str, float] = Field(description="Each scored factor's mutual information with the class.")
    gaps: list[FactorGap] = Field(description="The under-represented combinations, largest deficit first.")


class FactorGapsConfig(CombineConfig, MetadataConfigMixin):
    """A `factor-gaps` step's inputs and settings. Its `metadata:` policy should be the one `balance` read under."""

    input: str = Field(description="The Dataset whose Metadata the gaps are counted in.")
    balance: str = Field(description="A `balance` Output computed on exactly `input`.")
    mi_threshold: float = Field(
        default=0.1, ge=0.0, description="The least mutual information with the class a factor needs to be searched."
    )
    min_representation: int = Field(
        default=5,
        ge=1,
        description=("A combination is a gap where its count is under this while its expected count is over it."),
    )


def mi_from_balance(balance: BalanceOutput, factor_names: Sequence[str]) -> dict[str, float]:
    """Each named factor's mutual information with the class, read from Balance's class-to-factor rows; a factor
    Balance did not score is left out."""
    wanted = set(factor_names)
    return {
        str(row["factor_name"]): float(row["mi_value"])
        for row in balance.balance.to_dicts()
        if row["factor_name"] in wanted and row["mi_value"] is not None
    }


def _factor_gaps(
    fname: str, factor_col: pl.Series, class_labels: Any, label_map: Mapping[int, str], n_total: int, minimum: int
) -> list[FactorGap]:
    """The under-represented combinations of one factor's values with each class, as legacy data-coverage's gap analysis
    found them, values of equal count in value order, which legacy left to Polars and so to chance."""
    overall_vc = factor_col.value_counts().sort(["count", fname], descending=[True, False])
    if len(overall_vc) == 0:
        return []
    overall_dist: dict[Any, float] = {
        row[fname]: row["count"] / max(n_total, 1) for row in overall_vc.iter_rows(named=True)
    }
    gaps: list[FactorGap] = []
    for cls_id in sorted({int(c) for c in class_labels}):
        cls_mask = np.array(class_labels) == cls_id
        cls_size = int(cls_mask.sum())
        if cls_size == 0:
            continue
        cls_counts: dict[Any, int] = {
            row[fname]: row["count"]
            for row in factor_col.filter(pl.Series(cls_mask)).value_counts().iter_rows(named=True)
        }
        for fval, overall_prop in overall_dist.items():
            expected = overall_prop * cls_size
            actual = cls_counts.get(fval, 0)
            if actual < minimum and expected > minimum:
                deficit = 1.0 - (actual / max(expected, 1e-9))
                gaps.append(
                    FactorGap(
                        class_name=label_map.get(cls_id, str(cls_id)),
                        factor_name=fname,
                        factor_value=str(fval),
                        class_count=actual,
                        expected_count=round(expected, 1),
                        deficit=round(max(0.0, min(1.0, deficit)), 3),
                    )
                )
    return gaps


def find_gaps(
    metadata: "Metadata", mutual_information: Mapping[str, float], mi_threshold: float, minimum: int
) -> FactorGapsOutput:
    """The gaps among the factors at or over `mi_threshold`, counted at the metadata's label level, largest deficit
    first, as legacy data-coverage's gap analysis did."""
    mi = {name: mutual_information[name] for name in metadata.factor_names if name in mutual_information}
    df = metadata.rows_at(metadata.label_level)
    if df is None or len(df) == 0:
        return FactorGapsOutput(mutual_information=mi, gaps=[])
    class_labels = metadata.class_labels
    label_map = dict(metadata.index2label or {})
    gaps: list[FactorGap] = []
    for fname, score in mi.items():
        if score < mi_threshold or fname not in df.columns:
            continue
        gaps.extend(_factor_gaps(fname, df[fname], class_labels, label_map, len(class_labels), minimum))
    gaps.sort(key=lambda gap: gap.deficit, reverse=True)
    return FactorGapsOutput(mutual_information=mi, gaps=gaps)


class FactorGapsCombine(Combine[FactorGapsConfig]):
    """``factor-gaps``: the class-factor-value combinations under-represented among the factors balance ties to the
    class. It reads balance's mutual information, so it runs no Balance of its own (coverage spec §6.1)."""

    name: ClassVar[str] = "factor-gaps"
    title: ClassVar[str] = "Factor Gaps"
    description: ClassVar[str] = (
        "Class-factor-value combinations under-represented, among the factors tied to the class."
    )
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("input", DataType.DATASET, derives=frozenset({InputKind.METADATA})),
        Port("balance", DataType.OUTPUT, classes=(BalanceOutput,)),
    )
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(FactorGapsOutput,)),)
    same_node: ClassVar[tuple[str, ...]] = ("balance",)

    def run(self, config: FactorGapsConfig, inputs: Mapping[str, Any], context: CombineContext) -> Mapping[str, Any]:
        """The gaps in `input`'s Metadata, among the factors balance scored at or over `mi_threshold`."""
        metadata = context.derive_metadata(inputs["input"])
        mi = mi_from_balance(inputs["balance"].value, list(metadata.factor_names))
        return {"output": find_gaps(metadata, mi, config.mi_threshold, config.min_representation)}

    def section(self, record: Any) -> list["Block"]:
        """Each factor's mutual information with the class."""
        from dataeval_flow.evaluators.bias._report import ranked_table

        output: FactorGapsOutput = record.output
        if not output.mutual_information:
            return []
        return [ranked_table(output.mutual_information, headers=("Factor", "MI with the class"))]
