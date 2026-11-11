"""The ``feasibility`` evaluator's result: DataEval's own output object, typed, reported as DataEval prints it."""

__all__ = ["FeasibilityResult", "PopulatedFeasibilityOutput", "requirement_label"]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dataeval.performance import FeasibilityOutput

from dataeval_flow._blocks import Block
from dataeval_flow.evaluators._result import EvaluatorResult


def requirement_label(
    metric: str, threshold: float, class_name: str | None, aggregation: str | None, scope: str
) -> str:
    """A requirement as DataEval's printout names it, built from its fields alone: ``accuracy:cat ≥0.9 |sequence_mean
    @fixed_representation``, the class, the aggregation and the scope shown only where they are not the default."""
    text = f"{metric}:{class_name}" if class_name is not None else metric
    text += f" ≥{threshold:g}"
    if aggregation is not None:
        text += f" |{aggregation}"
    return text if scope == "full_input" else f"{text} @{scope}"


@dataclass(frozen=True, repr=False)
class PopulatedFeasibilityOutput(FeasibilityOutput):
    """DataEval's output, with the population Flow read it from, which ``data()`` leaves to the extras.

    DataEval states the population as "labeled items of <dataset>" and nothing more. These say how many items the
    source held, how many of them were labeled, and which declared classes no labeled item belongs to. They also say
    how the items were grouped into the independent units the bounds count: how many groups there are among the
    source's items, how many items the largest holds, and how many fewer groups there are because exact or near
    duplicates joined them. ``seed`` is the seed DataEval resolved, which the config states only when the user set it.
    """

    seed: int | None = None
    items: int = 0
    labeled: int = 0
    absent_classes: list[str] = field(default_factory=list)
    groups: int = 0
    largest_group: int = 0
    duplicate_merges: int = 0

    def data(self) -> dict[str, Any]:
        """What DataEval returned, the requirements and the allowances beside the two tables, and not the population."""
        kept = ("requirements", "allowances", "determinations", "bounds")
        return {key: value for key, value in super().data().items() if key in kept}

    def population(self) -> str:
        """The population every claim is about, in a line: the labeled items out of the source's, and the declared
        classes that have no labeled item, which are no part of any claim."""
        line = f"labeled items only, {self.labeled:,} of {self.items:,}"
        if self.absent_classes:
            line += f"; classes with no labeled item are in no claim: {', '.join(self.absent_classes)}"
        return line

    def grouping(self) -> str:
        """The independent units the bounds count, in a line: how many there are among the source's items, the
        largest, and how many fewer there are because exact or near duplicates joined them."""
        if self.groups == self.items:
            return f"each of the {self.items:,} items is its own unit"
        line = f"{self.groups:,} independent units among {self.items:,} items, the largest of {self.largest_group:,}"
        if self.duplicate_merges:
            line += f"; exact and near duplicates account for {self.duplicate_merges:,} fewer units"
        return line


class FeasibilityResult(EvaluatorResult[FeasibilityOutput]):
    """The result of a ``feasibility`` run: ``output`` is DataEval's ``FeasibilityOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``FeasibilityOutput``: ``requirements`` and ``allowances``, what was asked and what was assumed;
        ``determinations``, one row per requirement in the order given; and ``bounds``, one row per bound behind them,
        with ``is_determining`` marking those a determination rests on. ``to_dict()`` writes all four under ``data``,
        as DataEval returns them, and under ``extras`` the population Flow read them from: ``items``, the source's item
        count, ``labeled``, how many of them have a label, ``absent_classes``, the declared classes with no labeled
        item, the grouping, and ``seed``, the seed DataEval resolved. ``str(output)`` is DataEval's printout, which
        states the confidence and the assumptions and rounds a lower bound down and an upper bound up. The report
        draws the ``performance-feasibility`` finding's tables, and does not repeat the raw ones.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:  # noqa: ARG002
        """Nothing: the `performance-feasibility` finding draws each target's answer and bounds, and the rest is in
        ``output`` and ``result.json``, so the report does not repeat it."""
        return []
