"""The ``feasibility`` evaluator's result: DataEval's own output object, typed, reported as DataEval prints it."""

__all__ = ["FeasibilityResult", "PopulatedFeasibilityOutput"]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dataeval.performance import FeasibilityOutput

from dataeval_flow._blocks import Block
from dataeval_flow.evaluators._result import EvaluatorResult


@dataclass(frozen=True, repr=False)
class PopulatedFeasibilityOutput(FeasibilityOutput):
    """DataEval's output, with the population Flow read it from, which ``data()`` leaves to the extras.

    DataEval states the population as "labeled items of <dataset>" and nothing more. These say how many items the
    source held, how many of them were labeled, and which declared classes no labeled item belongs to.
    """

    items: int = 0
    labeled: int = 0
    absent_classes: list[str] = field(default_factory=list)

    def data(self) -> dict[str, Any]:
        """DataEval's determinations and evidence alone, not the population beside them."""
        return {key: value for key, value in super().data().items() if key in ("determinations", "evidence")}

    def population(self) -> str:
        """The population every claim is about, in a line: the labeled items out of the source's, and the declared
        classes that have no labeled item, which are no part of any claim."""
        line = f"labeled items only, {self.labeled:,} of {self.items:,}"
        if self.absent_classes:
            line += f"; classes with no labeled item are in no claim: {', '.join(self.absent_classes)}"
        return line


class FeasibilityResult(EvaluatorResult[FeasibilityOutput]):
    """The result of a ``feasibility`` run: ``output`` is DataEval's ``FeasibilityOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``FeasibilityOutput``: ``determinations``, one row per target in the order given, and ``evidence``,
        every bound of the call by id, with its basis, population, information scope, endpoints and assumptions.
        ``to_dict()`` writes both under ``data``, as DataEval returns them, and under ``extras`` the population Flow
        read them from: ``items``, the source's item count, ``labeled``, how many of them have a label, and
        ``absent_classes``, the declared classes with no labeled item. ``str(output)`` is DataEval's printout, which
        states the confidence and the assumptions and rounds a lower bound down and an upper bound up. The report
        draws the ``performance-feasibility`` finding's table, and does not repeat the raw tables.
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
