"""The quality evaluators' results: DataEval's own output objects, typed per evaluator."""

from collections.abc import Mapping, Sequence
from typing import Any

from dataeval.quality import DuplicatesOutput, OutliersOutput

from dataeval_flow._blocks import Block
from dataeval_flow.evaluators._core import CoreOutput
from dataeval_flow.evaluators._result import EvaluatorResult

__all__ = ["DuplicatesResult", "LabelHealthOutput", "LabelHealthResult", "OutliersResult"]


class DuplicatesResult(EvaluatorResult[DuplicatesOutput[Any, Any]]):
    """The result of a ``quality.duplicates`` run: ``output`` is DataEval's ``DuplicatesOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``DuplicatesOutput``, so its own methods work: ``data()`` for the table of groups,
        ``aggregate_by_image()``, ``aggregate_by_group()`` and the rest. Its JSON form is what ``to_dict()`` writes,
        with ``annotation_divergences`` and ``factor_cardinality`` under ``extras``, ``null`` unless the annotation or
        factor axis ran.
    metadata.evaluator
        The evaluator type, e.g. ``quality.duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:
        """Each duplicate group, largest first, its items named and pictured; then any extras."""
        from dataeval_flow.evaluators.quality._report import duplicate_section

        return duplicate_section(output, sources, detailed=detailed)


class OutliersResult(EvaluatorResult[OutliersOutput[Any]]):
    """The result of a ``quality.outliers`` run: ``output`` is DataEval's ``OutliersOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``OutliersOutput``, so its own methods work: ``data()`` for the table of flagged items,
        ``aggregate_by_item()``, ``aggregate_by_metric()`` and the rest. Its JSON form is what ``to_dict()`` writes.
    metadata.evaluator
        The evaluator type, e.g. ``quality.duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:  # noqa: ARG002
        """Each flagged image, then each flagged box, with every flag it raised and each metric's limits."""
        from dataeval_flow.evaluators.quality._report import outlier_section

        return outlier_section(output, sources)


class LabelHealthOutput(CoreOutput):
    """``quality.label-health``'s output: how a Dataset's labels spread over its classes.

    ``data()`` holds:

    - ``item_count``: the items the labels come from, labelled or not;
    - ``class_count``: the classes the Dataset declares, used or not;
    - ``label_count``: the labels, one per item for classification and one per box for detection;
    - ``label_counts_per_class`` and ``image_counts_per_class``: by class name, for the classes that occur;
    - ``empty_image_count``: the items with no label;
    - ``label_source``: where the labels came from, such as ``filepath``, or ``None`` where the source doesn't say.
    """


class LabelHealthResult(EvaluatorResult[LabelHealthOutput]):
    """The result of a ``quality.label-health`` run; ``output`` is a
    :class:`~dataeval_flow.evaluators.quality.LabelHealthOutput`.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        The counts: ``data()`` holds ``item_count``, ``class_count``, ``label_count``, ``label_counts_per_class``,
        ``image_counts_per_class``, ``empty_image_count`` and ``label_source``.
    metadata.evaluator
        The evaluator type, e.g. ``quality.duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """
