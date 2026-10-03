"""The scope evaluators' results: DataEval's own output objects, typed per evaluator."""

from collections.abc import Mapping, Sequence
from typing import Any

from dataeval.scope import CoverageOutput, PrioritizeOutput, RepresentationOutput

from dataeval_flow._blocks import Block
from dataeval_flow.evaluators._core import CoreOutput
from dataeval_flow.evaluators._result import EvaluatorResult

__all__ = [
    "CoverageResult",
    "LabelReconciliationOutput",
    "LabelReconciliationResult",
    "PrioritizeResult",
    "RepresentationResult",
]


class RepresentationResult(EvaluatorResult[RepresentationOutput]):
    """The result of a ``representation`` run: ``output`` is DataEval's ``RepresentationOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``RepresentationOutput``: ``data()`` is the worklist, one row per concept short of its target
        (``concept``, ``label``, ``parent``, ``action``, ``count``, ``target``, ``deficit``). ``leaf_coverage``,
        ``total_deficit``, ``violations`` and ``dark_branches`` are the summary ``to_dict()`` writes under ``extras``,
        and ``ignored_expected``, the ``expected`` names that resolve to no concept or to several, is a fifth.
        ``output.ontology_source`` is how the config named the ontology.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:  # noqa: ARG002
        """Leaf coverage, the deficit and how many concepts fall short."""
        from dataeval_flow.evaluators.scope._report import representation_section

        return representation_section(output)


class CoverageResult(EvaluatorResult[CoverageOutput]):
    """The result of a ``coverage`` run: ``output`` is DataEval's ``CoverageOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``CoverageOutput``: ``data()`` is the per-class table (``class``, ``count``, ``uncovered``,
        ``uncovered_fraction``, ``dispersion``, ``isotropy``, ``near_duplicate_fraction``, ``assessable``), and
        ``uncovered_indices``, ``coverage_radius`` and ``critical_value_radii`` are the source-wide results
        ``to_dict()`` writes under ``extras``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:
        """The uncovered items, pictured, then the per-class table and the rest."""
        from dataeval_flow.evaluators.scope._report import coverage_section

        return coverage_section(output, sources, detailed=detailed)


class PrioritizeResult(EvaluatorResult[PrioritizeOutput]):
    """The result of a ``prioritize`` run: ``output`` is DataEval's ``PrioritizeOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``PrioritizeOutput``, so its reorderings work: ``hard_first()``, ``easy_first()``, ``stratified()``
        and ``class_balanced()``. ``data()`` is the item indices in ranked order, which ``to_dict()`` writes as an
        array, with ``scores`` under ``extras``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:
        """The ranking's highest-priority and lowest-priority items, pictured."""
        from dataeval_flow.evaluators.scope._report import prioritize_section

        return prioritize_section(output, sources, detailed=detailed)


class LabelReconciliationOutput(CoreOutput):
    """``label-reconciliation``'s output: ``data()`` holds ``conforms``, ``matched`` (class name to concept id),
    ``unmatched`` (names no concept answers to) and ``ambiguous`` (each name several concepts answer to, with their
    ids)."""


class LabelReconciliationResult(EvaluatorResult[LabelReconciliationOutput]):
    """The result of a ``label-reconciliation`` run; ``output`` is a
    :class:`~dataeval_flow.evaluators.scope.LabelReconciliationOutput`.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        ``data()`` holds ``conforms``, true where no name is unmatched or ambiguous; ``matched``, each class name to
        the concept id it resolves to; ``unmatched``; and ``ambiguous``, each name to the ids of the concepts it
        names.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:  # noqa: ARG002
        """How many names matched, were unmatched, or were ambiguous."""
        from dataeval_flow._blocks import Fields

        data = output.get("data") or {}
        return [
            Fields(
                items=[
                    ("Matched", len(data.get("matched") or {})),
                    ("Unmatched", len(data.get("unmatched") or [])),
                    ("Ambiguous", len(data.get("ambiguous") or {})),
                ]
            )
        ]
