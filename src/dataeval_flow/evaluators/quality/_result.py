"""The quality evaluators' results: DataEval's own output objects, typed per evaluator."""

from collections.abc import Mapping, Sequence
from typing import Any

from dataeval.quality import DuplicatesOutput, OutliersOutput
from pydantic import BaseModel, Field

from dataeval_flow._blocks import Block
from dataeval_flow.evaluators._core import CoreOutput
from dataeval_flow.evaluators._result import EvaluatorResult

__all__ = [
    "ContentDigestOutput",
    "ContentDigestResult",
    "DuplicatesResult",
    "FactorTriageOutput",
    "FactorTriageResult",
    "LabelHealthOutput",
    "LabelHealthResult",
    "OutliersResult",
    "VerificationEntry",
]


class DuplicatesResult(EvaluatorResult[DuplicatesOutput[Any, Any]]):
    """The result of a ``duplicates`` run: ``output`` is DataEval's ``DuplicatesOutput``.

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
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:
        """Each duplicate group, largest first, its items named and pictured; then any extras."""
        from dataeval_flow.evaluators.quality._report import duplicate_section

        return duplicate_section(output, sources, detailed=detailed)


class OutliersResult(EvaluatorResult[OutliersOutput[Any]]):
    """The result of an ``outliers`` run: ``output`` is DataEval's ``OutliersOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``OutliersOutput``, so its own methods work: ``data()`` for the table of flagged items,
        ``aggregate_by_item()``, ``aggregate_by_metric()`` and the rest. Its JSON form is what ``to_dict()`` writes.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:  # noqa: ARG002
        """Each flagged image, then each flagged box, with every flag it raised and each metric's limits."""
        from dataeval_flow.evaluators.quality._report import outlier_section

        return outlier_section(output, sources)


class LabelHealthOutput(CoreOutput):
    """``label-health``'s output: how a Dataset's labels spread over its classes.

    ``data()`` holds:

    - ``item_count``: the items the labels come from, labelled or not;
    - ``class_count``: the classes the Dataset declares, used or not;
    - ``label_count``: the labels, one per item for classification and one per box for detection;
    - ``label_counts_per_class`` and ``image_counts_per_class``: by class name, for every declared class, at 0 where
      unseen;
    - ``empty_image_count`` and ``empty_image_indices``: the items with no label, counted and listed;
    - ``label_source``: where the labels came from, such as ``filepath``, or ``None`` where the source doesn't say.
    """


class LabelHealthResult(EvaluatorResult[LabelHealthOutput]):
    """The result of a ``label-health`` run; ``output`` is a
    :class:`~dataeval_flow.evaluators.quality.LabelHealthOutput`.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        The counts: ``data()`` holds ``item_count``, ``class_count``, ``label_count``, ``label_counts_per_class``,
        ``image_counts_per_class`` (each holding every declared class, at 0 where unseen), ``empty_image_count``,
        ``empty_image_indices`` and ``label_source``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:  # noqa: ARG002
        """The counts as fields, and a table of each class's labels and images."""
        from dataeval_flow.evaluators.quality._report import label_health_section

        return label_health_section(output)


class ContentDigestOutput(CoreOutput):
    """``content-digest``'s output: a Dataset's digests over every item.

    ``data()`` holds ``content``, the SHA-256 over each item's image and labels and the class names; ``metadata``, the
    SHA-256 over each item's metadata, as attached to that item; ``items``, how many items were read; and ``scheme``,
    the version of the digest scheme. Both digests are 64 hex characters, and neither depends on the items' order.
    """


class ContentDigestResult(EvaluatorResult[ContentDigestOutput]):
    """The result of a ``content-digest`` run; ``output`` is a
    :class:`~dataeval_flow.evaluators.quality.ContentDigestOutput`.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        The digests: ``data()`` holds ``content`` and ``metadata``, each 64 hex characters, ``items`` and ``scheme``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:  # noqa: ARG002
        """The item count and both digests, in full."""
        from dataeval_flow.evaluators.quality._report import content_digest_section

        return content_digest_section(output)


class VerificationEntry(BaseModel):
    """What one suggestion actually did when it was read back.

    ``recovered`` is the question worth asking.  A suggestion can be well-formed, run
    cleanly and still not work — upstream is explicit that a reading leaving every row
    holding its own value has not made the column a factor — so a stanza is worth checking
    before it is committed to a config.
    """

    factor: str = Field(description="The factor the suggestion repairs.")
    applied: bool = Field(description="Whether the suggestion was complete enough to apply.")
    recovered: bool = Field(description="Whether reading the metadata back under it recovered the factor.")
    detail: str = Field(description="One line saying what the reading produced.")


class FactorTriageOutput(CoreOutput):
    """``factor-triage``'s output: what a Dataset's metadata failed to read, and a policy that repairs it.

    ``data()`` holds:

    - ``findings``: each issue, worst first, a ``TriageFinding``: its ``factor``, ``category``, ``severity``,
      ``reasons``, ``remedy`` and ``detail``, and a ``suggestion`` where one repairs it;
    - ``suggested_policy``: every suggestion merged into one metadata policy body; ``suggested_policy_yaml``: the same
      as YAML, to paste under a config's ``metadata:`` key;
    - ``verification``: per suggestion, a :class:`VerificationEntry` saying whether reading the metadata back under it
      ``applied`` and ``recovered`` the factor; empty unless ``verify`` is on;
    - ``verification_error``: why verification raised, or ``None``. It tells a failed verification from
      ``verify: false`` and from nothing to verify, all of which leave ``verification`` empty;
    - ``recommended_policy``: the suggested fixes, each value triage could not read dropped to NaN, plus a pin for
      every factor the policy left unpinned, read from this data: explicit edges for a cut, as the binning record
      writes them (an infinite edge is the string ``"-inf"`` or ``"inf"``), levels for a vocabulary; ``None`` where
      the policy already pins everything and nothing was suggested. A ``floor_mass`` value is left in, not dropped.
      ``recommended_policy_yaml``: the same as YAML, headed by the caveat that a policy read from unrepresentative
      data can mislead;
    - ``recommendation_error``: why the read-back behind the recommendation raised, or ``None``;
    - ``counts``: how many issues fall in each category, and in each severity;
    - ``factor_count``: how many factors the metadata read;
    - ``places``: by factor, where each of a mixed column's problem values sits: the value, how many rows hold it, and
      the first of their items.
    """


class FactorTriageResult(EvaluatorResult[FactorTriageOutput]):
    """The result of a ``factor-triage`` run; ``output`` is a
    :class:`~dataeval_flow.evaluators.quality.FactorTriageOutput`.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        ``data()`` holds ``findings``, ``suggested_policy``, ``suggested_policy_yaml``, ``verification``,
        ``verification_error``, ``recommended_policy``, ``recommended_policy_yaml``, ``recommendation_error``,
        ``counts``, ``factor_count`` and ``places``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:  # noqa: ARG002
        """How many factors it read, and its issues by category and severity."""
        from dataeval_flow._triage_report import triage_section

        return triage_section(output)
