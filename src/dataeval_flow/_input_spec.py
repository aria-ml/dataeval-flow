"""The input vocabulary shared by workflows and evaluators: what a task's config consumes, and how many sources.

Private, at the top level alongside ``_kind.py``: both workflow and evaluator configs declare their ``inputs``
from this module, and imports nothing heavier than the standard library, so it sits beneath both kind packages.
"""

__all__ = ["InputKind", "InputSpec", "SourceCount"]

from dataclasses import dataclass
from enum import StrEnum


class InputKind(StrEnum):
    """What Flow hands a DataEval workflow or evaluator.

    The only vocabulary Flow's layers share about inputs: config validation, the producers,
    the registry and the CLI all key on it, and none of them names a DataEval method. The
    values use the names proposed for DataEval's own input protocols.
    """

    STATS = "stats"
    CLUSTERS = "clusters"
    METADATA = "metadata"
    LABELS = "labels"
    EMBEDDINGS = "embeddings"

    @property
    def needs_extractor(self) -> bool:
        """Whether producing this kind needs an extractor on the task."""
        return self in (InputKind.CLUSTERS, InputKind.EMBEDDINGS)


class SourceCount(StrEnum):
    """How many sources a task must name to run a given workflow or evaluator."""

    ONE = "1"
    ONE_OR_MORE = "1+"
    ONE_OR_TWO = "1-2"
    TWO = "2"
    TWO_OR_MORE = "2+"

    def allows(self, count: int) -> bool:
        """Whether a task naming *count* sources meets this rule."""
        low, high = _SOURCE_BOUNDS[self]
        return count >= low and (high is None or count <= high)

    @property
    def phrase(self) -> str:
        """The rule as an error message states it."""
        return _SOURCE_PHRASES[self]


_SOURCE_BOUNDS: dict[SourceCount, tuple[int, int | None]] = {
    SourceCount.ONE: (1, 1),
    SourceCount.ONE_OR_MORE: (1, None),
    SourceCount.ONE_OR_TWO: (1, 2),
    SourceCount.TWO: (2, 2),
    SourceCount.TWO_OR_MORE: (2, None),
}

_SOURCE_PHRASES: dict[SourceCount, str] = {
    SourceCount.ONE: "exactly one source",
    SourceCount.ONE_OR_MORE: "one or more sources",
    SourceCount.ONE_OR_TWO: "one or two sources",
    SourceCount.TWO: "exactly two sources",
    SourceCount.TWO_OR_MORE: "two or more sources",
}


@dataclass(frozen=True)
class InputSpec:
    """What a workflow or evaluator consumes, and how many sources feed it.

    A workflow or evaluator config declares one as its ``inputs`` class variable. Flow checks every task against
    it when the pipeline config loads, and again before a run: the task must name as many sources as ``sources``
    allows, and an extractor where a kind it wants is made with one.

    Examples
    --------
    >>> from dataeval_flow import InputKind, InputSpec, SourceCount
    >>> spec = InputSpec(
    ...     required=frozenset({InputKind.STATS}), sources=SourceCount.ONE, optional=frozenset({InputKind.CLUSTERS})
    ... )
    >>> spec.accepts_extractor
    True
    """

    required: frozenset[InputKind]
    """The kinds every run reads."""
    sources: SourceCount
    """How many sources a task must name."""
    optional: frozenset[InputKind] = frozenset()
    """The kinds a run reads only when its config's values switch them on, through the config's ``wanted_kinds``."""

    @property
    def kinds(self) -> frozenset[InputKind]:
        """Every kind this entry can consume."""
        return self.required | self.optional

    @property
    def accepts_extractor(self) -> bool:
        """Whether any kind this entry can consume is produced with an extractor."""
        return any(kind.needs_extractor for kind in self.kinds)
