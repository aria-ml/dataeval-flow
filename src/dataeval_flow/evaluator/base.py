"""Evaluator framework base: the input vocabulary and the parameters base every evaluator shares.

Imports nothing heavier than pydantic. ``config.schemas`` imports every evaluator's
parameters, and this module sits beneath all of them.
"""

__all__ = ["EvaluatorParametersBase", "InputKind", "InputSpec", "SourceCount", "task_problem"]

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class InputKind(StrEnum):
    """What Flow hands a DataEval evaluator.

    The only vocabulary Flow's layers share about inputs: config validation, the producers,
    the registry and the CLI all key on it, and none of them names a DataEval method. The
    values are the names proposed for DataEval's own input protocols, so adopting those would
    be a table swap rather than a rewrite.
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
    """How many sources an evaluator's task must name."""

    ONE = "1"
    ONE_OR_MORE = "1+"
    ONE_OR_TWO = "1-2"
    TWO = "2"

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
}

_SOURCE_PHRASES: dict[SourceCount, str] = {
    SourceCount.ONE: "exactly one source",
    SourceCount.ONE_OR_MORE: "one or more sources",
    SourceCount.ONE_OR_TWO: "one or two sources",
    SourceCount.TWO: "exactly two sources",
}


@dataclass(frozen=True)
class InputSpec:
    """What an evaluator consumes, and how many sources feed it."""

    required: frozenset[InputKind]
    sources: SourceCount
    optional: frozenset[InputKind] = frozenset()

    @property
    def kinds(self) -> frozenset[InputKind]:
        """Every kind this evaluator can consume."""
        return self.required | self.optional

    @property
    def accepts_extractor(self) -> bool:
        """Whether any kind this evaluator can consume is produced with an extractor."""
        return any(kind.needs_extractor for kind in self.kinds)


class EvaluatorParametersBase(BaseModel):
    """Base class for every evaluator's parameters.

    Unknown keys are rejected. Each field is a DataEval argument spelled as DataEval spells
    it, so a misspelling has to fail the config load rather than silently run DataEval's
    default.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    #: What this evaluator consumes. Declared on the parameters rather than on the evaluator,
    #: so config validation can read it without importing the evaluator.
    inputs: ClassVar[InputSpec]

    def wanted_kinds(self) -> frozenset[InputKind]:
        """The kinds this run needs: the required ones, plus any optional ones these values switch on."""
        return self.inputs.required

    def requires_extractor(self) -> bool:
        """Whether this run needs an extractor on its task."""
        return any(kind.needs_extractor for kind in self.wanted_kinds())

    def source_problem(self, count: int) -> str | None:  # noqa: ARG002
        """A rule these values place on the source count beyond ``inputs.sources``, or ``None``."""
        return None


def task_problem(params: EvaluatorParametersBase, *, source_count: int, has_extractor: bool) -> str | None:
    """What is wrong with a task that runs *params*, or ``None`` when the task can run it.

    Checked when the config loads, so a task that cannot run costs a config error rather
    than a walk over the dataset. The returned phrase completes the sentence
    ``"Task 't' runs <type>, which ..."``.

    Parameters
    ----------
    params : EvaluatorParametersBase
        The evaluator entry the task references.
    source_count : int
        How many sources the task names.
    has_extractor : bool
        Whether the task names an extractor.

    Returns
    -------
    str or None
        The problem, or ``None``.
    """
    spec = params.inputs
    if not spec.sources.allows(source_count):
        return f"takes {spec.sources.phrase}, but the task names {source_count}."
    if (problem := params.source_problem(source_count)) is not None:
        return problem
    if params.requires_extractor() and not has_extractor:
        kinds = ", ".join(sorted(kind for kind in params.wanted_kinds() if kind.needs_extractor))
        return f"needs an extractor to produce {kinds}; name one with `extractor:`."
    if has_extractor and not spec.accepts_extractor:
        return "does not use an extractor; remove `extractor:` from the task."
    return None
